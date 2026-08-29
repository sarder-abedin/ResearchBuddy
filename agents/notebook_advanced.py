"""
agents/notebook_advanced.py
───────────────────────────
Phase-2 advanced features for Mode 8 Research Notebook.

Each public function follows the same contract:
  feature(notebook_id, settings) -> (result, error_string)

An empty error_string means success.  All functions load the notebook from
NotebookMemory (JSON) and call the local Ollama LLM — no cloud dependencies.

Features
────────
  generate_cross_document_summary   Unified synthesis of all sources
  generate_faq                      Auto-generated Q&A pairs with citations
  generate_literature_review        Academic-style structured review
  generate_mindmap                  Concept tree → Graphviz DOT string
  generate_audio_summary            Spoken-word script (for TTS or playback)
  compare_sources                   Side-by-side analysis of two sources
  extract_knowledge_graph           Entity–relationship graph → DOT string
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents.notebook_memory import NotebookMemory
from config.settings import get_settings
from tools.citation_network import get_paper_abstract
from tools.search_tools import search_arxiv, search_semantic_scholar
from tools.temperature_levels import DEFAULT_TEMPERATURE_LEVEL, apply_temperature_level
from tools.text_parsing import (
    extract_references_section,
    format_page_label,
    join_chunks_with_headings,
)
from tools.writing_style import ANTI_AI_TELL_INSTRUCTION, ANTI_AI_TELL_NARRATIVE_INSTRUCTION, ANTI_AI_TELL_REVIEWER_INSTRUCTION

logger = logging.getLogger(__name__)
cfg = get_settings()

_MAX_CHARS_PER_DOC = 6_000   # chars sent to LLM per source
_MAX_TOTAL_CHARS = 20_000    # hard ceiling for the whole context block


# ── LLM factory ──────────────────────────────────────────────────────────────

def _max_predict(settings: dict) -> int:
    """Reserve 25% of context for the prompt; use the rest for output (min 4096)."""
    return max(4096, int(settings.get("num_ctx", cfg.num_ctx) * 0.75))


def _make_llm(settings: dict, temperature: float = 0.3, num_predict: int = 4096) -> ChatOllama:
    """Build a ChatOllama client whose temperature is adjusted by the user's response-tuning level."""
    import httpx
    from config.observability import get_langfuse_callbacks
    level = settings.get("temperature_level", DEFAULT_TEMPERATURE_LEVEL)
    return ChatOllama(
        model=settings.get("model", cfg.ollama_model),
        base_url=cfg.ollama_base_url,
        temperature=apply_temperature_level(temperature, level),
        num_predict=num_predict,
        num_ctx=settings.get("num_ctx", cfg.num_ctx),
        sync_client_kwargs={"timeout": httpx.Timeout(300.0)},
        callbacks=get_langfuse_callbacks(),
    )


def _invoke(llm: ChatOllama, system: str, human: str) -> str:
    """Invoke the LLM with a system/human message pair and return the stripped text content."""
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return resp.content.strip()


# ── Source context builder ────────────────────────────────────────────────────

def _sources_context(
    notebook: Dict[str, Any],
    max_chars_per_doc: int = _MAX_CHARS_PER_DOC,
    max_total_chars: int = _MAX_TOTAL_CHARS,
) -> str:
    """
    Build a numbered source block from all stored chunks.  Each source is
    capped at *max_chars_per_doc* and the whole block is capped at
    *max_total_chars* so we never blow the LLM context window.
    """
    sources = notebook.get("sources", [])
    chunks = notebook.get("chunks", [])

    by_doc: Dict[str, List[str]] = {}
    for ch in chunks:
        content_type = ch.get("content_type", "text")
        if content_type == "table" and ch.get("table_md"):
            body = "[TABLE]\n" + ch["table_md"].strip()
        elif content_type == "figure":
            body = "[FIGURE]\n" + ch.get("text", "")
        else:
            body = ch.get("text", "")
        by_doc.setdefault(ch["doc_id"], []).append(body)

    parts: List[str] = []
    total_chars = 0
    for i, src in enumerate(sources, 1):
        remaining = max_total_chars - total_chars
        if remaining <= 0:
            break
        cap = min(max_chars_per_doc, remaining)
        combined = " ".join(by_doc.get(src["doc_id"], []))
        excerpt = combined[:cap] + ("…" if len(combined) > cap else "")
        parts.append(f"[Source {i}: {src['filename']}]\n{excerpt}")
        total_chars += len(excerpt)

    return "\n\n".join(parts)


def _src_char_budget(settings: dict, reserved_output_tokens: int, reserved_prompt_chars: int = 1200) -> int:
    """Compute safe source-context char budget given the model's context window.

    Uses ~4 chars/token. Reserves space for output and prompt overhead so the
    combined input never overflows a small context window (e.g. 4 096 tokens
    for nemotron3:33b on 16 GB VRAM).
    """
    num_ctx = settings.get("num_ctx", cfg.num_ctx)
    available_tokens = max(0, num_ctx - reserved_output_tokens) - reserved_prompt_chars // 4
    return max(1500, min(_MAX_TOTAL_CHARS, available_tokens * 4))


def _build_numbered_excerpts(
    notebook: Dict[str, Any],
    max_chars_per_doc: int = _MAX_CHARS_PER_DOC,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build a context block with one numbered, page-tagged tag per CHUNK rather
    than per document — mirroring the Chat tab's citation convention
    (notebook_nodes.py::_build_context_block) instead of _sources_context's
    one-tag-per-document scheme.

    Without page-level tags, a multi-page source only ever has one citable
    number, so an LLM writing a "formal literature review" still invents its
    own per-claim bracket numbers (academic habit) that don't correspond to
    anything real — and then can't write an accurate References list for
    them either. Numbering by chunk gives the model real, distinct, citable
    excerpts, and lets the caller rebuild an accurate References list from
    whichever numbers it actually used (see _build_references_section)
    instead of trusting its self-written one.

    Returns (context_block, excerpts) where excerpts[i] is the chunk backing
    tag [i+1].
    """
    sources = notebook.get("sources", [])
    chunks = notebook.get("chunks", [])

    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for ch in chunks:
        by_doc.setdefault(ch["doc_id"], []).append(ch)

    excerpts: List[Dict[str, Any]] = []
    lines: List[str] = []
    total_chars = 0
    for src in sources:
        doc_chunks = sorted(by_doc.get(src["doc_id"], []), key=lambda c: c.get("chunk_index", 0))
        doc_chars = 0
        for ch in doc_chunks:
            if doc_chars >= max_chars_per_doc or total_chars >= _MAX_TOTAL_CHARS:
                break
            content_type = ch.get("content_type", "text")
            if content_type == "table" and ch.get("table_md"):
                type_tag = " [TABLE]"
                text = ch["table_md"].strip()
            elif content_type == "figure":
                type_tag = " [FIGURE]"
                text = ch.get("text", "").strip()
            else:
                type_tag = ""
                text = ch.get("text", "").strip()
            excerpts.append(ch)
            page_label = format_page_label(ch.get("page_num"))
            doc_name = ch.get("doc_name") or src.get("filename", "unknown")
            lines.append(f"[{len(excerpts)}] (source: {doc_name}, {page_label}){type_tag}\n{text}")
            doc_chars += len(text)
            total_chars += len(text)

    return "\n\n".join(lines), excerpts


_REF_HEADING_RE = re.compile(
    r"\n+(?:#{1,4}\s*References\b.*|\*\*References\*\*:?.*)",
    re.IGNORECASE | re.DOTALL,
)


def _strip_llm_references_section(body: str) -> str:
    """Cut off any References/Bibliography section the LLM wrote on its own.

    The model is told not to write one (see generate_literature_review's
    CITATION RULES), but instructions aren't guarantees — this is a
    defensive backstop so a stray self-written list never ends up coexisting
    with the accurate, code-generated one appended afterward.
    """
    match = _REF_HEADING_RE.search(body)
    return body[: match.start()].rstrip() if match else body.rstrip()


def _build_references_list(body: str, excerpts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rebuild the References list from the excerpt numbers actually cited in
    *body*, instead of trusting the LLM to write its own — which is what let
    inline citations like [1]-[7] coexist with a References list collapsed
    to one inaccurate, undifferentiated line.

    Structured output (one {"n", "doc_name", "page", "snippet", "doc_id"}
    dict per reference) for the live Streamlit view's snippet-expander UI —
    see ui/tabs/notebook.py::_render_citations. For a flattened single
    string (CLI / .md / .docx / .pdf export), pass the result through
    references_list_to_markdown().
    """
    cited_nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", body)})
    refs: List[Dict[str, Any]] = []
    for n in cited_nums:
        if 1 <= n <= len(excerpts):
            ch = excerpts[n - 1]
            refs.append({
                "n": n,
                "doc_name": ch.get("doc_name", "unknown"),
                "page": ch.get("page_num"),
                "snippet": ch.get("text", ""),
                "doc_id": ch.get("doc_id", ""),
            })
    if not refs:
        seen: List[str] = []
        for ch in excerpts:
            name = ch.get("doc_name", "unknown")
            if name not in seen:
                seen.append(name)
                refs.append({
                    "n": None,
                    "doc_name": name,
                    "page": None,
                    "snippet": "",
                    "doc_id": ch.get("doc_id", ""),
                })
    return refs


def references_list_to_markdown(refs: List[Dict[str, Any]]) -> str:
    """Flatten _build_references_list()'s structured output into the same
    '## References' Markdown block the pre-snippet-expander UI used to embed
    directly in the review body — for export paths (CLI / .md / .docx /
    .pdf) that need one complete standalone document rather than the live
    view's interactive expander."""
    lines: List[str] = []
    for ref in refs:
        if ref.get("n") is not None:
            page_label = format_page_label(ref.get("page"))
            lines.append(f"[{ref['n']}] {ref.get('doc_name', 'unknown')} ({page_label})")
        else:
            lines.append(f"- {ref.get('doc_name', 'unknown')}")
    return "## References\n" + "\n".join(lines)


# ── DOT helpers ───────────────────────────────────────────────────────────────

def _safe_dot(text: str, maxlen: int = 40) -> str:
    """Escape / trim a string for safe use in a Graphviz DOT label."""
    cleaned = re.sub(r'["\\\n\r\t]', " ", str(text)).strip()
    return cleaned[:maxlen]


def _mindmap_to_dot(data: Dict[str, Any]) -> str:
    """Convert the LLM mind-map JSON to a Graphviz DOT string."""
    central = _safe_dot(data.get("central", "Main Topic"), 60)
    branches = data.get("branches", [])

    lines = [
        "digraph mindmap {",
        '  graph [rankdir=LR, bgcolor="#0f172a", pad=0.4];',
        '  node [style=filled, fontname="Helvetica", fontsize=11, fontcolor=white];',
        f'  "root" [label="{central}", shape=ellipse, '
        f'fillcolor="#3b82f6", fontsize=14];',
    ]
    for i, branch in enumerate(branches):
        concept = _safe_dot(branch.get("concept", f"Branch {i}"))
        bid = f"b{i}"
        lines.append(
            f'  "{bid}" [label="{concept}", shape=box, fillcolor="#1e40af"];'
        )
        lines.append(f'  "root" -> "{bid}" [color="#60a5fa"];')
        for j, sub in enumerate(branch.get("sub_concepts", [])[:4]):
            sid = f"b{i}s{j}"
            lines.append(
                f'  "{sid}" [label="{_safe_dot(sub)}", shape=box, '
                f'fillcolor="#374151", fontsize=10];'
            )
            lines.append(f'  "{bid}" -> "{sid}" [color="#6b7280"];')
    lines.append("}")
    return "\n".join(lines)


def _knowledge_graph_to_dot(data: Dict[str, Any]) -> str:
    """Convert knowledge-graph JSON to a Graphviz DOT string."""
    type_colors: Dict[str, str] = {
        "concept": "#3b82f6",
        "method": "#10b981",
        "dataset": "#f59e0b",
        "author": "#8b5cf6",
        "institution": "#ef4444",
    }
    default_color = "#6b7280"

    lines = [
        "digraph knowledge_graph {",
        '  graph [bgcolor="#0f172a", rankdir=TB, pad=0.4];',
        '  node [style=filled, fontname="Helvetica", fontsize=10, fontcolor=white];',
        '  edge [fontname="Helvetica", fontsize=9, '
        'fontcolor="#9ca3af", color="#4b5563"];',
    ]
    for node in data.get("nodes", [])[:20]:
        nid = _safe_dot(node.get("id", ""))
        label = _safe_dot(node.get("label", nid))
        color = type_colors.get(node.get("type", ""), default_color)
        lines.append(f'  "{nid}" [label="{label}", fillcolor="{color}"];')
    for edge in data.get("edges", [])[:25]:
        src = _safe_dot(edge.get("from", ""))
        dst = _safe_dot(edge.get("to", ""))
        lbl = _safe_dot(edge.get("label", ""))
        lines.append(f'  "{src}" -> "{dst}" [label="{lbl}"];')
    lines.append("}")
    return "\n".join(lines)


def _parse_json_from_llm(raw: str) -> Any:
    """Extract a JSON array from an LLM response (array-first)."""
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    return json.loads(raw)


def _parse_json_object_from_llm(raw: str) -> Any:
    """Extract a JSON object from an LLM response (object-first, for mind map / KG)."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return json.loads(raw)


# ── Feature 1: Cross-document summary ────────────────────────────────────────

def generate_cross_document_summary(
    notebook_id: str, settings: dict
) -> Tuple[str, str]:
    """
    Synthesize all notebook sources into a unified markdown summary.

    For a single source: comprehensive summary with key points, methodology,
    and implications.
    For multiple sources: synthesis with common themes, complementary
    contributions, contradictions, and key takeaways.

    Returns (markdown_string, error_string).
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    sources = notebook.get("sources", [])
    if not sources:
        return "", "This notebook has no sources. Add documents first."

    context = _sources_context(notebook)

    if len(sources) == 1:
        system = (
            "You are a research analyst. Summarize the key points, methodology, "
            "findings, and implications of the provided source. "
            "Use clear markdown headings (##). Be thorough but concise.\n"
            + ANTI_AI_TELL_INSTRUCTION
        )
        human = f"SOURCE:\n{context}\n\nWrite a comprehensive summary in markdown."
    else:
        src_list = ", ".join(s["filename"] for s in sources)
        system = (
            "You are a research analyst synthesizing multiple documents.\n"
            "Write a cross-document synthesis in markdown with these sections:\n"
            "## Overview\nWhat the sources collectively cover.\n"
            "## Common Themes\nShared ideas, findings, or methods.\n"
            "## Complementary Contributions\nHow the sources add to each other.\n"
            "## Contradictions & Gaps\nWhere sources disagree or leave open questions.\n"
            "## Key Takeaways\n3–5 bullet conclusions.\n\n"
            "Attribute claims to specific sources by filename.\n"
            + ANTI_AI_TELL_INSTRUCTION
        )
        human = (
            f"SOURCES: {src_list}\n\n{context}\n\n"
            "Write the cross-document synthesis in markdown."
        )

    try:
        result = _invoke(_make_llm(settings, temperature=0.3, num_predict=_max_predict(settings)), system, human)
        return result, ""
    except Exception as e:
        logger.error("Cross-document summary failed: %s", e)
        return "", f"Summary generation failed: {e}"


# ── Feature 2: FAQ generation ─────────────────────────────────────────────────

def generate_faq(
    notebook_id: str,
    settings: dict,
    n_questions: int = 8,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Auto-generate FAQ items grounded in notebook sources.

    Returns (list_of_faq_dicts, error_string).
    Each dict: {"question": str, "answer": str, "sources": List[int]}.
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return [], f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return [], "No sources in this notebook."

    context = _sources_context(notebook)

    system = (
        f"You are a research expert. Based on the provided sources, generate "
        f"{n_questions} frequently asked questions with grounded answers.\n\n"
        "Output ONLY valid JSON — a JSON array with no surrounding text:\n"
        "[\n"
        '  {"question": "...", "answer": "...", "sources": [1, 2]},\n'
        "  ...\n"
        "]\n\n"
        "Rules:\n"
        "- Questions must cover the most important concepts in the sources.\n"
        "- Answers must be grounded ONLY in the provided sources.\n"
        "- 'sources' is an array of 1-based source numbers that support the answer.\n"
        "- Output ONLY the JSON array. No preamble, no code fences."
    )
    human = f"SOURCES:\n{context}\n\nGenerate {n_questions} FAQ items as a JSON array."

    raw = ""
    try:
        raw = _invoke(_make_llm(settings, temperature=0.3, num_predict=4096), system, human)
        items = _parse_json_from_llm(raw)
        if not isinstance(items, list):
            return [], "FAQ response was not a JSON array."
        return [i for i in items if isinstance(i, dict)], ""
    except json.JSONDecodeError as e:
        logger.error("FAQ JSON parse failed: %s | raw: %.200s", e, raw)
        return [], f"FAQ parsing failed: {e}"
    except Exception as e:
        logger.error("FAQ generation failed: %s", e)
        return [], f"FAQ generation failed: {e}"


# ── Feature 3: Literature review ──────────────────────────────────────────────

def generate_literature_review(
    notebook_id: str, settings: dict
) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Generate a formal academic-style literature review from notebook sources.

    Returns (review_body_markdown, references_list, error_string), where
    review_body_markdown excludes the References section (sections 1-6 only)
    and references_list is the structured form consumed by the live
    Streamlit snippet-expander view — see _build_references_list(). Callers
    that need a single flattened document (CLI / .md / .docx / .pdf export)
    should append _references_list_to_markdown(references_list).
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", [], f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return "", [], "No sources in this notebook."

    source_names = ", ".join(s["filename"] for s in notebook["sources"])
    context, excerpts = _build_numbered_excerpts(notebook)

    system = (
        "You are an academic researcher writing a formal literature review.\n"
        "The SOURCES below are numbered excerpts, each tagged with its source "
        "filename and page number.\n"
        "Structure the review in markdown with these sections:\n"
        "# Literature Review\n"
        "## 1. Introduction\n"
        "State the scope and purpose of this review.\n"
        "## 2. Background & Context\n"
        "Key background concepts established across the sources.\n"
        "## 3. Methodological Approaches\n"
        "Research methods and approaches described in the sources.\n"
        "## 4. Key Findings & Evidence\n"
        "Major findings organized thematically, attributed to source filenames.\n"
        "## 5. Critical Analysis\n"
        "Strengths, limitations, and gaps in the reviewed literature.\n"
        "## 6. Conclusion\n"
        "Synthesis of contributions and directions for future work.\n\n"
        "CITATION RULES:\n"
        "- Cite every claim inline with the bracketed excerpt number it came "
        "from, e.g. \"...reduces error [2].\" You may cite multiple excerpts "
        "for one claim, like [1][3].\n"
        "- Never cite a number that was not provided in SOURCES.\n"
        "- Do NOT write your own References or Bibliography section — one is "
        "generated automatically from the excerpt numbers you cite.\n"
        "Use formal academic tone.\n"
        + ANTI_AI_TELL_INSTRUCTION
    )
    human = (
        f"SOURCES: {source_names}\n\n{context}\n\n"
        "Write the formal literature review in markdown, sections 1-6 only "
        "(no References section)."
    )

    try:
        result = _invoke(_make_llm(settings, temperature=0.2, num_predict=_max_predict(settings)), system, human)
        body = _strip_llm_references_section(result)
        return body, _build_references_list(body, excerpts), ""
    except Exception as e:
        logger.error("Literature review generation failed: %s", e)
        return "", [], f"Literature review generation failed: {e}"


# ── Feature 4: Mind map ───────────────────────────────────────────────────────

def generate_mindmap(notebook_id: str, settings: dict) -> Tuple[str, str]:
    """
    Extract key concepts from notebook sources and return a Graphviz DOT string
    suitable for ``st.graphviz_chart``.

    Returns (dot_string, error_string).
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return "", "No sources in this notebook."

    # Use a smaller context for concept extraction — we need breadth, not depth.
    context = _sources_context(
        notebook, max_chars_per_doc=1_500,
        max_total_chars=_src_char_budget(settings, reserved_output_tokens=1024),
    )

    system = (
        "You are a knowledge analyst. Extract key concepts from the sources.\n"
        "Output ONLY minified single-line JSON (no newlines, no extra spaces) — no code fences, no other text.\n"
        'Example: {"central":"Main Topic","branches":[{"concept":"Branch","sub_concepts":["Sub1","Sub2"]}]}\n\n'
        "Rules:\n"
        "- Maximum 5 branches; maximum 3 sub-concepts per branch.\n"
        "- Labels: 2–4 words, no special characters.\n"
        "- Output ONLY the minified JSON object on a single line."
    )
    human = f"SOURCES:\n{context}\n\nExtract the mind map JSON."

    raw = ""
    try:
        raw = _invoke(_make_llm(settings, temperature=0.2, num_predict=1024), system, human)
        data = _parse_json_object_from_llm(raw)
        dot = _mindmap_to_dot(data)
        return dot, ""
    except json.JSONDecodeError as e:
        logger.error("Mind map JSON parse failed: %s | raw: %.200s", e, raw)
        return "", f"Mind map parsing failed: {e}"
    except Exception as e:
        logger.error("Mind map generation failed: %s", e)
        return "", f"Mind map generation failed: {e}"


# ── Feature 5: Audio summary ──────────────────────────────────────────────────

def generate_audio_summary(notebook_id: str, settings: dict) -> Tuple[str, str]:
    """
    Generate a spoken-word summary script — natural language optimised for
    text-to-speech playback (~300 words, ~2 minutes).

    Returns (script_text, error_string).  The text contains no markdown
    formatting so it reads cleanly when converted to audio.
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return "", "No sources in this notebook."

    context = _sources_context(notebook, max_chars_per_doc=2_000)
    nb_name = notebook.get("name", "Notebook")

    system = (
        "You are creating a spoken audio summary script. "
        "Write in clear, natural spoken language that sounds good when read aloud.\n\n"
        "Rules:\n"
        "- No markdown formatting — no #, *, _, backticks, or bullet dashes.\n"
        "- Short, complete sentences.\n"
        "- Natural spoken transitions: First, Additionally, Furthermore, Finally.\n"
        "- Approximately 280 to 320 words — about 2 minutes when read at a natural pace.\n"
        "- Start with an introduction (what this notebook covers).\n"
        "- End with a clear conclusion that ties everything together.\n"
        "- Do not list source filenames — integrate the content naturally.\n"
        + ANTI_AI_TELL_NARRATIVE_INSTRUCTION
    )
    human = (
        f'Create an audio summary script for a notebook called "{nb_name}".\n\n'
        f"SOURCES:\n{context}\n\n"
        "Write the spoken-word audio script now:"
    )

    try:
        result = _invoke(_make_llm(settings, temperature=0.5, num_predict=2048), system, human)
        return result, ""
    except Exception as e:
        logger.error("Audio summary generation failed: %s", e)
        return "", f"Audio summary generation failed: {e}"


# ── Feature 6: Source comparison ─────────────────────────────────────────────

def compare_sources(
    notebook_id: str,
    doc_id_a: str,
    doc_id_b: str,
    settings: dict,
) -> Tuple[str, str]:
    """
    Generate a side-by-side markdown comparison of two notebook sources.

    Returns (comparison_markdown, error_string).
    """
    if doc_id_a == doc_id_b:
        return "", "Please select two different sources to compare."

    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    src_map = {s["doc_id"]: s for s in notebook.get("sources", [])}
    src_a = src_map.get(doc_id_a)
    src_b = src_map.get(doc_id_b)
    if not src_a or not src_b:
        return "", "One or both selected sources were not found in this notebook."

    by_doc: Dict[str, List[str]] = {}
    for ch in notebook.get("chunks", []):
        by_doc.setdefault(ch["doc_id"], []).append(ch.get("text", ""))

    text_a = (" ".join(by_doc.get(doc_id_a, [])))[:_MAX_CHARS_PER_DOC]
    text_b = (" ".join(by_doc.get(doc_id_b, [])))[:_MAX_CHARS_PER_DOC]

    system = (
        "You are a research analyst comparing two documents.\n"
        "Write a detailed comparison in markdown with these sections:\n\n"
        "## Source Comparison\n\n"
        "### Overview\nOne paragraph per source describing its focus and main argument.\n\n"
        "### Common Ground\nShared themes, findings, or methods.\n\n"
        "### Unique Contributions\n"
        "Use a markdown table with columns: Aspect | Source A | Source B\n\n"
        "### Contradictions\nWhere the sources disagree or present conflicting evidence.\n\n"
        "### Synthesis\n"
        "How the two sources complement each other and what combined insight they provide."
    )
    human = (
        f"SOURCE A: {src_a['filename']}\n{text_a}\n\n"
        f"SOURCE B: {src_b['filename']}\n{text_b}\n\n"
        "Write the detailed comparison in markdown."
    )

    try:
        result = _invoke(_make_llm(settings, temperature=0.3, num_predict=_max_predict(settings)), system, human)
        return result, ""
    except Exception as e:
        logger.error("Source comparison failed: %s", e)
        return "", f"Source comparison failed: {e}"


# ── Feature 7: Knowledge graph ────────────────────────────────────────────────

def extract_knowledge_graph(notebook_id: str, settings: dict) -> Tuple[str, str]:
    """
    Extract entities and relationships from notebook sources and return a
    Graphviz DOT string suitable for ``st.graphviz_chart``.

    Returns (dot_string, error_string).
    Node types: concept | method | dataset | author | institution
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return "", "No sources in this notebook."

    context = _sources_context(
        notebook, max_chars_per_doc=1_500,
        max_total_chars=_src_char_budget(settings, reserved_output_tokens=1024),
    )

    system = (
        "You are a knowledge graph extractor.\n"
        "Output ONLY minified single-line JSON (no newlines, no extra spaces) — no code fences, no other text.\n"
        'Example: {"nodes":[{"id":"n1","label":"Entity","type":"concept"}],"edges":[{"from":"n1","to":"n2","label":"uses"}]}\n\n'
        "Rules:\n"
        "- Maximum 10 nodes, maximum 12 edges.\n"
        "- Node types: concept, method, dataset, author, institution.\n"
        "- Label: 2–4 words. No special characters.\n"
        "- Edge labels: 1–3 word verb phrases (uses, builds on, contradicts).\n"
        "- Output ONLY the minified JSON object on a single line."
    )
    human = f"SOURCES:\n{context}\n\nExtract the knowledge graph JSON."

    raw = ""
    try:
        raw = _invoke(_make_llm(settings, temperature=0.2, num_predict=1024), system, human)
        data = _parse_json_object_from_llm(raw)
        dot = _knowledge_graph_to_dot(data)
        return dot, ""
    except json.JSONDecodeError as e:
        logger.error("Knowledge graph JSON parse failed: %s | raw: %.200s", e, raw)
        return "", f"Knowledge graph parsing failed: {e}"
    except Exception as e:
        logger.error("Knowledge graph extraction failed: %s", e)
        return "", f"Knowledge graph extraction failed: {e}"


# ── Utility: DOT → raster/vector render ─────────────────────────────────────

def render_dot_bytes(dot_string: str, fmt: str = "png") -> Tuple[bytes, str]:
    """
    Render a Graphviz DOT string to the requested format.

    Parameters
    ----------
    dot_string : valid Graphviz DOT source
    fmt        : "png" | "svg" | "pdf"

    Returns (image_bytes, error_string).  Requires the *graphviz* Python package
    and the graphviz system tools (``apt install graphviz``).
    """
    try:
        import graphviz as gv
        src = gv.Source(dot_string)
        return src.pipe(format=fmt), ""
    except ImportError:
        return b"", "graphviz Python package not installed (pip install graphviz)."
    except Exception as e:
        logger.error("DOT render to %s failed: %s", fmt, e)
        return b"", f"Rendering to {fmt} failed: {e}"


# ── Utility: text → WAV audio ────────────────────────────────────────────────

def synthesize_speech(text: str, rate: int = 150) -> Tuple[bytes, str]:
    """
    Convert *text* to a WAV audio file using pyttsx3 (offline TTS).

    Returns (wav_bytes, error_string).  Requires the *pyttsx3* Python package
    and the espeak-ng system package (``apt install espeak-ng``).
    """
    try:
        import os
        import tempfile

        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "tts.wav")
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                return b"", "TTS engine produced no output. Is espeak-ng installed?"
            return open(tmp_path, "rb").read(), ""
    except ImportError:
        return b"", "pyttsx3 not installed (pip install pyttsx3)."
    except Exception as e:
        logger.error("TTS synthesis failed: %s", e)
        return b"", f"Speech synthesis failed: {e}"


# ── Feature 8: Citation timeline ─────────────────────────────────────────────

_REFERENCES_MAX_CHARS = 6_000   # chars of a source's bibliography sent to the LLM
_REFS_PER_SOURCE = 20           # max bibliography entries requested per source
_MAX_TOTAL_REFS = 30            # hard cap on the merged timeline

_YEAR_RE = re.compile(r"(\d{4})")


def _year_key(item: Dict[str, Any]) -> int:
    """Sort key for timeline items: ascending by 4-digit year, undated last."""
    m = _YEAR_RE.search(str(item.get("year", "")))
    return int(m.group(1)) if m else 9999


def extract_citation_timeline(
    notebook_id: str, enrich_with_abstracts: bool, settings: dict
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Build a chronological timeline of the works cited by the notebook's
    sources, by parsing each source's references/bibliography section.

    Returns (list_of_timeline_items, error_string). Each item:
      {"year": str, "title": str, "authors": str, "gist": str,
       "source": int, "url": str}

    By default each cited work's "gist" is a one-line summary the LLM infers
    from its title alone. If *enrich_with_abstracts* is True, each title is
    first looked up on Semantic Scholar and its TL;DR/abstract is used
    instead, falling back to the title-only gist for lookups that fail.
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return [], f"Notebook '{notebook_id}' not found."

    sources = notebook.get("sources", [])
    if not sources:
        return [], "No sources in this notebook."

    chunks = notebook.get("chunks", [])
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for ch in chunks:
        by_doc.setdefault(ch["doc_id"], []).append(ch)

    refs_by_source: List[List[Dict[str, Any]]] = []
    for i, src in enumerate(sources, 1):
        # Joined with paragraph breaks (not spaces) so a "References" heading
        # that lands at the end of one chunk still has a line break after it
        # for extract_references_section()'s heading regex to match, and via
        # join_chunks_with_headings() so a heading Docling stored in chunk
        # metadata rather than chunk text is restored before the search.
        combined = join_chunks_with_headings(by_doc.get(src["doc_id"], []))
        refs_section = extract_references_section(combined)[:_REFERENCES_MAX_CHARS]
        if not refs_section:
            continue

        system = (
            "You are extracting bibliography entries from an academic paper's "
            "references section.\n"
            "Output ONLY a JSON array — no code fences, no other text:\n"
            "[\n"
            '  {"year": "2017", "authors": "Smith et al.", "title": "Paper title"},\n'
            "  ...\n"
            "]\n\n"
            "Rules:\n"
            f"- Extract up to {_REFS_PER_SOURCE} distinct reference entries.\n"
            '- "year" is the 4-digit publication year, or "n.d." if not stated.\n'
            '- "authors" is a short author list, e.g. "Smith et al." or "Smith & Jones".\n'
            '- "title" is the cited work\'s title, without surrounding quotes.\n'
            "- Output ONLY the JSON array."
        )
        human = (
            f"REFERENCES SECTION (Source {i}: {src['filename']}):\n{refs_section}\n\n"
            "Extract the bibliography entries as a JSON array."
        )

        try:
            raw = _invoke(_make_llm(settings, temperature=0.1, num_predict=2048), system, human)
            items = _parse_json_from_llm(raw)
            if not isinstance(items, list):
                continue
            parsed = [
                {
                    "year": str(it.get("year") or "n.d."),
                    "authors": str(it.get("authors") or ""),
                    "title": str(it["title"]),
                    "gist": "",
                    "source": i,
                    "url": "",
                }
                for it in items[:_REFS_PER_SOURCE]
                if isinstance(it, dict) and it.get("title")
            ]
            if parsed:
                refs_by_source.append(parsed)
        except Exception as e:
            logger.debug("Reference extraction failed for source %d (%s): %s",
                          i, src.get("filename"), e)
            continue

    if not refs_by_source:
        return [], (
            "No references/bibliography section could be found or parsed in "
            "any source. Citation Timeline needs a references list at the "
            "end of at least one document."
        )

    # Round-robin merge so no single source's bibliography crowds out the rest.
    all_refs: List[Dict[str, Any]] = []
    for group in itertools.zip_longest(*refs_by_source):
        for item in group:
            if item is not None:
                all_refs.append(item)
        if len(all_refs) >= _MAX_TOTAL_REFS:
            break
    all_refs = all_refs[:_MAX_TOTAL_REFS]

    if enrich_with_abstracts:
        for item in all_refs:
            title = item["title"].strip()
            if not title:
                continue
            try:
                meta = get_paper_abstract(title)
            except Exception as e:
                logger.debug("Abstract enrichment failed for '%s': %s", title[:60], e)
                meta = None
            if meta:
                gist = (meta.get("tldr") or meta.get("abstract", "")[:240]).strip()
                if gist:
                    item["gist"] = gist
                if meta.get("year"):
                    item["year"] = str(meta["year"])
                if meta.get("url"):
                    item["url"] = meta["url"]
            time.sleep(0.4)

    needs_gist = [item for item in all_refs if not item["gist"]]
    if needs_gist:
        titles_block = "\n".join(
            f"{idx + 1}. {item['title']}" for idx, item in enumerate(needs_gist)
        )
        system_gist = (
            "You are a research assistant. For each numbered title below, write "
            "ONE short sentence (under 20 words) guessing its key idea or "
            "contribution, based on the title and your general knowledge if you "
            "recognize the work.\n"
            "Output ONLY a JSON array of strings, same order, no code fences: "
            '["gist 1", "gist 2", ...]'
        )
        human_gist = f"TITLES:\n{titles_block}\n\nOutput the JSON array of one-line gists."
        try:
            raw = _invoke(_make_llm(settings, temperature=0.4, num_predict=2048), system_gist, human_gist)
            gists = _parse_json_from_llm(raw)
            if isinstance(gists, list):
                for item, gist in zip(needs_gist, gists):
                    if isinstance(gist, str) and gist.strip():
                        item["gist"] = gist.strip()
        except Exception as e:
            logger.debug("Gist batch generation failed: %s", e)

    all_refs.sort(key=_year_key)
    return all_refs, ""


# ── Feature 9: Study comparison table ────────────────────────────────────────

def generate_study_comparison(notebook_id: str, settings: dict) -> Tuple[str, str]:
    """
    Generate a structured comparison table across all notebook sources —
    comparing research type, sample/data scope, methodology, key findings,
    and limitations.

    Returns (markdown_table_with_synthesis, error_string).
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    if not notebook.get("sources"):
        return "", "No sources in this notebook."

    source_names = [s["filename"] for s in notebook["sources"]]
    col_headers = " | ".join(f"**{n[:20]}**" for n in source_names)
    context = _sources_context(notebook)

    system = (
        "You are a systematic review analyst. Create a comparison table across all sources.\n\n"
        "Generate a markdown table with these rows and one column per source:\n"
        "| Dimension | Source 1 | Source 2 | ... |\n"
        "|-----------|----------|----------|\n"
        "Include rows for:\n"
        "- Research / Study type\n"
        "- Sample size / Data scope\n"
        "- Key methodology\n"
        "- Primary findings\n"
        "- Limitations\n"
        "- Year / Period\n\n"
        "After the table, write a **Synthesis** paragraph:\n"
        "- Strongest points of agreement\n"
        "- Most significant differences\n"
        "- What the sources collectively establish\n\n"
        "Use the exact source filenames as column headers."
    )
    human = (
        f"SOURCES: {', '.join(source_names)}\n\n{context}\n\n"
        "Generate the structured comparison table followed by the Synthesis paragraph."
    )

    try:
        result = _invoke(_make_llm(settings, temperature=0.2, num_predict=_max_predict(settings)), system, human)
        return result, ""
    except Exception as e:
        logger.error("Study comparison table generation failed: %s", e)
        return "", f"Study comparison table generation failed: {e}"


# ── Feature 10: IEEE-style paper reviewer ────────────────────────────────────

_REVIEWER_SEARCH_MAX = 3   # papers returned per search query
_REVIEWER_QUERIES = 3      # number of search queries generated from the critique


def _build_single_doc_context(notebook: Dict[str, Any], doc_id: str) -> Tuple[str, str]:
    """Return (context_text, filename) for one document from a loaded notebook."""
    src_map = {s["doc_id"]: s for s in notebook.get("sources", [])}
    src = src_map.get(doc_id)
    if not src:
        return "", ""
    by_doc: Dict[str, List[str]] = {}
    for ch in notebook.get("chunks", []):
        content_type = ch.get("content_type", "text")
        if content_type == "table" and ch.get("table_md"):
            body = "[TABLE]\n" + ch["table_md"].strip()
        elif content_type == "figure":
            body = "[FIGURE]\n" + ch.get("text", "")
        else:
            body = ch.get("text", "")
        by_doc.setdefault(ch["doc_id"], []).append(body)
    combined = " ".join(by_doc.get(doc_id, []))
    excerpt = combined[:_MAX_CHARS_PER_DOC] + ("…" if len(combined) > _MAX_CHARS_PER_DOC else "")
    return excerpt, src["filename"]


def generate_paper_review(
    notebook_id: str,
    doc_id: str,
    settings: dict,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Generate an IEEE-style peer review of a single uploaded paper.

    Steps:
    1. Loads the specified document's chunks from the notebook.
    2. Extracts 3 search queries from the document (topic, methodology, claimed contributions).
    3. Searches arXiv + Semantic Scholar; assigns [E1]–[En] reference numbers.
    4. Generates a structured critique grounded in BOTH the paper text AND the external
       paper abstracts — the LLM cites [En] inline wherever external evidence supports or
       contradicts a critique point.

    Returns (review_markdown, external_refs_list, error_string).
    Each external_ref dict: {ref_num, title, authors, year, url, source, abstract_snippet}.
    """
    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", [], f"Notebook '{notebook_id}' not found."

    context, filename = _build_single_doc_context(notebook, doc_id)
    if not context:
        return "", [], f"Document '{doc_id}' not found in this notebook."

    # Step 1: Extract search queries from the document before writing the review.
    system_queries = (
        "You are a literature search assistant helping a peer reviewer prepare for critique.\n"
        "Read the paper below and identify the 3 most important topics where published "
        "literature can substantiate or challenge a rigorous critique — e.g. the core "
        "methodology or algorithm, the claimed contribution or novelty, and the key "
        "experimental baselines or assumptions.\n"
        "For each topic, write ONE short academic search query (5–10 words) suitable for "
        "arXiv and Semantic Scholar.\n"
        "Output ONLY a JSON array of 3 query strings, no code fences:\n"
        '["query 1", "query 2", "query 3"]'
    )
    human_queries = (
        f"PAPER: {filename}\n\n{context[:4000]}\n\n"
        "Output the JSON array of 3 search queries."
    )

    search_queries: List[str] = []
    try:
        raw_q = _invoke(_make_llm(settings, temperature=0.0, num_predict=256), system_queries, human_queries)
        parsed = _parse_json_from_llm(raw_q)
        if isinstance(parsed, list):
            search_queries = [str(q).strip() for q in parsed if q][:_REVIEWER_QUERIES]
    except Exception as e:
        logger.debug("Search query extraction failed: %s", e)

    # Step 2: Search arXiv + Semantic Scholar; assign [E1]–[E9] reference numbers.
    seen_titles: set = set()
    external_refs: List[Dict[str, Any]] = []

    for query in search_queries:
        try:
            for paper in search_arxiv(query, max_results=_REVIEWER_SEARCH_MAX):
                key = paper.title.lower()[:60]
                if key not in seen_titles:
                    seen_titles.add(key)
                    external_refs.append({
                        "title": paper.title,
                        "authors": paper.authors[:3],
                        "year": paper.year,
                        "url": paper.url,
                        "source": "arXiv",
                        "abstract_snippet": paper.abstract[:300] if paper.abstract else "",
                    })
        except Exception:
            pass
        try:
            for paper in search_semantic_scholar(query, max_results=_REVIEWER_SEARCH_MAX):
                key = paper.title.lower()[:60]
                if key not in seen_titles:
                    seen_titles.add(key)
                    external_refs.append({
                        "title": paper.title,
                        "authors": paper.authors[:3],
                        "year": paper.year,
                        "url": paper.url,
                        "source": "Semantic Scholar",
                        "abstract_snippet": paper.abstract[:300] if paper.abstract else "",
                    })
        except Exception:
            pass
        if len(external_refs) >= 9:
            break

    external_refs = external_refs[:9]
    for i, ref in enumerate(external_refs):
        ref["ref_num"] = f"E{i + 1}"

    # Build the external reference block that goes into the review prompt.
    ext_block = _build_external_ref_block(external_refs)

    # Step 3: Generate the full review with both doc text and external abstracts.
    n_refs = len(external_refs)
    ref_range = f"[E1]–[E{n_refs}]" if n_refs > 1 else "[E1]" if n_refs == 1 else ""
    cite_rule = (
        f"- When citing external references use only {ref_range} as defined above. "
        "Cite inline immediately after the claim, e.g. [E2].\n"
        if n_refs else ""
    )
    system_review = (
        "You are an expert peer reviewer for a top-tier journal or conference. "
        "Your role is to act as a guardian of research quality — objective, fair, thorough, "
        "constructive, and evidence-based. Evaluate the paper below with the rigour, depth, "
        "and precision expected at a top-tier venue. Do not soften criticism — identify real "
        "problems clearly and directly, but always pair critique with a constructive suggestion.\n\n"
        + (
            "You have been given the paper text AND a set of external reference papers "
            f"{ref_range} retrieved from arXiv and Semantic Scholar.\n"
            "Use those external references to ground your critique:\n"
            "- If a methodology is non-standard, cite the correct established approach [En].\n"
            "- If a baseline is missing or outdated, name it and cite where it is used [En].\n"
            "- If a novelty claim overlaps with prior work, cite that prior work [En].\n"
            "- If an assumption is unjustified, cite literature that demonstrates its limits [En].\n"
            "- If a mathematical form is wrong, cite the reference that states the correct form [En].\n"
            "Only cite an external reference when it directly and specifically supports the point.\n\n"
            if n_refs else ""
        )
        + "Structure your review EXACTLY as follows (use these headings verbatim):\n\n"
        "## Summary\n"
        "2–3 sentences: what the paper claims to do and what its main contribution is. "
        "No evaluation here — pure description.\n\n"
        "## Strengths\n"
        "Numbered list. Each strength must be specific and backed by evidence from the "
        "text (section, equation, or result). No generic praise.\n\n"
        "## Weaknesses\n"
        "Numbered list, ordered from most to least critical. Each weakness is concrete, "
        "actionable, and tied to a specific location in the paper. Format: state the "
        "problem → cite the evidence → explain why it matters → suggest a concrete fix. "
        "Distinguish major issues (affect validity or claims) from minor ones (presentation). "
        "Cite external references [En] where they support the weakness.\n\n"
        "## Detailed Critique\n\n"
        "### 1. Novelty & Originality\n"
        "Is this a genuinely new contribution or an incremental variation of prior work? "
        "Evaluate: (a) what is truly new versus already established, (b) whether novelty "
        "claims are overstated — name the specific claim and cite the prior work that "
        "pre-empts it [En], (c) whether the contribution is incremental or transformative, "
        "(d) how the work advances the state of the art beyond existing methods.\n\n"
        "### 2. Significance & Impact\n"
        "Evaluate: (a) Is the research problem important? Why does it matter? "
        "(b) What is the potential academic impact — does this open new research directions? "
        "(c) What is the practical/industrial/societal impact? "
        "(d) Is the contribution substantial enough for a top-tier venue, or would it be "
        "better suited to a workshop or lower-tier venue? Justify your position with "
        "evidence from the paper and from external literature [En].\n\n"
        "### 3. Technical Soundness & Methodology\n"
        "This is the most important section. Examine the following explicitly:\n"
        "  a) MATHEMATICAL CORRECTNESS — Check every equation, derivation, proof, or "
        "theorem. Identify any error, incorrect step, missing condition, or unjustified "
        "simplification. Quote the expression and explain precisely what is wrong. "
        "Cite the correct form from external references [En].\n"
        "  b) LOGICAL VALIDITY — Identify any logical violation: circular reasoning, "
        "non-sequitur conclusions, invalid inference from data to claim, unsupported "
        "generalisations, or contradictions between the paper's own statements. Quote "
        "the passage and name the specific logical flaw.\n"
        "  c) MISLEADING OR INCORRECT CLAIMS — Flag any statement that is factually "
        "wrong, misleading, cherry-picked, or overgeneralised from narrow experiments. "
        "Provide the corrected fact or cite the contradicting source [En].\n"
        "  d) ASSUMPTIONS — Are all assumptions stated explicitly? Are they justified? "
        "Could they fail in practice? Name each critical assumption and evaluate it. "
        "Cite references that demonstrate limits or support [En].\n\n"
        "### 4. Experimental & Methodological Rigor\n"
        "Evaluate: (a) Are experiments comprehensive and well-designed? "
        "(b) Are baselines appropriate, fair, and up-to-date? Name important missing "
        "baselines and cite them from external references [En]. "
        "(c) Are metrics well-chosen and correctly applied? "
        "(d) Is statistical significance reported and valid? "
        "(e) Are ablation studies sufficient to isolate the individual contribution of each component? "
        "(f) Are datasets adequate, publicly available, and fairly used? "
        "(g) Is the work reproducible — are hyperparameters, code, and experimental details "
        "sufficient to replicate the results?\n\n"
        "### 5. Related Work Coverage\n"
        "Does the paper adequately survey the field? Are key prior works cited and fairly "
        "compared? Name missing important works from the external references [En] and "
        "explain what gap their omission creates in the narrative.\n\n"
        "### 6. Clarity & Presentation\n"
        "Note specific sections, figures, or tables that are unclear, ambiguous, or poorly "
        "structured (cite section numbers or headings). Evaluate whether figures and tables "
        "effectively communicate the key results. Identify any missing information that "
        "readers would need.\n\n"
        "## Questions for the Authors\n"
        "List 3–6 specific, well-formulated questions directed at the authors. Each question "
        "must: (a) quote or reference the specific passage, equation, or result that prompted "
        "it; (b) be genuinely clarifying — address a real ambiguity, missing detail, or "
        "unstated assumption in the paper as written; (c) be constructive — a good answer "
        "would improve the paper. Do NOT ask leading or rhetorical questions. "
        "Examples of well-formed questions: "
        "'In equation (4), the authors assume X — could they justify this assumption or show "
        "it holds under the experimental conditions?'; "
        "'The ablation in Table 2 isolates component A but not B — could the authors provide "
        "this result to confirm the individual contribution of B?'.\n\n"
        "## Recommendation\n"
        "State exactly one of: **Accept** / **Weak Accept** / **Borderline** / **Weak Reject** / **Reject**.\n"
        "Follow with 4–5 sentences of rationale that: (1) summarise the most critical findings "
        "from the Detailed Critique, (2) weigh strengths against weaknesses, (3) state what "
        "specific changes would be needed to move the recommendation up one level"
        + (" and (4) cite at least one external reference [En] where relevant." if n_refs else ".")
        + "\n\n"
        "CONTENT RULES:\n"
        "- Ground every critique in specific evidence: quote equations, cite sections, "
        "reference result tables — never refer abstractly to 'the methodology' or 'the experiments'.\n"
        + cite_rule
        + "- If you find no mathematical error, explicitly state that — do not omit the subsection.\n"
        "- Strengths and weaknesses must not repeat each other.\n"
        "- Each weakness must be accompanied by a concrete suggestion for improvement.\n"
        "- Within critique points, embed inline questions to the authors wherever a specific "
        "clarification would strengthen the critique — e.g., 'The authors claim X (Section 3) "
        "without stating how Y was controlled — could they clarify?' These inline questions "
        "are distinct from the 'Questions for the Authors' section and belong inside the "
        "Weaknesses or Detailed Critique where they are most relevant.\n"
        "- The Recommendation must follow logically from the Detailed Critique — do not "
        "contradict your own critique.\n"
        "- Distinguish between critical issues (affect validity of claims) and minor issues "
        "(presentation, typos, style) — critical issues carry more weight in the recommendation.\n"
        + ANTI_AI_TELL_REVIEWER_INSTRUCTION
    )
    human_review = (
        f"PAPER: {filename}\n\n"
        f"PAPER TEXT:\n{context}\n\n"
        + (f"{ext_block}\n\n" if ext_block else "")
        + "Write the full peer review now"
        + (f", citing {ref_range} inline where external references support the critique." if n_refs else ".")
    )

    try:
        review_text = _invoke(
            _make_llm(settings, temperature=0.2, num_predict=_max_predict(settings)),
            system_review,
            human_review,
        )
    except Exception as e:
        logger.error("Paper review generation failed: %s", e)
        return "", [], f"Paper review generation failed: {e}"

    return review_text, external_refs, ""


def _build_external_ref_block(external_refs: List[Dict[str, Any]]) -> str:
    """Format external references as a numbered block for LLM prompts."""
    if not external_refs:
        return ""
    lines = [
        "EXTERNAL REFERENCES "
        "(cite as [E1], [E2], … where they support or contradict a critique point):\n"
    ]
    for ref in external_refs:
        authors_str = ", ".join(ref.get("authors", [])[:3]) or "Unknown"
        year_str = f" ({ref['year']})" if ref.get("year") else ""
        snippet = ref.get("abstract_snippet", "")
        lines.append(
            f"[{ref['ref_num']}] {ref['title']}{year_str}. {authors_str}. "
            f"Source: {ref['source']}.\n"
            f"  Abstract: {snippet}\n"
        )
    return "\n".join(lines)


def reviewer_chat(
    notebook_id: str,
    doc_id: str,
    review_text: str,
    chat_history: List[Dict[str, str]],
    user_message: str,
    settings: dict,
    external_refs: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    """
    Expert peer discussion within the Reviewer tab.

    Both the user and the agent are domain experts on the paper's subject matter.
    They discuss the paper as intellectual peers — raising points, challenging
    claims, probing reasoning, and exploring the evidence together.  The agent
    contributes independent analysis, agrees or disagrees based on evidence,
    and can introduce new angles that the user has not yet raised.

    The full chat history is persisted in NotebookMemory so the discussion can
    continue across page reloads.

    chat_history is a list of {"role": "user"|"assistant", "content": "..."} dicts
    (server-persisted history; full history sent on every call).

    Returns (response_text, error_string).
    """
    from langchain_core.messages import AIMessage

    mem = NotebookMemory()
    notebook = mem.load(notebook_id)
    if not notebook:
        return "", f"Notebook '{notebook_id}' not found."

    context, filename = _build_single_doc_context(notebook, doc_id)
    if not context:
        return "", f"Document '{doc_id}' not found in this notebook."

    ext_block = _build_external_ref_block(external_refs or [])

    system = (
        f"You are a senior researcher and domain expert engaging in a collegial peer "
        f"discussion about the paper '{filename}' with another expert — your co-reviewer. "
        "Both of you have deep knowledge of the subject matter and are reviewing this "
        "paper together as intellectual equals. The goal is a rigorous, evidence-driven "
        "discussion that helps both of you reach well-grounded conclusions about the "
        "paper's quality, correctness, novelty, and significance.\n\n"
        "HOW TO ENGAGE:\n"
        "1. Respond as a peer expert, not as an assistant. You may agree, disagree, "
        "qualify, or extend what your co-reviewer says — always with specific evidence "
        "from the paper text or external references.\n"
        "2. Bring your own expert perspective: raise aspects your co-reviewer has not yet "
        "noticed, point out implications they may have missed, or suggest how a given "
        "weakness connects to a deeper methodological issue.\n"
        "3. When your co-reviewer proposes a critique, engage analytically:\n"
        "   a. If it is accurate and well-grounded: confirm it, cite the specific evidence, "
        "and add your own supporting analysis or extend the point further.\n"
        "   b. If it is partially correct: say precisely what holds and what does not, and "
        "why — do not soften the disagreement.\n"
        "   c. If it is unsupported or incorrect: push back directly with specific "
        "counter-evidence quoted from the paper or from an external reference [En]. "
        "A good co-reviewer does not flatter — they sharpen each other's thinking.\n"
        "4. For mathematical and formal aspects of the paper:\n"
        "   a. When an equation, derivation, or proof is under discussion, check it against "
        "the paper text for internal coherence (dimensional consistency, index notation, "
        "boundary conditions, whether the stated results follow from the assumptions).\n"
        "   b. If a formulation is ambiguous, engage with the ambiguity explicitly — quote "
        "the expression and share your interpretation, then ask your co-reviewer how they "
        "read it.\n"
        "   c. When you spot a mathematical error, explore it jointly rather than just "
        "announcing it: ask your co-reviewer what they think happens when a specific "
        "condition is changed (e.g., 'What does equation (3) give when X → 0?'). This "
        "is the hallmark of expert collegial discussion.\n"
        "   d. Distinguish computational errors (wrong arithmetic, sign flips) from "
        "conceptual errors (flawed assumptions, misapplied theorems) — both matter but "
        "for different reasons, and an expert knows which is which.\n"
        "5. Keep the following reviewer considerations active throughout the discussion:\n"
        "   • Is the problem genuinely important to the field?\n"
        "   • Is the claimed contribution truly novel relative to prior work?\n"
        "   • Are the methods technically sound?\n"
        "   • Do the experiments validate the core claims?\n"
        "   • Are critical baselines present and fair?\n"
        "   • Is the work reproducible?\n"
        "   • Are the limitations honestly acknowledged?\n"
        "   • Is the impact substantial enough for the target venue?\n"
        "6. Distinguish critical issues (affect validity of claims) from minor ones "
        "(presentation, style). Both matter; make clear which is which.\n"
        "7. When citing external references, use their [En] labels.\n\n"
        "TONE: Collegial, direct, and evidence-driven. The best expert discussions are "
        "intellectually honest — neither deferential nor combative. Treat your co-reviewer "
        "as a smart peer who benefits from precise pushback as much as from agreement.\n\n"
        f"PAPER EXCERPT:\n{context[:2000]}\n\n"
        f"GENERATED REVIEW (shared context — either party may reference it):\n{review_text[:2000]}\n\n"
        + (f"{ext_block}\n\n" if ext_block else "")
        + ANTI_AI_TELL_REVIEWER_INSTRUCTION
    )

    messages: List[Any] = [SystemMessage(content=system)]
    for turn in chat_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=user_message))

    try:
        llm = _make_llm(settings, temperature=0.3, num_predict=2048)
        resp = llm.invoke(messages)
        return resp.content.strip(), ""
    except Exception as e:
        logger.error("Reviewer chat failed: %s", e)
        return "", f"Reviewer chat failed: {e}"
