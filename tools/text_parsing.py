"""
tools/text_parsing.py
──────────────────────
Helpers for cleaning up raw LLM text output and source documents.

extract_suggested_questions() pulls the trailing "suggested_questions" JSON
array that Notebook Chat and Explain prompts ask the LLM to append, tolerating
the various formats and minor JSON quirks local models tend to produce.

extract_references_section() locates the bibliography at the end of an
academic paper, for the Notebook "Citation Timeline" feature.

prepend_heading() and join_chunks_with_headings() restore the section heading
Docling stores in a chunk's metadata rather than its text, so a document
rebuilt from stored chunks still has the section structure the extractors
above search for.

format_page_label() converts a chunk's internal 0-based page_num into the
1-based label a user would actually see in their PDF viewer.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

_SQ_PATTERNS = [
    # Full JSON object  {"suggested_questions": [...]}
    re.compile(r'\{[^{}]*"suggested_questions"\s*:\s*(\[.*?\])\s*\}', re.DOTALL),
    # Quoted key without outer braces  "suggested_questions": [...]
    re.compile(r'"suggested_questions"\s*:\s*(\[.*?\])', re.DOTALL),
    # Bare key  suggested_questions: [...] or suggested_questions = [...]
    re.compile(r'suggested_questions\s*[:=]\s*(\[.*?\])', re.DOTALL | re.IGNORECASE),
]

# LLMs often bold/italicize the key (e.g. **suggested_questions**:), which would
# otherwise prevent every pattern above from matching.
_BOLD_KEY = re.compile(r'[*_]{1,2}(suggested_questions)[*_]{1,2}', re.IGNORECASE)

# Trailing comma before the closing bracket — common LLM JSON mistake.
_TRAILING_COMMA = re.compile(r',\s*\]')

# Curly/smart quotes used as string delimiters instead of straight ASCII quotes.
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def extract_suggested_questions(raw: str, max_questions: int = 3) -> Tuple[str, List[str]]:
    """
    Split a trailing ``suggested_questions`` JSON array off LLM output.

    Returns ``(body, questions)``:
      - ``body``: ``raw`` with the matched block (and anything after it) removed
      - ``questions``: up to ``max_questions`` strings, or ``[]`` if no block
        could be parsed (in which case ``body == raw``, unchanged)
    """
    # Normalize a markdown-bolded/italicized key so the patterns below can match it.
    text = _BOLD_KEY.sub(r"\1", raw)

    for pat in _SQ_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        block = m.group(1).translate(_SMART_QUOTES)
        block = _TRAILING_COMMA.sub("]", block)
        try:
            candidates = json.loads(block)
        except Exception:
            continue
        if isinstance(candidates, list) and candidates:
            questions = [str(q) for q in candidates if q][:max_questions]
            return text[:m.start()].strip(), questions

    return raw, []


# Bibliography sections always start with one of these headings, alone on
# their own line (optionally followed by a colon and/or a page number).
# Requiring end-of-line avoids false positives like "...the actual
# bibliography entry follows".
_REFERENCES_HEADING = re.compile(
    r'\b(references|bibliography|works\s+cited|literature\s+cited)\b'
    r'[ \t]*:?[ \t]*\d{0,4}[ \t]*(?=\n|$)',
    re.IGNORECASE,
)

# Standard numbered-bibliography entry marker, e.g. "[1] A. Author, ...",
# anchored to the start of a line. Used as a fallback when no heading
# matches (PDF text extraction sometimes mangles letter-spaced headings
# like "R E F E R E N C E S" into something extract_references_section
# can no longer recognize as a word).
_BRACKET_NUM = re.compile(r'(?:^|\n)[ \t]*\[(\d{1,3})\][ \t]+\S')

# Minimum run of consecutive [1], [2], [3]... markers required before the
# fallback trusts that it has found a real reference list (rather than,
# say, a short numbered list of contributions in the introduction).
_MIN_BRACKET_RUN = 5


def _numbered_bibliography_start(text: str) -> int:
    """
    Find the start of a run of >= _MIN_BRACKET_RUN consecutive [1] [2] [3]...
    markers, each beginning a line, and return the character offset of the
    "[1]" that starts the run -- or -1 if no such run exists.

    Like extract_references_section()'s heading search, candidates in the
    first 40% of the document are ignored unless nothing later qualifies.
    """
    by_num: Dict[int, List[int]] = {}
    for m in _BRACKET_NUM.finditer(text):
        by_num.setdefault(int(m.group(1)), []).append(m.start())

    cutoff = len(text) * 0.4
    ones = [p for p in by_num.get(1, []) if p >= cutoff] or by_num.get(1, [])
    for start_pos in reversed(ones):
        cursor = start_pos
        run_len = 1
        for target in range(2, _MIN_BRACKET_RUN + 3):
            later = [p for p in by_num.get(target, []) if p > cursor]
            if not later:
                break
            cursor = min(later)
            run_len += 1
        if run_len >= _MIN_BRACKET_RUN:
            return start_pos
    return -1


def extract_references_section(text: str) -> str:
    """
    Return the text following the last "References"/"Bibliography"/etc.
    heading in ``text``, or ``""`` if no such heading is found.

    Used by the Notebook Citation Timeline feature to isolate a source
    document's bibliography from its body text. Heading matches in the
    first 40% of the document (e.g. a table-of-contents entry or an
    in-text mention) are ignored unless nothing later qualifies.

    If no heading can be matched at all (or it only yields a tiny
    fragment), falls back to _numbered_bibliography_start() in case the
    heading itself was mangled by PDF text extraction but the reference
    list still uses standard "[1] ... [2] ..." numbering.
    """
    section = ""
    matches = list(_REFERENCES_HEADING.finditer(text))
    if matches:
        cutoff = len(text) * 0.4
        candidates = [m for m in matches if m.start() >= cutoff] or matches
        section = text[candidates[-1].end():].strip()

    if len(section) < 200:
        start = _numbered_bibliography_start(text)
        if start >= 0:
            fallback = text[start:].strip()
            if len(fallback) > len(section):
                section = fallback

    return section


def prepend_heading(text: str, heading: str) -> str:
    """
    Return ``text`` with ``heading`` restored as its own leading line.

    Returns ``text`` unchanged when ``heading`` is empty or the text already
    opens with it, so this is safe to apply to chunks that were ingested
    before headings were baked in as well as to ones that already carry them.
    """
    heading = (heading or "").strip()
    if not heading:
        return text
    if text.lstrip().casefold().startswith(heading.casefold()):
        return text
    return f"{heading}\n{text}"


def join_chunks_with_headings(chunks, separator: str = "\n\n") -> str:
    """
    Rejoin stored chunk dicts into one document string, restoring the section
    heading each chunk was extracted under.

    Docling reports a chunk's heading in ``chunk.meta.headings`` rather than in
    the chunk body, and the processor stores it as ``metadata["heading"]``. A
    plain ``separator.join(c["text"] ...)`` therefore rebuilds a document with
    no section headings at all — which is why extract_references_section()
    could find no "References" line and silently skipped whole sources whose
    bibliography wasn't numbered ``[1] [2] [3]``.

    A heading is emitted once, at the point it changes, rather than on every
    chunk of a section: repeating it would leave extract_references_section()
    matching the *last* "References" occurrence and returning only the final
    fragment of the bibliography.
    """
    parts: List[str] = []
    prev_heading = ""
    for chunk in chunks:
        text = chunk.get("text", "") or ""
        meta = chunk.get("metadata") or {}
        heading = (meta.get("heading") or "").strip() if isinstance(meta, dict) else ""
        if heading and heading != prev_heading:
            text = prepend_heading(text, heading)
        prev_heading = heading
        parts.append(text)
    return separator.join(parts)


def format_page_label(page_num) -> str:
    """
    Convert a chunk's internal 0-based ``page_num`` into the 1-based label a
    user would see in their actual PDF viewer (e.g. ``0`` -> ``"p. 1"``).

    Returns ``"n/a"`` for the "unknown page" sentinel (``-1``/``None``) or
    any non-int input, mirroring the existing convention across the
    document-processing pipelines.
    """
    if isinstance(page_num, bool) or not isinstance(page_num, int) or page_num < 0:
        return "n/a"
    return f"p. {page_num + 1}"
