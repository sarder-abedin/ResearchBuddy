"""backend/tests/test_notebook_pipeline_api.py
─────────────────────────────────────────────────
API-level tests for the Mode 2 Phase B (7-agent Research Notebook analysis
pipeline) HTTP endpoints.

Both the router's NotebookMemory (via notebook_service) and the pipeline's
own NotebookMemory (agents.notebook_pipeline_nodes -- a *separate* lazy
singleton, distinct from agents.notebook_nodes used by Phase A's chat) are
swapped for the same temp-file instance -- mirroring test_notebook_api.py's
`mem` fixture.

Two real-pipeline runs are exercised end to end:
  - an empty notebook, which needs no LLM mocking at all -- every one of the
    7 nodes takes its own "no sources/chunks" early-return guard clause, so
    the whole pipeline completes with zero LLM calls;
  - a one-source notebook, which patches ``langchain_ollama.ChatOllama``
    (the local-import boundary inside
    ``agents.notebook_pipeline_nodes._make_llm``) with a content-aware mock
    that inspects the system prompt to return a plausible response for each
    of the 5 LLM-calling nodes. Agent 3 (retrieval)'s own self-reflective
    grading call is intentionally left unmocked: it hits a real (failing)
    connection attempt that agents/self_reflective_rag.py already catches
    and falls back safely from, the same reliance
    test_notebook_api.py::test_chat_with_source_returns_grounded_answer_and_citations
    already makes on that fallback path.

Every export endpoint operates on an *already-completed* job, so rather than
re-running the pipeline for each one, those tests fabricate a finished/
errored/running Job directly via backend.app.jobs -- mirroring
test_systematic_review_api.py's `_finished_job`/`_errored_job`/`_running_job`
helpers.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import upload_source

import agents.notebook_pipeline_nodes as notebook_pipeline_nodes_module
from agents.notebook_memory import NotebookMemory
from backend.app import jobs as jobs_module
from backend.app.services import notebook_service

_BASE = "/api/notebook/pipeline"
_NB_BASE = "/api/notebook"


@pytest.fixture()
def mem(tmp_path, monkeypatch) -> NotebookMemory:
    instance = NotebookMemory(tmp_path / "notebooks.db")
    monkeypatch.setattr(notebook_service, "_memory", instance)
    monkeypatch.setattr(notebook_pipeline_nodes_module, "_memory", instance)
    return instance


def _poll_until_terminal(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    data: dict = {}
    while time.monotonic() < deadline:
        r = client.get(f"{_BASE}/jobs/{job_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal state: {data}")


def _finished_job(result: dict) -> str:
    job = jobs_module.create_job()
    job.status = "done"
    job.result = result
    return job.id


def _errored_job(error: str) -> str:
    job = jobs_module.create_job()
    job.status = "error"
    job.error = error
    return job.id


def _running_job() -> str:
    job = jobs_module.create_job()
    job.status = "running"
    return job.id


def _mock_pipeline_llm() -> MagicMock:
    """Content-aware ChatOllama stand-in for the 5 LLM-calling pipeline nodes."""
    llm = MagicMock()

    def _side_effect(messages):
        system = messages[0].content
        msg = MagicMock()
        if "You are a research analyst. Summarize this document" in system:
            msg.content = "This document explores Rayleigh scattering across three mock paragraphs."
        elif "Summarize this document in 3" in system:
            msg.content = "Mock per-document summary sentence covering the source."
        elif "synthesising multiple sources" in system:
            msg.content = "**Overview**\nMock cross-document synthesis overview."
        elif "research fact-checker" in system:
            msg.content = json.dumps([
                {
                    "claim": "The sky is blue due to Rayleigh scattering.",
                    "source_name": "sky.txt",
                    "confidence": "HIGH",
                    "supporting_text": "The sky is blue because of Rayleigh scattering.",
                }
            ])
        elif "Extract a knowledge graph" in system:
            msg.content = json.dumps({
                "nodes": [
                    {"id": "1", "label": "Rayleigh scattering", "type": "concept"},
                    {"id": "2", "label": "Sky color", "type": "concept"},
                ],
                "edges": [{"from": "1", "to": "2", "label": "causes"}],
            })
        elif "expert tutor" in system:
            msg.content = (
                "## Key Concepts\n- **Rayleigh Scattering** — mock definition.\n\n"
                "## Glossary\n| Term | Definition | Source |\n|---|---|---|\n"
                "| Rayleigh scattering | Mock def | sky.txt |\n\n"
                "## Review Questions\n**Q:** Why is the sky blue?\n**A:** Rayleigh scattering.\n\n"
                "## Quick Summary\nMock quick summary paragraph."
            )
        elif "podcast script writer" in system:
            msg.content = "HOST: Why is the sky blue?\nEXPERT: It's Rayleigh scattering."
        else:
            msg.content = "Mock response."
        return msg

    llm.invoke.side_effect = _side_effect
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline run (background job + polling)
# ─────────────────────────────────────────────────────────────────────────────

def test_run_pipeline_unknown_notebook_returns_404(client: TestClient, mem):
    r = client.post(f"{_BASE}/run", json={"notebook_id": "does-not-exist"})
    assert r.status_code == 404


def test_get_unknown_pipeline_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist")
    assert r.status_code == 404


def test_run_pipeline_blank_notebook_id_returns_422(client: TestClient, mem):
    r = client.post(f"{_BASE}/run", json={"notebook_id": "   "})
    assert r.status_code == 422


def test_run_pipeline_empty_notebook_completes_with_no_llm_calls(client: TestClient, mem):
    """Every node's own "no sources/chunks" guard clause fires -- exercises the
    real (unmocked) pipeline end to end with zero LLM calls."""
    nb_id = client.post(f"{_NB_BASE}/notebooks", json={"name": "Empty"}).json()["notebook_id"]

    r = client.post(f"{_BASE}/run", json={"notebook_id": nb_id})
    assert r.status_code == 202
    data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done"
    assert data["error"] is None
    result = data["result"]
    assert result["notebook_id"] == nb_id
    assert result["doc_count"] == 0
    assert result["cross_summary"] == ""
    assert result["per_doc_summaries"] == {}
    assert result["retrieval_mode"] == "empty"
    assert result["retrieved_chunks"] == []
    assert result["citation_report"] == "No summary available for verification."
    assert result["verified_citations"] == []
    assert result["knowledge_graph_dot"] == ""
    assert result["kg_data"] == {}
    assert result["study_guide"] == ""
    assert result["podcast_script"] == ""
    assert result["eval_result"] == {}
    assert result["progress_pct"] == 100
    assert len(result["errors"]) == 6
    assert "ingest" in result["completed_steps"]
    assert "notebook_pipeline_eval" not in result["completed_steps"]


def test_run_pipeline_with_sources_populates_all_agent_outputs(client: TestClient, mem):
    """Three short single-chunk sources -- not one -- deliberately: with only
    1-2 documents in the corpus, BM25's IDF formula (log((N-n+0.5)/(n+0.5)))
    is mathematically <= 0 for every term (a term occurring in n of N <= 2
    docs can never satisfy n < N/2), so Agent 3 would deterministically
    retrieve zero chunks regardless of query wording -- not a pipeline bug,
    just BM25 with a near-empty corpus. A 3rd, unrelated document gives the
    query's distinctive term a positive IDF. This also exercises
    summarization_node's multi-doc branch (per-doc + cross-doc synthesis)
    instead of its single-doc branch.
    """
    nb_id = client.post(f"{_NB_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    sources = {
        "sky.txt": b"The sky appears blue due to Rayleigh scattering of sunlight in the atmosphere.",
        "ocean.txt": b"The ocean looks blue because water absorbs red light and reflects blue light back.",
        "forest.txt": b"Forests support biodiversity by providing habitat for countless plant and animal species.",
    }
    for filename, content in sources.items():
        files = {"file": (filename, content, "text/plain")}
        upload_source(client, nb_id, files)

    with patch("langchain_ollama.ChatOllama", return_value=_mock_pipeline_llm()):
        r = client.post(f"{_BASE}/run", json={"notebook_id": nb_id, "query": "Rayleigh scattering"})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done", data.get("error")
    result = data["result"]
    assert result["doc_count"] == 3
    assert set(result["per_doc_summaries"]) == set(sources)
    assert result["cross_summary"]
    assert result["retrieval_mode"] == "fallback"
    assert len(result["retrieved_chunks"]) >= 1
    assert result["retrieved_chunks"][0]["doc_name"] == "sky.txt"
    assert result["verified_citations"][0]["confidence"] == "HIGH"
    assert "Citation Verification Report" in result["citation_report"]
    assert "digraph" in result["knowledge_graph_dot"]
    assert result["kg_data"]["nodes"]
    assert "Key Concepts" in result["study_guide"]
    assert "HOST:" in result["podcast_script"]
    assert result["progress_pct"] == 100
    assert result["errors"] == []


def test_run_pipeline_node_exception_surfaces_as_job_error(client: TestClient, mem):
    """Confirms the job-error path itself works -- a node raising rather than
    catching is surfaced as status="error", not silently swallowed."""
    nb_id = client.post(f"{_NB_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    files = {"file": ("sky.txt", b"The sky is blue.", "text/plain")}
    upload_source(client, nb_id, files)

    with patch.object(
        notebook_pipeline_nodes_module, "_get_memory", side_effect=RuntimeError("DB exploded")
    ):
        r = client.post(f"{_BASE}/run", json={"notebook_id": nb_id})
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "error"
    assert "DB exploded" in data["error"]
    assert data["result"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Export: plain text (summary / citations / study-guide / podcast)
# ─────────────────────────────────────────────────────────────────────────────

def test_export_text_unknown_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist/export/text/summary")
    assert r.status_code == 404


def test_export_text_errored_job_returns_409(client: TestClient, mem):
    job_id = _errored_job("pipeline blew up")
    r = client.get(f"{_BASE}/jobs/{job_id}/export/text/summary")
    assert r.status_code == 409
    assert "pipeline blew up" in r.json()["detail"]


def test_export_text_unfinished_job_returns_409(client: TestClient, mem):
    job_id = _running_job()
    r = client.get(f"{_BASE}/jobs/{job_id}/export/text/summary")
    assert r.status_code == 409


def test_export_text_invalid_artifact_returns_422(client: TestClient, mem):
    job_id = _finished_job({"cross_summary": "Summary."})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/text/not-a-real-artifact")
    assert r.status_code == 422


def test_export_text_missing_content_returns_404(client: TestClient, mem):
    job_id = _finished_job({"cross_summary": ""})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/text/summary")
    assert r.status_code == 404


@pytest.mark.parametrize(
    "artifact,field,content",
    [
        ("summary", "cross_summary", "Mock cross summary."),
        ("citations", "citation_report", "Mock citation report."),
        ("study-guide", "study_guide", "Mock study guide."),
        ("podcast", "podcast_script", "HOST: hi\nEXPERT: hello"),
    ],
)
def test_export_text_happy_path(client: TestClient, mem, artifact, field, content):
    job_id = _finished_job({field: content})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/text/{artifact}")
    assert r.status_code == 200
    assert r.text == content
    assert r.headers["content-type"].startswith("text/markdown")


# ─────────────────────────────────────────────────────────────────────────────
# Export: study guide DOCX / PDF
# ─────────────────────────────────────────────────────────────────────────────

def test_export_study_guide_unknown_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist/export/study-guide/docx")
    assert r.status_code == 404


def test_export_study_guide_missing_content_returns_404(client: TestClient, mem):
    job_id = _finished_job({"study_guide": ""})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/study-guide/docx")
    assert r.status_code == 404


def test_export_study_guide_invalid_format_returns_422(client: TestClient, mem):
    job_id = _finished_job({"study_guide": "## Guide"})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/study-guide/epub")
    assert r.status_code == 422


def test_export_study_guide_docx_happy_path(client: TestClient, mem):
    job_id = _finished_job({"study_guide": "## Guide"})
    with patch("tools.export_tools.build_docx", return_value=b"FAKE-DOCX-BYTES") as mock_build:
        r = client.get(f"{_BASE}/jobs/{job_id}/export/study-guide/docx")

    assert r.status_code == 200
    assert r.content == b"FAKE-DOCX-BYTES"
    assert "wordprocessingml" in r.headers["content-type"]
    assert "study_guide.docx" in r.headers["content-disposition"]
    mock_build.assert_called_once_with("## Guide", [])


def test_export_study_guide_pdf_happy_path(client: TestClient, mem):
    job_id = _finished_job({"study_guide": "## Guide"})
    with patch("tools.export_tools.build_pdf", return_value=b"FAKE-PDF-BYTES") as mock_build:
        r = client.get(f"{_BASE}/jobs/{job_id}/export/study-guide/pdf")

    assert r.status_code == 200
    assert r.content == b"FAKE-PDF-BYTES"
    assert r.headers["content-type"] == "application/pdf"
    assert "study_guide.pdf" in r.headers["content-disposition"]
    mock_build.assert_called_once_with("## Guide", [])


# ─────────────────────────────────────────────────────────────────────────────
# Export: knowledge graph PNG / SVG
# ─────────────────────────────────────────────────────────────────────────────

def test_export_knowledge_graph_unknown_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist/export/knowledge-graph/png")
    assert r.status_code == 404


def test_export_knowledge_graph_missing_content_returns_404(client: TestClient, mem):
    job_id = _finished_job({"knowledge_graph_dot": ""})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/knowledge-graph/png")
    assert r.status_code == 404


def test_export_knowledge_graph_invalid_format_returns_422(client: TestClient, mem):
    job_id = _finished_job({"knowledge_graph_dot": "digraph{}"})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/knowledge-graph/gif")
    assert r.status_code == 422


def test_export_knowledge_graph_png_happy_path(client: TestClient, mem):
    job_id = _finished_job({"knowledge_graph_dot": "digraph{}"})
    with patch(
        "agents.notebook_advanced.render_dot_bytes", return_value=(b"FAKE-PNG-BYTES", "")
    ) as mock_render:
        r = client.get(f"{_BASE}/jobs/{job_id}/export/knowledge-graph/png")

    assert r.status_code == 200
    assert r.content == b"FAKE-PNG-BYTES"
    assert r.headers["content-type"] == "image/png"
    assert "knowledge_graph.png" in r.headers["content-disposition"]
    mock_render.assert_called_once_with("digraph{}", "png")


def test_export_knowledge_graph_svg_happy_path(client: TestClient, mem):
    job_id = _finished_job({"knowledge_graph_dot": "digraph{}"})
    with patch(
        "agents.notebook_advanced.render_dot_bytes", return_value=(b"<svg/>", "")
    ):
        r = client.get(f"{_BASE}/jobs/{job_id}/export/knowledge-graph/svg")

    assert r.status_code == 200
    assert r.content == b"<svg/>"
    assert r.headers["content-type"] == "image/svg+xml"


def test_export_knowledge_graph_render_failure_returns_503(client: TestClient, mem):
    job_id = _finished_job({"knowledge_graph_dot": "digraph{}"})
    with patch(
        "agents.notebook_advanced.render_dot_bytes", return_value=(b"", "graphviz not installed")
    ):
        r = client.get(f"{_BASE}/jobs/{job_id}/export/knowledge-graph/png")

    assert r.status_code == 503
    assert "graphviz not installed" in r.json()["detail"]
