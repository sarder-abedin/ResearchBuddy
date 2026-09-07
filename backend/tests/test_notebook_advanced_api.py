"""backend/tests/test_notebook_advanced_api.py
────────────────────────────────────────────────
API-level tests for Mode 2 Phase C (9 standalone Research Notebook advanced
tools) HTTP endpoints.

Both the router's NotebookMemory (via notebook_service) and
agents.notebook_advanced's own *uncached* ``NotebookMemory()`` calls are
swapped for the same temp-file instance, mirroring test_notebook_api.py's
``mem`` fixture -- but since agents/notebook_advanced.py instantiates
``NotebookMemory()`` fresh inline in every function (no module-level
singleton to monkeypatch), the ``NotebookMemory`` class name itself is
monkeypatched in that module's namespace, ignoring constructor args and
always returning the shared instance.

Real end-to-end runs are exercised for all 9 features, with
``agents.notebook_advanced.ChatOllama`` (the local-import boundary inside
``_make_llm``) replaced by a content-aware mock keyed on each feature's
distinct system prompt -- mirrors test_notebook_pipeline_api.py's
``_mock_pipeline_llm`` pattern.

Export endpoints operate on an already-completed job, so rather than
re-running a feature for each one, those tests fabricate a finished/
errored/running Job directly via backend.app.jobs -- mirrors
test_notebook_pipeline_api.py's ``_finished_job``/``_errored_job``/
``_running_job`` helpers.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import upload_source

import agents.notebook_advanced as notebook_advanced_module
from agents.notebook_memory import NotebookMemory
from backend.app import jobs as jobs_module
from backend.app.services import notebook_service

_BASE = "/api/notebook/advanced"
_NB_BASE = "/api/notebook"


@pytest.fixture()
def mem(tmp_path, monkeypatch) -> NotebookMemory:
    instance = NotebookMemory(tmp_path / "notebooks.db")
    monkeypatch.setattr(notebook_service, "_memory", instance)
    monkeypatch.setattr(notebook_advanced_module, "NotebookMemory", lambda *a, **kw: instance)
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


def _mock_advanced_llm() -> MagicMock:
    """Content-aware ChatOllama stand-in for all 9 Phase C tools' LLM calls."""
    llm = MagicMock()

    def _side_effect(messages):
        system = messages[0].content
        msg = MagicMock()
        if "synthesizing multiple documents" in system:
            msg.content = "## Overview\nMock cross-document synthesis."
        elif "Summarize the key points, methodology" in system:
            msg.content = "## Summary\nMock single-source summary."
        elif "frequently asked questions" in system:
            msg.content = json.dumps([
                {"question": "What is the main finding?", "answer": "Mock answer [1].", "sources": [1]}
            ])
        elif "formal literature review" in system:
            msg.content = (
                "# Literature Review\n## 1. Introduction\nMock intro [1].\n"
                "## 6. Conclusion\nMock conclusion."
            )
        elif "knowledge analyst" in system:
            msg.content = json.dumps({
                "central": "Rayleigh Scattering",
                "branches": [{"concept": "Light", "sub_concepts": ["Wavelength"]}],
            })
        elif "spoken audio summary script" in system:
            msg.content = "This notebook covers Rayleigh scattering and why the sky looks blue."
        elif "comparing two documents" in system:
            msg.content = "## Source Comparison\n### Overview\nMock comparison."
        elif "knowledge graph extractor" in system:
            msg.content = json.dumps({
                "nodes": [{"id": "n1", "label": "Scattering", "type": "concept"}],
                "edges": [],
            })
        elif "extracting bibliography entries" in system:
            msg.content = json.dumps([
                {"year": "2018", "authors": "Smith et al.", "title": "A Study of Light Scattering"}
            ])
        elif "For each numbered title below" in system:
            msg.content = json.dumps(["Mock one-line gist."])
        elif "Create a comparison table across all sources" in system:
            msg.content = "| Dimension | sky.txt |\n|---|---|\n| Year | 2018 |\n\n**Synthesis**\nMock synthesis."
        else:
            msg.content = "Mock response."
        return msg

    llm.invoke.side_effect = _side_effect
    return llm


def _make_notebook_with_source(
    client: TestClient, text: bytes = b"The sky is blue because of Rayleigh scattering."
):
    nb_id = client.post(f"{_NB_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    files = {"file": ("sky.txt", text, "text/plain")}
    upload = upload_source(client, nb_id, files)
    return nb_id, upload["source"]["doc_id"]


_REFERENCES_FIXTURE = (
    "Rayleigh scattering explains why the sky appears blue during the day. "
    "Sunlight is scattered by gases in Earth's atmosphere, and blue light is "
    "scattered more than other colors because it travels in shorter, smaller "
    "waves. This phenomenon was first described in detail in the nineteenth "
    "century and remains a standard example in atmospheric optics courses "
    "around the world today.\n\n"
    "References\n"
    "[1] Smith, J. (2018). A Study of Light Scattering. Journal of Optics, 12(3), 45-67.\n"
    "[2] Doe, A., & Lee, B. (2019). Atmospheric Phenomena Review. Physics Today, 5(2), 100-120.\n"
    "[3] Brown, C. (2020). Color Perception in the Sky. Nature Physics, 8(1), 12-29.\n"
).encode()


# ─────────────────────────────────────────────────────────────────────────────
# Validation common to every trigger endpoint
# ─────────────────────────────────────────────────────────────────────────────

_TRIGGER_PATHS = [
    ("/cross-document-summary", {}),
    ("/faq", {}),
    ("/literature-review", {}),
    ("/mindmap", {}),
    ("/audio-summary", {}),
    ("/compare-sources", {"doc_id_a": "a", "doc_id_b": "b"}),
    ("/knowledge-graph", {}),
    ("/citation-timeline", {}),
    ("/study-comparison", {}),
]


@pytest.mark.parametrize("path,payload", _TRIGGER_PATHS)
def test_trigger_unknown_notebook_returns_404(client: TestClient, mem, path, payload):
    r = client.post(f"{_BASE}{path}", json={"notebook_id": "does-not-exist", **payload})
    assert r.status_code == 404


@pytest.mark.parametrize("path,payload", _TRIGGER_PATHS)
def test_trigger_blank_notebook_id_returns_422(client: TestClient, mem, path, payload):
    r = client.post(f"{_BASE}{path}", json={"notebook_id": "   ", **payload})
    assert r.status_code == 422


def test_get_unknown_advanced_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist")
    assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Run: end-to-end happy paths (background job + polling)
# ─────────────────────────────────────────────────────────────────────────────

def test_cross_document_summary_round_trip(client: TestClient, mem):
    nb_id, _ = _make_notebook_with_source(client)
    with patch.object(notebook_advanced_module, "ChatOllama", return_value=_mock_advanced_llm()):
        r = client.post(f"{_BASE}/cross-document-summary", json={"notebook_id": nb_id})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done", data.get("error")
    result = data["result"]
    assert result["notebook_id"] == nb_id
    assert "Mock single-source summary" in result["summary"]


def test_faq_round_trip(client: TestClient, mem):
    nb_id, _ = _make_notebook_with_source(client)
    with patch.object(notebook_advanced_module, "ChatOllama", return_value=_mock_advanced_llm()):
        r = client.post(f"{_BASE}/faq", json={"notebook_id": nb_id, "n_questions": 1})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done", data.get("error")
    result = data["result"]
    assert result["faqs"][0]["question"] == "What is the main finding?"
    assert result["faqs"][0]["sources"] == [1]


def test_literature_review_round_trip(client: TestClient, mem):
    nb_id, _ = _make_notebook_with_source(client)
    with patch.object(notebook_advanced_module, "ChatOllama", return_value=_mock_advanced_llm()):
        r = client.post(f"{_BASE}/literature-review", json={"notebook_id": nb_id})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done", data.get("error")
    result = data["result"]
    assert "Literature Review" in result["review"]
    assert result["references"][0]["n"] == 1


def test_mindmap_round_trip(client: TestClient, mem):
    nb_id, _ = _make_notebook_with_source(client)
    with patch.object(notebook_advanced_module, "ChatOllama", return_value=_mock_advanced_llm()):
        r = client.post(f"{_BASE}/mindmap", json={"notebook_id": nb_id})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done", data.get("error")
    result = data["result"]
    assert "digraph mindmap" in result["mindmap_dot"]
    assert "Rayleigh Scattering" in result["mindmap_dot"]


def test_audio_summary_round_trip(client: TestClient, mem):
    nb_id, _ = _make_notebook_with_source(client)
    with patch.object(notebook_advanced_module, "ChatOllama", return_value=_mock_advanced_llm()):
        r = client.post(f"{_BASE}/audio-summary", json={"notebook_id": nb_id})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done", data.get("error")
    assert "Rayleigh scattering" in data["result"]["audio_script"]


def test_compare_sources_round_trip(client: TestClient, mem):
    nb_id, doc_id_a = _make_notebook_with_source(client, text=b"Source A is about light.")
    files_b = {"file": ("ocean.txt", b"Source B is about water.", "text/plain")}
    doc_id_b = upload_source(client, nb_id, files_b)["source"]["doc_id"]

    with patch.object(notebook_advanced_module, "ChatOllama", return_value=_mock_advanced_llm()):
        r = client.post(
            f"{_BASE}/compare-sources",
            json={"notebook_id": nb_id, "doc_id_a": doc_id_a, "doc_id_b": doc_id_b},
        )
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done", data.get("error")
    assert "Source Comparison" in data["result"]["comparison"]


def test_compare_sources_same_doc_id_surfaces_as_job_error(client: TestClient, mem):
    nb_id, doc_id_a = _make_notebook_with_source(client)
    r = client.post(
        f"{_BASE}/compare-sources",
        json={"notebook_id": nb_id, "doc_id_a": doc_id_a, "doc_id_b": doc_id_a},
    )
    assert r.status_code == 202
    data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "error"
    assert "two different sources" in data["error"]


def test_compare_sources_unknown_doc_id_surfaces_as_job_error(client: TestClient, mem):
    nb_id, _ = _make_notebook_with_source(client)
    r = client.post(
        f"{_BASE}/compare-sources",
        json={"notebook_id": nb_id, "doc_id_a": "missing-a", "doc_id_b": "missing-b"},
    )
    assert r.status_code == 202
    data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "error"
    assert "not found in this notebook" in data["error"]


def test_knowledge_graph_round_trip(client: TestClient, mem):
    nb_id, _ = _make_notebook_with_source(client)
    with patch.object(notebook_advanced_module, "ChatOllama", return_value=_mock_advanced_llm()):
        r = client.post(f"{_BASE}/knowledge-graph", json={"notebook_id": nb_id})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done", data.get("error")
    assert "digraph knowledge_graph" in data["result"]["knowledge_graph_dot"]


def test_study_comparison_round_trip(client: TestClient, mem):
    nb_id, _ = _make_notebook_with_source(client)
    with patch.object(notebook_advanced_module, "ChatOllama", return_value=_mock_advanced_llm()):
        r = client.post(f"{_BASE}/study-comparison", json={"notebook_id": nb_id})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done", data.get("error")
    assert "Dimension" in data["result"]["study_comparison"]


def test_citation_timeline_round_trip(client: TestClient, mem):
    nb_id, _ = _make_notebook_with_source(client, text=_REFERENCES_FIXTURE)
    with patch.object(notebook_advanced_module, "ChatOllama", return_value=_mock_advanced_llm()):
        r = client.post(f"{_BASE}/citation-timeline", json={"notebook_id": nb_id})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done", data.get("error")
    timeline = data["result"]["timeline"]
    assert timeline
    assert timeline[0]["title"] == "A Study of Light Scattering"
    assert timeline[0]["year"] == "2018"


def test_citation_timeline_no_references_section_surfaces_as_job_error(client: TestClient, mem):
    nb_id, _ = _make_notebook_with_source(client, text=b"The sky is blue because of Rayleigh scattering.")
    with patch.object(notebook_advanced_module, "ChatOllama", return_value=_mock_advanced_llm()):
        r = client.post(f"{_BASE}/citation-timeline", json={"notebook_id": nb_id})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "error"
    assert "No references/bibliography section" in data["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Export: plain text (summary / review / audio-script / comparison / study-comparison)
# ─────────────────────────────────────────────────────────────────────────────

def test_export_text_unknown_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist/export/text/summary")
    assert r.status_code == 404


def test_export_text_errored_job_returns_409(client: TestClient, mem):
    job_id = _errored_job("tool blew up")
    r = client.get(f"{_BASE}/jobs/{job_id}/export/text/summary")
    assert r.status_code == 409
    assert "tool blew up" in r.json()["detail"]


def test_export_text_unfinished_job_returns_409(client: TestClient, mem):
    job_id = _running_job()
    r = client.get(f"{_BASE}/jobs/{job_id}/export/text/summary")
    assert r.status_code == 409


def test_export_text_invalid_artifact_returns_422(client: TestClient, mem):
    job_id = _finished_job({"summary": "Summary."})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/text/not-a-real-artifact")
    assert r.status_code == 422


def test_export_text_missing_content_returns_404(client: TestClient, mem):
    job_id = _finished_job({"summary": ""})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/text/summary")
    assert r.status_code == 404


@pytest.mark.parametrize(
    "artifact,result,content",
    [
        ("summary", {"summary": "Mock summary."}, "Mock summary."),
        ("audio-script", {"audio_script": "Mock script."}, "Mock script."),
        ("comparison", {"comparison": "Mock comparison."}, "Mock comparison."),
        ("study-comparison", {"study_comparison": "Mock table."}, "Mock table."),
    ],
)
def test_export_text_happy_path(client: TestClient, mem, artifact, result, content):
    job_id = _finished_job(result)
    r = client.get(f"{_BASE}/jobs/{job_id}/export/text/{artifact}")
    assert r.status_code == 200
    assert r.text == content
    assert r.headers["content-type"].startswith("text/markdown")


def test_export_text_review_without_references_returns_body_only(client: TestClient, mem):
    job_id = _finished_job({"review": "## Review body", "references": []})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/text/review")
    assert r.status_code == 200
    assert r.text == "## Review body"


def test_export_text_review_with_references_appends_markdown_list(client: TestClient, mem):
    job_id = _finished_job({
        "review": "## Review body",
        "references": [{"n": 1, "doc_name": "sky.txt", "page": 0, "snippet": "...", "doc_id": "d1"}],
    })
    r = client.get(f"{_BASE}/jobs/{job_id}/export/text/review")
    assert r.status_code == 200
    assert "## Review body" in r.text
    assert "## References" in r.text
    assert "[1] sky.txt (p. 1)" in r.text


# ─────────────────────────────────────────────────────────────────────────────
# Export: document (DOCX / PDF) for summary / review / study-comparison
# ─────────────────────────────────────────────────────────────────────────────

def test_export_document_unknown_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist/export/document/summary/docx")
    assert r.status_code == 404


def test_export_document_missing_content_returns_404(client: TestClient, mem):
    job_id = _finished_job({"summary": ""})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/document/summary/docx")
    assert r.status_code == 404


def test_export_document_invalid_artifact_returns_422(client: TestClient, mem):
    job_id = _finished_job({"mindmap_dot": "digraph{}"})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/document/mindmap/docx")
    assert r.status_code == 422


def test_export_document_invalid_format_returns_422(client: TestClient, mem):
    job_id = _finished_job({"summary": "Summary."})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/document/summary/epub")
    assert r.status_code == 422


def test_export_document_docx_happy_path(client: TestClient, mem):
    job_id = _finished_job({"summary": "Summary."})
    with patch("tools.export_tools.build_docx", return_value=b"FAKE-DOCX-BYTES") as mock_build:
        r = client.get(f"{_BASE}/jobs/{job_id}/export/document/summary/docx")

    assert r.status_code == 200
    assert r.content == b"FAKE-DOCX-BYTES"
    assert "wordprocessingml" in r.headers["content-type"]
    assert "summary.docx" in r.headers["content-disposition"]
    mock_build.assert_called_once_with("Summary.", [])


def test_export_document_pdf_happy_path(client: TestClient, mem):
    job_id = _finished_job({"study_comparison": "Table."})
    with patch("tools.export_tools.build_pdf", return_value=b"FAKE-PDF-BYTES") as mock_build:
        r = client.get(f"{_BASE}/jobs/{job_id}/export/document/study-comparison/pdf")

    assert r.status_code == 200
    assert r.content == b"FAKE-PDF-BYTES"
    assert r.headers["content-type"] == "application/pdf"
    assert "study-comparison.pdf" in r.headers["content-disposition"]
    mock_build.assert_called_once_with("Table.", [])


def test_export_document_review_composes_body_and_references(client: TestClient, mem):
    job_id = _finished_job({
        "review": "## Review body",
        "references": [{"n": 1, "doc_name": "sky.txt", "page": 0, "snippet": "...", "doc_id": "d1"}],
    })
    with patch("tools.export_tools.build_docx", return_value=b"FAKE-DOCX-BYTES") as mock_build:
        r = client.get(f"{_BASE}/jobs/{job_id}/export/document/review/docx")

    assert r.status_code == 200
    composed = mock_build.call_args.args[0]
    assert "## Review body" in composed
    assert "[1] sky.txt (p. 1)" in composed


# ─────────────────────────────────────────────────────────────────────────────
# Export: dot (PNG / SVG) for mindmap / knowledge-graph
# ─────────────────────────────────────────────────────────────────────────────

def test_export_dot_unknown_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist/export/dot/mindmap/png")
    assert r.status_code == 404


def test_export_dot_missing_content_returns_404(client: TestClient, mem):
    job_id = _finished_job({"mindmap_dot": ""})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/dot/mindmap/png")
    assert r.status_code == 404


def test_export_dot_invalid_artifact_returns_422(client: TestClient, mem):
    job_id = _finished_job({"summary": "Summary."})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/dot/summary/png")
    assert r.status_code == 422


def test_export_dot_invalid_format_returns_422(client: TestClient, mem):
    job_id = _finished_job({"mindmap_dot": "digraph{}"})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/dot/mindmap/gif")
    assert r.status_code == 422


def test_export_dot_mindmap_png_happy_path(client: TestClient, mem):
    job_id = _finished_job({"mindmap_dot": "digraph mindmap{}"})
    with patch(
        "agents.notebook_advanced.render_dot_bytes", return_value=(b"FAKE-PNG-BYTES", "")
    ) as mock_render:
        r = client.get(f"{_BASE}/jobs/{job_id}/export/dot/mindmap/png")

    assert r.status_code == 200
    assert r.content == b"FAKE-PNG-BYTES"
    assert r.headers["content-type"] == "image/png"
    assert "mindmap.png" in r.headers["content-disposition"]
    mock_render.assert_called_once_with("digraph mindmap{}", "png")


def test_export_dot_knowledge_graph_svg_happy_path(client: TestClient, mem):
    job_id = _finished_job({"knowledge_graph_dot": "digraph knowledge_graph{}"})
    with patch(
        "agents.notebook_advanced.render_dot_bytes", return_value=(b"<svg/>", "")
    ):
        r = client.get(f"{_BASE}/jobs/{job_id}/export/dot/knowledge-graph/svg")

    assert r.status_code == 200
    assert r.content == b"<svg/>"
    assert r.headers["content-type"] == "image/svg+xml"


def test_export_dot_render_failure_returns_503(client: TestClient, mem):
    job_id = _finished_job({"mindmap_dot": "digraph{}"})
    with patch(
        "agents.notebook_advanced.render_dot_bytes", return_value=(b"", "graphviz not installed")
    ):
        r = client.get(f"{_BASE}/jobs/{job_id}/export/dot/mindmap/png")

    assert r.status_code == 503
    assert "graphviz not installed" in r.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# Export: audio WAV
# ─────────────────────────────────────────────────────────────────────────────

def test_export_audio_unknown_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist/export/audio/wav")
    assert r.status_code == 404


def test_export_audio_missing_content_returns_404(client: TestClient, mem):
    job_id = _finished_job({"audio_script": ""})
    r = client.get(f"{_BASE}/jobs/{job_id}/export/audio/wav")
    assert r.status_code == 404


def test_export_audio_happy_path(client: TestClient, mem):
    job_id = _finished_job({"audio_script": "Mock script."})
    with patch(
        "agents.notebook_advanced.synthesize_speech", return_value=(b"FAKE-WAV-BYTES", "")
    ) as mock_synth:
        r = client.get(f"{_BASE}/jobs/{job_id}/export/audio/wav")

    assert r.status_code == 200
    assert r.content == b"FAKE-WAV-BYTES"
    assert r.headers["content-type"] == "audio/wav"
    assert "audio_summary.wav" in r.headers["content-disposition"]
    mock_synth.assert_called_once_with("Mock script.")


def test_export_audio_synthesis_failure_returns_503(client: TestClient, mem):
    job_id = _finished_job({"audio_script": "Mock script."})
    with patch(
        "agents.notebook_advanced.synthesize_speech", return_value=(b"", "pyttsx3 not installed")
    ):
        r = client.get(f"{_BASE}/jobs/{job_id}/export/audio/wav")

    assert r.status_code == 503
    assert "pyttsx3 not installed" in r.json()["detail"]
