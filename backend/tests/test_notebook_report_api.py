"""backend/tests/test_notebook_report_api.py
─────────────────────────────────────────────────
API-level tests for the Mode 2 Phase E (Research Report workflow) HTTP
endpoints: run submission (background job + polling) and citation export.

Two stores are swapped for the same temp-file NotebookMemory instance,
mirroring test_notebook_explain_api.py's pattern:
  - notebook_service._memory                 (so /api/notebook/notebooks
                                              -created notebooks are visible
                                              to notebook_exists)
  - notebook_report_service.NotebookMemory   (class swap -- run_report
                                              instantiates a fresh one per
                                              call)

``agents/graph.py`` imports ``ChatOllama`` directly at module level (``from
langchain_ollama import ChatOllama``), so it's patched there -- a different
boundary than ``tools.search_tools.AcademicSearcher``/``WebSearcher``, which
``agents/graph.py`` imports lazily inside its own step functions (the same
boundary ``mock_search.py``'s own docstring documents).
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import upload_source

import agents.graph as graph_module
import tools.search_tools as search_tools_module
from agents.notebook_memory import NotebookMemory
from backend.app.mock_llm import MockChatOllama
from backend.app.mock_search import MockAcademicSearcher, MockWebSearcher
from backend.app.services import notebook_report_service, notebook_service

_BASE = "/api/notebook/report"
_NOTEBOOK_BASE = "/api/notebook"


@pytest.fixture()
def mem(tmp_path, monkeypatch):
    notebook_instance = NotebookMemory(tmp_path / "notebooks.db")
    monkeypatch.setattr(notebook_service, "_memory", notebook_instance)
    monkeypatch.setattr(notebook_report_service, "NotebookMemory", lambda *a, **kw: notebook_instance)
    monkeypatch.setattr(search_tools_module, "AcademicSearcher", MockAcademicSearcher)
    monkeypatch.setattr(search_tools_module, "WebSearcher", MockWebSearcher)
    return notebook_instance


def _poll_until_terminal(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    data: dict = {}
    while time.monotonic() < deadline:
        r = client.get(f"{_BASE}/jobs/{job_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("done", "error"):
            return data
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach a terminal state: {data}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation / 404s
# ─────────────────────────────────────────────────────────────────────────────

def test_run_unknown_notebook_returns_404(client: TestClient, mem):
    r = client.post(f"{_BASE}/run", json={"notebook_id": "nope", "goal": "What is X?"})
    assert r.status_code == 404


def test_run_blank_goal_returns_422(client: TestClient, mem):
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    r = client.post(f"{_BASE}/run", json={"notebook_id": nb_id, "goal": "   "})
    assert r.status_code == 422


def test_get_unknown_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist")
    assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Run (background job + polling)
# ─────────────────────────────────────────────────────────────────────────────

def test_run_search_mode_with_no_sources(client: TestClient, mem):
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]

    with patch.object(graph_module, "ChatOllama", MockChatOllama):
        r = client.post(f"{_BASE}/run", json={"notebook_id": nb_id, "goal": "What is X?"})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done"
    assert data["error"] is None
    result = data["result"]
    assert result["notebook_id"] == nb_id
    assert result["mode"] == "search"
    assert "[Paper 1]" in result["report"]
    assert result["references"]
    assert result["references"][0]["year"] == "2023"
    assert result["key_findings"]
    assert result["eval_result"]["overall"] == 4
    assert result["progress_pct"] == 100


def test_run_document_mode_with_uploaded_source_and_no_academic(client: TestClient, mem):
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "Attention"}).json()["notebook_id"]
    files = {
        "file": (
            "paper.txt",
            b"Self-attention lets a model weigh every token against every other token.",
            "text/plain",
        )
    }
    upload_source(client, nb_id, files)

    with patch.object(graph_module, "ChatOllama", MockChatOllama):
        r = client.post(
            f"{_BASE}/run",
            json={"notebook_id": nb_id, "goal": "What is self-attention?", "include_academic": False},
        )
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done"
    result = data["result"]
    assert result["mode"] == "document"
    assert "[Source 1]" in result["report"]
    assert result["references"] == []


def test_run_hybrid_mode_with_source_and_academic(client: TestClient, mem):
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "Attention"}).json()["notebook_id"]
    files = {
        "file": (
            "paper.txt",
            b"Self-attention lets a model weigh every token against every other token.",
            "text/plain",
        )
    }
    upload_source(client, nb_id, files)

    with patch.object(graph_module, "ChatOllama", MockChatOllama):
        r = client.post(
            f"{_BASE}/run",
            json={"notebook_id": nb_id, "goal": "What is self-attention?", "include_academic": True},
        )
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done"
    result = data["result"]
    assert result["mode"] == "hybrid"
    assert "[Source 1]" in result["report"]
    assert result["references"]


def test_run_with_web_search_sets_status_ok_and_adds_web_reference(client: TestClient, mem):
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]

    with patch.object(graph_module, "ChatOllama", MockChatOllama):
        r = client.post(
            f"{_BASE}/run",
            json={"notebook_id": nb_id, "goal": "What is X?", "include_web": True},
        )
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done"
    result = data["result"]
    assert result["web_search_status"] == "ok"
    assert any(ref["source"] == "web" for ref in result["references"])


# ─────────────────────────────────────────────────────────────────────────────
# Export: BibTeX / RIS
# ─────────────────────────────────────────────────────────────────────────────

def test_export_citations_unknown_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist/export/citations/bibtex")
    assert r.status_code == 404


def test_export_citations_unfinished_job_returns_409(client: TestClient, mem):
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    job_id = client.post(f"{_BASE}/run", json={"notebook_id": nb_id, "goal": "What is X?"}).json()["job_id"]

    r = client.get(f"{_BASE}/jobs/{job_id}/export/citations/bibtex")
    assert r.status_code == 409


def test_export_citations_bibtex_and_ris(client: TestClient, mem):
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]

    with patch.object(graph_module, "ChatOllama", MockChatOllama):
        r = client.post(f"{_BASE}/run", json={"notebook_id": nb_id, "goal": "What is X?"})
        job_id = r.json()["job_id"]
        data = _poll_until_terminal(client, job_id)
    assert data["status"] == "done"

    bibtex = client.get(f"{_BASE}/jobs/{job_id}/export/citations/bibtex")
    assert bibtex.status_code == 200
    assert bibtex.headers["content-type"].startswith("text/plain")
    assert "@misc{" in bibtex.text or "@article{" in bibtex.text

    ris = client.get(f"{_BASE}/jobs/{job_id}/export/citations/ris")
    assert ris.status_code == 200
    assert ris.headers["content-type"].startswith("text/plain")
    assert "TY  - " in ris.text
    assert "ER  - " in ris.text
