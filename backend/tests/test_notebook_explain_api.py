"""backend/tests/test_notebook_explain_api.py
─────────────────────────────────────────────────
API-level tests for the Mode 2 Phase D (Explain tab / storyteller pipeline)
HTTP endpoints: turn submission (background job + polling) and history.

Three stores are swapped for the same temp-file instances, mirroring
test_notebook_api.py's dual-NotebookMemory pattern but extended to the extra
StorytellerMemory access point this pipeline introduces:
  - notebook_service._memory          (so /api/notebook/notebooks-created
                                        notebooks are visible to _require_notebook)
  - notebook_explain_service.NotebookMemory  (class swap -- _ensure_session
                                        instantiates a fresh one per call)
  - notebook_explain_service._story_memory + agents.story_nodes._memory
                                        (both must point at the same
                                        StorytellerMemory so the service's
                                        _ensure_session and the pipeline's own
                                        context_loader/memory_saver nodes agree)

The LLM is mocked at agents.story_nodes.ChatOllama using the real
MockChatOllama class (not a single-return-value MagicMock): one Explain turn
can make up to four distinct LLM calls with different expected response
shapes (coverage scoring, the main explanation, its concept-extraction
micro-call, and -- on a detected repeat -- concept-map extraction), so the
content-aware dispatch in backend/app/mock_llm.py is needed to give each call
the right canned shape.

A notebook with no uploaded sources makes source_router_node skip the LLM
coverage call entirely and hard-code coverage_score=0 (see story_nodes.py),
which always triggers its online-search fallback -- so AcademicSearcher/
WebSearcher are mocked too (reusing mock_search.py's own fakes), the same
network-free boundary BEESEARCH_MOCK_LLM=1 installs for real.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import upload_source

import agents.story_nodes as story_nodes_module
import tools.search_tools as search_tools_module
from agents.notebook_memory import NotebookMemory
from agents.story_memory import StorytellerMemory
from backend.app.mock_llm import MockChatOllama
from backend.app.mock_search import MockAcademicSearcher, MockWebSearcher
from backend.app.services import notebook_explain_service, notebook_service

_BASE = "/api/notebook/explain"
_NOTEBOOK_BASE = "/api/notebook"


@pytest.fixture()
def mem(tmp_path, monkeypatch):
    notebook_instance = NotebookMemory(tmp_path / "notebooks.db")
    story_instance = StorytellerMemory(tmp_path / "story.db")
    monkeypatch.setattr(notebook_service, "_memory", notebook_instance)
    monkeypatch.setattr(notebook_explain_service, "NotebookMemory", lambda *a, **kw: notebook_instance)
    monkeypatch.setattr(notebook_explain_service, "_story_memory", story_instance)
    monkeypatch.setattr(story_nodes_module, "_memory", story_instance)
    monkeypatch.setattr(search_tools_module, "AcademicSearcher", MockAcademicSearcher)
    monkeypatch.setattr(search_tools_module, "WebSearcher", MockWebSearcher)
    return notebook_instance, story_instance


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

def test_turn_unknown_notebook_returns_404(client: TestClient, mem):
    r = client.post(f"{_BASE}/turn", json={"notebook_id": "nope", "message": "hi"})
    assert r.status_code == 404


def test_turn_blank_message_returns_422(client: TestClient, mem):
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    r = client.post(f"{_BASE}/turn", json={"notebook_id": nb_id, "message": "   "})
    assert r.status_code == 422


def test_get_unknown_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist")
    assert r.status_code == 404


def test_history_unknown_notebook_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/does-not-exist/history")
    assert r.status_code == 404


def test_history_empty_before_first_turn(client: TestClient, mem):
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    r = client.get(f"{_BASE}/{nb_id}/history")
    assert r.status_code == 200
    assert r.json() == []


# ─────────────────────────────────────────────────────────────────────────────
# Turn (background job + polling)
# ─────────────────────────────────────────────────────────────────────────────

def test_turn_with_document_returns_grounded_explanation_and_citations(client: TestClient, mem):
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "Attention"}).json()["notebook_id"]
    files = {
        "file": (
            "paper.txt",
            b"Self-attention lets a model weigh every token against every other token.",
            "text/plain",
        )
    }
    upload_source(client, nb_id, files)

    with patch.object(story_nodes_module, "ChatOllama", MockChatOllama):
        r = client.post(f"{_BASE}/turn", json={"notebook_id": nb_id, "message": "What is self-attention?"})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done"
    assert data["error"] is None
    result = data["result"]
    assert result["notebook_id"] == nb_id
    assert result["user_message"] == "What is self-attention?"
    assert "[1]" in result["assistant_response"]
    assert result["citations"]
    assert result["citations"][0]["n"] == "1"
    assert result["citations"][0]["page_label"] == "p. 1"
    assert len(result["suggested_questions"]) == 3
    assert result["is_repeat_clarification"] is False

    history = client.get(f"{_BASE}/{nb_id}/history").json()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["citations"][0]["page_label"] == "p. 1"


def test_turn_without_documents_falls_back_to_online_search_and_cites_sources(client: TestClient, mem):
    # No sources uploaded -- source_router_node hard-codes coverage_score=0 and
    # always triggers the online-search fallback (see module docstring), so
    # citations come from the mocked AcademicSearcher/WebSearcher results
    # instead of being empty.
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]

    with patch.object(story_nodes_module, "ChatOllama", MockChatOllama):
        r = client.post(f"{_BASE}/turn", json={"notebook_id": nb_id, "message": "What is X?"})
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done"
    result = data["result"]
    assert "[Source 1]" in result["assistant_response"]
    assert result["citations"]
    assert result["citations"][0]["n"] == "Source 1"
    assert result["citations"][0]["page_label"] == "n/a"


def test_repeat_confusion_phrase_sets_is_repeat_clarification(client: TestClient, mem):
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]

    with patch.object(story_nodes_module, "ChatOllama", MockChatOllama):
        r1 = client.post(f"{_BASE}/turn", json={"notebook_id": nb_id, "message": "What is attention?"})
        first = _poll_until_terminal(client, r1.json()["job_id"])
        assert first["status"] == "done"

        r2 = client.post(
            f"{_BASE}/turn",
            json={"notebook_id": nb_id, "message": "I don't understand, can you explain again?"},
        )
        second = _poll_until_terminal(client, r2.json()["job_id"])

    # concept_visualizer fails safe (e.g. pyvis not installed) -- never a job error.
    assert second["status"] == "done"
    assert second["error"] is None
    assert second["result"]["is_repeat_clarification"] is True

    history = client.get(f"{_BASE}/{nb_id}/history").json()
    assert len(history) == 4


def test_turn_pipeline_exception_surfaces_in_errors_not_as_job_error(client: TestClient, mem):
    nb_id = client.post(f"{_NOTEBOOK_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]

    # storyteller_node builds its ChatOllama instance (_llm()) *outside* its
    # try/except -- only the subsequent .invoke() (via _call()) is guarded --
    # so the failure must come from .invoke(), not from the ChatOllama
    # constructor itself (unlike notebook_nodes.py's answer_node, which calls
    # _llm(state).invoke(...) as a single guarded expression).
    failing_llm = MagicMock()
    failing_llm.invoke.side_effect = RuntimeError("LLM exploded")
    with patch.object(story_nodes_module, "ChatOllama", return_value=failing_llm):
        r = client.post(f"{_BASE}/turn", json={"notebook_id": nb_id, "message": "What is X?"})
        data = _poll_until_terminal(client, r.json()["job_id"])

    # storyteller_node catches LLM errors itself and folds the failure into
    # assistant_response/errors rather than raising -- same pattern as
    # notebook_nodes.py's answer_node (see test_notebook_api.py's analogous test).
    assert data["status"] == "done"
    result = data["result"]
    assert "Error generating response" in result["assistant_response"]
    assert any("LLM exploded" in e for e in result["errors"])
