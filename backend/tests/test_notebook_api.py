"""backend/tests/test_notebook_api.py
───────────────────────────────────────
API-level tests for the Mode 2 Phase A (Research Notebook core) HTTP
endpoints: notebook CRUD, source upload/removal, history, and chat.

Both the router's NotebookMemory (via notebook_service) and the pipeline's
own NotebookMemory (agents.notebook_nodes -- a *separate* lazy singleton)
are swapped for the same temp-file instance, so a notebook created through
the API is visible to agents.notebook_graph.run_notebook_turn exactly as it
would be against the real on-disk store. The chat LLM call is mocked at
agents.notebook_nodes.ChatOllama, the same boundary
test_research_assistant_api.py uses for Mode 3's ChatOllama.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import upload_source

import agents.notebook_nodes as notebook_nodes_module
from agents.notebook_memory import NotebookMemory
from backend.app.services import notebook_service

_BASE = "/api/notebook"


@pytest.fixture()
def mem(tmp_path, monkeypatch) -> NotebookMemory:
    instance = NotebookMemory(tmp_path / "notebooks.db")
    monkeypatch.setattr(notebook_service, "_memory", instance)
    monkeypatch.setattr(notebook_nodes_module, "_memory", instance)
    return instance


def _poll_until_terminal(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
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


def _mock_llm(answer_and_questions: str) -> MagicMock:
    llm = MagicMock()
    msg = MagicMock()
    msg.content = answer_and_questions
    llm.invoke.return_value = msg
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# Notebook CRUD
# ─────────────────────────────────────────────────────────────────────────────

def test_create_notebook(client: TestClient, mem):
    r = client.post(f"{_BASE}/notebooks", json={"name": "My Notes"})
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "My Notes"
    assert data["source_count"] == 0


def test_list_notebooks(client: TestClient, mem):
    client.post(f"{_BASE}/notebooks", json={"name": "A"})
    client.post(f"{_BASE}/notebooks", json={"name": "B"})
    r = client.get(f"{_BASE}/notebooks")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_get_notebook_unknown_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/notebooks/does-not-exist")
    assert r.status_code == 404


def test_get_notebook_round_trip(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    r = client.get(f"{_BASE}/notebooks/{nb_id}")
    assert r.status_code == 200
    assert r.json()["notebook_id"] == nb_id


def test_delete_notebook_unknown_returns_404(client: TestClient, mem):
    r = client.delete(f"{_BASE}/notebooks/does-not-exist")
    assert r.status_code == 404


def test_delete_notebook_round_trip(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    r = client.delete(f"{_BASE}/notebooks/{nb_id}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    assert client.get(f"{_BASE}/notebooks/{nb_id}").status_code == 404


def test_rename_notebook_unknown_returns_404(client: TestClient, mem):
    r = client.post(f"{_BASE}/notebooks/does-not-exist/rename", json={"new_name": "New"})
    assert r.status_code == 404


def test_rename_notebook_round_trip(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "Old"}).json()["notebook_id"]
    r = client.post(f"{_BASE}/notebooks/{nb_id}/rename", json={"new_name": "New"})
    assert r.status_code == 200
    assert r.json()["name"] == "New"


def test_get_history_empty_for_new_notebook(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    r = client.get(f"{_BASE}/notebooks/{nb_id}/history")
    assert r.status_code == 200
    assert r.json() == []


# ─────────────────────────────────────────────────────────────────────────────
# Source upload / removal
# ─────────────────────────────────────────────────────────────────────────────

def test_upload_source_unknown_notebook_returns_404(client: TestClient, mem):
    files = {"file": ("a.txt", b"hello", "text/plain")}
    r = client.post(f"{_BASE}/notebooks/does-not-exist/sources", files=files)
    assert r.status_code == 404


def test_upload_source_unsupported_type_returns_400(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    files = {"file": ("a.exe", b"binary junk", "application/octet-stream")}
    r = client.post(f"{_BASE}/notebooks/{nb_id}/sources", files=files)
    assert r.status_code == 400


def test_upload_source_returns_job_id(client: TestClient, mem):
    """The POST returns 202 + a job id rather than the finished result: processing
    is far too slow to hold the request open."""
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    files = {"file": ("notes.txt", b"Hello world, this is a source.", "text/plain")}
    r = client.post(f"{_BASE}/notebooks/{nb_id}/sources", files=files)
    assert r.status_code == 202
    assert r.json()["job_id"]


def test_upload_job_status_unknown_id_returns_404(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    r = client.get(f"{_BASE}/notebooks/{nb_id}/sources/jobs/nope")
    assert r.status_code == 404


def test_upload_source_round_trip(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    files = {"file": ("notes.txt", b"Hello world, this is a source.", "text/plain")}
    data = upload_source(client, nb_id, files)
    assert data["added"] is True
    assert data["duplicate"] is False
    assert data["source"]["filename"] == "notes.txt"

    detail = client.get(f"{_BASE}/notebooks/{nb_id}").json()
    assert detail["source_count"] == 1
    assert detail["sources"][0]["filename"] == "notes.txt"


def test_upload_duplicate_source_returns_duplicate_true(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    files = {"file": ("notes.txt", b"Same content.", "text/plain")}
    upload_source(client, nb_id, files)

    files2 = {"file": ("notes.txt", b"Same content.", "text/plain")}
    assert upload_source(client, nb_id, files2) == {
        "added": False, "duplicate": True, "source": None
    }


def test_remove_source_round_trip(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    files = {"file": ("notes.txt", b"Hello world.", "text/plain")}
    doc_id = upload_source(client, nb_id, files)["source"]["doc_id"]

    r = client.delete(f"{_BASE}/notebooks/{nb_id}/sources/{doc_id}")
    assert r.status_code == 200
    assert r.json()["removed"] is True
    assert client.get(f"{_BASE}/notebooks/{nb_id}").json()["source_count"] == 0


def test_remove_source_unknown_doc_id_returns_removed_false(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    r = client.delete(f"{_BASE}/notebooks/{nb_id}/sources/does-not-exist")
    assert r.status_code == 200
    assert r.json()["removed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Chat (background job + polling)
# ─────────────────────────────────────────────────────────────────────────────

def test_chat_unknown_notebook_returns_404(client: TestClient, mem):
    r = client.post(f"{_BASE}/chat", json={"notebook_id": "does-not-exist", "message": "hi"})
    assert r.status_code == 404


def test_chat_blank_message_returns_422(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    r = client.post(f"{_BASE}/chat", json={"notebook_id": nb_id, "message": "   "})
    assert r.status_code == 422


def test_get_unknown_job_returns_404(client: TestClient, mem):
    r = client.get(f"{_BASE}/jobs/does-not-exist")
    assert r.status_code == 404


def test_chat_no_sources_returns_canned_message_with_no_citations(client: TestClient, mem):
    """The answer_node's own zero-source early return -- no LLM call needed,
    so this exercises the real (unmocked) pipeline end to end."""
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]

    r = client.post(f"{_BASE}/chat", json={"notebook_id": nb_id, "message": "What is this about?"})
    assert r.status_code == 202
    data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done"
    assert data["error"] is None
    result = data["result"]
    assert "no sources yet" in result["assistant_response"]
    assert result["citations"] == []


def test_chat_with_source_returns_grounded_answer_and_citations(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    files = {"file": ("sky.txt", b"The sky is blue because of Rayleigh scattering.", "text/plain")}
    upload_source(client, nb_id, files)

    llm = _mock_llm(
        'The sky is blue due to Rayleigh scattering [1].\n'
        '{"suggested_questions": ["What is Rayleigh scattering?"]}'
    )
    with patch.object(notebook_nodes_module, "ChatOllama", return_value=llm):
        r = client.post(f"{_BASE}/chat", json={"notebook_id": nb_id, "message": "Why is the sky blue?"})
        assert r.status_code == 202
        data = _poll_until_terminal(client, r.json()["job_id"])

    assert data["status"] == "done"
    assert data["error"] is None
    result = data["result"]
    assert "[1]" in result["assistant_response"]
    assert result["citations"][0]["n"] == 1
    assert result["citations"][0]["page_label"] == "p. 1"
    assert result["suggested_questions"] == ["What is Rayleigh scattering?"]

    # The turn was persisted -- visible via both the live result and history.
    history = client.get(f"{_BASE}/notebooks/{nb_id}/history").json()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["citations"][0]["page_label"] == "p. 1"


def test_chat_pipeline_exception_surfaces_as_job_error(client: TestClient, mem):
    nb_id = client.post(f"{_BASE}/notebooks", json={"name": "X"}).json()["notebook_id"]
    files = {"file": ("sky.txt", b"The sky is blue.", "text/plain")}
    upload_source(client, nb_id, files)

    with patch.object(
        notebook_nodes_module, "ChatOllama", side_effect=RuntimeError("LLM exploded")
    ):
        r = client.post(f"{_BASE}/chat", json={"notebook_id": nb_id, "message": "Why?"})
        data = _poll_until_terminal(client, r.json()["job_id"])

    # Notebook answer_node catches LLM errors itself and returns a normal
    # (non-job-error) result with the failure folded into assistant_response.
    assert data["status"] == "done"
    result = data["result"]
    assert "Error generating answer" in result["assistant_response"]
    assert any("LLM exploded" in e for e in result["errors"])
