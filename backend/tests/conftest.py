"""Shared pytest fixtures for backend service/API tests."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

_NB_BASE = "/api/notebook"
_UPLOAD_TIMEOUT_S = 30.0


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def upload_source(
    client: TestClient,
    notebook_id: str,
    files: Dict[str, Any],
    **data: Any,
) -> Optional[Dict[str, Any]]:
    """Upload a source and block until its background job finishes.

    Uploads return 202 + a job id (processing is far too slow to hold the
    request open), so every test that just wants the finished source goes
    through here instead of reading a result straight off the POST.

    Returns the job's ``UploadSourceResult`` payload, or ``None`` if the job
    ended in error. Non-2xx POST responses are returned to the caller's
    assertions untouched by raising nothing -- use ``client.post`` directly
    when asserting on rejection status codes.
    """
    resp = client.post(
        f"{_NB_BASE}/notebooks/{notebook_id}/sources", files=files, data=data or None
    )
    assert resp.status_code == 202, f"upload rejected: {resp.status_code} {resp.text}"
    job_id = resp.json()["job_id"]

    deadline = time.monotonic() + _UPLOAD_TIMEOUT_S
    while time.monotonic() < deadline:
        status = client.get(
            f"{_NB_BASE}/notebooks/{notebook_id}/sources/jobs/{job_id}"
        ).json()
        if status["status"] == "done":
            return status["result"]
        if status["status"] == "error":
            return None
        time.sleep(0.01)
    raise AssertionError(f"upload job {job_id} did not finish within {_UPLOAD_TIMEOUT_S}s")
