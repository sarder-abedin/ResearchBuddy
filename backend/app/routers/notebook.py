"""backend/app/routers/notebook.py
───────────────────────────────────
Mode 2 Phase A (Research Notebook core) over HTTP: notebook CRUD, source
upload/removal, conversation history, and chat turns.

Chat follows the same background-job + polling pattern as Mode 1 / Mode 3:
``POST /chat`` kicks off ``agents.notebook_graph.run_notebook_turn`` on a
background thread and returns a job id immediately (202 Accepted); the
frontend polls ``GET /jobs/{job_id}`` for the same ``stream_callback``
progress (retrieve/answer/save/notebook_eval) the Streamlit tab already
renders live, then the final result once ``status == "done"``.

Everything else (notebook CRUD, source upload/removal, history) is a small,
fast, non-LLM operation against ``NotebookMemory``, so those run synchronously.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .. import jobs

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
from ..schemas.jobs import JobCreated
from ..schemas.notebook import (
    ChatJobStatus,
    ChatRequest,
    ConversationTurn,
    CreateNotebookRequest,
    DeleteNotebookResult,
    NotebookDetail,
    NotebookSummary,
    RemoveSourceResult,
    RenameNotebookRequest,
    UploadSourceResult,
)
from ..services import notebook_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/notebook", tags=["notebook"])


# ─────────────────────────────────────────────────────────────────────────────
# Notebook CRUD
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/notebooks", response_model=NotebookSummary, status_code=201)
def create_notebook(req: CreateNotebookRequest) -> NotebookSummary:
    return notebook_service.create_notebook(req)


@router.get("/notebooks", response_model=List[NotebookSummary])
def list_notebooks() -> List[NotebookSummary]:
    return notebook_service.list_notebooks()


@router.get("/notebooks/{notebook_id}", response_model=NotebookDetail)
def get_notebook(notebook_id: str) -> NotebookDetail:
    detail = notebook_service.get_notebook_detail(notebook_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
    return detail


@router.delete("/notebooks/{notebook_id}", response_model=DeleteNotebookResult)
def delete_notebook(notebook_id: str) -> DeleteNotebookResult:
    if not notebook_service.delete_notebook(notebook_id):
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
    return DeleteNotebookResult(deleted=True)


@router.post("/notebooks/{notebook_id}/rename", response_model=NotebookSummary)
def rename_notebook(notebook_id: str, req: RenameNotebookRequest) -> NotebookSummary:
    summary = notebook_service.rename_notebook(notebook_id, req.new_name)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
    return summary


@router.get("/notebooks/{notebook_id}/history", response_model=List[ConversationTurn])
def get_history(notebook_id: str, max_turns: int = 8) -> List[ConversationTurn]:
    return notebook_service.get_history(notebook_id, max_turns=max_turns)


# ─────────────────────────────────────────────────────────────────────────────
# Source upload / removal
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/notebooks/{notebook_id}/sources", response_model=UploadSourceResult)
async def upload_source(
    notebook_id: str,
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None, gt=0),
    chunk_overlap: Optional[int] = Form(None, ge=0),
    use_docling: bool = Form(False),
    use_ocr: bool = Form(False),
    large_doc_page_threshold: int = Form(50, ge=1, le=500),
    vision_model: str = Form(""),
) -> UploadSourceResult:
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower() or ".bin"

    # Stream upload directly to disk in 64 KiB chunks — never hold the full
    # file in RAM alongside the extraction buffers (avoids OOM on large PDFs).
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    try:
        file_size = 0
        while chunk := await file.read(1 << 16):
            file_size += len(chunk)
            if file_size > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                )
            tmp.write(chunk)
        tmp.close()

        try:
            # Offloaded to a worker thread: notebook_service.upload_source() is
            # fully synchronous and slow (Docling ML conversion, then one vision
            # LLM call per figure). Awaiting it inline on the event loop froze
            # the whole app for the duration -- health checks, job polling and
            # every other request included -- so an upload looked like it had
            # stalled. Every other endpoint in this router is a plain `def`,
            # which FastAPI already runs in a threadpool; this one has to stay
            # `async def` for the streaming `await file.read()` above.
            return await run_in_threadpool(
                notebook_service.upload_source,
                notebook_id, filename, tmp_path,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                use_docling=use_docling, use_ocr=use_ocr,
                large_doc_page_threshold=large_doc_page_threshold,
                vision_model=vision_model,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            tmp.close()
        except Exception:
            pass
        tmp_path.unlink(missing_ok=True)


@router.delete("/notebooks/{notebook_id}/sources/{doc_id}", response_model=RemoveSourceResult)
def remove_source(notebook_id: str, doc_id: str) -> RemoveSourceResult:
    return RemoveSourceResult(removed=notebook_service.remove_source(notebook_id, doc_id))


# ─────────────────────────────────────────────────────────────────────────────
# Chat (background job + polling)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=JobCreated, status_code=202)
def chat(req: ChatRequest) -> JobCreated:
    if not notebook_service.notebook_exists(req.notebook_id):
        raise HTTPException(status_code=404, detail=f"Notebook '{req.notebook_id}' not found.")
    job = jobs.create_job()
    jobs.run_in_background(job, lambda cb: notebook_service.run_chat_turn(req, cb))
    return JobCreated(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=ChatJobStatus)
def get_job_status(job_id: str) -> ChatJobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ChatJobStatus(
        id=job.id,
        status=job.status,
        stage=job.stage,
        stage_info=job.stage_info,
        error=job.error,
        result=job.result,
    )
