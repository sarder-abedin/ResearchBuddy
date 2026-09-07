"""backend/app/schemas/notebook.py
───────────────────────────────────
Pydantic request/response shapes for Mode 2 (Research Notebook) Phase A:
notebook CRUD, source upload/removal, conversation history, and chat turns.

Mirrors ``NotebookMemory``'s on-disk shapes (``list_notebooks()``/``load()``'s
dict keys) and ``NotebookState``'s final-state keys (``agents/notebook_state.py``)
so a raw dict returned by the service layer can be handed straight to a
response model and validated/coerced field-by-field, the same pattern
``AskJobStatus``/``SRJobStatus`` already use for their ``result`` field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .jobs import JobStatusBase

TemperatureLevel = Literal["precise", "focused", "balanced", "creative"]


# ─────────────────────────────────────────────────────────────────────────────
# Notebook CRUD
# ─────────────────────────────────────────────────────────────────────────────

class CreateNotebookRequest(BaseModel):
    name: str = Field(
        "", description="Display name; blank defaults to 'Untitled Notebook' (mirrors NotebookMemory.new_notebook)."
    )


class RenameNotebookRequest(BaseModel):
    new_name: str = Field(
        "", description="New display name; blank keeps the current name (mirrors NotebookMemory.rename)."
    )


class NotebookSummary(BaseModel):
    notebook_id: str
    name: str
    source_count: int
    turn_count: int
    source_names: List[str] = Field(default_factory=list)
    created_at: str
    last_modified: str


class SourceMeta(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    source_type: str
    url: str = ""
    total_pages: int
    total_chunks: int
    content_md5: str = ""
    added_at: str = ""


class CitationItem(BaseModel):
    n: int
    doc_name: str = "unknown"
    page: int = 0
    page_label: str = "n/a"
    snippet: str = ""
    url: str = ""


class ConversationTurn(BaseModel):
    role: str
    content: str
    timestamp: str = ""
    citations: Optional[List[CitationItem]] = None
    suggested_questions: Optional[List[str]] = None


class SavedReview(BaseModel):
    doc_id: str = ""
    doc_filename: str = ""
    review_text: str = ""
    external_refs: List[Dict[str, Any]] = Field(default_factory=list)
    generated_at: str = ""


class NotebookDetail(BaseModel):
    notebook_id: str
    name: str
    source_count: int
    turn_count: int
    sources: List[SourceMeta] = Field(default_factory=list)
    conversation: List[ConversationTurn] = Field(default_factory=list)
    created_at: str
    last_modified: str
    saved_reviews: Dict[str, SavedReview] = Field(default_factory=dict)
    reviewer_chats: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


class UploadSourceResult(BaseModel):
    added: bool
    duplicate: bool = False
    source: Optional[SourceMeta] = None


class UploadJobStatus(JobStatusBase):
    """Progress of a source upload running on the background job runner.

    Uploads are slow enough (Docling layout ML, then one vision LLM call per
    figure) that holding the HTTP request open for them left the browser
    waiting minutes with no feedback, so they follow the same 202 + poll
    pattern as chat.
    """

    result: Optional[UploadSourceResult] = None


class RemoveSourceResult(BaseModel):
    removed: bool


class DeleteNotebookResult(BaseModel):
    deleted: bool


# ─────────────────────────────────────────────────────────────────────────────
# Chat (background job + polling, same pattern as Mode 1 / Mode 3)
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    notebook_id: str = Field(..., description="Target notebook id, returned by /notebooks (POST).")
    message: str = Field(..., description="The user's question for this turn.")
    include_web_search: bool = Field(
        False, description="Also search the web (DuckDuckGo) alongside this notebook's sources."
    )
    model: Optional[str] = Field(None, description="Ollama model override; omit to use the server's configured default.")
    num_ctx: Optional[int] = Field(None, gt=0, description="Context window override (tokens).")
    embed_model: Optional[str] = Field(None, description="Ollama embedding model override for Hybrid RAG retrieval.")
    top_k: Optional[int] = Field(None, gt=0, description="Number of chunks to retrieve; omit to use the server default.")
    temperature_level: Optional[TemperatureLevel] = Field(
        None, description="Response tuning level; omit to use the module default ('focused')."
    )

    @field_validator("notebook_id")
    @classmethod
    def notebook_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("notebook_id is required.")
        return v

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Please enter a question.")
        return v


class ChatResult(BaseModel):
    notebook_id: str
    user_message: str = ""
    assistant_response: str = ""
    citations: List[CitationItem] = Field(default_factory=list)
    suggested_questions: List[str] = Field(default_factory=list)
    source_count: int = 0
    retrieval_mode: str = "empty"
    eval_result: Dict[str, Any] = Field(default_factory=dict)
    rag_reflection_info: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    progress_pct: int = 100


class ChatJobStatus(JobStatusBase):
    result: Optional[ChatResult] = None
