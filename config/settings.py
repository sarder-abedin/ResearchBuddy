"""
config/settings.py
──────────────────
Centralised configuration loaded from environment variables (.env).
Using Pydantic BaseSettings so every value is typed and validated at
startup — no silent misconfigurations at runtime.

TUTORIAL NOTE
─────────────
All defaults here work out-of-the-box with a local Ollama installation.
For academic APIs (Semantic Scholar, CrossRef, arXiv) no API key is
required; optional keys only unlock higher rate limits.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings  # pip install pydantic-settings

# Load .env from project root (two levels up from this file)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


class Settings(BaseSettings):
    """Typed application configuration, populated from environment variables / `.env`.

    Every field has a sensible local-first default (no API keys required for
    Ollama or the academic search APIs), so the app runs out-of-the-box.
    Access via `get_settings()` rather than instantiating directly, so the
    whole app shares one cached instance.
    """

    # ── Local LLM ───────────────────────────────────────────
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.1:8b", alias="OLLAMA_MODEL")
    num_ctx: int = Field(32768, alias="NUM_CTX")

    # ── Research Notebook temperature tuning ─────────────────
    # One of "precise" | "focused" | "balanced" | "creative" — see
    # tools/temperature_levels.py. "focused" preserves BeeSearch's
    # original per-call temperature tuning unchanged.
    temperature_level: str = Field("focused", alias="TEMPERATURE_LEVEL")

    # ── Hybrid RAG ───────────────────────────────────────────
    embedding_model: str = Field("nomic-embed-text", alias="EMBED_MODEL")
    chroma_persist_dir: str = Field("./outputs/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("research_embeddings", alias="CHROMA_COLLECTION_NAME")
    hybrid_top_k: int = Field(8, alias="HYBRID_TOP_K")

    # ── Docker / Deployment ──────────────────────────────────
    app_port: int = Field(8501, alias="APP_PORT")

    # ── Semantic Scholar ─────────────────────────────────────
    semantic_scholar_api_key: str = Field("", alias="SEMANTIC_SCHOLAR_API_KEY")
    semantic_scholar_base_url: str = Field(
        "https://api.semanticscholar.org/graph/v1",
        alias="SEMANTIC_SCHOLAR_BASE_URL",
    )

    # ── CrossRef ─────────────────────────────────────────────
    crossref_base_url: str = Field(
        "https://api.crossref.org/works", alias="CROSSREF_BASE_URL"
    )
    crossref_email: str = Field("researcher@example.com", alias="CROSSREF_EMAIL")

    # ── arXiv ────────────────────────────────────────────────
    arxiv_max_results: int = Field(10, alias="ARXIV_MAX_RESULTS")

    # ── Document Processing ──────────────────────────────────
    max_document_chunks: int = Field(500, alias="MAX_DOCUMENT_CHUNKS")
    chunk_size: int = Field(800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(150, alias="CHUNK_OVERLAP")
    docling_models_path: str = Field("models/docling", alias="DOCLING_MODELS_PATH")
    # PDFs larger than this page count auto-switch from Docling to pdfplumber
    # to avoid loading Docling's ML models (~500 MB) on resource-constrained machines.
    large_doc_page_threshold: int = Field(50, alias="LARGE_DOC_PAGE_THRESHOLD")

    # ── Google Search FastAPI Service ────────────────────────
    google_search_service_url: str = Field(
        "http://localhost:8000", alias="GOOGLE_SEARCH_SERVICE_URL"
    )

    # ── Search ───────────────────────────────────────────────
    max_search_results: int = Field(8, alias="MAX_SEARCH_RESULTS")

    # ── Output ───────────────────────────────────────────────
    output_dir: str = Field("./outputs", alias="OUTPUT_DIR")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ── Vision model for figure captioning (optional) ────────
    # Set to an Ollama vision model (e.g. "llava:7b", "llama3.2-vision:11b")
    # to enable automatic captioning of figures in uploaded PDFs.
    # Leave blank to skip figure extraction (default, no overhead).
    vision_model: str = Field("", alias="VISION_MODEL")
    # Per-figure captioning budget. Captioning is one LLM call per figure and
    # runs inline during upload, so both a timeout and a cap are needed to keep
    # a figure-heavy paper from stalling the upload indefinitely.
    vision_timeout: float = Field(120.0, alias="VISION_TIMEOUT")
    max_figure_captions: int = Field(12, alias="MAX_FIGURE_CAPTIONS")

    # ── Langfuse Observability (optional) ────────────────────
    langfuse_public_key: str = Field("", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field("", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("http://localhost:3000", alias="LANGFUSE_HOST")

    class Config:
        """Pydantic settings config: load `.env` from the project root and
        allow constructing `Settings` with either field names or their
        `alias=` env-var names."""

        env_file = str(_ROOT / ".env")
        env_file_encoding = "utf-8"
        populate_by_name = True

    # ── Helpers ──────────────────────────────────────────────
    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for d in [self.output_dir, self.chroma_persist_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    The `lru_cache` means `.env` is only read once per process — restart the
    app (not just rerun the Streamlit script) to pick up `.env` changes.
    """
    s = Settings()
    s.ensure_output_dirs()
    return s
