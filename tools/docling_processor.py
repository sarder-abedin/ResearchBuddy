"""
tools/docling_processor.py
──────────────────────────
Docling-based document processor providing advanced PDF layout understanding,
table extraction (as Markdown), OCR for scanned documents, and support for
PPTX, XLSX, HTML, and common image formats.

Produces the same ProcessedDocument / DocumentChunk schema as DocumentProcessor
so it slots in to HybridStore and all agent pipelines without any changes there.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union

from tools.document_tools import (
    DocumentChunk,
    ProcessedDocument,
    _clean_text,
    _stable_id,
)
from tools.text_parsing import prepend_heading

logger = logging.getLogger(__name__)

# ── Format support ─────────────────────────────────────────────────────────────

_DOCLING_EXTENSIONS = frozenset({
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".html", ".htm",
    ".md", ".txt", ".rst",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp",
})

_CSV_EXTENSIONS = frozenset({".csv"})

SUPPORTED_EXTENSIONS: frozenset = _DOCLING_EXTENSIONS | _CSV_EXTENSIONS


# ── Converter singleton ────────────────────────────────────────────────────────

_converter_cache: Dict[tuple, Any] = {}


def _set_cache_env(models_path: Path) -> None:
    """Point HuggingFace + Docling model downloads to the project models dir."""
    models_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(models_path / "hf"))
    os.environ.setdefault("DOCLING_ARTIFACTS_PATH", str(models_path))
    os.environ.setdefault("TORCH_HOME", str(models_path / "torch"))


def _build_converter(use_ocr: bool, models_path: Path, generate_images: bool = False):
    """Build a Docling DocumentConverter (expensive — cached per (use_ocr, path, generate_images))."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ImportError(
            "Docling is not installed. Run: pip install docling"
        ) from exc

    _set_cache_env(models_path)

    # Try to configure PDF pipeline options with OCR
    try:
        from docling.document_converter import PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        opts_kwargs: Dict[str, Any] = {
            "do_ocr": use_ocr,
            "do_table_structure": True,
        }
        if generate_images:
            opts_kwargs["generate_picture_images"] = True
        pdf_opts = PdfPipelineOptions(**opts_kwargs)
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
        )
    except (ImportError, Exception):
        # Fallback: basic converter without explicit options
        return DocumentConverter()


def _get_converter(use_ocr: bool, models_path: Path, generate_images: bool = False):
    """Return the cached DocumentConverter for this combination of options, building it on first use."""
    key = (use_ocr, str(models_path.resolve()), generate_images)
    if key not in _converter_cache:
        _converter_cache[key] = _build_converter(use_ocr, models_path, generate_images)
    return _converter_cache[key]


def _caption_image(
    image_b64: str,
    vision_model: str,
    ollama_base_url: str,
    timeout: float = 120.0,
) -> str:
    """Caption a base64-encoded PNG image using the configured Ollama vision model.

    ``timeout`` bounds a single captioning call. Without it a vision model that
    stalls on one image blocks the upload forever, since captioning runs inline
    over every figure in the document.
    """
    try:
        import httpx
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage

        llm = ChatOllama(
            model=vision_model,
            base_url=ollama_base_url,
            temperature=0.0,
            sync_client_kwargs={"timeout": httpx.Timeout(timeout)},
        )
        resp = llm.invoke([
            HumanMessage(content=[
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"},
                {
                    "type": "text",
                    "text": (
                        "Describe this figure from an academic document. "
                        "Include: what type of figure it is (chart, diagram, photograph, etc.), "
                        "what it shows, any visible labels, titles, axes, or key values. "
                        "Be concise but complete."
                    ),
                },
            ])
        ])
        return resp.content.strip()
    except Exception as exc:
        logger.debug("Vision model captioning failed: %s", exc)
        return ""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _table_md_to_plain(md: str) -> str:
    """Convert Markdown table to plain pipe-delimited rows suitable for embedding."""
    lines = []
    for line in md.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip Markdown separator rows (|---|---|)
        if stripped.startswith("|") and all(c in "|-: " for c in stripped):
            continue
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            lines.append(" | ".join(cells))
        else:
            lines.append(stripped)
    return "\n".join(lines)


def _extract_page_num(item) -> int:
    """Safely extract 0-based page number from a Docling doc item."""
    try:
        prov = getattr(item, "prov", None) or []
        if prov:
            return max(0, int(prov[0].page_no) - 1)
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return 0


def _table_to_markdown(item, docling_doc) -> Optional[str]:
    """Export a Docling TableItem to Markdown across docling-core versions.

    Newer docling-core takes the owning document (``export_to_markdown(doc)``)
    and warns on every call made without it; older versions accept no argument.
    Try the current signature first and fall back, so neither version floods
    the logs with deprecation warnings nor breaks outright.
    """
    if hasattr(item, "export_to_markdown"):
        if docling_doc is not None:
            try:
                return item.export_to_markdown(docling_doc)
            except Exception:
                # Not just TypeError: any failure here should still fall
                # through to the remaining strategies rather than lose the
                # table, since the caller only keeps a non-None result.
                pass
        try:
            return item.export_to_markdown()
        except Exception:
            pass
    if hasattr(item, "to_dataframe"):
        try:
            return item.to_dataframe().to_markdown(index=False)
        except Exception:
            pass
    return None


def _map_docling_chunks(
    raw_chunks: list,
    doc_id: str,
    doc_name: str,
    docling_doc=None,
) -> List[DocumentChunk]:
    """Map Docling ChunkWithMetadata objects to DocumentChunk dataclass."""
    chunks: List[DocumentChunk] = []
    # Docling repeats a section's heading on every chunk within it; emitting it
    # only when it changes reproduces the document's real section structure.
    prev_heading = ""

    for i, chunk in enumerate(raw_chunks):
        text = (getattr(chunk, "text", "") or "").strip()
        if not text:
            continue

        page_num = 0
        table_md: Optional[str] = None
        heading: str = ""
        content_type = "text"

        meta = getattr(chunk, "meta", None)
        if meta is not None:
            headings = getattr(meta, "headings", None)
            if headings:
                heading = headings[-1]

            doc_items = getattr(meta, "doc_items", None) or []
            for item in doc_items:
                page_num = _extract_page_num(item)

                # Attempt to get Markdown from table items
                try:
                    from docling_core.types.doc import TableItem
                    if isinstance(item, TableItem):
                        content_type = "table"
                        md: Optional[str] = _table_to_markdown(item, docling_doc)
                        if md:
                            table_md = md
                            # Plain rows for BM25 / embeddings
                            text = _table_md_to_plain(md) or text
                except (ImportError, Exception):
                    pass

        # Bake the section heading into the chunk body. Docling keeps it only
        # in chunk.meta.headings, so without this every consumer that rebuilds
        # a document from chunk text (citation timeline, section detection,
        # Research Report context) sees a document with no section boundaries.
        if heading and heading != prev_heading:
            text = prepend_heading(text, heading)
        prev_heading = heading

        cid = _stable_id(f"{doc_id}:{i}:{text[:50]}")
        metadata: dict = {
            "source": doc_name,
            "page": page_num + 1,
            "chunk_index": i,
            "content_type": content_type,
        }
        if heading:
            metadata["heading"] = heading
        if table_md:
            metadata["table_md"] = table_md  # Markdown kept for UI rendering

        chunks.append(DocumentChunk(
            chunk_id=cid,
            doc_id=doc_id,
            doc_name=doc_name,
            page_num=page_num,
            chunk_index=i,
            text=_clean_text(text),
            metadata=metadata,
        ))

    return chunks


def _process_csv(
    file_obj: IO[bytes],
    path: Path,
    doc_id: str,
) -> List[DocumentChunk]:
    """Convert a CSV file into DocumentChunks (50 rows per chunk)."""
    content = file_obj.read().decode("utf-8", errors="replace")
    reader = csv.reader(content.splitlines())
    rows = [" | ".join(str(c) for c in row) for row in reader if any(str(c).strip() for c in row)]

    chunks: List[DocumentChunk] = []
    chunk_size = 50
    for start in range(0, max(len(rows), 1), chunk_size):
        chunk_text = "\n".join(rows[start:start + chunk_size])
        idx = start // chunk_size
        cid = _stable_id(f"{doc_id}:{idx}:{chunk_text[:50]}")
        chunks.append(DocumentChunk(
            chunk_id=cid,
            doc_id=doc_id,
            doc_name=path.name,
            page_num=0,
            chunk_index=idx,
            text=chunk_text,
            metadata={
                "source": path.name,
                "page": 1,
                "chunk_index": idx,
                "content_type": "table",
            },
        ))
    return chunks


# ── Main processor ─────────────────────────────────────────────────────────────

class DoclingProcessor:
    """
    Docling-backed document processor.

    Drop-in replacement for DocumentProcessor — returns the same
    ProcessedDocument / DocumentChunk schema consumed by HybridStore.

    Supported formats
    -----------------
    PDF (text + table extraction, optional OCR), DOCX, PPTX, XLSX,
    HTML, Markdown, TXT, PNG/JPG/JPEG (via OCR), CSV.
    """

    def __init__(
        self,
        use_ocr: bool = False,
        max_raw_chars: int = 0,
        models_path: Optional[Union[str, Path]] = None,
        vision_model: str = "",
        ollama_base_url: str = "",
        vision_timeout: Optional[float] = None,
        max_figure_captions: Optional[int] = None,
    ):
        """
        Configure OCR, the raw-text size cap, where Docling's ML model
        weights are cached on disk, and an optional vision model for figure
        captioning (defaults to disabled).

        ``vision_timeout`` and ``max_figure_captions`` bound figure captioning;
        both fall back to the configured defaults when not given.
        """
        self.use_ocr = use_ocr
        self.max_raw_chars = max_raw_chars
        self.models_path = Path(models_path or _default_models_path())
        self.vision_model = vision_model
        self.ollama_base_url = ollama_base_url
        self.vision_timeout, self.max_figure_captions = _default_vision_limits(
            vision_timeout, max_figure_captions
        )

    # ── Public API ────────────────────────────────────────────

    def process_file(
        self,
        file_path: Union[str, Path],
        file_obj: Optional[IO[bytes]] = None,
    ) -> ProcessedDocument:
        """
        Process one uploaded file into a ProcessedDocument.

        CSV files are handled directly (no Docling model needed); every other
        supported extension is routed through the Docling conversion pipeline.

        Raises:
            ValueError: if the file extension isn't in SUPPORTED_EXTENSIONS.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        doc_id = _stable_id(path.name)

        logger.info(
            "DoclingProcessor: %s (ext=%s, ocr=%s)",
            path.name, ext, self.use_ocr,
        )

        if ext in _CSV_EXTENSIONS:
            if file_obj is None:
                with open(path, "rb") as fh:
                    buf = io.BytesIO(fh.read())
            else:
                buf = file_obj
            chunks = _process_csv(buf, path, doc_id)
            raw = "\n".join(c.text for c in chunks)
            return ProcessedDocument(
                doc_id=doc_id,
                filename=path.name,
                file_type="CSV",
                total_pages=1,
                total_chunks=len(chunks),
                chunks=chunks,
                raw_text=raw,
                content_md5=hashlib.md5(raw[:50000].encode()).hexdigest(),
            )

        if ext not in _DOCLING_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            )

        return self._process_with_docling(path, file_obj, doc_id)

    def process_raw_text(self, text: str, name: str = "pasted_text") -> ProcessedDocument:
        """Plain text input — delegates to DocumentProcessor (no Docling needed)."""
        from tools.document_tools import DocumentProcessor
        return DocumentProcessor().process_raw_text(text, name)

    # ── Docling pipeline ──────────────────────────────────────

    def _process_with_docling(
        self,
        path: Path,
        file_obj: Optional[IO[bytes]],
        doc_id: str,
    ) -> ProcessedDocument:
        """Run the Docling conversion + chunking pipeline and assemble a ProcessedDocument."""
        converter = _get_converter(self.use_ocr, self.models_path, bool(self.vision_model))

        # Docling requires a real file path — write in-memory streams to a temp file
        tmp_path: Optional[Path] = None
        try:
            if file_obj is not None:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=path.suffix
                ) as tmp:
                    tmp.write(file_obj.read())
                    tmp_path = Path(tmp.name)
                source = tmp_path
            else:
                source = path

            result = converter.convert(source)
            docling_doc = result.document
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        # Structure-aware chunking
        chunks = self._chunk_document(docling_doc, doc_id, path.name)
        if self.vision_model:
            figure_chunks = self._extract_figure_chunks(docling_doc, doc_id, path.name, len(chunks))
            chunks.extend(figure_chunks)

        # Export raw text as Markdown (best quality, preserves headings)
        try:
            raw_text = docling_doc.export_to_markdown()
        except Exception:
            raw_text = "\n\n".join(c.text for c in chunks)

        if self.max_raw_chars and len(raw_text) > self.max_raw_chars:
            raw_text = raw_text[: self.max_raw_chars]

        # Page count
        try:
            total_pages = len(docling_doc.pages)
        except Exception:
            total_pages = max((c.page_num for c in chunks), default=0) + 1

        content_md5 = hashlib.md5(raw_text[:50000].encode()).hexdigest()

        return ProcessedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_type=path.suffix.lstrip(".").upper(),
            total_pages=total_pages,
            total_chunks=len(chunks),
            chunks=chunks,
            raw_text=raw_text,
            content_md5=content_md5,
        )

    def _chunk_document(
        self, docling_doc, doc_id: str, doc_name: str
    ) -> List[DocumentChunk]:
        """Chunk with Docling HybridChunker; fall back to char-based on failure."""
        try:
            from docling.chunking import HybridChunker
            try:
                chunker = HybridChunker(max_tokens=256)
            except TypeError:
                chunker = HybridChunker()
            raw_chunks = list(chunker.chunk(docling_doc))
            mapped = _map_docling_chunks(raw_chunks, doc_id, doc_name, docling_doc)
            if mapped:
                return mapped
        except Exception as exc:
            logger.warning(
                "HybridChunker failed (%s) — using text-export fallback", exc
            )
        return self._fallback_chunks(docling_doc, doc_id, doc_name)

    def _fallback_chunks(
        self, docling_doc, doc_id: str, doc_name: str
    ) -> List[DocumentChunk]:
        """Export to Markdown then split with the char-based chunker."""
        from tools.document_tools import _chunk_text

        try:
            text = docling_doc.export_to_markdown()
        except Exception:
            text = str(docling_doc)

        text = _clean_text(text)
        raw_chunks = _chunk_text(text, chunk_size=800, overlap=150)
        result: List[DocumentChunk] = []
        for i, chunk_text in enumerate(raw_chunks):
            cid = _stable_id(f"{doc_id}:{i}:{chunk_text[:50]}")
            result.append(DocumentChunk(
                chunk_id=cid,
                doc_id=doc_id,
                doc_name=doc_name,
                page_num=0,
                chunk_index=i,
                text=chunk_text,
                metadata={"source": doc_name, "page": 0, "chunk_index": i},
            ))
        return result

    def _extract_figure_chunks(
        self,
        docling_doc,
        doc_id: str,
        doc_name: str,
        start_index: int,
    ) -> List[DocumentChunk]:
        """Caption each PictureItem in the Docling document using the vision model.

        Returns one DocumentChunk per figure, with content_type="figure" and the
        caption as the chunk text. Silently skips any figure that fails extraction
        or captioning — never blocks the main pipeline.

        Captioning is one LLM call per figure, run sequentially, so a
        figure-heavy paper is capped at ``max_figure_captions`` figures to bound
        how long an upload can take. Remaining figures are skipped with a log
        line rather than silently.
        """
        if not self.vision_model:
            return []

        try:
            from docling_core.types.doc import PictureItem
        except ImportError:
            return []

        import base64
        import io as _io

        chunks: List[DocumentChunk] = []
        figure_index = 0
        skipped = 0
        try:
            for item, _level in docling_doc.iterate_items():
                if not isinstance(item, PictureItem):
                    continue
                if figure_index >= self.max_figure_captions:
                    skipped += 1
                    continue
                try:
                    pil_img = item.get_image(docling_doc)
                    if pil_img is None:
                        continue
                    buf = _io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    image_b64 = base64.b64encode(buf.getvalue()).decode()

                    caption = _caption_image(
                        image_b64, self.vision_model, self.ollama_base_url,
                        timeout=self.vision_timeout,
                    )
                    if not caption:
                        continue

                    page_num = _extract_page_num(item)
                    idx = start_index + figure_index
                    cid = _stable_id(f"{doc_id}:figure:{figure_index}:{caption[:50]}")
                    chunks.append(DocumentChunk(
                        chunk_id=cid,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page_num=page_num,
                        chunk_index=idx,
                        text=caption,
                        metadata={
                            "source": doc_name,
                            "page": page_num + 1,
                            "chunk_index": idx,
                            "content_type": "figure",
                        },
                    ))
                    figure_index += 1
                except Exception as exc:
                    logger.debug("Figure captioning skipped for one item: %s", exc)
        except Exception as exc:
            logger.debug("Figure extraction pass failed: %s", exc)

        if skipped:
            logger.info(
                "Captioned %d figures; skipped %d over the MAX_FIGURE_CAPTIONS "
                "limit of %d (raise it in .env to caption more)",
                figure_index, skipped, self.max_figure_captions,
            )

        return chunks


# ── Helpers ────────────────────────────────────────────────────────────────────

def _default_models_path() -> Path:
    """Return the configured Docling models directory, or a relative fallback if settings can't be read."""
    try:
        from config.settings import get_settings
        return Path(get_settings().docling_models_path)
    except Exception:
        return Path("models/docling")


def _default_vision_limits(
    timeout: Optional[float], max_captions: Optional[int]
) -> tuple:
    """Resolve the figure-captioning budget, falling back to configured defaults.

    Mirrors _default_models_path()'s tolerance of an unreadable settings file so
    the processor stays constructible in tests and minimal environments.
    """
    if timeout is not None and max_captions is not None:
        return float(timeout), int(max_captions)
    try:
        from config.settings import get_settings
        cfg = get_settings()
        default_timeout, default_max = cfg.vision_timeout, cfg.max_figure_captions
    except Exception:
        default_timeout, default_max = 120.0, 12
    return (
        float(default_timeout if timeout is None else timeout),
        int(default_max if max_captions is None else max_captions),
    )
