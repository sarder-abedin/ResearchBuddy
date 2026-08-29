"""
agents/notebook_memory.py
──────────────────────────
Long-term persistence for the Research Notebook (Mode 8) — a NotebookLM-style
mode where users build a notebook from their own sources and chat with it
using grounded, cited retrieval.

State is split into two parts in SQLite:
  • meta_json column in notebooks table stores:
      {name, sources, conversation, created_at, last_modified}  (NO chunks)
  • Chunks stored separately in notebook_chunks table

This allows efficient querying without loading all chunk text when only
metadata is needed (e.g. list_notebooks).
"""

from __future__ import annotations

import logging
import random
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.session_db import _tx, init_db, pack, unpack
from tools.text_parsing import join_chunks_with_headings

logger = logging.getLogger(__name__)


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _short_id(length: int = 8) -> str:
    """Generate a short random lowercase-alphanumeric id for a new notebook."""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


class NotebookMemory:
    """
    Persistent store for Research Notebook sessions.

    Typical lifecycle
    -----------------
    mem = NotebookMemory()

    # Create a notebook
    nb_id = mem.new_notebook(name="Antibiotic Resistance")

    # Add a processed source (after DocumentProcessor + HybridStore indexing)
    mem.add_source(nb_id, processed_document, source_type="file")

    # Chat: append turns with citations
    mem.add_turn(nb_id, "user", "What datasets are used?")
    mem.add_turn(nb_id, "assistant", "The study uses ...[1]",
                 citations=[{"n": 1, "doc_name": "paper.pdf", "page": 4}])

    # Reload later
    notebook = mem.load(nb_id)
    """

    def __init__(self, db_path: Path | None = None):
        """Open (and lazily initialise) the notebooks SQLite database.

        Parameters
        ----------
        db_path : Override path for the SQLite file; defaults to the
                  package-standard location inside `init_db`/`_tx` when None.
        """
        self._db_path = db_path
        init_db(self._db_path)

    # ── Notebook management ───────────────────────────────────

    def new_notebook(self, name: str, notebook_id: str = "") -> str:
        """Create a new, empty notebook and return its id."""
        nb_id = notebook_id or _short_id()
        now = _now()
        meta: Dict[str, Any] = {
            "name": name.strip() or "Untitled Notebook",
            "sources": [],          # list of source metadata dicts
            "conversation": [],     # list of {role, content, timestamp, citations, ...}
            "created_at": now,
            "last_modified": now,
        }
        with _tx(self._db_path) as conn:
            conn.execute(
                "INSERT INTO notebooks VALUES (?,?,?,?,?,?,?)",
                (nb_id, now, now, meta["name"], 0, 0, pack(meta)),
            )
        logger.info("New notebook: %s — %s", nb_id, name[:60])
        return nb_id

    def load(self, notebook_id: str) -> Optional[Dict[str, Any]]:
        """Load a notebook by id. Returns None if not found."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT meta_json FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
            if row is None:
                return None
            meta = unpack(row["meta_json"])
            chunks = conn.execute(
                """SELECT chunk_id, doc_id, doc_name, page_num, chunk_index, text
                   FROM notebook_chunks WHERE notebook_id=?""",
                (notebook_id,),
            ).fetchall()
        meta["chunks"] = [dict(c) for c in chunks]
        meta["notebook_id"] = notebook_id
        return meta

    def list_notebooks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return summary info for all notebooks, newest first."""
        with _tx(self._db_path) as conn:
            rows = conn.execute(
                """SELECT notebook_id, created_at, updated_at, name,
                          source_count, turn_count, meta_json
                   FROM notebooks
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        notebooks = []
        for row in rows:
            meta = unpack(row["meta_json"])
            notebooks.append({
                "notebook_id": row["notebook_id"],
                "name": row["name"] or "Untitled",
                "source_count": row["source_count"],
                "turn_count": row["turn_count"],
                "source_names": [s.get("filename", "") for s in meta.get("sources", [])],
                "created_at": row["created_at"],
                "last_modified": row["updated_at"],
            })
        return notebooks

    def delete(self, notebook_id: str) -> bool:
        """Delete a notebook (and, via FK cascade, its chunks). Returns True if a row was removed."""
        with _tx(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            )
            return cursor.rowcount > 0

    def rename(self, notebook_id: str, new_name: str) -> bool:
        """Update a notebook's display name. Returns False if the notebook doesn't exist."""
        nb = self.load(notebook_id)
        if nb is None:
            return False
        cleaned = new_name.strip() or nb.get("name", "Untitled Notebook")
        now = _now()
        # Rebuild meta without notebook_id/chunks keys
        meta = {k: v for k, v in nb.items() if k not in ("notebook_id", "chunks")}
        meta["name"] = cleaned
        meta["last_modified"] = now
        with _tx(self._db_path) as conn:
            conn.execute(
                "UPDATE notebooks SET name=?, updated_at=?, meta_json=? WHERE notebook_id=?",
                (cleaned, now, pack(meta), notebook_id),
            )
        return True

    # ── Source management ─────────────────────────────────────

    def add_source(
        self,
        notebook_id: str,
        processed_doc: Any,
        source_type: str = "file",
        url: str = "",
    ) -> bool:
        """
        Append a processed source's metadata + chunks to the notebook.

        Parameters
        ----------
        processed_doc : a tools.document_tools.ProcessedDocument
        source_type   : "file" or "url"
        url           : original URL when source_type == "url"
        """
        with _tx(self._db_path) as conn:
            # Check notebook exists
            row = conn.execute(
                "SELECT meta_json, source_count FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
            if row is None:
                logger.warning("add_source: notebook %s not found", notebook_id)
                return False

            # Duplicate check
            dup = conn.execute(
                "SELECT 1 FROM notebook_chunks WHERE notebook_id=? AND doc_id=? LIMIT 1",
                (notebook_id, processed_doc.doc_id),
            ).fetchone()
            if dup is not None:
                logger.info(
                    "Source %s already in notebook %s — skipping",
                    processed_doc.filename, notebook_id,
                )
                return False

            meta = unpack(row["meta_json"])
            meta.setdefault("sources", []).append({
                "doc_id": processed_doc.doc_id,
                "filename": processed_doc.filename,
                "file_type": processed_doc.file_type,
                "source_type": source_type,
                "url": url,
                "total_pages": processed_doc.total_pages,
                "total_chunks": processed_doc.total_chunks,
                "content_md5": processed_doc.content_md5,
                "added_at": _now(),
            })
            now = _now()
            meta["last_modified"] = now
            source_count = len(meta["sources"])

            conn.execute(
                "UPDATE notebooks SET source_count=?, updated_at=?, meta_json=? WHERE notebook_id=?",
                (source_count, now, pack(meta), notebook_id),
            )

            # Insert chunks
            conn.executemany(
                """INSERT INTO notebook_chunks
                   (chunk_id, notebook_id, doc_id, doc_name, page_num, chunk_index, text)
                   VALUES (?,?,?,?,?,?,?)""",
                [
                    (ch.chunk_id, notebook_id, ch.doc_id, ch.doc_name,
                     ch.page_num, ch.chunk_index, ch.text)
                    for ch in processed_doc.chunks
                ],
            )

        logger.info(
            "Added source '%s' (%d chunks) to notebook %s",
            processed_doc.filename, processed_doc.total_chunks, notebook_id,
        )
        return True

    def remove_source(self, notebook_id: str, doc_id: str) -> bool:
        """Remove a source and all of its chunks from the notebook."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT meta_json, source_count FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
            if row is None:
                return False

            meta = unpack(row["meta_json"])
            before = len(meta.get("sources", []))
            meta["sources"] = [
                s for s in meta.get("sources", []) if s.get("doc_id") != doc_id
            ]
            if len(meta["sources"]) == before:
                return False

            now = _now()
            meta["last_modified"] = now
            source_count = len(meta["sources"])

            conn.execute(
                "DELETE FROM notebook_chunks WHERE notebook_id=? AND doc_id=?",
                (notebook_id, doc_id),
            )
            conn.execute(
                "DELETE FROM notebook_source_files WHERE notebook_id=? AND doc_id=?",
                (notebook_id, doc_id),
            )
            conn.execute(
                "UPDATE notebooks SET source_count=?, updated_at=?, meta_json=? WHERE notebook_id=?",
                (source_count, now, pack(meta), notebook_id),
            )

        logger.info("Removed source %s from notebook %s", doc_id, notebook_id)
        return True

    # ── Raw source-file bytes (PDF jump-navigation) ────────────

    def add_source_file(self, notebook_id: str, doc_id: str, filename: str,
                         file_bytes: bytes, mime_type: str = "application/pdf") -> bool:
        """
        Persist a source's original uploaded bytes, so the UI can later jump
        to a specific page in its real PDF instead of just showing a text
        excerpt. Skipped (returns False) when file_bytes is empty — e.g.
        non-PDF sources, which process_uploads() never sets raw_bytes for.
        """
        if not file_bytes:
            return False
        with _tx(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO notebook_source_files
                   (doc_id, notebook_id, filename, mime_type, file_bytes, added_at)
                   VALUES (?,?,?,?,?,?)""",
                (doc_id, notebook_id, filename, mime_type, file_bytes, _now()),
            )
        return True

    def get_source_file(self, notebook_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return {"filename", "mime_type", "file_bytes"} for a stored source, or None."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                """SELECT filename, mime_type, file_bytes FROM notebook_source_files
                   WHERE notebook_id=? AND doc_id=?""",
                (notebook_id, doc_id),
            ).fetchone()
        return dict(row) if row else None

    def delete_source_file(self, notebook_id: str, doc_id: str) -> bool:
        """Remove a stored source file's bytes. Returns True if a row was removed."""
        with _tx(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM notebook_source_files WHERE notebook_id=? AND doc_id=?",
                (notebook_id, doc_id),
            )
            return cursor.rowcount > 0

    # ── Conversation ──────────────────────────────────────────

    def add_turn(
        self,
        notebook_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        suggested_questions: Optional[List[str]] = None,
    ) -> None:
        """Append a single conversation turn."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT meta_json, turn_count FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
            if row is None:
                logger.warning("add_turn: notebook %s not found", notebook_id)
                return
            meta = unpack(row["meta_json"])
            meta.setdefault("conversation", []).append({
                "role": role,
                "content": content,
                "timestamp": _now(),
                "citations": citations,
                "suggested_questions": suggested_questions,
            })
            now = _now()
            meta["last_modified"] = now
            turn_count = len(meta["conversation"])
            conn.execute(
                "UPDATE notebooks SET turn_count=?, updated_at=?, meta_json=? WHERE notebook_id=?",
                (turn_count, now, pack(meta), notebook_id),
            )

    def get_history(self, notebook_id: str, max_turns: int = 8) -> List[Dict]:
        """Return the last `max_turns` conversation turns."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT meta_json FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
        if row is None:
            return []
        meta = unpack(row["meta_json"])
        return meta.get("conversation", [])[-max_turns:]

    # ── Paper reviews ─────────────────────────────────────────

    def save_review(
        self,
        notebook_id: str,
        doc_id: str,
        doc_filename: str,
        review_text: str,
        external_refs: List[Dict[str, Any]],
    ) -> bool:
        """Persist a generated paper review in the notebook's meta_json under reviews[doc_id]."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT meta_json FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
            if row is None:
                return False
            meta = unpack(row["meta_json"])
            meta.setdefault("reviews", {})[doc_id] = {
                "doc_id": doc_id,
                "doc_filename": doc_filename,
                "review_text": review_text,
                "external_refs": external_refs,
                "generated_at": _now(),
            }
            now = _now()
            meta["last_modified"] = now
            conn.execute(
                "UPDATE notebooks SET updated_at=?, meta_json=? WHERE notebook_id=?",
                (now, pack(meta), notebook_id),
            )
        return True

    def get_review(self, notebook_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored review for a given doc, or None if none was saved."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT meta_json FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
        if row is None:
            return None
        meta = unpack(row["meta_json"])
        return meta.get("reviews", {}).get(doc_id)

    # ── Reviewer critique-validation chat ─────────────────────

    def save_reviewer_chat(
        self,
        notebook_id: str,
        doc_id: str,
        chat_history: List[Dict[str, Any]],
    ) -> bool:
        """Persist the full critique-validation chat history for a given doc."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT meta_json FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
            if row is None:
                return False
            meta = unpack(row["meta_json"])
            meta.setdefault("reviewer_chats", {})[doc_id] = chat_history
            now = _now()
            meta["last_modified"] = now
            conn.execute(
                "UPDATE notebooks SET updated_at=?, meta_json=? WHERE notebook_id=?",
                (now, pack(meta), notebook_id),
            )
        return True

    def get_reviewer_chat(self, notebook_id: str, doc_id: str) -> List[Dict[str, Any]]:
        """Return the stored critique-validation chat history for a given doc, or []."""
        with _tx(self._db_path) as conn:
            row = conn.execute(
                "SELECT meta_json FROM notebooks WHERE notebook_id=?",
                (notebook_id,),
            ).fetchone()
        if row is None:
            return []
        meta = unpack(row["meta_json"])
        return meta.get("reviewer_chats", {}).get(doc_id, [])

    # ── Cross-notebook search ─────────────────────────────────

    def search_all_notebooks(
        self,
        query: str,
        limit: int = 30,
        notebook_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Keyword-search every chunk in every notebook (or a chosen subset) —
        "search across everything I've ever uploaded".

        This runs a lightweight, dependency-free substring/keyword search
        directly over the shared `notebook_chunks` table (the same SQLite
        store every notebook's text already lives in), then ranks candidates
        in Python by how many distinct query terms each chunk contains and
        how often they occur. It deliberately avoids requiring each
        notebook's FAISS/BM25 index to be built or loaded — those are
        per-notebook, in-memory, and expensive to construct on demand —
        so results return instantly even for notebooks you haven't opened
        in this session.

        Returns hits ordered by relevance, each shaped as:
            {notebook_id, notebook_name, doc_id, doc_name, page_num,
             chunk_index, text, snippet, matched_terms, score}
        """
        terms = sorted({t.lower() for t in query.split() if t.strip()})
        if not terms:
            return []

        where_terms = " OR ".join(["LOWER(c.text) LIKE ?"] * len(terms))
        params: List[Any] = [f"%{t}%" for t in terms]

        sql = (
            "SELECT c.notebook_id, c.doc_id, c.doc_name, c.page_num, c.chunk_index, c.text, "
            "       n.name AS notebook_name "
            "FROM notebook_chunks c "
            "JOIN notebooks n ON n.notebook_id = c.notebook_id "
            f"WHERE ({where_terms})"
        )
        if notebook_ids:
            sql += f" AND c.notebook_id IN ({','.join('?' for _ in notebook_ids)})"
            params.extend(notebook_ids)
        sql += " LIMIT 1000"  # cap candidates fetched before in-Python ranking

        with _tx(self._db_path) as conn:
            rows = conn.execute(sql, params).fetchall()

        hits: List[Dict[str, Any]] = []
        for row in rows:
            text = row["text"] or ""
            lowered = text.lower()
            matched_terms = sum(1 for t in terms if t in lowered)
            if not matched_terms:
                continue
            occurrences = sum(lowered.count(t) for t in terms)
            hits.append({
                "notebook_id": row["notebook_id"],
                "notebook_name": row["notebook_name"] or "Untitled",
                "doc_id": row["doc_id"],
                "doc_name": row["doc_name"],
                "page_num": row["page_num"],
                "chunk_index": row["chunk_index"],
                "text": text,
                "snippet": _make_snippet(text, terms),
                "matched_terms": matched_terms,
                "score": matched_terms * 1000 + occurrences,
            })

        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:limit]


def rebuild_processed_documents(notebook: Dict[str, Any]) -> List[Any]:
    """Reconstruct ProcessedDocument objects from a loaded notebook's stored chunks.

    For callers that need the full `tools.document_tools.ProcessedDocument`
    shape (e.g. the Research Report workflow's `agents.graph._build_doc_context`)
    rather than the raw chunk/source dicts `NotebookMemory.load()` returns.
    """
    from tools.document_tools import DocumentChunk, ProcessedDocument
    chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for c in notebook.get("chunks", []):
        chunks_by_doc.setdefault(c["doc_id"], []).append(c)
    src_by_id = {s["doc_id"]: s for s in notebook.get("sources", [])}
    docs = []
    for doc_id, raw_chunks in chunks_by_doc.items():
        src = src_by_id.get(doc_id, {})
        filename = src.get("filename", doc_id)
        sorted_chunks = sorted(raw_chunks, key=lambda c: (c.get("page_num", 0), c.get("chunk_index", 0)))
        doc_chunks = [
            DocumentChunk(
                chunk_id=c["chunk_id"],
                doc_id=doc_id,
                doc_name=filename,
                page_num=c.get("page_num", 0),
                chunk_index=c.get("chunk_index", 0),
                text=c.get("text", ""),
                metadata=c.get("metadata", {}),
            )
            for c in sorted_chunks
        ]
        # Restores section headings Docling stored in chunk metadata rather
        # than chunk text, so consumers of raw_text still see the document's
        # section structure.
        raw_text = join_chunks_with_headings(sorted_chunks)
        docs.append(ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=src.get("file_type", Path(filename).suffix.lstrip(".").lower() or "unknown"),
            total_pages=src.get("total_pages", len(set(c.get("page_num", 0) for c in sorted_chunks))),
            total_chunks=len(doc_chunks),
            chunks=doc_chunks,
            raw_text=raw_text,
            content_md5=src.get("content_md5", ""),
        ))
    return docs


def _make_snippet(text: str, terms: List[str], radius: int = 110) -> str:
    """Return a short excerpt centred on the first matched term, for result previews."""
    lowered = text.lower()
    pos = -1
    for t in terms:
        idx = lowered.find(t)
        if idx != -1 and (pos == -1 or idx < pos):
            pos = idx
    if pos == -1:
        snippet = text[: radius * 2].strip()
        return snippet + ("…" if len(text) > radius * 2 else "")
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    snippet = text[start:end].strip()
    return ("…" if start > 0 else "") + snippet + ("…" if end < len(text) else "")
