"""
tools/hybrid_store.py
──────────────────────
Hybrid RAG retrieval engine: dense vector search + BM25 sparse search,
fused with Reciprocal Rank Fusion (RRF).

Architecture
────────────
  ChromaDB  — persistent embedding cache across sessions.
              Same document chunks are never re-embedded on the next run.

  FAISS     — in-memory index for the current session.
              Built from the current documents for ultra-fast retrieval.

  BM25      — in-memory keyword index for the current session.
              Provides complementary signal for exact-term matches.

  RRF       — Reciprocal Rank Fusion merges dense and sparse result lists
              without requiring score normalisation.

Typical retrieval call:
  store = HybridStore(...)
  store.add_documents(processed_docs)
  chunks = store.search_hybrid("attention mechanism in transformers", k=8)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tools.embeddings import OllamaEmbedder

logger = logging.getLogger(__name__)


class _NoopEF:
    """
    Sentinel embedding function passed to ChromaDB to prevent it from
    instantiating DefaultEmbeddingFunction (which downloads all-MiniLM-L6-v2
    via sentence-transformers). HybridStore always passes precomputed
    embeddings to upsert/add, so this function is never actually called.
    """

    def __call__(self, input):  # noqa: A002
        raise RuntimeError(
            "HybridStore always provides precomputed Ollama embeddings; "
            "the ChromaDB embedding function should never be invoked."
        )

_RRF_K = 60   # standard RRF constant; higher → smooths rank differences


class HybridStore:
    """
    Per-session hybrid retrieval store.

    Parameters
    ----------
    session_id      : Unique identifier for this run (used for FAISS/BM25 isolation).
    embed_model     : Ollama embedding model name, e.g. "nomic-embed-text".
    ollama_base_url : Ollama server URL, e.g. "http://localhost:11434".
    persist_dir     : Directory for ChromaDB storage (embedding cache).
    """

    def __init__(
        self,
        session_id: str,
        embed_model: str,
        ollama_base_url: str,
        persist_dir: str,
    ):
        """Initialise an empty store; FAISS/BM25 indices are built lazily by add_documents*."""
        self.session_id = session_id
        self.persist_dir = persist_dir

        self._embedder = OllamaEmbedder(embed_model, ollama_base_url)
        self._collection_name = _safe_collection_name(embed_model)

        # ChromaDB: persistent embedding cache
        self._chroma_client = None
        self._chroma_col = None

        # FAISS: fast in-memory dense index (session-scoped)
        self._faiss_index = None
        self._faiss_chunks: List[Dict[str, Any]] = []

        # BM25: sparse keyword index (session-scoped)
        self._bm25 = None
        self._bm25_chunks: List[Dict[str, Any]] = []

        self._is_indexed: bool = False

        # Content-hash manifest: maps doc filename → content_md5 for cache invalidation
        self._manifest: Dict[str, str] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def add_documents_bm25_only(self, docs: list) -> None:
        """
        Build a BM25 keyword index from *docs* without embedding or importing FAISS.

        This is the low-memory path used when the embedding model is unavailable
        or when memory-constrained environments cannot afford FAISS.  Search
        quality is lower than full hybrid RAG but the index build never OOMs.
        """
        chunks = _flatten_chunks(docs)
        if not chunks:
            logger.warning("HybridStore.add_documents_bm25_only: no chunks found.")
            return
        logger.info(
            "HybridStore: BM25-only index for %d chunks (no FAISS/embedding).", len(chunks)
        )
        self._build_bm25(chunks)
        self._faiss_index = None   # explicitly no dense index
        self._faiss_chunks = []
        self._is_indexed = True

    def add_documents(
        self,
        docs: list,
        warning_callback=None,
    ) -> None:
        """
        Embed and index all chunks from a list of ProcessedDocuments.
        Embeddings are cached in ChromaDB; FAISS and BM25 are rebuilt in-memory.

        *warning_callback* — optional callable(str) invoked whenever the
        embedder auto-reduces its batch size due to low VRAM.  Pass a
        Streamlit ``st.warning`` call (or similar) to surface this in the UI.
        """
        chunks = _flatten_chunks(docs)
        if not chunks:
            logger.warning("HybridStore.add_documents: no chunks found in documents.")
            return

        logger.info(
            "HybridStore: indexing %d chunks from %d document(s)…",
            len(chunks), len(docs),
        )

        # Invalidate stale cached embeddings for documents whose content has changed
        for doc in docs:
            if doc.content_md5 and doc.content_md5 != self._manifest.get(doc.filename, ""):
                self._invalidate_doc_cache(doc.filename)
                self._manifest[doc.filename] = doc.content_md5

        try:
            embeddings = self._embed_with_cache(chunks, warning_callback=warning_callback)
            self._build_faiss(chunks, embeddings)
            self._build_bm25(chunks)
            self._is_indexed = True
            logger.info(
                "HybridStore: indexed %d chunks (FAISS dim=%d).",
                len(chunks), self._faiss_index.d if self._faiss_index else 0,
            )
        except RuntimeError as exc:
            msg = (
                f"Embedding unavailable ({exc}). "
                "Falling back to BM25 keyword search — "
                "run `ollama pull nomic-embed-text` for full hybrid RAG."
            )
            logger.warning("HybridStore: %s", msg)
            if warning_callback:
                warning_callback(msg)
            self.add_documents_bm25_only(docs)

    def search_hybrid(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        """
        Run dense + sparse retrieval and merge results with RRF.
        Falls back to BM25-only when no FAISS index is present.

        Returns up to *k* chunk dicts, each with keys:
            chunk_id, doc_name, page_num, chunk_index, text, rrf_score,
            dense_score (optional), sparse_score (optional)
        """
        if not self._is_indexed:
            return []

        # BM25-only path: no FAISS index built (e.g. low-memory environments)
        if self._faiss_index is None:
            return self._search_sparse(query, k)

        k_candidates = min(k * 4, len(self._faiss_chunks))
        dense = self._search_dense(query, k_candidates)
        sparse = self._search_sparse(query, k_candidates)
        return _rrf_merge(dense, sparse, k)

    def is_indexed(self) -> bool:
        """Whether add_documents (or the BM25-only variant) has populated this store."""
        return self._is_indexed

    def embedder_available(self) -> bool:
        """Whether the configured Ollama embedding model is reachable and pulled."""
        return self._embedder.is_available()

    def list_indexed_documents(self) -> List[str]:
        """Return unique document names currently in the FAISS index."""
        seen: set[str] = set()
        docs = []
        for c in self._faiss_chunks:
            name = c.get("doc_name", "unknown")
            if name not in seen:
                seen.add(name)
                docs.append(name)
        return docs

    def clear_cache(self) -> None:
        """Delete the ChromaDB collection (clears the persistent embedding cache)."""
        try:
            client = self._get_chroma_client()
            client.delete_collection(self._collection_name)
            self._chroma_col = None
            logger.info("HybridStore: cleared ChromaDB collection '%s'.", self._collection_name)
        except Exception as e:
            logger.warning("HybridStore: could not clear ChromaDB: %s", e)

    # ── Dense retrieval (FAISS) ───────────────────────────────────────────────

    def _search_dense(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Embed *query* and return the top-k FAISS nearest-neighbour chunks with their cosine score."""
        import faiss as _faiss

        q_emb = np.array([self._embedder.embed_query(query)], dtype=np.float32)
        _faiss.normalize_L2(q_emb)

        scores, indices = self._faiss_index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = dict(self._faiss_chunks[idx])
                chunk["dense_score"] = float(score)
                results.append(chunk)
        return results

    # ── Sparse retrieval (BM25) ───────────────────────────────────────────────

    def _search_sparse(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Tokenise *query* and return the top-k BM25-scoring chunks (score > 0 only)."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                chunk = dict(self._bm25_chunks[idx])
                chunk["sparse_score"] = float(scores[idx])
                results.append(chunk)
        return results

    # ── Index builders ────────────────────────────────────────────────────────

    def _build_faiss(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        """Build a fresh in-memory FAISS IndexFlatIP over *embeddings*, replacing any prior session index."""
        import faiss as _faiss

        emb_arr = np.array(embeddings, dtype=np.float32)
        _faiss.normalize_L2(emb_arr)
        dim = emb_arr.shape[1]
        index = _faiss.IndexFlatIP(dim)   # cosine similarity after L2 normalisation
        index.add(emb_arr)
        self._faiss_index = index
        self._faiss_chunks = chunks

    def _build_bm25(self, chunks: List[Dict]) -> None:
        """Build a fresh in-memory BM25Okapi index over *chunks*, replacing any prior session index."""
        from rank_bm25 import BM25Okapi
        tokenized = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_chunks = chunks

    # ── Embedding cache (ChromaDB) ────────────────────────────────────────────

    def _embed_with_cache(
        self,
        chunks: List[Dict],
        warning_callback=None,
    ) -> List[List[float]]:
        """
        Return embeddings for all chunks, reading from ChromaDB cache where
        available and only calling Ollama for unseen chunks.

        *warning_callback* is forwarded to the embedder so low-VRAM
        auto-batch-reduction warnings reach the caller.
        """
        chunk_ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        embeddings: List[Optional[List[float]]] = [None] * len(chunks)
        to_embed_indices: List[int] = []

        # Check cache
        try:
            col = self._get_chroma_col()
            existing = col.get(ids=chunk_ids, include=["embeddings"])
            cached_map = {
                cid: emb
                for cid, emb in zip(existing["ids"], existing["embeddings"])
                if emb is not None
            }
            for i, cid in enumerate(chunk_ids):
                if cid in cached_map:
                    embeddings[i] = cached_map[cid]
                else:
                    to_embed_indices.append(i)
        except Exception as e:
            logger.debug("ChromaDB cache read failed (%s) — embedding all chunks.", e)
            to_embed_indices = list(range(len(chunks)))

        cache_hits = len(chunks) - len(to_embed_indices)
        logger.info(
            "  Embeddings: %d from cache, %d new to embed.",
            cache_hits, len(to_embed_indices),
        )

        # Embed uncached chunks
        if to_embed_indices:
            new_texts = [texts[i] for i in to_embed_indices]
            new_embs = self._embedder.embed_texts(new_texts, warning_callback=warning_callback)
            for i, emb in zip(to_embed_indices, new_embs):
                embeddings[i] = emb

            # Persist to cache
            try:
                col = self._get_chroma_col()
                col.upsert(
                    ids=[chunk_ids[i] for i in to_embed_indices],
                    documents=[texts[i] for i in to_embed_indices],
                    embeddings=new_embs,
                    metadatas=[
                        {"doc_name": chunks[i]["doc_name"], "session_id": self.session_id}
                        for i in to_embed_indices
                    ],
                )
            except Exception as e:
                logger.warning("ChromaDB cache write failed: %s", e)

        return embeddings  # type: ignore[return-value]

    # ── Cache invalidation ────────────────────────────────────────────────────

    def _invalidate_doc_cache(self, doc_name: str) -> int:
        """Delete all cached ChromaDB embeddings for *doc_name* (called when its content_md5 changes)."""
        try:
            col = self._get_chroma_col()
            existing = col.get(where={"doc_name": doc_name})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
                logger.info("Cache invalidated: removed %d stale chunks for '%s'", len(existing["ids"]), doc_name)
                return len(existing["ids"])
        except Exception as e:
            logger.warning("Cache invalidation failed for '%s': %s", doc_name, e)
        return 0

    # ── ChromaDB helpers ──────────────────────────────────────────────────────

    def _get_chroma_client(self):
        """Return (lazily creating) the persistent ChromaDB client rooted at persist_dir."""
        if self._chroma_client is None:
            import chromadb
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._chroma_client

    def _get_chroma_col(self):
        """Return (lazily creating) this embed model's ChromaDB collection."""
        if self._chroma_col is None:
            client = self._get_chroma_client()
            try:
                self._chroma_col = client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=_NoopEF(),
                )
            except Exception:
                # Existing collection was created without a custom embedding_function,
                # causing ChromaDB to record "default" as the EF type. Reopening with
                # _NoopEF raises a conflict error on chromadb < 1.0. Delete and recreate
                # so future startups no longer trigger the sentence-transformers download.
                logger.info(
                    "HybridStore: recreating ChromaDB collection '%s' to remove "
                    "DefaultEmbeddingFunction (one-time migration; cached embeddings cleared).",
                    self._collection_name,
                )
                client.delete_collection(self._collection_name)
                self._chroma_col = client.create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=_NoopEF(),
                )
        return self._chroma_col


# ── Module-level session cache ─────────────────────────────────────────────────

_stores: Dict[str, HybridStore] = {}


def get_or_create_store(
    session_id: str,
    embed_model: str,
    ollama_base_url: str,
    persist_dir: str,
) -> HybridStore:
    """Return (or create) the HybridStore for this session."""
    global _stores
    if session_id not in _stores:
        _stores[session_id] = HybridStore(
            session_id=session_id,
            embed_model=embed_model,
            ollama_base_url=ollama_base_url,
            persist_dir=persist_dir,
        )
    return _stores[session_id]


def clear_all_stores() -> None:
    """Discard all in-memory session stores (ChromaDB cache is unaffected)."""
    global _stores
    _stores.clear()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _flatten_chunks(docs: list) -> List[Dict[str, Any]]:
    """Convert a list of ProcessedDocuments into a flat list of chunk dicts."""
    chunks = []
    for doc in docs:
        for chunk in doc.chunks:
            chunks.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "page_num": chunk.page_num,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            })
    return chunks


def _rrf_merge(
    dense: List[Dict],
    sparse: List[Dict],
    k: int,
) -> List[Dict]:
    """Reciprocal Rank Fusion of dense and sparse result lists."""
    scores: Dict[str, Dict] = {}

    for rank, result in enumerate(dense):
        cid = result["chunk_id"]
        if cid not in scores:
            scores[cid] = {"chunk": result, "rrf_score": 0.0}
        scores[cid]["rrf_score"] += 1.0 / (_RRF_K + rank + 1)

    for rank, result in enumerate(sparse):
        cid = result["chunk_id"]
        if cid not in scores:
            scores[cid] = {"chunk": result, "rrf_score": 0.0}
        scores[cid]["rrf_score"] += 1.0 / (_RRF_K + rank + 1)

    ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    out = []
    for item in ranked[:k]:
        chunk = dict(item["chunk"])
        chunk["rrf_score"] = item["rrf_score"]
        out.append(chunk)
    return out


def _safe_collection_name(embed_model: str) -> str:
    """Derive a valid ChromaDB collection name from an embed model string."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", embed_model)[:60]
    return f"emb_{name}"
