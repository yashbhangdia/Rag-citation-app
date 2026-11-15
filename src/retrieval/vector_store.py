"""
src/retrieval/vector_store.py

Manages the ChromaDB vector store:
  - Build / update index from chunked documents
  - Similarity search with metadata returned intact
  - Delete documents by doc_id
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple
from pathlib import Path
from loguru import logger

from src.langchain_compat import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from configs.settings import ensure_dirs, resolve_device, settings


# ── Embedding model (local, no API key needed) ──────────────────────────────

class SentenceTransformerEmbeddings(Embeddings):
    """Minimal LangChain Embeddings adapter over sentence-transformers."""

    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        # bge models retrieve better with this instruction on the query side
        q = "Represent this sentence for searching relevant passages: " + text
        return self._model.encode([q], normalize_embeddings=True)[0].tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Cached embedding model (default BAAI/bge-small-en-v1.5: 384-d, MIT licence,
    ~90 MB). Runs on cuda / mps / cpu automatically.
    """
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    return SentenceTransformerEmbeddings(settings.embedding_model, resolve_device())


# ── VectorStore wrapper ──────────────────────────────────────────────────────

class VectorStoreManager:
    """
    Thin wrapper around ChromaDB for RAG use-cases.

    Usage
    -----
    >>> vsm = VectorStoreManager()
    >>> vsm.add_documents(chunks)
    >>> results = vsm.similarity_search("What is photosynthesis?", k=5)
    """

    def __init__(self) -> None:
        ensure_dirs()
        self._embeddings = get_embeddings()
        self._docs_cache: List[dict] | None = None
        persist_dir = settings.chroma_persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
        )
        logger.info(
            f"ChromaDB ready — collection '{settings.collection_name}' "
            f"at '{persist_dir}' "
            f"({self.count()} vectors stored)"
        )

    # ── Indexing ─────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """Index a list of chunked documents. Duplicate chunk_ids are ignored."""
        if not documents:
            logger.warning("add_documents called with empty list — nothing to do.")
            return

        # Use chunk_id as stable ChromaDB id to prevent duplicates
        ids = [doc.metadata["chunk_id"] for doc in documents]

        self._docs_cache = None
        # Batch to stay under Chroma's max batch size on large ingests
        for i in range(0, len(documents), 256):
            self._store.add_documents(documents=documents[i:i + 256], ids=ids[i:i + 256])
        logger.info(f"Indexed {len(documents)} chunks into ChromaDB.")

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
        results = self._store.get(where={"doc_id": doc_id})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self._docs_cache = None
            self._store.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id='{doc_id}'")
        else:
            logger.warning(f"No chunks found for doc_id='{doc_id}'")

    def reset(self) -> None:
        """Wipe the entire collection (use with caution)."""
        self._docs_cache = None
        self._store.delete_collection()
        logger.warning("Vector store collection wiped.")

    # ── Search ────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Document]:
        """
        Return the top-k most relevant chunks for a query.
        Documents include full metadata for citation building.
        """
        k = k or settings.top_k_results
        results = self._store.similarity_search(query, k=k)
        logger.debug(f"similarity_search('{query[:60]}…') → {len(results)} results")
        return results

    def similarity_search_with_scores(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """Same as above but also returns cosine similarity scores."""
        k = k or settings.top_k_results
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Number of vectors currently stored."""
        return self._store._collection.count()

    def list_documents(self) -> List[dict]:
        """Deduplicated list of indexed source documents (cached until the index changes)."""
        if self._docs_cache is not None:
            return self._docs_cache
        results = self._store.get(include=["metadatas"])
        seen, docs = set(), []
        for meta in results.get("metadatas") or []:
            did = meta.get("doc_id", "unknown")
            if did in seen:
                continue
            seen.add(did)
            docs.append({
                "doc_id":      did,
                "file_name":   meta.get("file_name", "unknown"),
                "file_type":   meta.get("file_type", "unknown"),
                "source":      meta.get("source", "unknown"),
                "company":     meta.get("company", "Unknown"),
                "fiscal_year": meta.get("fiscal_year", "Unknown"),
            })
        self._docs_cache = docs
        return docs

    def all_documents(self, where: dict | None = None) -> List[Document]:
        """Every stored chunk (text + metadata) — used to build the BM25 index."""
        res = self._store.get(where=where, include=["documents", "metadatas"])
        return [
            Document(page_content=t, metadata=m)
            for t, m in zip(res.get("documents") or [], res.get("metadatas") or [])
        ]
