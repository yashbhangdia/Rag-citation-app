"""
src/retrieval/vector_store.py

Manages the ChromaDB vector store:
  - Build / update index from chunked documents
  - Similarity search with metadata returned intact
  - Delete documents by doc_id
"""

from __future__ import annotations

from typing import List, Tuple
from pathlib import Path
from loguru import logger

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from configs.settings import settings


# ── Embedding model (local, no API key needed) ──────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFace embedding model.
    Downloads once (~90 MB) then cached in ~/.cache/huggingface/.

    Default: BAAI/bge-small-en-v1.5
      - 384 dimensions
      - State-of-the-art retrieval quality for its size
      - MIT licence, commercial use OK
    """
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},        # swap to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True},
    )


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
        self._embeddings = get_embeddings()
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

        self._store.add_documents(documents=documents, ids=ids)
        logger.info(f"Indexed {len(documents)} chunks into ChromaDB.")

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
        results = self._store.get(where={"doc_id": doc_id})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self._store.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id='{doc_id}'")
        else:
            logger.warning(f"No chunks found for doc_id='{doc_id}'")

    def reset(self) -> None:
        """Wipe the entire collection (use with caution)."""
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
        """Return a deduplicated list of indexed source documents."""
        results = self._store.get()
        seen, docs = set(), []
        for meta in results.get("metadatas", []):
            did = meta.get("doc_id", "unknown")
            if did not in seen:
                seen.add(did)
                docs.append({
                    "doc_id":    did,
                    "file_name": meta.get("file_name", "unknown"),
                    "file_type": meta.get("file_type", "unknown"),
                    "source":    meta.get("source", "unknown"),
                })
        return docs
