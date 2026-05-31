"""
src/retrieval/retriever.py

Advanced retrieval strategies on top of the VectorStoreManager:
  - Basic similarity search
  - MMR (Maximal Marginal Relevance) for diversity
  - Metadata filtering
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any

from src.langchain_compat import Document
from loguru import logger

from .vector_store import VectorStoreManager


class RAGRetriever:
    """
    Retriever with multiple strategies for optimal context selection.

    Usage
    -----
    >>> retriever = RAGRetriever(vsm)
    >>> docs = retriever.retrieve("Explain transformer attention", strategy="mmr")
    """

    def __init__(self, vector_store: VectorStoreManager) -> None:
        self._vsm = vector_store

    def retrieve(
        self,
        query: str,
        k: int = 5,
        strategy: str = "similarity",
        filter_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve relevant chunks for a given query.

        Args:
            query:       User question.
            k:           Number of chunks to retrieve.
            strategy:    "similarity" | "mmr"
            filter_meta: Optional ChromaDB metadata filter dict.
                         e.g. {"file_name": "report.pdf"}

        Returns:
            List of Documents ordered by relevance.
        """
        if strategy == "mmr":
            return self._mmr_search(query, k=k, filter_meta=filter_meta)
        return self._similarity_search(query, k=k, filter_meta=filter_meta)

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
    ) -> List[tuple[Document, float]]:
        """Returns (Document, relevance_score) tuples. Score ∈ [0, 1]."""
        return self._vsm.similarity_search_with_scores(query, k=k)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _similarity_search(
        self,
        query: str,
        k: int,
        filter_meta: Optional[Dict[str, Any]],
    ) -> List[Document]:
        store = self._vsm._store
        kwargs: Dict[str, Any] = {"k": k}
        if filter_meta:
            kwargs["filter"] = filter_meta
        results = store.similarity_search(query, **kwargs)
        logger.debug(f"[similarity] '{query[:50]}' → {len(results)} docs")
        return results

    def _mmr_search(
        self,
        query: str,
        k: int,
        filter_meta: Optional[Dict[str, Any]],
        fetch_k_multiplier: int = 4,
    ) -> List[Document]:
        """
        Maximal Marginal Relevance balances relevance with diversity.
        Prevents returning k very similar chunks from the same paragraph.
        """
        store = self._vsm._store
        fetch_k = min(k * fetch_k_multiplier, self._vsm.count())
        kwargs: Dict[str, Any] = {"k": k, "fetch_k": fetch_k}
        if filter_meta:
            kwargs["filter"] = filter_meta
        results = store.max_marginal_relevance_search(query, **kwargs)
        logger.debug(f"[MMR] '{query[:50]}' → {len(results)} diverse docs")
        return results
