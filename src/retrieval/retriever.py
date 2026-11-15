"""
src/retrieval/retriever.py

Retrieval strategies on top of VectorStoreManager:
  - similarity  : dense cosine search
  - mmr         : dense + diversity
  - hybrid      : dense + BM25 fused with reciprocal rank fusion (best for exact
                  figures / tickers, which dense embeddings handle poorly)
  - metadata filters (company / fiscal year) inferred from the question
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from loguru import logger
from rank_bm25 import BM25Okapi

from src.ingestion.document_loader import detect_companies
from src.langchain_compat import Document

from .vector_store import VectorStoreManager

_TOKEN = re.compile(r"[a-z0-9₹%.,]+")


def _tok(text: str) -> List[str]:
    return [t.strip(".,") for t in _TOKEN.findall(text.lower()) if t.strip(".,")]


def _key(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or doc.page_content[:80]


def reciprocal_rank_fusion(rankings: List[List[Document]], k: int = 60) -> List[Document]:
    """Fuse several ranked lists; pure function (unit-tested)."""
    scores: Dict[str, float] = {}
    docs: Dict[str, Document] = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking):
            key = _key(d)
            docs[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


def build_where(companies: List[str]) -> Optional[Dict[str, Any]]:
    """Chroma `where` clause when the question names exactly one company.

    Fiscal years are deliberately *not* hard-filtered: "FY25" in a question and
    "FY2024-25" in a filename are the same year but different strings; the
    reranker handles year relevance instead.
    """
    return {"company": companies[0]} if len(companies) == 1 else None


class RAGRetriever:
    def __init__(self, vector_store: VectorStoreManager) -> None:
        self._vsm = vector_store
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Document] = []
        self._bm25_count = -1

    # ── public ───────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 5,
        strategy: str = "hybrid",
        filter_meta: Optional[Dict[str, Any]] = None,
        auto_filter: bool = True,
    ) -> List[Document]:
        """
        Retrieve candidate chunks. Comparison questions naming several companies
        are retrieved per company so no company is crowded out.
        """
        companies = detect_companies(query) if auto_filter and not filter_meta else []

        if len(companies) > 1:
            per = max(2, k // len(companies) + 1)
            out: List[Document] = []
            for c in companies:
                where = build_where([c])
                out.extend(self._search(query, per, strategy, where))
            return out

        where = filter_meta or build_where(companies)
        docs = self._search(query, k, strategy, where)
        if not docs and where:  # filter too strict (e.g. year not indexed) → retry open
            logger.info("Filter returned nothing — retrying without metadata filter.")
            docs = self._search(query, k, strategy, None)
        return docs

    def retrieve_with_scores(self, query: str, k: int = 5):
        return self._vsm.similarity_search_with_scores(query, k=k)

    # ── internals ────────────────────────────────────────────────────────────

    def _search(self, query, k, strategy, where) -> List[Document]:
        if self._vsm.count() == 0:
            return []
        if strategy == "hybrid":
            return self._hybrid(query, k, where)
        if strategy == "mmr":
            return self._mmr(query, k, where)
        return self._dense(query, k, where)

    def _dense(self, query, k, where) -> List[Document]:
        kwargs: Dict[str, Any] = {"k": k}
        if where:
            kwargs["filter"] = where
        return self._vsm._store.similarity_search(query, **kwargs)

    def _mmr(self, query, k, where, fetch_k_multiplier: int = 4) -> List[Document]:
        kwargs: Dict[str, Any] = {
            "k": k,
            "fetch_k": min(k * fetch_k_multiplier, self._vsm.count()),
        }
        if where:
            kwargs["filter"] = where
        return self._vsm._store.max_marginal_relevance_search(query, **kwargs)

    def _ensure_bm25(self) -> None:
        n = self._vsm.count()
        if self._bm25 is not None and n == self._bm25_count:
            return
        self._bm25_docs = self._vsm.all_documents()
        self._bm25 = BM25Okapi([_tok(d.page_content) for d in self._bm25_docs]) if self._bm25_docs else None
        self._bm25_count = n
        logger.info(f"BM25 index built over {len(self._bm25_docs)} chunks")

    def _matches(self, meta: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(self._matches(meta, w) for w in where["$and"])
        (field, cond), = where.items()
        if isinstance(cond, dict) and "$in" in cond:
            return meta.get(field) in cond["$in"]
        return meta.get(field) == cond

    def _hybrid(self, query, k, where) -> List[Document]:
        dense = self._dense(query, k * 2, where)
        self._ensure_bm25()
        sparse: List[Document] = []
        if self._bm25 is not None:
            scores = self._bm25.get_scores(_tok(query))
            order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
            for i in order:
                if scores[i] <= 0 or len(sparse) >= k * 2:
                    break
                d = self._bm25_docs[i]
                if self._matches(d.metadata, where):
                    sparse.append(d)
        fused = reciprocal_rank_fusion([dense, sparse])
        logger.debug(f"[hybrid] dense={len(dense)} bm25={len(sparse)} fused={len(fused)}")
        return fused[:k * 2]
