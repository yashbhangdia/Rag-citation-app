"""
src/retrieval/reranker.py

Cross-encoder reranker (sentence-transformers). The model is loaded once per
process and shared (``get_reranker``), because it is reused by the pipeline and
by citation verification.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

from loguru import logger
from sentence_transformers import CrossEncoder

from configs.settings import resolve_device, settings
from src.langchain_compat import Document


class CrossEncoderReranker:
    """
    Model options by size:
        "cross-encoder/ms-marco-MiniLM-L-6-v2"   →  80MB, fastest
        "cross-encoder/ms-marco-MiniLM-L-12-v2"  → 130MB, better quality
        "BAAI/bge-reranker-base"                  → 280MB, best quality
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        model_name = model_name or settings.reranker_model
        logger.info(f"Loading reranker: {model_name}")
        self._model = CrossEncoder(model_name, device=resolve_device())

    def score(self, query: str, text: str) -> float:
        return float(self._model.predict([[query, text]])[0])

    def rerank_with_scores(
        self, query: str, documents: List[Document], top_n: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        if not documents:
            return []
        scores = self._model.predict([[query, d.page_content] for d in documents])
        ranked = sorted(zip(documents, map(float, scores)), key=lambda x: x[1], reverse=True)
        return ranked[:top_n] if top_n else ranked

    def rerank(
        self, query: str, documents: List[Document], top_n: Optional[int] = None
    ) -> List[Document]:
        top = self.rerank_with_scores(query, documents, top_n)
        logger.debug(
            f"Reranker: {len(documents)} → {len(top)} | scores: "
            f"{[round(s, 2) for _, s in top]}"
        )
        return [d for d, _ in top]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
