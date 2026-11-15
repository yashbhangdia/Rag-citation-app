"""
src/generation/rag_pipeline.py

Retrieve → rerank → generate → verify citations.
`query` and `query_stream` share one retrieval path so they number and rank
citations identically.
"""

from __future__ import annotations

import re
from typing import Dict, Iterator, List, Optional, Tuple

from loguru import logger

from configs.settings import settings
from src.retrieval.reranker import CrossEncoderReranker, get_reranker
from src.retrieval.retriever import RAGRetriever
from src.retrieval.vector_store import VectorStoreManager
from src.utils.citation_builder import (
    Citation,
    CitationResult,
    build_citations,
    build_context_block,
    parse_citations_from_answer,
    verify_citations,
)

from .llm_client import BaseLLM, get_llm
from .prompts import build_messages, build_rewrite_messages

NO_ANSWER = "I couldn't find any relevant information in the indexed documents."
History = List[Dict[str, str]]


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> style reasoning blocks some models emit."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[thinking\].*?\[/thinking\]", "", text, flags=re.DOTALL)
    return text.strip()


class RAGPipeline:
    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        llm_client: Optional[BaseLLM] = None,
        retrieval_strategy: str = "hybrid",
        top_k: Optional[int] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.vsm = vector_store or VectorStoreManager()
        self.retriever = RAGRetriever(self.vsm)
        self.llm = llm_client or get_llm()
        self.strategy = retrieval_strategy
        self.top_k = top_k or settings.top_k_results
        self.reranker = reranker or get_reranker()
        logger.info("RAGPipeline initialised.")

    # ── shared retrieval path ────────────────────────────────────────────────

    def _standalone_question(self, question: str, history: Optional[History]) -> str:
        """Resolve follow-ups ('and for Infosys?') into a standalone search query."""
        if not history:
            return question
        try:
            out = _strip_thinking(self.llm.generate(
                build_rewrite_messages(question, history), max_new_tokens=80
            )).strip().splitlines()[0]
            return out or question
        except Exception as exc:  # rewriting is best-effort
            logger.warning(f"Query rewrite failed ({exc}); using original question.")
            return question

    def retrieve(
        self, question: str, top_k: Optional[int] = None, strategy: Optional[str] = None
    ) -> Tuple[list, list]:
        """Returns (reranked chunks, citations) — numbering is final from here on."""
        k = top_k or self.top_k
        candidates = self.retriever.retrieve(
            question,
            k=max(settings.retrieval_candidates, k),
            strategy=strategy or self.strategy,
        )
        chunks = self.reranker.rerank(question, candidates, top_n=k)
        return chunks, build_citations(chunks)

    def _support_scorer(self):
        return lambda claim, text: self.reranker.score(claim, text)

    def _finish(self, raw_answer: str, citations: List[Citation]) -> CitationResult:
        result = parse_citations_from_answer(_strip_thinking(raw_answer), citations)
        return verify_citations(
            result, scorer=self._support_scorer(), min_score=settings.min_support_score
        )

    # ── entry points ─────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
        history: Optional[History] = None,
    ) -> CitationResult:
        search_q = self._standalone_question(question, history)
        chunks, citations = self.retrieve(search_q, top_k, strategy)
        if not chunks:
            return CitationResult(answer=NO_ANSWER, citations=[])

        messages = build_messages(build_context_block(chunks), question, history)
        raw = self.llm.generate(messages)
        logger.info(f"Generated answer ({len(raw)} chars)")
        return self._finish(raw, citations)

    def query_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
        history: Optional[History] = None,
    ) -> Tuple[Iterator[str], List[Citation]]:
        """
        Returns (token_iterator, citations). When the stream is exhausted, call
        ``finish(full_text, citations)`` to get the verified CitationResult.
        """
        search_q = self._standalone_question(question, history)
        chunks, citations = self.retrieve(search_q, top_k, strategy)
        if not chunks:
            return iter([NO_ANSWER]), []
        messages = build_messages(build_context_block(chunks), question, history)
        return self.llm.generate_stream(messages), citations

    def finish(self, full_text: str, citations: List[Citation]) -> CitationResult:
        """Verify a streamed answer (the streaming counterpart of ``query``)."""
        if not citations:
            return CitationResult(answer=full_text, citations=[])
        return self._finish(full_text, citations)
