"""
src/generation/rag_pipeline.py

The main RAG pipeline that ties retrieval + generation together.
Exposes a simple .query() method for the frontend to call.
"""

from __future__ import annotations
from typing import Optional, Iterator
from loguru import logger

from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.retriever import RAGRetriever
from src.utils.citation_builder import (
    build_citations,
    build_context_block,
    parse_citations_from_answer,
    CitationResult,
)
from .llm_client import HFInferenceClient
from .prompts import build_prompt_for_model
from configs.settings import settings


class RAGPipeline:
    """
    End-to-end RAG pipeline with citation support.

    Usage
    -----
    >>> pipeline = RAGPipeline()
    >>> result = pipeline.query("What is the deadline for tax filing?")
    >>> print(result.format_full_response())
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        llm_client: Optional[HFInferenceClient] = None,
        retrieval_strategy: str = "mmr",
        top_k: Optional[int] = None,
    ) -> None:
        self.vsm = vector_store or VectorStoreManager()
        self.retriever = RAGRetriever(self.vsm)
        self.llm = llm_client or HFInferenceClient()
        self.strategy = retrieval_strategy
        self.top_k = top_k or settings.top_k_results
        logger.info("RAGPipeline initialised.")

    # ── Main entry points ─────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> CitationResult:
        """
        Full RAG cycle: retrieve → format context → generate → parse citations.

        Args:
            question: Natural-language question.
            top_k:    Override number of retrieved chunks.
            strategy: Override retrieval strategy ("similarity" | "mmr").

        Returns:
            CitationResult with .answer, .citations, .format_full_response()
        """
        k = top_k or self.top_k
        strat = strategy or self.strategy

        # 1. Retrieve relevant chunks
        chunks = self.retriever.retrieve(question, k=k, strategy=strat)
        if not chunks:
            logger.warning("No chunks retrieved — returning empty response.")
            return CitationResult(
                answer="I couldn't find any relevant information in the indexed documents.",
                citations=[],
            )

        # 2. Build citation objects & context string
        citations = build_citations(chunks)
        context = build_context_block(chunks)

        # 3. Build model-specific prompt
        prompt = build_prompt_for_model(
            model_name=self.llm.model,
            context=context,
            question=question,
        )
        logger.debug(f"Prompt length: {len(prompt)} chars")

        # 4. Generate answer
        raw_answer = self.llm.generate(prompt)
        logger.info(f"Generated answer ({len(raw_answer)} chars)")

        # 5. Parse and return
        return parse_citations_from_answer(raw_answer, citations)

    def query_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> tuple[Iterator[str], list]:
        """
        Streaming version — returns (token_iterator, citations_list).
        The iterator yields tokens as they stream from the model.

        Usage in Streamlit:
            token_stream, citations = pipeline.query_stream(question)
            answer = st.write_stream(token_stream)
        """
        k = top_k or self.top_k
        chunks = self.retriever.retrieve(question, k=k, strategy=self.strategy)

        if not chunks:
            def _empty():
                yield "I couldn't find any relevant information in the indexed documents."
            return _empty(), []

        citations = build_citations(chunks)
        context = build_context_block(chunks)
        prompt = build_prompt_for_model(
            model_name=self.llm.model,
            context=context,
            question=question,
        )

        return self.llm.generate_stream(prompt), citations
