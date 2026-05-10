"""
src/generation/llm_client.py

Wraps the HuggingFace Inference API (free tier).
Supports streaming and non-streaming responses.

Free models that work well (no approval needed):
  - HuggingFaceH4/zephyr-7b-beta
  - microsoft/Phi-3-mini-4k-instruct
  - mistralai/Mistral-7B-Instruct-v0.3  (may need approval)
"""

from __future__ import annotations
import os
from typing import Optional, Iterator
from loguru import logger

import requests
from huggingface_hub import InferenceClient

from configs.settings import settings


class HFInferenceClient:
    """
    Client for HuggingFace Inference API.

    Obtains token from:
      1. settings.huggingface_api_token  (from .env)
      2. HF_TOKEN environment variable (HF standard)

    Free tier limits (as of 2025):
      - ~1,000 requests / day per model
      - Max 8,000 input tokens
      - Max 1,024 output tokens per call
    """

    def __init__(
        self,
        model: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self.model = model or settings.llm_model
        self.token = (
            token
            or settings.huggingface_api_token
            or os.getenv("HF_TOKEN", "")
        )
        if not self.token:
            raise ValueError(
                "HuggingFace API token not found. "
                "Set HUGGINGFACE_API_TOKEN in your .env file. "
                "Get a free token at: https://huggingface.co/settings/tokens"
            )
        self._client = InferenceClient(model=self.model, token=self.token)
        logger.info(f"HF Inference client ready — model: {self.model}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Generate a response. Returns the full text string.

        Lower temperature (0.1–0.4) is recommended for RAG tasks
        to keep answers grounded in the retrieved context.
        """
        max_tokens = max_new_tokens or settings.max_new_tokens
        try:
            response = self._client.text_generation(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                return_full_text=False,   # only return generated part
            )
            return response.strip()
        except Exception as exc:
            logger.error(f"HF Inference API error: {exc}")
            raise

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """
        Streaming version — yields text tokens as they arrive.
        Use with Streamlit's st.write_stream().
        """
        max_tokens = max_new_tokens or settings.max_new_tokens
        try:
            for token in self._client.text_generation(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
                stream=True,
            ):
                yield token
        except Exception as exc:
            logger.error(f"HF streaming error: {exc}")
            raise
