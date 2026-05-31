"""
src/generation/llm_client.py

Uses ChatHuggingFace + HuggingFaceEndpoint with provider="auto".
Follows the exact pattern from official LangChain HuggingFace docs.
"""

from __future__ import annotations
import os
from typing import Optional, Iterator
from loguru import logger

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage

from configs.settings import settings


class HFInferenceClient:

    def __init__(self, model: Optional[str] = None, token: Optional[str] = None) -> None:
        self.model = model or settings.llm_model
        self.token = (
            token
            or settings.huggingface_api_token
            or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
            or os.getenv("HF_TOKEN", "")
        )
        if not self.token:
            raise ValueError(
                "HuggingFace API token not found. "
                "Set HUGGINGFACEHUB_API_TOKEN in your .env file. "
                "Get a free token at: https://huggingface.co/settings/tokens"
            )

        os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.token

        _endpoint = HuggingFaceEndpoint(
            repo_id=self.model,
            task="text-generation",
            max_new_tokens=settings.max_new_tokens,
            provider="auto",          # ← lets HF pick the best working provider
            huggingfacehub_api_token=self.token,
        )
        self._chat_model = ChatHuggingFace(llm=_endpoint)
        logger.info(f"ChatHuggingFace ready — model: {self.model}")

    def generate(self, prompt: str, **kwargs) -> str:
        system, human = _split_prompt(prompt)
        try:
            response = self._chat_model.invoke([
                SystemMessage(content=system),
                HumanMessage(content=human),
            ])
            return response.content.strip()
        except Exception as exc:
            logger.error(f"ChatHuggingFace error: {exc}")
            raise

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        system, human = _split_prompt(prompt)
        try:
            for chunk in self._chat_model.stream([
                SystemMessage(content=system),
                HumanMessage(content=human),
            ]):
                if chunk.content:
                    yield chunk.content
        except Exception as exc:
            logger.error(f"ChatHuggingFace stream error: {exc}")
            raise


def _split_prompt(prompt: str) -> tuple[str, str]:
    separator = "QUESTION:"
    if separator in prompt:
        parts = prompt.split(separator, 1)
        return parts[0].strip(), "QUESTION:" + parts[1].strip()
    return "You are a helpful, citation-aware research assistant.", prompt.strip()