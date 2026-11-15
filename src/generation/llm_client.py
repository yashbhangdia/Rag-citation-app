"""
src/generation/llm_client.py

Pluggable, free LLM backends. Every backend takes proper chat messages
(system / user / assistant dicts) and applies the model's own chat template
exactly once.

  transformers : open model run in-process on a GPU (Colab T4, 4-bit). No API key.
  hf_api       : HuggingFace Inference Providers via huggingface_hub (free tier).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from threading import Thread
from typing import Dict, Iterator, List, Optional

from loguru import logger

from configs.settings import resolve_device, settings

Messages = List[Dict[str, str]]


class BaseLLM(ABC):
    model: str

    @abstractmethod
    def generate(self, messages: Messages, **kwargs) -> str: ...

    @abstractmethod
    def generate_stream(self, messages: Messages, **kwargs) -> Iterator[str]: ...


# ── HF Inference Providers ───────────────────────────────────────────────────

class HFApiLLM(BaseLLM):
    def __init__(self, model: Optional[str] = None, token: Optional[str] = None) -> None:
        from huggingface_hub import InferenceClient

        self.model = model or settings.llm_model
        self.token = (
            token
            or settings.huggingface_api_token
            or os.getenv("HF_TOKEN", "")
        )
        if not self.token:
            raise ValueError(
                "HuggingFace token not found. Set HUGGINGFACEHUB_API_TOKEN in .env "
                "(free token: https://huggingface.co/settings/tokens), or use "
                "LLM_BACKEND=transformers on a GPU (Colab)."
            )
        self._client = InferenceClient(model=self.model, token=self.token)
        logger.info(f"HFApiLLM ready — model: {self.model}")

    def _call(self, messages: Messages, stream: bool, **kwargs):
        from huggingface_hub.errors import HfHubHTTPError

        try:
            return self._client.chat_completion(
                messages=messages,
                max_tokens=kwargs.get("max_new_tokens", settings.max_new_tokens),
                temperature=kwargs.get("temperature", 0.1),
                stream=stream,
            )
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 402:
                raise RuntimeError(
                    "HuggingFace Inference credits are exhausted (402). Use "
                    "LLM_BACKEND=transformers on a GPU (Colab notebook) instead."
                ) from exc
            if "model_not_supported" in str(exc):
                raise RuntimeError(
                    f"'{self.model}' is not served by any provider enabled on your HF "
                    "account. Pick another LLM_MODEL (e.g. meta-llama/Llama-3.1-8B-Instruct)."
                ) from exc
            raise

    def generate(self, messages: Messages, **kwargs) -> str:
        out = self._call(messages, False, **kwargs)
        return (out.choices[0].message.content or "").strip()

    def generate_stream(self, messages: Messages, **kwargs) -> Iterator[str]:
        stream = self._call(messages, True, **kwargs)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ── In-process open model (Colab GPU) ────────────────────────────────────────

class TransformersLLM(BaseLLM):
    """Qwen2.5-Instruct (4-bit on CUDA). Falls back to a small model on CPU/MPS."""

    def __init__(self, model: Optional[str] = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = resolve_device()
        self.model = model or (
            settings.local_llm_model if device == "cuda" else settings.local_llm_model_small
        )
        logger.info(f"Loading {self.model} on {device} (first run downloads weights)…")

        self._tok = AutoTokenizer.from_pretrained(self.model)
        kwargs: dict = {}
        if device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                kwargs["device_map"] = "auto"
            except ImportError:
                logger.warning("bitsandbytes missing — loading fp16 on GPU.")
                kwargs.update(torch_dtype=torch.float16, device_map="auto")
        else:
            kwargs["torch_dtype"] = torch.float16 if device == "mps" else torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(self.model, **kwargs)
        if device != "cuda":
            self._model.to(device)
        self._model.eval()
        self._device = self._model.device
        logger.info("TransformersLLM ready.")

    def _inputs(self, messages: Messages):
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._tok(text, return_tensors="pt").to(self._device)

    def _gen_kwargs(self, **kwargs) -> dict:
        return dict(
            max_new_tokens=kwargs.get("max_new_tokens", settings.max_new_tokens),
            do_sample=False,  # deterministic — we want faithful, repeatable answers
            pad_token_id=self._tok.eos_token_id,
        )

    def generate(self, messages: Messages, **kwargs) -> str:
        import torch

        inputs = self._inputs(messages)
        with torch.no_grad():
            out = self._model.generate(**inputs, **self._gen_kwargs(**kwargs))
        new = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(new, skip_special_tokens=True).strip()

    def generate_stream(self, messages: Messages, **kwargs) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        inputs = self._inputs(messages)
        streamer = TextIteratorStreamer(
            self._tok, skip_prompt=True, skip_special_tokens=True
        )
        thread = Thread(
            target=self._model.generate,
            kwargs=dict(**inputs, streamer=streamer, **self._gen_kwargs(**kwargs)),
            daemon=True,
        )
        thread.start()
        for piece in streamer:
            if piece:
                yield piece
        thread.join()


def get_llm(backend: Optional[str] = None) -> BaseLLM:
    backend = (backend or settings.llm_backend).lower()
    if backend == "transformers":
        return TransformersLLM()
    if backend == "hf_api":
        return HFApiLLM()
    raise ValueError(f"Unknown LLM_BACKEND '{backend}' (use 'transformers' or 'hf_api').")
