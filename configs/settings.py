"""
configs/settings.py
Central configuration using pydantic-settings for validation & env loading.
"""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # HuggingFace
    huggingface_api_token: str = Field(
        default="",
        validation_alias=AliasChoices(
            "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_TOKEN", "HF_TOKEN"
        ),
    )

    # Models
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDING_MODEL"
    )
    # "transformers": open model run in-process on a GPU (Colab T4). Free.
    # "hf_api":       HuggingFace Inference Providers (laptop use; free-tier limits).
    llm_backend: str = Field(default="transformers", alias="LLM_BACKEND")
    # Model used by the hf_api backend (Inference Providers)
    llm_model: str = Field(default="meta-llama/Llama-3.1-8B-Instruct", alias="LLM_MODEL")
    # Models used by the transformers backend (run on the Colab GPU)
    local_llm_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct", alias="LOCAL_LLM_MODEL"
    )
    local_llm_model_small: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct", alias="LOCAL_LLM_MODEL_SMALL"
    )  # used when no CUDA GPU is available
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    device: str = Field(default="auto", alias="DEVICE")  # auto | cuda | cpu | mps

    # ChromaDB
    chroma_persist_dir: str = Field(
        default="./data/chroma_db",
        alias="CHROMA_PERSIST_DIR"
    )
    collection_name: str = Field(
        default="rag_documents",
        alias="COLLECTION_NAME"
    )

    # RAG params
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")
    retrieval_candidates: int = Field(default=20, alias="RETRIEVAL_CANDIDATES")
    min_support_score: float = Field(default=-2.0, alias="MIN_SUPPORT_SCORE")
    max_new_tokens: int = Field(default=1024, alias="MAX_NEW_TOKENS")

    # App
    app_title: str = Field(default="RAG Citation Assistant", alias="APP_TITLE")
    app_port: int = Field(default=8501, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    documents_dir: str = Field(default="./data/documents",alias="DOCUMENTS_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True
        extra = "ignore"


# Singleton
settings = Settings()

def ensure_dirs() -> None:
    """Create data directories on demand (not as an import side effect)."""
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.documents_dir).mkdir(parents=True, exist_ok=True)


def resolve_device() -> str:
    """Resolve DEVICE=auto to cuda / mps / cpu."""
    if settings.device != "auto":
        return settings.device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
