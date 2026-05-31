"""
configs/settings.py
Central configuration using pydantic-settings for validation & env loading.
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # HuggingFace
    huggingface_api_token: str = Field(default="", alias="HUGGINGFACEHUB_API_TOKEN")

    # Models
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDING_MODEL"
    )
    llm_model: str = Field(
        default="HuggingFaceH4/zephyr-7b-beta",
        alias="LLM_MODEL"
    )

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


# Singleton
settings = Settings()

# Ensure directories exist
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
Path(settings.documents_dir).mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
