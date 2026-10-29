from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://secondbrain:secondbrain@localhost:5432/secondbrain"
    redis_url: str = "redis://localhost:6379/0"

    api_key: str = ""  # empty = auth disabled
    upload_dir: Path = Path("data/uploads")
    max_upload_mb: int = 25

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    chunk_max_words: int = 300
    chunk_overlap_words: int = 40

    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_entailment_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
