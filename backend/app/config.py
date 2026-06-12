"""Markaziy konfiguratsiya. Barcha sozlamalar .env faylidan oʻqiladi."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    LLM_MODEL: str = "gpt-oss:20b"
    EMBEDDING_MODEL: str = "bge-m3"

    # Qdrant
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION_LAWS: str = "laws"
    QDRANT_COLLECTION_REPORTS: str = "audit_reports"
    QDRANT_COLLECTION_UPLOADS: str = "uploads"

    # Redis cache
    REDIS_URL: str = "redis://redis:6379/0"
    CACHE_TTL_SECONDS: int = 86400
    CACHE_SIMILARITY_THRESHOLD: float = 0.95

    # RAG
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 4

    # LLM
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024

    # Auth
    JWT_SECRET: str = "change-me-in-prod-please"
    JWT_EXPIRE_HOURS: int = 720  # 30 kun
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin123"

    # Monitoring (ixtiyoriy)
    GPU_STATS_URL: str | None = None

    # Yuklamalar
    DATA_DIR: str = "/data"
    UPLOAD_DIR: str = "/data/uploads"
    LAWS_DIR: str = "/data/laws"
    REPORTS_DIR: str = "/data/reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
