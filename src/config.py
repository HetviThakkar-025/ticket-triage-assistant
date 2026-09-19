"""Application configuration, loaded from environment / .env."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = ROOT_DIR / "knowledge_base"
CACHE_DIR = ROOT_DIR / ".cache"

load_dotenv(ROOT_DIR / ".env")


class Settings:
    """Environment-backed settings.

    Values are read lazily so tests can override environment variables
    before the first access (see ``get_settings.cache_clear``).
    """

    @property
    def gemini_api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "").strip()

    @property
    def jwt_secret(self) -> str:
        secret = os.getenv("JWT_SECRET", "").strip()
        if not secret:
            raise RuntimeError(
                "JWT_SECRET is not set. Copy .env.example to .env and fill it in."
            )
        return secret

    @property
    def jwt_algorithm(self) -> str:
        return os.getenv("JWT_ALGORITHM", "HS256")

    @property
    def jwt_expiry_minutes(self) -> int:
        return int(os.getenv("JWT_EXPIRY_MINUTES", "60"))

    @property
    def db_path(self) -> Path:
        return Path(os.getenv("TICKET_DB_PATH", str(ROOT_DIR / "tickets.db")))

    @property
    def gemini_model(self) -> str:
        return os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    @property
    def gemini_embedding_model(self) -> str:
        return os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

    @property
    def embedder_backend(self) -> str:
        """auto | gemini | sentence-transformers | tfidf"""
        return os.getenv("EMBEDDER_BACKEND", "auto").strip().lower()

    @property
    def top_k(self) -> int:
        return int(os.getenv("RETRIEVAL_TOP_K", "8"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
