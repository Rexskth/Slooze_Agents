"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    llm_api_key: str
    llm_model: str
    llm_base_url: str | None
    embedding_api_key: str
    embedding_model: str
    embedding_base_url: str | None
    tavily_api_key: str
    tavily_base_url: str
    max_search_results: int
    context_char_limit: int
    llm_temperature: float
    cache_ttl_seconds: int
    log_level: str
    documents_dir: str
    vector_store_dir: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    retrieval_top_k: int
    summary_max_chunks: int


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def load_settings() -> Settings:
    """Load runtime settings from the environment."""

    llm_api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not llm_api_key:
        raise ConfigError("Missing required environment variable: LLM_API_KEY")

    llm_model = os.getenv("LLM_MODEL", "").strip() or os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    llm_base_url = os.getenv("LLM_BASE_URL", "").strip() or None
    embedding_api_key = os.getenv("EMBEDDING_API_KEY", "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
    if not embedding_api_key:
        raise ConfigError("Missing required environment variable: EMBEDDING_API_KEY")
    embedding_model = os.getenv(
        "EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    ).strip()
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1").strip() or None

    return Settings(
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        tavily_api_key=_get_required_env("TAVILY_API_KEY"),
        tavily_base_url=os.getenv("TAVILY_BASE_URL", "https://api.tavily.com").strip(),
        max_search_results=_get_int_env("MAX_SEARCH_RESULTS", 5),
        context_char_limit=_get_int_env("CONTEXT_CHAR_LIMIT", 6000),
        llm_temperature=_get_float_env("LLM_TEMPERATURE", 0.2),
        cache_ttl_seconds=_get_int_env("CACHE_TTL_SECONDS", 300),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        documents_dir=os.getenv("DOCUMENTS_DIR", "data/documents").strip(),
        vector_store_dir=os.getenv("VECTOR_STORE_DIR", "data/vector_store").strip(),
        chunk_size_tokens=_get_int_env("CHUNK_SIZE_TOKENS", 700),
        chunk_overlap_tokens=_get_int_env("CHUNK_OVERLAP_TOKENS", 80),
        retrieval_top_k=_get_int_env("RETRIEVAL_TOP_K", 4),
        summary_max_chunks=_get_int_env("SUMMARY_MAX_CHUNKS", 8),
    )
