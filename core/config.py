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

    openai_api_key: str
    openai_model: str
    tavily_api_key: str
    tavily_base_url: str
    max_search_results: int
    context_char_limit: int
    llm_temperature: float
    cache_ttl_seconds: int
    log_level: str


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

    return Settings(
        openai_api_key=_get_required_env("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        tavily_api_key=_get_required_env("TAVILY_API_KEY"),
        tavily_base_url=os.getenv("TAVILY_BASE_URL", "https://api.tavily.com").strip(),
        max_search_results=_get_int_env("MAX_SEARCH_RESULTS", 5),
        context_char_limit=_get_int_env("CONTEXT_CHAR_LIMIT", 6000),
        llm_temperature=_get_float_env("LLM_TEMPERATURE", 0.2),
        cache_ttl_seconds=_get_int_env("CACHE_TTL_SECONDS", 300),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
    )
