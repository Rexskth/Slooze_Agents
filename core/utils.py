"""Shared utility helpers for logging, retries, caching, and text cleanup."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable


class AgentError(RuntimeError):
    """Base error for expected agent failures."""


class SearchProviderError(AgentError):
    """Raised when the search provider fails."""


class LLMProviderError(AgentError):
    """Raised when the language model provider fails."""


class EmbeddingProviderError(AgentError):
    """Raised when the embedding provider fails."""


class DocumentProcessingError(AgentError):
    """Raised when PDF ingestion or processing fails."""


class DocumentNotFoundError(AgentError):
    """Raised when a requested document is unavailable."""


class TTLCache:
    """A small in-memory TTL cache for repeated queries."""

    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None

        expires_at, value = entry
        if time.time() >= expires_at:
            self._store.pop(key, None)
            return None

        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time() + self.ttl_seconds, value)


def configure_logging(level: str) -> None:
    """Configure process-wide logging once."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""

    return logging.getLogger(name)


def normalize_query(query: str) -> str:
    """Normalize incoming user queries for caching and validation."""

    normalized = re.sub(r"\s+", " ", query).strip()
    return normalized


def make_cache_key(query: str) -> str:
    """Build a deterministic cache key from a normalized query."""

    return hashlib.sha256(normalize_query(query).lower().encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """Remove repeated whitespace and noisy line breaks."""

    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text conservatively to stay inside prompt budgets."""

    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def file_sha256(content: bytes) -> str:
    """Hash file bytes deterministically."""

    return hashlib.sha256(content).hexdigest()


def safe_filename(name: str) -> str:
    """Build a filesystem-safe filename."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "document.pdf"


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON to disk with stable formatting."""

    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    """Read JSON from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def estimate_tokens(text: str) -> int:
    """Estimate token count without a tokenizer dependency."""

    words = len((text or "").split())
    return max(1, int(words * 1.3))


def resolve_data_path(path: str) -> Path:
    """Resolve a configured data path relative to the project root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(os.getcwd()) / candidate


async def async_retry(
    operation: Callable[[], Awaitable[Any]],
    *,
    retries: int = 2,
    base_delay_seconds: float = 1.0,
    retriable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Any:
    """Retry an async operation with small exponential backoff."""

    attempt = 0
    while True:
        try:
            return await operation()
        except retriable_exceptions:
            attempt += 1
            if attempt > retries:
                raise
            await asyncio.sleep(base_delay_seconds * attempt)
