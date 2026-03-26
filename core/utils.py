"""Shared utility helpers for logging, retries, caching, and text cleanup."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from typing import Any, Awaitable, Callable


class AgentError(RuntimeError):
    """Base error for expected agent failures."""


class SearchProviderError(AgentError):
    """Raised when the search provider fails."""


class LLMProviderError(AgentError):
    """Raised when the language model provider fails."""


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
