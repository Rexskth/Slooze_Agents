"""Reusable embeddings client for OpenAI-compatible embedding APIs."""

from __future__ import annotations

from typing import Any

import httpx

from core.config import Settings
from core.utils import EmbeddingProviderError, async_retry, get_logger


logger = get_logger(__name__)


class EmbeddingsClient:
    """Thin wrapper around an embeddings endpoint."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = (settings.embedding_base_url or "").rstrip("/")
        if not self.base_url:
            raise EmbeddingProviderError("Embedding base URL is not configured.")

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text inputs."""

        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.settings.embedding_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
        }

        async def _request() -> list[list[float]]:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{self.base_url}/embeddings", headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            items = data.get("data", [])
            if not items:
                raise EmbeddingProviderError("Embedding provider returned no embedding data.")

            embeddings: list[list[float]] = []
            for item in items:
                embedding = item.get("embedding")
                if not embedding:
                    raise EmbeddingProviderError("Embedding provider returned an invalid embedding payload.")
                embeddings.append(embedding)

            return embeddings

        try:
            return await async_retry(
                _request,
                retries=2,
                base_delay_seconds=1.0,
                retriable_exceptions=(httpx.HTTPError, EmbeddingProviderError),
            )
        except Exception as exc:
            logger.exception("Embedding request failed.")
            raise EmbeddingProviderError("Failed to generate embeddings from the configured provider.") from exc

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""

        embeddings = await self.embed_texts([query])
        if not embeddings:
            raise EmbeddingProviderError("Embedding provider returned no vectors.")
        return embeddings[0]
