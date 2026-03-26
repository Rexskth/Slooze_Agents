"""External tooling for web search retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from core.config import Settings
from core.utils import SearchProviderError, async_retry, clean_text, get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Normalized Tavily search result."""

    id: int
    title: str
    url: str
    content: str


class TavilySearchTool:
    """Async Tavily search client."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """Fetch and normalize search results from Tavily."""

        max_results = max_results or self.settings.max_search_results

        async def _request() -> list[SearchResult]:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{self.settings.tavily_base_url.rstrip('/')}/search",
                    json={
                        "api_key": self.settings.tavily_api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "advanced",
                        "include_answer": False,
                        "include_raw_content": False,
                    },
                )
                response.raise_for_status()
                payload = response.json()

            results = []
            for index, item in enumerate(payload.get("results", []), start=1):
                title = clean_text(item.get("title", ""))
                url = clean_text(item.get("url", ""))
                content = clean_text(item.get("content", "") or item.get("snippet", ""))

                if not (title and url and content):
                    continue

                results.append(
                    SearchResult(
                        id=index,
                        title=title,
                        url=url,
                        content=content,
                    )
                )

            return results

        try:
            return await async_retry(
                _request,
                retries=2,
                base_delay_seconds=1.0,
                retriable_exceptions=(httpx.HTTPError,),
            )
        except Exception as exc:
            logger.exception("Tavily search request failed.")
            raise SearchProviderError("Failed to fetch search results from Tavily.") from exc
