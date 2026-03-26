"""Grounded web search agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from core.config import Settings
from core.llm import OpenAILLMClient
from core.utils import TTLCache, clean_text, get_logger, make_cache_key, normalize_query, truncate_text
from agent.web_search_agent.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from agent.web_search_agent.tools import SearchResult, TavilySearchTool


logger = get_logger(__name__)


@dataclass(frozen=True)
class WebSearchResponse:
    """Structured web search agent output."""

    answer: str
    sources: list[str]


class WebSearchAgent:
    """Coordinates search retrieval and grounded answer generation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.search_tool = TavilySearchTool(settings)
        self.llm_client = OpenAILLMClient(settings)
        self.cache = TTLCache(ttl_seconds=settings.cache_ttl_seconds)

    async def answer(self, query: str) -> WebSearchResponse:
        """Answer a user query using web search plus grounded generation."""

        normalized_query = normalize_query(query)
        cache_key = make_cache_key(normalized_query)
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            logger.info("Cache hit for query: %s", normalized_query)
            return cached_response

        results = await self.search_tool.search(normalized_query)
        if not results:
            response = WebSearchResponse(answer="Insufficient information", sources=[])
            self.cache.set(cache_key, response)
            return response

        context = self._build_context(results)
        llm_payload = await self.llm_client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT_TEMPLATE.format(context=context, query=normalized_query),
        )

        response = self._build_response(llm_payload=llm_payload, results=results)
        self.cache.set(cache_key, response)
        return response

    def _build_context(self, results: list[SearchResult]) -> str:
        """Serialize retrieved results into a compact grounded prompt context."""

        sections: list[str] = []
        remaining_chars = self.settings.context_char_limit

        for result in results:
            snippet = clean_text(result.content)
            section = (
                f"[Source {result.id}]\n"
                f"Title: {result.title}\n"
                f"URL: {result.url}\n"
                f"Content: {snippet}\n"
            )

            if len(section) > remaining_chars:
                section = truncate_text(section, remaining_chars)

            if not section:
                continue

            sections.append(section)
            remaining_chars -= len(section)
            if remaining_chars <= 0:
                break

        return "\n".join(sections).strip()

    def _build_response(self, *, llm_payload: dict, results: list[SearchResult]) -> WebSearchResponse:
        """Map model output back to answer text and concrete URLs."""

        answer = clean_text(str(llm_payload.get("answer", "") or "Insufficient information"))
        requested_ids = llm_payload.get("source_ids", [])
        result_map = {result.id: result.url for result in results}

        sources: list[str] = []
        for source_id in requested_ids:
            if isinstance(source_id, int) and source_id in result_map and result_map[source_id] not in sources:
                sources.append(result_map[source_id])

        if answer == "Insufficient information":
            sources = []
        elif not sources:
            # Fall back to top retrieved results so every grounded answer carries sources.
            sources = [result.url for result in results[: min(3, len(results))]]
            logger.warning(
                "Model returned an answer without valid source_ids. Falling back to top %s retrieved URLs.",
                len(sources),
            )

        return WebSearchResponse(answer=answer, sources=sources)
