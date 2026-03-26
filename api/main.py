"""FastAPI entrypoint for the web search agent service."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from core.config import ConfigError, load_settings
from core.utils import AgentError, configure_logging, normalize_query
from agent.web_search_agent.agent import WebSearchAgent


class SearchRequest(BaseModel):
    """Incoming query payload."""

    query: str = Field(..., description="Natural language web search query.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = normalize_query(value)
        if not normalized:
            raise ValueError("Query must be a non-empty string.")
        return normalized


class SearchResponse(BaseModel):
    """API response payload."""

    answer: str
    sources: list[str]


@lru_cache
def get_agent() -> WebSearchAgent:
    """Create a singleton agent instance for the process."""

    settings = load_settings()
    configure_logging(settings.log_level)
    return WebSearchAgent(settings)


app = FastAPI(
    title="AI Web Search Agent",
    version="1.0.0",
    description="Grounded web search agent powered by Tavily and OpenAI.",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health endpoint."""

    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Search the web and return a grounded answer with sources."""

    try:
        agent = get_agent()
        result = await agent.answer(request.query)
        return SearchResponse(answer=result.answer, sources=result.sources)
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected server error.") from exc
