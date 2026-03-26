"""Central controller for routing and delegating user queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.orchestrator.router import PDF_ROUTE, QueryRouter
from agent.pdf_rag_agent.qa import PDFQuestionAnsweringService
from agent.web_search_agent.agent import WebSearchAgent
from core.utils import AgentError, DocumentNotFoundError, get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class UnifiedAgentResponse:
    """Unified response returned by the orchestrator."""

    route: str
    answer: str
    sources: list[Any]


class AgentController:
    """Delegate user queries to the appropriate specialist agent."""

    def __init__(
        self,
        *,
        router: QueryRouter,
        web_agent: WebSearchAgent,
        pdf_qa_service: PDFQuestionAnsweringService,
    ) -> None:
        self.router = router
        self.web_agent = web_agent
        self.pdf_qa_service = pdf_qa_service

    async def handle_query(self, *, query: str, document_id: str | None = None) -> UnifiedAgentResponse:
        """Route and execute a query across specialist agents."""

        decision = self.router.decide(query=query, document_id=document_id)

        if decision.route == PDF_ROUTE:
            if not document_id:
                raise DocumentNotFoundError(
                    "PDF route selected, but no document_id was provided. Upload a PDF first and pass document_id."
                )

            if decision.is_summary_request:
                result = await self.pdf_qa_service.summarize_document(document_id=document_id)
            else:
                result = await self.pdf_qa_service.answer_question(document_id=document_id, query=query)

            response = UnifiedAgentResponse(
                route=decision.route,
                answer=result.answer,
                sources=[{"page": source.page, "chunk_id": source.chunk_id} for source in result.sources],
            )
        else:
            result = await self.web_agent.answer(query)
            response = UnifiedAgentResponse(
                route=decision.route,
                answer=result.answer,
                sources=result.sources,
            )

        logger.info(
            "Controller response | route=%s | sources=%s | answer_preview=%s",
            response.route,
            len(response.sources),
            response.answer[:120],
        )
        return response
