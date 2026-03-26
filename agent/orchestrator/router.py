"""Rule-based routing for the unified AI agent platform."""

from __future__ import annotations

from dataclasses import dataclass

from core.utils import get_logger, normalize_query


logger = get_logger(__name__)


PDF_ROUTE = "pdf"
WEB_ROUTE = "web"

PDF_KEYWORDS = {
    "pdf",
    "document",
    "file",
    "paper",
    "report",
    "resume",
    "according to the document",
    "according to the pdf",
    "in the document",
    "in the pdf",
    "from the document",
    "from the pdf",
}

PDF_SUMMARY_KEYWORDS = {
    "summarize document",
    "summarise document",
    "summarize pdf",
    "summarise pdf",
    "summarize this file",
    "summary of the document",
    "summary of the pdf",
}

WEB_KEYWORDS = {
    "latest",
    "news",
    "today",
    "recent",
    "current",
    "launch",
    "launched",
    "market",
    "specs",
    "price",
    "stock",
    "weather",
    "score",
    "update",
}


@dataclass(frozen=True)
class RouteDecision:
    """Outcome of query routing."""

    route: str
    reason: str
    is_summary_request: bool = False


class QueryRouter:
    """Determine which specialist agent should handle a query."""

    def decide(self, *, query: str, document_id: str | None = None) -> RouteDecision:
        """Route a query to either the web or PDF agent."""

        normalized_query = normalize_query(query).lower()
        is_summary_request = any(keyword in normalized_query for keyword in PDF_SUMMARY_KEYWORDS)

        if is_summary_request:
            decision = RouteDecision(
                route=PDF_ROUTE,
                reason="Matched document summarization keywords.",
                is_summary_request=True,
            )
        elif document_id and any(keyword in normalized_query for keyword in PDF_KEYWORDS):
            decision = RouteDecision(
                route=PDF_ROUTE,
                reason="Document id provided and query contains document-oriented keywords.",
            )
        elif document_id and not any(keyword in normalized_query for keyword in WEB_KEYWORDS):
            decision = RouteDecision(
                route=PDF_ROUTE,
                reason="Document id provided and query does not strongly indicate a live web lookup.",
            )
        elif any(keyword in normalized_query for keyword in PDF_KEYWORDS):
            decision = RouteDecision(
                route=PDF_ROUTE,
                reason="Matched document-oriented keywords.",
            )
        elif any(keyword in normalized_query for keyword in WEB_KEYWORDS):
            decision = RouteDecision(
                route=WEB_ROUTE,
                reason="Matched live web-search keywords.",
            )
        else:
            decision = RouteDecision(
                route=WEB_ROUTE,
                reason="Defaulted to web route for general open-domain questions.",
            )

        logger.info(
            "Routing decision | route=%s | summary=%s | document_id_present=%s | reason=%s | query=%s",
            decision.route,
            decision.is_summary_request,
            bool(document_id),
            decision.reason,
            normalized_query,
        )
        return decision
