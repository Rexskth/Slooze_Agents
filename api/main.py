"""FastAPI entrypoint for the agent platform."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from agent.orchestrator.agent_controller import AgentController
from agent.orchestrator.router import QueryRouter
from agent.pdf_rag_agent.chunking import chunk_document
from agent.pdf_rag_agent.ingestion import PDFIngestionService
from agent.pdf_rag_agent.qa import PDFQuestionAnsweringService
from agent.pdf_rag_agent.retrieval import PDFVectorStore
from core.config import ConfigError, load_settings
from core.utils import (
    AgentError,
    DocumentNotFoundError,
    DocumentProcessingError,
    configure_logging,
    normalize_query,
)
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


class QueryRequest(BaseModel):
    """Unified orchestrator query request."""

    query: str = Field(..., description="Natural language query for the platform.")
    document_id: str | None = Field(
        default=None,
        description="Optional document identifier used for PDF-specific queries.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = normalize_query(value)
        if not normalized:
            raise ValueError("Query must be a non-empty string.")
        return normalized


class UnifiedQueryResponse(BaseModel):
    """Unified orchestrated response across agents."""

    route: str
    answer: str
    sources: list[Any]


class UploadResponse(BaseModel):
    """Response for uploaded documents."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    reused_existing: bool


class AskRequest(BaseModel):
    """Question answering request."""

    document_id: str = Field(..., description="Document identifier returned by /upload.")
    query: str = Field(..., description="Question about the uploaded document.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = normalize_query(value)
        if not normalized:
            raise ValueError("Query must be a non-empty string.")
        return normalized


class SummarizeRequest(BaseModel):
    """Document summarization request."""

    document_id: str = Field(..., description="Document identifier returned by /upload.")


class SourceResponse(BaseModel):
    """Chunk source metadata for PDF responses."""

    page: int
    chunk_id: str


class DocumentAnswerResponse(BaseModel):
    """Response for PDF QA and summarization."""

    answer: str
    sources: list[SourceResponse]


@lru_cache
def get_agent() -> WebSearchAgent:
    """Create a singleton agent instance for the process."""

    settings = load_settings()
    configure_logging(settings.log_level)
    return WebSearchAgent(settings)


@lru_cache
def get_pdf_ingestion_service() -> PDFIngestionService:
    """Create a singleton PDF ingestion service."""

    settings = load_settings()
    configure_logging(settings.log_level)
    return PDFIngestionService(settings)


@lru_cache
def get_pdf_vector_store() -> PDFVectorStore:
    """Create a singleton PDF vector store service."""

    settings = load_settings()
    configure_logging(settings.log_level)
    return PDFVectorStore(settings)


@lru_cache
def get_pdf_qa_service() -> PDFQuestionAnsweringService:
    """Create a singleton PDF QA service."""

    settings = load_settings()
    configure_logging(settings.log_level)
    return PDFQuestionAnsweringService(settings)


@lru_cache
def get_router() -> QueryRouter:
    """Create a singleton query router."""

    settings = load_settings()
    configure_logging(settings.log_level)
    return QueryRouter()


@lru_cache
def get_agent_controller() -> AgentController:
    """Create a singleton orchestrator controller."""

    settings = load_settings()
    configure_logging(settings.log_level)
    return AgentController(
        router=get_router(),
        web_agent=WebSearchAgent(settings),
        pdf_qa_service=PDFQuestionAnsweringService(settings),
    )


app = FastAPI(
    title="AI Agent Platform",
    version="1.0.0",
    description="Grounded web search and PDF RAG agents powered by FastAPI.",
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


@app.post("/query", response_model=UnifiedQueryResponse)
async def query_platform(request: QueryRequest) -> UnifiedQueryResponse:
    """Route a query dynamically to the web or PDF agent."""

    try:
        controller = get_agent_controller()
        result = await controller.handle_query(query=request.query, document_id=request.document_id)
        return UnifiedQueryResponse(route=result.route, answer=result.answer, sources=result.sources)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected server error.") from exc


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """Upload, ingest, chunk, and index a PDF document."""

    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        content = await file.read()
        ingestion_service = get_pdf_ingestion_service()
        vector_store = get_pdf_vector_store()
        settings = load_settings()

        ingested = ingestion_service.ingest(filename=file.filename, content=content)
        chunks = chunk_document(
            document_id=ingested.document_id,
            pages=ingested.pages,
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks could be generated from the uploaded PDF.")

        if not (ingested.reused_existing and vector_store.has_document(ingested.document_id)):
            await vector_store.build_document_index(document_id=ingested.document_id, chunks=chunks)
        return UploadResponse(
            document_id=ingested.document_id,
            filename=ingested.filename,
            page_count=ingested.page_count,
            chunk_count=len(chunks),
            reused_existing=ingested.reused_existing,
        )
    except HTTPException:
        raise
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected server error.") from exc


@app.post("/ask", response_model=DocumentAnswerResponse)
async def ask_document(request: AskRequest) -> DocumentAnswerResponse:
    """Answer a question about an uploaded PDF document."""

    try:
        qa_service = get_pdf_qa_service()
        result = await qa_service.answer_question(document_id=request.document_id, query=request.query)
        return DocumentAnswerResponse(
            answer=result.answer,
            sources=[SourceResponse(page=source.page, chunk_id=source.chunk_id) for source in result.sources],
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected server error.") from exc


@app.post("/summarize", response_model=DocumentAnswerResponse)
async def summarize_document(request: SummarizeRequest) -> DocumentAnswerResponse:
    """Summarize an uploaded PDF document."""

    try:
        qa_service = get_pdf_qa_service()
        result = await qa_service.summarize_document(document_id=request.document_id)
        return DocumentAnswerResponse(
            answer=result.answer,
            sources=[SourceResponse(page=source.page, chunk_id=source.chunk_id) for source in result.sources],
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except AgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected server error.") from exc
