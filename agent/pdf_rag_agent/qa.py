"""Grounded QA and summarization logic for PDFs."""

from __future__ import annotations

from dataclasses import dataclass

from agent.pdf_rag_agent.chunking import DocumentChunk
from agent.pdf_rag_agent.prompts import (
    QA_SYSTEM_PROMPT,
    QA_USER_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT_TEMPLATE,
)
from agent.pdf_rag_agent.retrieval import PDFVectorStore, RetrievedChunk
from core.config import Settings
from core.llm import OpenAILLMClient
from core.utils import clean_text, truncate_text


@dataclass(frozen=True)
class ChunkSource:
    """Source metadata returned to clients."""

    page: int
    chunk_id: str


@dataclass(frozen=True)
class PDFAnswer:
    """Structured answer for QA and summarization."""

    answer: str
    sources: list[ChunkSource]


class PDFQuestionAnsweringService:
    """Orchestrate grounded retrieval and generation for PDFs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm_client = OpenAILLMClient(settings)
        self.vector_store = PDFVectorStore(settings)

    async def answer_question(self, *, document_id: str, query: str) -> PDFAnswer:
        """Answer a question using retrieved document chunks."""

        retrieved = await self.vector_store.retrieve(document_id=document_id, query=query, top_k=self.settings.retrieval_top_k)
        if not retrieved:
            return PDFAnswer(answer="Not found in document", sources=[])

        context = self._format_retrieved_context(retrieved)
        answer = clean_text(
            await self.llm_client.generate_text(
                system_prompt=QA_SYSTEM_PROMPT,
                user_prompt=QA_USER_PROMPT_TEMPLATE.format(context=context, query=query),
            )
        )
        if answer == "Not found in document":
            return PDFAnswer(answer=answer, sources=[])

        return PDFAnswer(answer=answer, sources=self._sources_from_retrieved(retrieved))

    async def summarize_document(self, *, document_id: str) -> PDFAnswer:
        """Summarize a document using representative chunks."""

        chunks = self.vector_store.load_chunks(document_id)
        if not chunks:
            return PDFAnswer(answer="Not found in document", sources=[])

        selected_chunks = self._select_summary_chunks(chunks)
        context = self._format_summary_context(selected_chunks)
        answer = clean_text(
            await self.llm_client.generate_text(
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                user_prompt=SUMMARY_USER_PROMPT_TEMPLATE.format(context=context),
            )
        )
        if answer == "Not found in document":
            return PDFAnswer(answer=answer, sources=[])

        sources = [ChunkSource(page=chunk.page, chunk_id=chunk.chunk_id) for chunk in selected_chunks]
        return PDFAnswer(answer=answer, sources=self._dedupe_sources(sources))

    def _format_retrieved_context(self, retrieved: list[RetrievedChunk]) -> str:
        ordered = sorted(retrieved, key=lambda item: (item.page, item.chunk_id))
        sections = [
            f"[Chunk {chunk.chunk_id} | Page {chunk.page}]\n{truncate_text(chunk.text, 1800)}"
            for chunk in ordered
        ]
        return "\n\n".join(sections)

    def _format_summary_context(self, chunks: list[DocumentChunk]) -> str:
        sections = [
            f"[Chunk {chunk.chunk_id} | Page {chunk.page}]\n{truncate_text(chunk.text, 1800)}"
            for chunk in chunks
        ]
        return "\n\n".join(sections)

    def _select_summary_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        if len(chunks) <= self.settings.summary_max_chunks:
            return chunks

        step = max(1, len(chunks) // self.settings.summary_max_chunks)
        selected = [chunks[index] for index in range(0, len(chunks), step)]
        return selected[: self.settings.summary_max_chunks]

    def _sources_from_retrieved(self, retrieved: list[RetrievedChunk]) -> list[ChunkSource]:
        return self._dedupe_sources(
            [ChunkSource(page=chunk.page, chunk_id=chunk.chunk_id) for chunk in retrieved]
        )

    def _dedupe_sources(self, sources: list[ChunkSource]) -> list[ChunkSource]:
        seen: set[tuple[int, str]] = set()
        deduped: list[ChunkSource] = []
        for source in sources:
            key = (source.page, source.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        return deduped
