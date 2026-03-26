"""Chunking logic for extracted PDF text."""

from __future__ import annotations

from dataclasses import dataclass

from core.utils import clean_text, estimate_tokens


@dataclass(frozen=True)
class PageText:
    """Text extracted from a single PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    """Chunked document unit stored in the vector store."""

    chunk_id: str
    document_id: str
    page: int
    text: str
    token_estimate: int


def _token_budget_to_word_budget(token_budget: int) -> int:
    return max(50, int(token_budget / 1.3))


def chunk_document(
    *,
    document_id: str,
    pages: list[PageText],
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[DocumentChunk]:
    """Split extracted page text into overlapping chunks."""

    chunk_word_budget = _token_budget_to_word_budget(chunk_size_tokens)
    overlap_word_budget = min(_token_budget_to_word_budget(chunk_overlap_tokens), chunk_word_budget // 2)

    chunks: list[DocumentChunk] = []
    chunk_index = 1

    for page in pages:
        page_text = clean_text(page.text)
        if not page_text:
            continue

        words = page_text.split()
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_word_budget)
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"chunk_{chunk_index}",
                        document_id=document_id,
                        page=page.page_number,
                        text=chunk_text,
                        token_estimate=estimate_tokens(chunk_text),
                    )
                )
                chunk_index += 1

            if end >= len(words):
                break
            start = max(0, end - overlap_word_budget)

    return chunks
