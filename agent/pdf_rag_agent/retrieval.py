"""FAISS-backed retrieval for PDF chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from agent.pdf_rag_agent.chunking import DocumentChunk
from core.config import Settings
from core.embeddings import EmbeddingsClient
from core.utils import DocumentNotFoundError, ensure_directory, read_json, write_json


@dataclass(frozen=True)
class RetrievedChunk:
    """Chunk returned from vector search."""

    chunk_id: str
    document_id: str
    page: int
    text: str
    score: float


class PDFVectorStore:
    """Persist and query document chunk embeddings using FAISS."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vector_store_dir = ensure_directory(settings.vector_store_dir)
        self.embeddings_client = EmbeddingsClient(settings)

    async def build_document_index(self, *, document_id: str, chunks: list[DocumentChunk]) -> None:
        """Create and persist a document-specific FAISS index."""

        if not chunks:
            raise DocumentNotFoundError("No chunks available to index for this document.")

        embeddings = await self.embeddings_client.embed_texts([chunk.text for chunk in chunks])
        matrix = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(matrix)

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        faiss.write_index(index, str(self._index_path(document_id)))
        write_json(
            self._metadata_path(document_id),
            {
                "document_id": document_id,
                "chunks": [asdict(chunk) for chunk in chunks],
            },
        )

    def has_document(self, document_id: str) -> bool:
        """Check whether a persisted FAISS index exists for a document."""

        return self._index_path(document_id).exists() and self._metadata_path(document_id).exists()

    async def retrieve(self, *, document_id: str, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve the most similar chunks for a document query."""

        index_path = self._index_path(document_id)
        metadata_path = self._metadata_path(document_id)
        if not index_path.exists() or not metadata_path.exists():
            raise DocumentNotFoundError(f"No vector store found for document_id: {document_id}")

        index = faiss.read_index(str(index_path))
        metadata = read_json(metadata_path)
        chunks = metadata.get("chunks", [])
        if not chunks:
            raise DocumentNotFoundError(f"No chunk metadata found for document_id: {document_id}")

        query_embedding = np.array([await self.embeddings_client.embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_embedding)
        scores, indices = index.search(query_embedding, top_k or self.settings.retrieval_top_k)

        retrieved: list[RetrievedChunk] = []
        for score, index_value in zip(scores[0], indices[0], strict=False):
            if index_value < 0 or index_value >= len(chunks):
                continue
            chunk = chunks[index_value]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    page=chunk["page"],
                    text=chunk["text"],
                    score=float(score),
                )
            )

        return retrieved

    def load_chunks(self, document_id: str) -> list[DocumentChunk]:
        """Load stored chunks for summarization and inspection."""

        metadata_path = self._metadata_path(document_id)
        if not metadata_path.exists():
            raise DocumentNotFoundError(f"No metadata found for document_id: {document_id}")

        payload = read_json(metadata_path)
        return [DocumentChunk(**chunk) for chunk in payload.get("chunks", [])]

    def _index_path(self, document_id: str) -> Path:
        return self.vector_store_dir / f"{document_id}.faiss"

    def _metadata_path(self, document_id: str) -> Path:
        return self.vector_store_dir / f"{document_id}_metadata.json"
