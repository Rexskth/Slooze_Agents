"""PDF ingestion and extraction workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from agent.pdf_rag_agent.chunking import PageText
from core.config import Settings
from core.utils import (
    DocumentNotFoundError,
    DocumentProcessingError,
    ensure_directory,
    file_sha256,
    read_json,
    safe_filename,
    write_json,
)


REGISTRY_FILENAME = "documents_registry.json"


@dataclass(frozen=True)
class IngestedDocument:
    """Metadata returned after a PDF is ingested."""

    document_id: str
    filename: str
    file_path: Path
    file_hash: str
    page_count: int
    pages: list[PageText]
    reused_existing: bool


class PDFIngestionService:
    """Handle PDF upload persistence and text extraction."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.documents_dir = ensure_directory(settings.documents_dir)
        self.vector_store_dir = ensure_directory(settings.vector_store_dir)
        self.registry_path = self.vector_store_dir / REGISTRY_FILENAME

    def ingest(self, *, filename: str, content: bytes) -> IngestedDocument:
        """Persist a PDF locally and extract its text."""

        if not content:
            raise DocumentProcessingError("Uploaded PDF is empty.")

        file_hash = file_sha256(content)
        existing = self._get_existing_document(file_hash)
        if existing is not None:
            return self._load_existing_document(existing["document_id"], reused_existing=True)

        document_id = f"doc_{file_hash[:12]}"
        safe_name = safe_filename(filename or f"{document_id}.pdf")
        file_path = self.documents_dir / f"{document_id}_{safe_name}"
        file_path.write_bytes(content)

        pages = self._extract_pages(file_path)
        if not pages:
            file_path.unlink(missing_ok=True)
            raise DocumentProcessingError("No extractable text found in the uploaded PDF.")

        record = {
            "document_id": document_id,
            "filename": safe_name,
            "file_hash": file_hash,
            "file_path": str(file_path),
        }
        self._upsert_registry_record(record)

        return IngestedDocument(
            document_id=document_id,
            filename=safe_name,
            file_path=file_path,
            file_hash=file_hash,
            page_count=len(pages),
            pages=pages,
            reused_existing=False,
        )

    def _extract_pages(self, file_path: Path) -> list[PageText]:
        """Extract page-wise text using PyMuPDF."""

        try:
            document = fitz.open(file_path)
        except Exception as exc:
            raise DocumentProcessingError("Invalid or unreadable PDF file.") from exc

        pages: list[PageText] = []
        with document:
            for index, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append(PageText(page_number=index, text=text))
        return pages

    def _load_existing_document(self, document_id: str, *, reused_existing: bool) -> IngestedDocument:
        """Load document metadata and extracted text from an existing local file."""

        record = self.get_document_record(document_id)
        file_path = Path(record["file_path"])
        pages = self._extract_pages(file_path)
        if not pages:
            raise DocumentProcessingError("Stored PDF no longer contains extractable text.")

        return IngestedDocument(
            document_id=document_id,
            filename=record["filename"],
            file_path=file_path,
            file_hash=record["file_hash"],
            page_count=len(pages),
            pages=pages,
            reused_existing=reused_existing,
        )

    def get_document_record(self, document_id: str) -> dict:
        """Fetch a document record from the registry."""

        registry = self._read_registry()
        for item in registry.get("documents", []):
            if item["document_id"] == document_id:
                return item
        raise DocumentNotFoundError(f"Document not found for id: {document_id}")

    def _get_existing_document(self, file_hash: str) -> dict | None:
        registry = self._read_registry()
        for item in registry.get("documents", []):
            if item["file_hash"] == file_hash:
                return item
        return None

    def _read_registry(self) -> dict:
        if not self.registry_path.exists():
            return {"documents": []}
        return read_json(self.registry_path)

    def _upsert_registry_record(self, record: dict) -> None:
        registry = self._read_registry()
        documents = [item for item in registry.get("documents", []) if item["document_id"] != record["document_id"]]
        documents.append(record)
        write_json(self.registry_path, {"documents": documents})
