"""Streamlit UI for the PDF RAG agent."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload"
ASK_ENDPOINT = f"{API_BASE_URL}/ask"
SUMMARIZE_ENDPOINT = f"{API_BASE_URL}/summarize"


def upload_pdf(file_name: str, content: bytes) -> dict:
    """Upload a PDF to the backend."""

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            UPLOAD_ENDPOINT,
            files={"file": (file_name, content, "application/pdf")},
        )
        response.raise_for_status()
        return response.json()


def ask_document(document_id: str, query: str) -> dict:
    """Ask a question about an uploaded document."""

    with httpx.Client(timeout=120.0) as client:
        response = client.post(ASK_ENDPOINT, json={"document_id": document_id, "query": query})
        response.raise_for_status()
        return response.json()


def summarize_document(document_id: str) -> dict:
    """Summarize an uploaded document."""

    with httpx.Client(timeout=120.0) as client:
        response = client.post(SUMMARIZE_ENDPOINT, json={"document_id": document_id})
        response.raise_for_status()
        return response.json()


def render_pdf_sources(sources: list[dict]) -> None:
    """Render PDF chunk sources."""

    if not sources:
        return

    st.markdown("**Sources**")
    for source in sources:
        page = source.get("page", "?")
        chunk_id = source.get("chunk_id", "unknown")
        st.markdown(f"- Page `{page}` | Chunk `{chunk_id}`")


def handle_http_error(exc: httpx.HTTPStatusError) -> None:
    """Display backend error details."""

    try:
        detail = exc.response.json().get("detail", exc.response.text)
    except ValueError:
        detail = exc.response.text
    st.error(f"Backend error: {detail}")


def reset_pdf_state() -> None:
    """Clear the PDF UI state."""

    st.session_state.pdf_chat = []
    st.session_state.document_id = None
    st.session_state.document_name = None
    st.session_state.upload_result = None


st.set_page_config(
    page_title="Slooze Agents PDF",
    page_icon="P",
    layout="centered",
)

if "pdf_chat" not in st.session_state:
    st.session_state.pdf_chat = []
if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

st.title("Slooze Agents")
st.caption("AI Agent for PDF Summarization and Question Answering")

with st.sidebar:
    st.subheader("Connection")
    st.code(API_BASE_URL)
    st.caption("Start the FastAPI backend before using this UI.")
    if st.button("Reset PDF Session", use_container_width=True):
        reset_pdf_state()
        st.rerun()

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    st.write(f"Selected file: `{uploaded_file.name}`")
    if st.button("Upload PDF", type="primary", use_container_width=True):
        try:
            with st.spinner("Uploading and indexing PDF..."):
                result = upload_pdf(uploaded_file.name, uploaded_file.getvalue())
            st.session_state.document_id = result["document_id"]
            st.session_state.document_name = uploaded_file.name
            st.session_state.upload_result = result
            st.session_state.pdf_chat = []
            st.success("PDF uploaded and indexed successfully.")
        except httpx.HTTPStatusError as exc:
            handle_http_error(exc)
        except httpx.HTTPError as exc:
            st.error("Could not reach the backend. Make sure the API server is running.")
            st.caption(str(exc))

if st.session_state.upload_result:
    result = st.session_state.upload_result
    st.info(
        f"Document ID: `{result['document_id']}` | "
        f"Pages: `{result['page_count']}` | "
        f"Chunks: `{result['chunk_count']}` | "
        f"Reused existing: `{result['reused_existing']}`"
    )

if st.session_state.document_id:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Summarize Document", use_container_width=True):
            try:
                with st.spinner("Generating grounded summary..."):
                    result = summarize_document(st.session_state.document_id)
                st.session_state.pdf_chat.append(
                    {
                        "role": "assistant",
                        "label": "Document Summary",
                        "content": result.get("answer", "No summary returned."),
                        "sources": result.get("sources", []),
                    }
                )
                st.rerun()
            except httpx.HTTPStatusError as exc:
                handle_http_error(exc)
            except httpx.HTTPError as exc:
                st.error("Could not reach the backend. Make sure the API server is running.")
                st.caption(str(exc))
    with col2:
        st.caption(f"Active document: `{st.session_state.document_name or st.session_state.document_id}`")

for item in st.session_state.pdf_chat:
    with st.chat_message(item["role"]):
        if item.get("label"):
            st.markdown(f"**{item['label']}**")
        st.markdown(item["content"])
        if item["role"] == "assistant":
            render_pdf_sources(item.get("sources", []))

if not st.session_state.document_id:
    st.warning("Upload a PDF first to ask questions or generate a summary.")
else:
    query = st.chat_input("Ask a question about the uploaded PDF")
    if query:
        cleaned_query = query.strip()
        if not cleaned_query:
            st.warning("Please enter a non-empty question.")
        else:
            st.session_state.pdf_chat.append({"role": "user", "content": cleaned_query})
            with st.chat_message("user"):
                st.markdown(cleaned_query)

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving relevant chunks and answering..."):
                        result = ask_document(st.session_state.document_id, cleaned_query)
                    answer = result.get("answer", "No answer returned.")
                    sources = result.get("sources", [])
                    st.markdown(answer)
                    render_pdf_sources(sources)
                st.session_state.pdf_chat.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except httpx.HTTPStatusError as exc:
                handle_http_error(exc)
            except httpx.HTTPError as exc:
                st.error("Could not reach the backend. Make sure the API server is running.")
                st.caption(str(exc))
