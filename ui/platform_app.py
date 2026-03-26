"""Unified Streamlit UI for the orchestrated AI agent platform."""

from __future__ import annotations

import os

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
QUERY_ENDPOINT = f"{API_BASE_URL}/query"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload"
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


def query_platform(query: str, document_id: str | None = None) -> dict:
    """Call the unified orchestrator endpoint."""

    payload = {"query": query}
    if document_id:
        payload["document_id"] = document_id

    with httpx.Client(timeout=120.0) as client:
        response = client.post(QUERY_ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json()


def summarize_document(document_id: str) -> dict:
    """Call the direct PDF summarization endpoint for convenience."""

    with httpx.Client(timeout=120.0) as client:
        response = client.post(SUMMARIZE_ENDPOINT, json={"document_id": document_id})
        response.raise_for_status()
        return response.json()


def handle_http_error(exc: httpx.HTTPStatusError) -> None:
    """Display backend error details."""

    try:
        detail = exc.response.json().get("detail", exc.response.text)
    except ValueError:
        detail = exc.response.text
    st.error(f"Backend error: {detail}")


def reset_platform_state() -> None:
    """Clear orchestrator UI state."""

    st.session_state.platform_chat = []
    st.session_state.document_id = None
    st.session_state.document_name = None
    st.session_state.upload_result = None
    st.session_state.use_document_context = False


def render_unified_sources(route: str, sources: list) -> None:
    """Render sources based on the routed agent type."""

    if not sources:
        return

    st.markdown("**Sources**")
    if route == "web":
        for source in sources:
            st.markdown(f"- [{source}]({source})")
        return

    for source in sources:
        if isinstance(source, dict):
            page = source.get("page", "?")
            chunk_id = source.get("chunk_id", "unknown")
            st.markdown(f"- Page `{page}` | Chunk `{chunk_id}`")
        else:
            st.markdown(f"- `{source}`")


st.set_page_config(
    page_title="Slooze Agents Platform",
    page_icon="A",
    layout="centered",
)

if "platform_chat" not in st.session_state:
    st.session_state.platform_chat = []
if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "upload_result" not in st.session_state:
    st.session_state.upload_result = None
if "use_document_context" not in st.session_state:
    st.session_state.use_document_context = False

st.title("Slooze Agents")
st.caption("Unified AI Agent Platform")

with st.sidebar:
    st.subheader("Connection")
    st.code(API_BASE_URL)
    st.caption("This UI sends questions to the unified `/query` endpoint.")
    if st.button("Reset Platform Session", use_container_width=True):
        reset_platform_state()
        st.rerun()

st.subheader("Document Context")
uploaded_file = st.file_uploader("Upload a PDF to enable document-aware routing", type=["pdf"])

if uploaded_file is not None:
    st.write(f"Selected file: `{uploaded_file.name}`")
    if st.button("Upload PDF for Platform", type="primary", use_container_width=True):
        try:
            with st.spinner("Uploading and indexing PDF..."):
                result = upload_pdf(uploaded_file.name, uploaded_file.getvalue())
            st.session_state.document_id = result["document_id"]
            st.session_state.document_name = uploaded_file.name
            st.session_state.upload_result = result
            st.session_state.use_document_context = True
            st.success("PDF uploaded successfully. The platform can now route document-aware queries.")
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

st.session_state.use_document_context = st.checkbox(
    "Include current document_id when sending queries",
    value=st.session_state.use_document_context,
    disabled=st.session_state.document_id is None,
)

if st.session_state.document_id:
    if st.button("Summarize Current Document", use_container_width=True):
        try:
            with st.spinner("Generating grounded document summary..."):
                result = summarize_document(st.session_state.document_id)
            st.session_state.platform_chat.append(
                {
                    "role": "assistant",
                    "route": "pdf",
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

st.subheader("Platform Chat")
st.caption(
    "Ask open-domain questions for web routing, or ask document-aware questions after uploading a PDF."
)

for item in st.session_state.platform_chat:
    with st.chat_message(item["role"]):
        if item["role"] == "assistant":
            route = item.get("route", "unknown")
            st.caption(f"Route: `{route}`")
        if item.get("label"):
            st.markdown(f"**{item['label']}**")
        st.markdown(item["content"])
        if item["role"] == "assistant":
            render_unified_sources(item.get("route", "web"), item.get("sources", []))

query = st.chat_input("Ask the unified agent platform a question")

if query:
    cleaned_query = query.strip()
    if not cleaned_query:
        st.warning("Please enter a non-empty question.")
    else:
        document_id = st.session_state.document_id if st.session_state.use_document_context else None
        st.session_state.platform_chat.append({"role": "user", "content": cleaned_query})
        with st.chat_message("user"):
            st.markdown(cleaned_query)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Routing query and generating answer..."):
                    result = query_platform(cleaned_query, document_id=document_id)

                route = result.get("route", "unknown")
                answer = result.get("answer", "No answer returned.")
                sources = result.get("sources", [])

                st.caption(f"Route: `{route}`")
                st.markdown(answer)
                render_unified_sources(route, sources)

            st.session_state.platform_chat.append(
                {
                    "role": "assistant",
                    "route": route,
                    "content": answer,
                    "sources": sources,
                }
            )
        except httpx.HTTPStatusError as exc:
            handle_http_error(exc)
        except httpx.HTTPError as exc:
            st.error("Could not reach the backend. Make sure the API server is running.")
            st.caption(str(exc))
