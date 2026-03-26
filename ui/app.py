"""Streamlit UI for interacting with the web search agent."""

from __future__ import annotations

import os

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
SEARCH_ENDPOINT = f"{API_BASE_URL}/search"


def fetch_answer(query: str) -> dict:
    """Call the backend search endpoint and return the JSON payload."""

    with httpx.Client(timeout=60.0) as client:
        response = client.post(SEARCH_ENDPOINT, json={"query": query})
        response.raise_for_status()
        return response.json()


def reset_chat() -> None:
    """Clear UI chat history."""

    st.session_state.messages = []


def render_sources(sources: list[str]) -> None:
    """Render answer sources as clickable links."""

    if not sources:
        return

    st.markdown("**Sources**")
    for source in sources:
        st.markdown(f"- [{source}]({source})")


st.set_page_config(
    page_title="Slooze Agents",
    page_icon="S",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Slooze Agents")
st.caption("Web Search Agent Demo")

with st.sidebar:
    st.subheader("Connection")
    st.code(API_BASE_URL)
    st.caption("Start the FastAPI server before sending queries from this UI.")
    if st.button("Clear Chat", use_container_width=True):
        reset_chat()
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

query = st.chat_input("Ask the web search agent a question")

if query:
    cleaned_query = query.strip()
    if not cleaned_query:
        st.warning("Please enter a non-empty question.")
    else:
        st.session_state.messages.append({"role": "user", "content": cleaned_query})
        with st.chat_message("user"):
            st.markdown(cleaned_query)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the web and grounding an answer..."):
                    result = fetch_answer(cleaned_query)

                answer = result.get("answer", "No answer returned.")
                sources = result.get("sources", [])

                st.markdown(answer)
                render_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except httpx.HTTPStatusError as exc:
                try:
                    detail = exc.response.json().get("detail", exc.response.text)
                except ValueError:
                    detail = exc.response.text
                st.error(f"Backend error: {detail}")
            except httpx.HTTPError as exc:
                st.error(
                    "Could not reach the FastAPI backend. "
                    "Make sure the API server is running and API_BASE_URL is correct."
                )
                st.caption(str(exc))
