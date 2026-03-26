# AI Agent Platform

This repository contains a modular FastAPI-based AI backend with two independent agents:

- `Web Search Agent`
- `AI Agent for PDF Summarization and Question Answering`

The platform is designed to look like a production-ready backend, not a one-off script. Shared configuration, LLM access, embeddings, and utilities live in `core/`, while each agent keeps its own retrieval and prompting logic isolated.

## Project Structure

```text
agent/
  web_search_agent/
    agent.py
    tools.py
    prompts.py
  pdf_rag_agent/
    ingestion.py
    chunking.py
    retrieval.py
    qa.py
    prompts.py

core/
  embeddings.py
  llm.py
  config.py
  utils.py

api/
  main.py

ui/
  app.py

data/
  documents/
  vector_store/

orchestrator/

requirements.txt
.env
.env.example
README.md
```

## Architecture

### 1. API Layer

- `api/main.py` exposes the FastAPI app and all agent endpoints.
- Request validation and HTTP error mapping stay at the edge of the system.
- `ui/app.py` provides a thin Streamlit chat interface for the web search agent.

### 2. Agent Layer

- `agent/web_search_agent/` contains grounded web retrieval and answer generation.
- `agent/pdf_rag_agent/` contains PDF ingestion, chunking, FAISS retrieval, question answering, and summarization.

### 3. Core Layer

- `core/config.py` loads environment-backed settings.
- `core/llm.py` centralizes OpenAI-compatible LLM calls for reuse.
- `core/embeddings.py` centralizes embedding generation for the PDF agent.
- `core/utils.py` provides logging, retries, caching, hashing, filesystem helpers, text cleanup, and shared exceptions.

This separation keeps the search tool, reasoning layer, and API layer independent and easy to extend.

## Design Decisions

- Grounding first: both agents are instructed to answer only from retrieved context.
- Source enforcement: normal answers should be returned with supporting sources.
- Minimal abstraction: no LangChain or heavy orchestration framework is used.
- Async by default: provider calls are asynchronous.
- Resilience: retries are applied around network calls.
- Reuse: LLM, embeddings, configuration, and utilities are shared through core modules.
- PDF persistence: uploaded PDFs and FAISS indexes are stored locally in `data/`.
- Duplicate handling: PDFs are hashed so the same file can reuse an existing `document_id`.

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Update `.env` with valid provider keys.

### Example LLM + Embeddings Setup

Groq for chat and OpenRouter for embeddings:

```env
LLM_API_KEY=your_groq_api_key
LLM_MODEL=openai/gpt-oss-120b
LLM_BASE_URL=https://api.groq.com/openai/v1

EMBEDDING_API_KEY=your_openrouter_api_key
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1

TAVILY_API_KEY=your_tavily_api_key
TAVILY_BASE_URL=https://api.tavily.com

API_BASE_URL=http://127.0.0.1:8000
DOCUMENTS_DIR=data/documents
VECTOR_STORE_DIR=data/vector_store
CHUNK_SIZE_TOKENS=700
CHUNK_OVERLAP_TOKENS=80
RETRIEVAL_TOP_K=4
SUMMARY_MAX_CHUNKS=8
```

## Quick Start

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Update `.env` with your real credentials, then start the backend:

```bash
uvicorn api.main:app --reload
```

If you want to demo the web search agent from a UI, open a second terminal and run:

```bash
source .venv/bin/activate
streamlit run ui/app.py
```

If you want to demo the PDF RAG agent from a UI, run:

```bash
source .venv/bin/activate
streamlit run ui/pdf_app.py
```

Default local URLs:

- FastAPI: `http://127.0.0.1:8000`
- Streamlit: `http://127.0.0.1:8501`

## API Usage

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Web Search Agent

### Endpoint

- `POST /search`

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "latest MacBook specs"}'
```

### Example Response

```json
{
  "answer": "Apple's latest MacBook lineup includes...",
  "sources": [
    "https://www.apple.com/macbook-air/",
    "https://support.apple.com/..."
  ]
}
```

## AI Agent for PDF Summarization and Question Answering

### Workflow

1. Upload a PDF to `/upload`
2. Receive a `document_id`
3. Use that `document_id` with `/ask` or `/summarize`

### Upload a PDF

- Endpoint: `POST /upload`
- Content type: `multipart/form-data`

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@/absolute/path/to/your/document.pdf"
```

Example response:

```json
{
  "document_id": "doc_abc123def456",
  "filename": "document.pdf",
  "page_count": 8,
  "chunk_count": 14,
  "reused_existing": false
}
```

### Ask a Question About a PDF

- Endpoint: `POST /ask`

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_abc123def456",
    "query": "What methodology was used?"
  }'
```

Example response:

```json
{
  "answer": "The document describes a survey-based methodology with ...",
  "sources": [
    {
      "page": 2,
      "chunk_id": "chunk_5"
    }
  ]
}
```

### Summarize a PDF

- Endpoint: `POST /summarize`

```bash
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_abc123def456"
  }'
```

Example response:

```json
{
  "answer": "This document discusses ...",
  "sources": [
    {
      "page": 1,
      "chunk_id": "chunk_1"
    }
  ]
}
```

## UI Usage

- Open the Streamlit app after starting the FastAPI server.
- Enter a natural language question in the chat input.
- Review the grounded answer and clickable source links.
- Use the sidebar clear button to reset the conversation.

Example questions:

- `latest MacBook specs`
- `latest launched cars`
- `top richest people`

## PDF UI Usage

- Open the PDF Streamlit app after starting the FastAPI server:
  `streamlit run ui/pdf_app.py`
- Upload a PDF file.
- Wait for indexing to complete and note the displayed `document_id`.
- Use `Summarize Document` to generate a document summary.
- Ask follow-up questions in the chat input.
- Review the answer and page/chunk-based sources.

## How To Verify It Works

### 1. Verify web search

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"latest MacBook specs"}'
```

Confirm the response includes:

- an answer
- one or more source URLs

### 2. Verify PDF upload and QA

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@/absolute/path/to/your/document.pdf"
```

Take the returned `document_id`, then run:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_abc123def456",
    "query": "What is this document about?"
  }'
```

Confirm the response includes:

- a grounded answer
- `sources` with page and chunk references

### 3. Verify PDF summarization

```bash
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_abc123def456"
  }'
```

Confirm the response includes:

- a concise summary
- `sources` with page and chunk references

### 4. Verify the Streamlit UI

- Open `http://127.0.0.1:8501`
- Ask a web search question in the chat box
- Confirm the response includes a grounded answer and source links

## Working Behavior

The web search agent is designed to enforce grounded output:

- If enough evidence is found, it returns a summarized answer plus sources.
- If the evidence is insufficient, it returns `Insufficient information`.
- A normal answer should not be shown without source URLs.

The PDF RAG agent is designed to enforce document-grounded output:

- PDFs are parsed with PyMuPDF and chunked before indexing.
- Each chunk gets an embedding and is stored in a local FAISS index.
- `/ask` retrieves the top relevant chunks for the query.
- `/summarize` summarizes representative document chunks.
- If the answer is not supported by document context, it returns `Not found in document`.

## Error Handling

- Empty queries return validation errors.
- Invalid or unreadable PDFs return client errors.
- Asking about a missing `document_id` returns not found.
- Missing configuration returns a clear server-side configuration message.
- Tavily, embedding provider, or LLM provider failures return meaningful upstream error messages.
- Unhandled failures are converted into a generic 500 response.

## Extensibility

This codebase is intentionally shaped so a future orchestrator can route between multiple agents:

- keep each agent in its own package
- reuse `core/` for shared clients and utilities
- let the API layer remain thin while orchestration grows separately

That makes this a solid base for expanding into a unified multi-agent backend.
