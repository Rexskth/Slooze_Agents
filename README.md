# Unified AI Agent Platform

This project is a production-style AI backend built in Python with FastAPI. It combines:

- a `Web Search Agent` for real-time internet queries
- an `AI Agent for PDF Summarization and Question Answering`
- a thin `Orchestrator Layer` that routes queries to the correct specialist agent

The system is designed to be modular, grounded, and easy to extend.

## What This Platform Can Do

### 1. Web Search Agent

- accepts a natural language query
- retrieves live web results using Tavily
- uses an LLM to produce a grounded answer
- returns answer plus source URLs

### 2. PDF RAG Agent

- accepts a PDF upload
- extracts text using PyMuPDF
- chunks text and stores embeddings in FAISS
- answers questions grounded in the uploaded document
- summarizes the uploaded document
- returns answer plus page/chunk sources

### 3. Orchestrator

- exposes a unified `/query` endpoint
- decides whether a query should go to:
  - the web agent
  - the PDF agent
- returns a unified response:

```json
{
  "route": "web",
  "answer": "...",
  "sources": [...]
}
```

## Tech Stack

- Python
- FastAPI
- Streamlit
- Tavily
- FAISS
- PyMuPDF
- OpenAI-compatible chat API
- OpenRouter embeddings
- python-dotenv

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
  orchestrator/
    router.py
    agent_controller.py

core/
  embeddings.py
  llm.py
  config.py
  utils.py

api/
  main.py

ui/
  app.py
  pdf_app.py
  platform_app.py

data/
  documents/
  vector_store/

requirements.txt
.env.example
README.md
```

## Architecture Overview

### API Layer

- [main.py](/Users/mac/Documents/Me/Slooze_Agents/api/main.py) exposes:
  - `/query`
  - `/search`
  - `/upload`
  - `/ask`
  - `/summarize`

### Specialist Agents

- [agent.py](/Users/mac/Documents/Me/Slooze_Agents/agent/web_search_agent/agent.py) handles grounded web search
- [qa.py](/Users/mac/Documents/Me/Slooze_Agents/agent/pdf_rag_agent/qa.py) handles PDF question answering and summarization

### Orchestrator Layer

- [router.py](/Users/mac/Documents/Me/Slooze_Agents/agent/orchestrator/router.py) makes routing decisions
- [agent_controller.py](/Users/mac/Documents/Me/Slooze_Agents/agent/orchestrator/agent_controller.py) delegates work to specialist agents

### Shared Core

- [config.py](/Users/mac/Documents/Me/Slooze_Agents/core/config.py)
- [llm.py](/Users/mac/Documents/Me/Slooze_Agents/core/llm.py)
- [embeddings.py](/Users/mac/Documents/Me/Slooze_Agents/core/embeddings.py)
- [utils.py](/Users/mac/Documents/Me/Slooze_Agents/core/utils.py)

## Before You Run

You need API keys for:

- Tavily
- chat LLM provider
- embedding provider

This project is currently configured to work well with:

- Groq for chat completion
- OpenRouter for embeddings

You can also switch the chat model/base URL if you use another OpenAI-compatible provider.

## Environment Setup

Create your local `.env` file:

```bash
cp .env.example .env
```

Then update it with your real values.

### Recommended `.env` Setup

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

MAX_SEARCH_RESULTS=5
CONTEXT_CHAR_LIMIT=6000
LLM_TEMPERATURE=0.2
CACHE_TTL_SECONDS=300
LOG_LEVEL=INFO
```

## Install and Run

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the backend

```bash
uvicorn api.main:app --reload
```

Backend default URL:

- `http://127.0.0.1:8000`

## Streamlit UIs

This repo includes three Streamlit apps.

### 1. Web Search UI

Use this to test only the web search agent.

```bash
streamlit run ui/app.py
```

### 2. PDF Agent UI

Use this to test only the PDF upload / ask / summarize flow.

```bash
streamlit run ui/pdf_app.py
```

### 3. Unified Platform UI

Use this to test the full orchestrated system from one place.

```bash
streamlit run ui/platform_app.py
```

This is the recommended UI for evaluators.

## Best Way For Evaluators To Test

### Step 1. Start backend

```bash
uvicorn api.main:app --reload
```

### Step 2. Start unified platform UI

```bash
streamlit run ui/platform_app.py
```

### Step 3. Open the UI in browser

Usually:

- `http://127.0.0.1:8501`

### Step 4. Test web routing

In the unified UI, ask:

- `latest MacBook specs`
- `latest launched cars`
- `top richest people`

Expected:

- route should behave like `web`
- answer should include web sources

### Step 5. Test PDF routing

In the unified UI:

1. upload a PDF
2. enable document context
3. ask:
   - `What methodology was used in this document?`
   - `Summarize this document`

Expected:

- route should behave like `pdf`
- answer should include page/chunk sources

## API Endpoints

### Health

```bash
GET /health
```

### Unified Query

```bash
POST /query
```

Example web request:

```json
{
  "query": "What are the latest MacBook specs?"
}
```

Example PDF request:

```json
{
  "query": "What methodology was used in this document?",
  "document_id": "doc_abc123def456"
}
```

Unified response format:

```json
{
  "route": "web",
  "answer": "...",
  "sources": [...]
}
```

### Direct Specialist Endpoints

These remain available for debugging and isolated testing:

- `POST /search`
- `POST /upload`
- `POST /ask`
- `POST /summarize`

## Important Behavior Notes

### Orchestrator

- `/query` uses rule-based routing
- if a PDF-oriented query is detected, `document_id` is required
- if `document_id` is missing for a PDF route, the API returns a clear error

### Web Agent

- answers are grounded in Tavily search results
- source URLs are returned

### PDF Agent

- uploaded PDFs are stored locally
- extracted text is chunked and embedded
- vectors are stored in FAISS
- answers are grounded in retrieved chunks
- sources include page number and chunk id

## Local Data

The system stores runtime artifacts in:

- `data/documents/`
- `data/vector_store/`

These are local runtime files and are not meant to be committed as actual document data.

## Design Choices

- specialist agents remain independent
- orchestration is added on top, not mixed into agent logic
- no LangChain or hidden framework abstractions
- provider configuration is environment-driven
- direct endpoints are preserved for debugging

## Troubleshooting

### Backend starts but answers fail

Check:

- `LLM_API_KEY`
- `EMBEDDING_API_KEY`
- `TAVILY_API_KEY`
- base URLs in `.env`

### PDF upload works but questions fail

Check:

- embedding provider settings
- that the PDF actually contains extractable text

### Unified UI does not route to PDF

Check:

- a PDF was uploaded
- document context is enabled in the UI
- the query is document-oriented

### Streamlit cannot reach backend

Check:

- backend is running
- `API_BASE_URL` matches the backend URL

## Recommended Demo Flow

For an evaluator, the simplest demo flow is:

1. clone the repo
2. create `.env`
3. install dependencies
4. run backend
5. run `streamlit run ui/platform_app.py`
6. test one web query
7. upload one PDF
8. test one PDF question
9. test PDF summarization

That path demonstrates the full platform with minimal setup friction.
