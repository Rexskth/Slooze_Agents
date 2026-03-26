# AI Web Search Agent

A production-style AI web search agent built with FastAPI, Tavily, and an OpenAI-compatible LLM API. The service accepts a natural language query, retrieves real-time web results, constrains the model to those results, and returns a grounded answer with source URLs.

## Project Structure

```text
agent/
  web_search_agent/
    agent.py
    tools.py
    prompts.py
  pdf_rag_agent/

core/
  llm.py
  config.py
  utils.py

api/
  main.py

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

- `api/main.py` exposes the FastAPI app and `/search` endpoint.
- Request validation and HTTP error mapping stay at the edge of the system.

### 2. Agent Layer

- `agent/web_search_agent/tools.py` contains the Tavily integration only.
- `agent/web_search_agent/prompts.py` stores prompt templates separately from code.
- `agent/web_search_agent/agent.py` orchestrates search, context assembly, LLM invocation, caching, and response shaping.

### 3. Core Layer

- `core/config.py` loads environment-backed settings.
- `core/llm.py` centralizes OpenAI-compatible LLM calls for reuse.
- `core/utils.py` provides logging, retries, caching, text cleanup, and shared exceptions.

This separation keeps the search tool, reasoning layer, and API layer independent and easy to extend.

## Design Decisions

- Grounding first: the model is instructed to use only Tavily-provided context.
- Minimal abstraction: no heavy orchestration framework is used.
- Async by default: Tavily and LLM provider calls are asynchronous.
- Resilience: retries are applied around network calls.
- Reuse: LLM integration and configuration are shared through core modules.
- Performance: repeated queries are cached in memory with TTL.

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

Then update `.env` with valid `LLM_API_KEY` and `TAVILY_API_KEY` values.

Example Groq configuration:

```env
LLM_API_KEY=your_groq_api_key
LLM_MODEL=openai/gpt-oss-120b
LLM_BASE_URL=https://api.groq.com/openai/v1
TAVILY_API_KEY=your_tavily_api_key
```

Example OpenAI configuration:

```env
LLM_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
TAVILY_API_KEY=your_tavily_api_key
```

## Running the API

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Usage

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

### Search Request

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

## Error Handling

- Empty queries return validation errors.
- Missing configuration returns a clear server-side configuration message.
- Tavily or LLM provider failures return meaningful upstream error messages.
- Unhandled failures are converted into a generic 500 response.

## Extensibility

This codebase is intentionally shaped so a future orchestrator can route between multiple agents:

- keep each agent in its own package
- reuse `core/` for shared clients and utilities
- let the API layer remain thin while orchestration grows separately

That makes this a solid base for expanding into a unified multi-agent backend.
