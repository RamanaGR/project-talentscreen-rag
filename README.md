# TalentScreen — Production RAG Recruitment Assistant

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://react.dev/)
[![Milvus](https://img.shields.io/badge/Milvus-00A1EA?style=for-the-badge&logo=milvus&logoColor=white)](https://milvus.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python_3.10-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-DD0031?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)

**Enterprise-grade Recruitment Knowledge Assistant powered by Retrieval-Augmented Generation (RAG)**

*Based on the Whitebox Learning TalentScreen RAG Study Guide*

</div>

---

## Overview

**TalentScreen** is an AI-powered recruitment assistant that replaces traditional keyword-based ATS search with deep semantic talent discovery. Recruiters interact in natural language to find, evaluate, and compare candidates across resumes, job descriptions, interview guides, hiring policies, and SOPs — all in one unified platform.

The system is built on a **production RAG architecture** combining:
- **Hybrid Retrieval** (BM25 + Dense Vector Search via Milvus) with Reciprocal Rank Fusion (RRF)
- **BGE Cross-Encoder Reranking** for precision-ranked results
- **Provider-Agnostic LLM Generation** (local Ollama, Claude via Bedrock/Anthropic, OpenAI, GPT4Free)
- **Redis Caching**, **PostgreSQL Audit Logs**, and **Langfuse Observability**

---

## Quick Start (Setup & Run)

### Prerequisites
- **Docker Desktop** installed and running
- At least **6 GB RAM** allocated to Docker (Milvus requires sufficient memory)

### 1. Configure Environment
Clone the repository, create your `.env` file from the template, and configure your keys:

```bash
git clone <repo-url>
cd talent-screen
cp .env.example .env
```

Open `.env` and choose a provider. For local development without any cloud
credentials, use Ollama:
```env
# Local provider (no API key required)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT_SECONDS=120
```

Install and start Ollama on the host before starting the Docker stack:

```bash
ollama serve
ollama pull llama3.2
```

The backend uses Ollama for query understanding, query expansion, resume
metadata extraction, recruiter chat, and evaluation. If Ollama is unavailable,
the application uses its existing deterministic local fallbacks; it does not
silently send requests to a configured remote provider when `LLM_PROVIDER=ollama`.

For a cloud provider, keep the existing options:

```env
# AWS Bedrock Claude
# LLM_PROVIDER=bedrock
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# BEDROCK_MODEL_ID=anthropic.claude-3-7-sonnet-20250219-v1:0

# Anthropic Direct API
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-your_api_key

# OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-your_api_key

# GPT4Free (no API key required, provider availability varies)
# LLM_PROVIDER=g4f
```

When running the backend outside Docker, use
`OLLAMA_BASE_URL=http://localhost:11434`. When the backend runs inside Docker
and Ollama runs on the host, use `http://host.docker.internal:11434`.

### 2. Start all Containers
Launch the backend, React frontend, database, Redis cache, and Milvus vector search engine:

```bash
docker-compose up -d --build
```

Verify that all services started cleanly:
```bash
docker-compose ps
```

| Service | Access URL |
|---------|------------|
| **Recruiter App UI** | [http://localhost:3000](http://localhost:3000) |
| **Backend REST API** | [http://localhost:8000](http://localhost:8000) |
| **Interactive API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) |

---

## Local Ollama troubleshooting

- Check that Ollama is running with `ollama list`.
- Verify the model name matches `OLLAMA_MODEL`.
- Test the local API with `curl http://localhost:11434/api/tags`.
- For Docker, verify the backend can resolve `host.docker.internal`.
- Increase `OLLAMA_TIMEOUT_SECONDS` for larger models or slower hardware.
- Ollama runs the chat model locally, but embeddings and reranking still use
  the configured local SentenceTransformer models.

## How to Test & Ingest Resumes

The repository includes a dedicated `sample_resumes/` folder containing resumes that can be used to test the ingestion pipeline:
- `sample_resumes/SriRoshini_Resume.pdf`
- `sample_resumes/Nadipally_Sriroshini_Resume.pdf`

Follow the curl command examples below to test every component of the RAG pipeline.

### Step 1: Check System Health
Ensure the backend and database components are fully ready:
```bash
curl -i http://localhost:8000/health
```
*Expected output: `{"status":"healthy","service":"talent-screen","version":"2.0.0"}`*

### Step 2: Ingest Resumes (Document Processing)
Upload the provided sample resumes to the backend. The files are processed asynchronously in the background (parsed using IBM Docling, split semantically, embedded, and indexed in Milvus):

```bash
# Upload SriRoshini's Resume
curl -i -X POST -F "file=@sample_resumes/SriRoshini_Resume.pdf" http://localhost:8000/upload

# Upload Nadipally's Resume
curl -i -X POST -F "file=@sample_resumes/Nadipally_Sriroshini_Resume.pdf" http://localhost:8000/upload
```

Check the status of the background indexing tasks:
```bash
curl -s http://localhost:8000/documents
```
*Wait a few seconds until the status changes from `"processing"` to `"completed"`.*

### Step 3: Run Candidate Search (Hybrid RAG Search)
Perform a semantic search query. The backend expands your query with AI and ranks candidates using RRF and the BGE Cross-Encoder reranker:

```bash
curl -i "http://localhost:8000/search?query=Python+developer+with+FastAPI+and+AWS+experience&top_k=2"
```

### Step 4: Test Recruiter Streaming Assistant (Grounded Chat)
Chat with the recruiter assistant about the uploaded resumes. Responses are grounded in the context of the resumes and automatically redact PII (emails/phone numbers):

```bash
curl -i -X POST -H "Content-Type: application/json" \
  -d '{"query":"What are the primary programming languages and skills for SriRoshini?"}' \
  http://localhost:8000/chat
```

### Step 5: Check Dashboard Analytics
View aggregate metrics about uploaded documents, top skills parsed, and query activity:
```bash
curl -i http://localhost:8000/analytics
```

---

## Technical Architecture (RAG Study Guide Alignment)

| Part | Component | Technical Implementation Details |
|------|-----------|----------------------------------|
| **Part 4** | Document Ingestion | Uses **IBM Docling** for high-accuracy PDF parsing (`app/ingestion/parser.py`) and semantic chunking with overlap (`app/ingestion/chunker.py`). |
| **Part 5** | Embedding & Indexing | Maps chunks to vectors using `all-MiniLM-L6-v2` (`app/services/embedding_service.py`) and stores them in **Milvus** with an HNSW index (`app/db/vector_db.py`). |
| **Part 6** | Query Processing | LLM extracts search intent, cleans queries, and expands tech terms (`app/services/llm_service.py`). |
| **Part 7** | Hybrid Retrieval | Combines **BM25 keyword search** and **Milvus vector search** using Reciprocal Rank Fusion (RRF) (`app/services/retrieval_service.py`). |
| **Part 8** | Reranking | Reranks fused search results with `BAAI/bge-reranker-base` to optimize precision (`app/services/retrieval_service.py`). |
| **Part 9** | AI Recruiter Assistant | Builds grounded contexts for the configured LLM provider, instructing it to restrict answers to the context and avoid hallucinations (`app/services/llm_service.py`). |
| **Part 10** | Caching & Data Management | Implements **Redis** caching for query responses (`app/services/cache_service.py`) and **PostgreSQL** for storing chat histories and audit records (`app/models/database.py`). |
| **Part 12** | Security | Automatic regex-based email/phone redaction and guardrails against prompt injection (`app/core/guardrails.py`). |
| **Part 13** | Observability | Traces generation, search queries, and latencies via **Langfuse** (`app/services/observability.py`). |
| **Part 14** | Deployment | Multi-stage production `Dockerfile` configurations for backend and frontend packages. |

---

## Useful Development Commands

```bash
# View live backend logs
docker-compose logs -f backend

# Restart the backend service to apply code updates
docker-compose restart backend

# View logs for the background ingestion worker
docker-compose logs backend | grep -i worker

# Shut down all services and clean database volumes
docker-compose down -v
```

---

*This repository for TalentScreen RAG *
