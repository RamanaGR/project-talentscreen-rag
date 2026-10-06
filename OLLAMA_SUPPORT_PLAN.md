# TalentScreen Local Ollama Support Plan

## Status

The plan was confirmed on 2026-10-06 and implemented in the application.

## Repository findings

- The project is a FastAPI backend with a React frontend, Milvus vector search,
  PostgreSQL, Redis, local SentenceTransformer embeddings, and a background
  document-ingestion worker.
- The main LLM integration is
  [`backend/app/services/llm_service.py`](./backend/app/services/llm_service.py).
  It currently contains Bedrock, Anthropic, OpenAI, G4F, and mock behavior,
  including separate non-streaming and streaming paths.
- [`backend/app/core/config.py`](./backend/app/core/config.py) exposes
  `LLM_PROVIDER`, but the current client initialization does not consistently
  use it to select one provider.
- [`backend/app/ingestion/metadata.py`](./backend/app/ingestion/metadata.py)
  calls Anthropic directly and falls back to regex extraction. It does not use
  the central LLM service.
- [`backend/app/services/evaluation_service.py`](./backend/app/services/evaluation_service.py)
  also calls Anthropic directly and uses a heuristic response when unavailable.
- Query understanding, query optimization, query expansion, and recruiter chat
  already flow through the central LLM service.
- The React chat screen consumes the backend's plain-text streaming response, so
  Ollama support can remain backend-focused without changing the frontend API.
- Both the root and backend dependency files exist. The backend dependency file
  is used by [`backend/Dockerfile`](./backend/Dockerfile), so it is the primary
  dependency manifest for the API container.
- The current README primarily documents Bedrock/Anthropic setup and does not
  explain a no-key local provider workflow.

## Proposed behavior

### Provider selection

Add `ollama` as an explicit `LLM_PROVIDER` option.

When `LLM_PROVIDER=ollama`:

1. Connect to the configured local Ollama HTTP endpoint.
2. Use the configured chat model for query processing and recruiter chat.
3. Use Ollama streaming for `/chat` when available.
4. Fail clearly in logs and use the existing safe fallback behavior when the
   Ollama server or model is unavailable.

Existing providers remain supported:

- `bedrock`: keep AWS credentials and Bedrock model configuration.
- `anthropic`: keep direct Anthropic API support.
- `openai`: keep OpenAI API support.
- `g4f`: keep the existing no-key fallback option.
- mock/heuristic fallbacks: retain them for offline failure handling.

The provider selection should be explicit and deterministic rather than
silently trying unrelated configured credentials first. Any fallback chain
should be documented and logged so a local setup never unexpectedly sends
prompts to a remote provider.

### Ollama configuration

Add settings and `.env.example` entries similar to:

- `OLLAMA_BASE_URL` (default suitable for local development)
- `OLLAMA_MODEL` (a documented example such as `llama3.2`)
- optional request timeout configuration

The final names should be defined once in `Settings` and reused everywhere.
The README should explain:

```bash
ollama serve
ollama pull <model>
```

and the corresponding `.env` values. It should explicitly state that Ollama
does not require an API key and that the model must already be available
locally.

### Central service reuse

Use the central LLM abstraction for all text-generation features:

- query understanding
- query optimization
- query expansion
- grounded recruiter chat
- document metadata extraction
- optional RAG evaluation

This prevents Ollama support from being implemented once in the main chat
service but omitted from ingestion or evaluation. If a shared abstraction is
introduced, preserve the existing method behavior and return shapes.

### Streaming

Add an Ollama streaming implementation that converts Ollama response chunks
into the existing internal text-chunk shape consumed by the FastAPI streaming
endpoint. Handle:

- successful token streaming
- malformed or incomplete response chunks
- connection errors
- missing model errors
- clean stream termination

The endpoint should continue masking PII chunk-by-chunk and saving the final
conversation transcript as it does today.

## Planned file changes

### Backend implementation

- [`backend/app/core/config.py`](./backend/app/core/config.py)
  - Add typed Ollama settings and provider documentation.
  - Keep existing API keys and Bedrock settings unchanged.
- [`backend/app/services/llm_service.py`](./backend/app/services/llm_service.py)
  - Add the Ollama client/transport.
  - Make provider selection explicit.
  - Add non-streaming and streaming Ollama paths.
  - Preserve existing provider behavior and safe fallbacks.
  - Add short class-level and function-level comments/docstrings where the
    provider flow or fallback order is not self-evident.
- [`backend/app/ingestion/metadata.py`](./backend/app/ingestion/metadata.py)
  - Route LLM metadata extraction through the shared service, or introduce a
    small shared interface that supports the configured provider.
  - Preserve regex extraction as a local/offline fallback.
  - Add focused comments/docstrings for extraction and fallback responsibilities.
- [`backend/app/services/evaluation_service.py`](./backend/app/services/evaluation_service.py)
  - Use the configured provider through the shared abstraction.
  - Preserve heuristic evaluation when no provider is available.
- [`backend/requirements.txt`](./backend/requirements.txt)
  - Add only the dependency needed by the selected Ollama integration, unless
    the existing LangChain OpenAI-compatible client can support Ollama without
    another package.
- [`requirements.txt`](./requirements.txt)
  - Keep aligned with backend dependency changes if this manifest is intended
    to remain installable for local development.

### Configuration and deployment

- [`.env.example`](./.env.example)
  - Add `ollama` to provider choices and document local model/base URL values.
  - Keep Bedrock, Anthropic, and OpenAI variables available.
- [`docker-compose.yml`](./docker-compose.yml)
  - Document the host-to-container networking choice for Ollama.
  - If desired, add an optional Ollama service profile rather than forcing an
    Ollama container on users who run Ollama natively.
  - Ensure the backend can reach a host-installed Ollama from Docker on the
    supported platform.
- [`README.md`](./README.md)
  - Add a no-key Ollama quick-start path.
  - Explain local model download, resource requirements, model selection,
    Docker networking, troubleshooting, and how to switch back to remote
    providers.
  - Correct provider descriptions so they match the actual implementation.

### Comments and documentation style

Add crisp, useful comments/docstrings at class and function boundaries,
especially for:

- provider selection and initialization
- Ollama request/response translation
- streaming chunk handling
- shared metadata/evaluation LLM calls
- fallback and error behavior

Avoid comments that merely restate a line of code. Keep existing project
terminology and document the reason for non-obvious branches.

## Validation plan

After approval and implementation:

1. Run Python syntax and import checks for changed backend modules.
2. Run the backend's available tests, adding focused tests if no suitable tests
   exist.
3. Test provider configuration without any API keys:
   - `LLM_PROVIDER=ollama` initializes without Bedrock/Anthropic/OpenAI keys.
   - a configured Ollama model can answer a non-streaming request.
   - `/chat` streams text and persists the transcript.
4. Test failure paths:
   - Ollama is unreachable.
   - the configured model is missing.
   - Ollama returns an invalid response.
   - the service falls back according to the documented policy without
     silently using a remote provider.
5. Run the frontend build because the streaming contract must remain compatible.
6. Verify existing provider configuration remains accepted and documented.

## Decisions to confirm before implementation

The proposed default is to keep Ollama as a backend connection to an Ollama
instance running on the developer machine, while leaving the existing
Docker-based infrastructure unchanged.

Please confirm:

1. Should Ollama be the selected default in `.env.example`, or should the
   example continue to default to a remote provider while showing Ollama as
   the recommended no-key option?
2. Should Ollama be run natively on the host, or should the Compose file also
   provide an optional Ollama container/profile?
3. When `LLM_PROVIDER=ollama` is unavailable, should the service use only local
   regex/mock/heuristic fallbacks, or may it fall back to configured remote
   providers?
4. Which default Ollama model should the documentation use? A practical
   starting point is `llama3.2`, but the setting will remain configurable.

Implementation will begin only after these choices and the overall plan are
confirmed.
