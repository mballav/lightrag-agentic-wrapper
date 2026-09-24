# lightrag-agentic-wrapper

Agentic wrapper around [LightRAG](https://github.com/HKUDS/LightRAG). This first slice is the adapter layer: one Python API that can talk to **embedded** LightRAG (SDK in-process) or a **standalone LightRAG server** (REST), selected by `LIGHTRAG_MODE`.

## Layout

- `lightrag_wrapper/config.py` — `LightRAGConfig`, loaded from split env files
- `lightrag_wrapper/adapter/` — `create_adapter()`, embedded SDK backend, HTTP server backend
- `lightrag_wrapper/app.py` — FastAPI surface (`/ingest`, `/ingest/file`, `/ingest/status`, `/query`)
- `docker-compose.yaml` — wrapper app in **embedded** LightRAG mode (dev)
- `docker-compose.prod.yaml` — wrapper + standalone LightRAG server
- `env.example/` — committed templates; copy to `env/` (gitignored)

## Configuration

Env is split by concern and by environment:

```
env.example/
  wrapper/.env.dev
  wrapper/.env.prod
  lightrag-config/.env.dev
  lightrag-config/.env.prod
```

Copy the templates into the gitignored `env/` tree, then fill in secrets:

```bash
# Windows
xcopy /E /I env.example env

# macOS / Linux
cp -R env.example env
```

`APP_ENV=dev` (default) or `APP_ENV=prod` selects which pair `LightRAGConfig.from_env()` loads:

1. `env/lightrag-config/.env.{APP_ENV}` — LLM, embedding, storage, LightRAG server
2. `env/wrapper/.env.{APP_ENV}` — adapter mode, wrapper HTTP, ingest/query defaults

Embedded and server stores are separate on disk. Document graph/KV/vector data now lives in Dockerized **Postgres**, **Milvus**, and **Neo4j**. The wrapper still has its own upload staging directory (`WRAPPER_UPLOAD_DIR`), which is not LightRAG's `INPUT_DIR`.

| Role | Variable | Default |
|---|---|---|
| KV store | `LIGHTRAG_KV_STORAGE` | `PGKVStorage` (Postgres) |
| Document status | `LIGHTRAG_DOC_STATUS_STORAGE` | `PGDocStatusStorage` (Postgres) |
| Vector store | `LIGHTRAG_VECTOR_STORAGE` | `MilvusVectorDBStorage` |
| Graph store | `LIGHTRAG_GRAPH_STORAGE` | `Neo4JStorage` (Bloom enabled) |
| Wrapper API multipart staging | `WRAPPER_UPLOAD_DIR` | `./data/wrapper/uploads` → `/app/data/wrapper/uploads` |

LightRAG still uses `WORKING_DIR` / `INPUT_DIR` for pipeline scratch files. Compose mounts those plus `./data/wrapper/uploads`. Storage services persist in named Docker volumes.

Dev Compose waits for Postgres, Neo4j, and Milvus before starting the wrapper. Incoming `/ingest/file` bodies are written to `WRAPPER_UPLOAD_DIR` first, then ingested. Prod Compose starts the official LightRAG container against the same three backends.

Process environment variables still override file values. Optional paths: `WRAPPER_ENV_FILE`, `LIGHTRAG_ENV_FILE`.

Switching backends (wrapper file):

```env
LIGHTRAG_MODE=embedded
# or
LIGHTRAG_MODE=server
LIGHTRAG_SERVER_URL=http://127.0.0.1:9621
```

## Local run (embedded)

Start the storage containers first so Postgres, Milvus, and Neo4j are on localhost:

```bash
docker compose up -d postgres neo4j milvus milvus-etcd milvus-minio
```

Then run the wrapper:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
xcopy /E /I env.example env
set APP_ENV=dev
python -m lightrag_wrapper
```

The API listens on `APP_HOST`/`APP_PORT` (dev default `127.0.0.1:8000`).

```bash
curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d "{\"text\":\"LightRAG is a graph-based RAG framework.\",\"source\":\"notes.txt\"}"
curl -X POST http://127.0.0.1:8000/ingest/file -F "files=@notes.txt" -F "files=@guide.md"
curl -X POST http://127.0.0.1:8000/ingest/status -H "Content-Type: application/json" -d "{\"track_ids\":[\"upload_20250729_170612_abc123\"]}"
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"query\":\"What is LightRAG?\"}"
```

`/query` first runs an LLM-backed mode router, then calls the adapter with the chosen mode, `only_need_context=true`, `include_references=true`, and a blank `user_prompt`. Routable modes: `local`, `global`, `hybrid`, `naive`, `mix` (fallback).

## Docker Compose

**Dev** (`docker compose up --build`) runs the wrapper in `LIGHTRAG_MODE=embedded` plus Postgres, Milvus, and Neo4j (Bloom).

```bash
xcopy /E /I env.example env
docker compose up --build
```

**Production overlay** switches the wrapper to server mode and starts the official LightRAG container against the same storage services. LightRAG WebUI is bound to localhost only.

```bash
docker compose -f docker-compose.yaml -f docker-compose.prod.yaml up --build
```

Point leftover file-scratch dirs at a host disk with a project `.env` (Compose interpolation):

```env
LIGHTRAG_RAG_STORAGE=/var/lib/lightrag/rag_storage
LIGHTRAG_INPUTS=/var/lib/lightrag/inputs
WRAPPER_UPLOADS=/var/lib/wrapper/uploads
```

- Wrapper API: http://localhost:8000
- LightRAG server / WebUI (prod overlay): http://127.0.0.1:9621
- Neo4j Bloom: http://127.0.0.1:7474/bloom/ (bolt: `neo4j://127.0.0.1:7687`, user `neo4j`)
- Postgres: `127.0.0.1:5432` (user/db `rag`)
- Milvus: `http://127.0.0.1:19530`

Neo4j uses the Enterprise image with Bloom and a 30-day evaluation agreement (`NEO4J_ACCEPT_LICENSE_AGREEMENT=eval`). Drop a Bloom license into `./licenses` (mounted at `/licenses`) if you have one.

A host-side `python -m lightrag_wrapper` uses `localhost` in `env/lightrag-config`; start the storage services first (`docker compose up postgres neo4j milvus milvus-etcd milvus-minio`).

If the LLM or embedding endpoint runs on the host (for example Ollama), use `host.docker.internal` in `LLM_BINDING_HOST` / `EMBEDDING_BINDING_HOST`.

## Using the adapter in code

```python
import asyncio
from lightrag_wrapper import LightRAGConfig, create_adapter

async def main():
    config = await LightRAGConfig.from_env()
    async with create_adapter(config) as rag:
        await rag.ingest_files(["notes.txt", "guide.md"])
        status = await rag.ingest_status(["upload_20250729_170612_abc123"])
        print(status)
        result = await rag.query("What is this about?", mode="mix")
        print(result.response)

asyncio.run(main())
```

## LLM service (LiteLLM)

`lightrag_wrapper.llm.LLMService` routes chat, embeddings, images, audio, rerank, moderation, and Responses-API calls through [LiteLLM](https://docs.litellm.ai/), using the same `LLM_*` / `EMBEDDING_*` env vars as LightRAG. Embedded mode wires this service into the LightRAG SDK as `llm_model_func` / `embedding_func` (server mode still uses the standalone LightRAG process for model calls).

LiteLLM has built-in `num_retries` for completion/responses, but coverage is uneven for other APIs. The service therefore applies a uniform tenacity policy (retries + exponential backoff) to every operation and disables LiteLLM's inner retries to avoid double-retrying.

```python
from lightrag_wrapper import LLMService, LightRAGConfig

config = await LightRAGConfig.from_env()
llm = LLMService(config)

# Chat
resp = await llm.complete(prompt="Summarize LightRAG in one sentence.")

# Embeddings
emb = await llm.embed(["doc one", "doc two"])

# Any LiteLLM operation by name (kwargs forwarded as-is)
out = await llm.invoke("rerank", query="q", documents=["a", "b"])
```

Reliability knobs (in `env/lightrag-config`):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_NUM_RETRIES` | `3` | Total attempts per call |
| `LLM_RETRY_STRATEGY` | `exponential_backoff_retry` | Or `constant_retry` |
| `LLM_RETRY_MIN_WAIT` | `1` | Min backoff seconds |
| `LLM_RETRY_MAX_WAIT` | `10` | Max backoff seconds |
| `LLM_DROP_PARAMS` | `true` | Drop unsupported OpenAI params |
