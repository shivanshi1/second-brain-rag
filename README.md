# Second Brain RAG

Ask questions across your own Markdown notes, PDFs and saved articles, with cited answers.
Runs fully locally and for free: FastAPI, PostgreSQL + pgvector, Celery/Redis, local
embeddings (sentence-transformers) and a local LLM (Ollama).

> Status: **Phase 2 plus an initial Phase 3 sentence-support prototype, pending verification.**
> `/ask` returns separate citation validity and NLI-estimated sentence support fields. Strict
> mode retries once, then drops unsupported sentences or refuses; open mode labels unsupported
> text. This local NLI estimate can be wrong and must be evaluated and tuned. Retrieval reranking,
> a relevance gate, and the evaluation harness remain to be implemented.

## Quickstart

```bash
cp .env.example .env
docker compose up --build          # db, redis, api (+ migrations), worker
# in another terminal (Ollama needed from Phase 2 on): https://ollama.com
ollama pull llama3.1:8b
# The sentence-support model downloads from Hugging Face on the first /ask request.
```

API docs: http://localhost:8000/docs

```bash
pip install -e .                                  # installs the `secondbrain` CLI
secondbrain ingest eval/corpus --tags demo
secondbrain search "when should I use a monolith?"
secondbrain ask "When should I use a monolith?"
```

## Local development (without Docker for the app)

```bash
python -m venv .venv && source .venv/bin/activate
make install        # pip install -e ".[ml,dev]" + pre-commit
make up             # only db + redis in Docker
make migrate
make api            # terminal 1
make worker         # terminal 2
make test lint typecheck
```

## What is already implemented

- Upload `.md` / `.txt` / `.pdf` via API or CLI, content-hash deduplication
- Async ingestion (Celery): parse -> heading-aware chunk -> embed -> store
- Markdown frontmatter tags/title/date; PDF page numbers
- Hybrid search: pgvector cosine + Postgres full-text, fused with RRF
- PostgreSQL keyword search combines sanitized query terms with OR for broader recall
- Tag / source-type / date / document filters applied in SQL
- `/ask` with hybrid retrieval, Ollama generation, source citations, and query logging
- NLI-based per-sentence support estimates, separate `citations_valid` and `supported_ratio`
  fields, strict-mode retry/filter/refusal, and open-mode labels for unsupported sentences
- Duplicate-title cleanup in citation headings and three-chunk default for Q&A retrieval
- `secondbrain ask` CLI command with strict/open mode
- LLM provider interface + Ollama client, strict/open prompt templates
- Tests, ruff, mypy, pre-commit, GitHub Actions CI

## Next steps (maps to the plan's phases)

| Phase | TODO |
|---|---|
| 2 | Run the updated tests and verify the `/ask` flow and CLI command with Ollama |
| 3 | Evaluate and tune the NLI support threshold; add a golden-set runner and metrics; implement retrieval reranker and relevance gate |
| 4 | URL ingestion (trafilatura), `/sources`, `/summarize`, flashcards + Anki export |
| 5 | Streamlit UI, Groq/Gemini fallback client, streaming, README polish and demo GIF |

## Layout

```
src/secondbrain/
  api/          FastAPI app and routes
  db/           SQLAlchemy models, session, repositories
  ingestion/    loaders, chunking, embedder, Celery tasks
  retrieval/    filters, vector, keyword, RRF fusion, pipeline
  generation/   LLM clients and prompts
  evaluation/   (Phase 3)
alembic/        migrations
tests/          unit tests (no DB or models needed)
```
