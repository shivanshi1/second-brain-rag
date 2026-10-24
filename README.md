# Second Brain RAG

Ask questions across your own Markdown notes, PDFs and saved articles, with cited answers.
Runs fully locally and for free: FastAPI, PostgreSQL + pgvector, Celery/Redis, local
embeddings (sentence-transformers) and a local LLM (Ollama).

> Status: **Phase 0-1 boilerplate.** Ingestion and hybrid `/search` work; `/ask`, reranking,
> strict mode, flashcards and evaluation are the next phases (see the implementation plan).

## Quickstart

```bash
cp .env.example .env
docker compose up --build          # db, redis, api (+ migrations), worker
# in another terminal (Ollama needed from Phase 2 on): https://ollama.com
ollama pull llama3.1:8b
```

API docs: http://localhost:8000/docs

```bash
pip install -e .                                  # installs the `secondbrain` CLI
secondbrain ingest eval/corpus --tags demo
secondbrain search "when should I use a monolith?"
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
- Tag / source-type / date / document filters applied in SQL
- LLM provider interface + Ollama client, strict/open prompt templates
- Tests, ruff, mypy, pre-commit, GitHub Actions CI

## Next steps (maps to the plan's phases)

| Phase | TODO |
|---|---|
| 2 | Implement `/ask` in `api/routes/ask.py`, citation parsing, query logging |
| 3 | Golden set in `eval/`, metrics runner in `evaluation/`, cross-encoder reranker, strict-mode gate |
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
