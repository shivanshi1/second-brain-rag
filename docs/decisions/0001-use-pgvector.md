# ADR 0001: Use PostgreSQL + pgvector for everything

**Status:** accepted

**Context:** We need vector search, keyword search, metadata filters and relational data.

**Decision:** One Postgres instance: pgvector (HNSW) for embeddings, `tsvector` + GIN for
full-text search, normal tables for documents/tags/logs.

**Consequences:** Filters run in SQL before ranking; one system to deploy and back up.
At much larger scale a dedicated vector DB (e.g. Qdrant) could be revisited.
