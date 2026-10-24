"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-01-01
"""

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 384  # keep in sync with EMBEDDING_DIM / the model you use


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tags",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
    )
    op.create_index("ix_tags_name", "tags", ["name"], unique=True)

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("source_type", sa.String(20), nullable=False),
        sa.Column("source_uri", sa.Text),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error", sa.Text),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("document_date", sa.DateTime(timezone=True)),
        sa.Column("meta", postgresql.JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_documents_source_type", "documents", ["source_type"])
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"], unique=True)
    op.create_index("ix_documents_status", "documents", ["status"])

    op.create_table(
        "document_tags",
        sa.Column(
            "document_id",
            sa.Integer,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "tag_id", sa.Integer, sa.ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True
        ),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "document_id",
            sa.Integer,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("heading_path", sa.Text, nullable=False, server_default=""),
        sa.Column("page_number", sa.Integer),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(EMBEDDING_DIM)),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR,
            sa.Computed("to_tsvector('english', text)", persisted=True),
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    op.create_index(
        "ix_chunks_embedding",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "query_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("filters", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("answer", sa.Text),
        sa.Column("chunk_ids", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("model", sa.String(100)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    for table in ("query_logs", "chunks", "document_tags", "documents", "tags"):
        op.drop_table(table)
