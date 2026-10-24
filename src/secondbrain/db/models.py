from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from secondbrain.config import settings
from secondbrain.db.base import Base

DocumentTag = Table(
    "document_tags",
    Base.metadata,
    Column("document_id", ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    source_type: Mapped[str] = mapped_column(String(20), index=True)  # md | pdf | text | web
    source_uri: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    document_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )
    tags: Mapped[list[Tag]] = relationship(secondary=DocumentTag)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    heading_path: Mapped[str] = mapped_column(Text, default="")
    page_number: Mapped[int | None] = mapped_column(Integer)
    embedding = mapped_column(Vector(settings.embedding_dim))
    tsv = mapped_column(TSVECTOR, Computed("to_tsvector('english', text)", persisted=True))

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class QueryLog(Base):
    """Every question is logged so failures can be replayed into the eval set."""

    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    filters: Mapped[dict] = mapped_column(JSONB, default=dict)
    answer: Mapped[str | None] = mapped_column(Text)
    chunk_ids: Mapped[list] = mapped_column(JSONB, default=list)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
