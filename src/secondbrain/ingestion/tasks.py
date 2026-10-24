from pathlib import Path

from sqlalchemy import delete

from secondbrain.config import settings
from secondbrain.db.models import Chunk, Document
from secondbrain.db.repositories import get_or_create_tags
from secondbrain.db.session import SessionLocal
from secondbrain.ingestion.celery_app import celery_app
from secondbrain.ingestion.chunking import chunk_sections
from secondbrain.ingestion.embedder import embed_texts
from secondbrain.ingestion.loaders import load_file
from secondbrain.logging import get_logger

log = get_logger()


@celery_app.task(name="ingest.process_document")
def process_document(document_id: int) -> None:
    """parse -> chunk -> embed -> store. Idempotent: old chunks are replaced."""
    with SessionLocal() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            return
        try:
            doc.status = "processing"
            db.commit()

            loaded = load_file(Path(doc.source_uri or ""), doc.source_type)
            chunks = chunk_sections(
                loaded.sections, settings.chunk_max_words, settings.chunk_overlap_words
            )
            if not chunks:
                raise ValueError("Document produced no chunks (empty file?).")
            vectors = embed_texts([c.embed_text for c in chunks])

            db.execute(delete(Chunk).where(Chunk.document_id == doc.id))
            db.add_all(
                Chunk(
                    document_id=doc.id,
                    chunk_index=c.index,
                    text=c.text,
                    heading_path=c.heading_path,
                    page_number=c.page_number,
                    embedding=v,
                )
                for c, v in zip(chunks, vectors, strict=True)
            )
            doc.title = loaded.title or doc.title
            doc.document_date = loaded.document_date or doc.document_date
            doc.meta = {**(doc.meta or {}), **loaded.meta}
            known = {t.name for t in doc.tags}
            doc.tags.extend(t for t in get_or_create_tags(db, loaded.tags) if t.name not in known)
            doc.status, doc.error = "ready", None
            db.commit()
            log.info("ingest.done", document_id=doc.id, chunks=len(chunks))
        except Exception as exc:
            db.rollback()
            doc = db.get(Document, document_id)
            if doc is not None:
                doc.status, doc.error = "failed", str(exc)[:1000]
                db.commit()
            log.error("ingest.failed", document_id=document_id, error=str(exc))
