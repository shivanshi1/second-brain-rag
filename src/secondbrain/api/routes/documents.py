import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from secondbrain.config import settings
from secondbrain.db.models import Document, DocumentTag, Tag
from secondbrain.db.repositories import get_or_create_tags, normalize_tag
from secondbrain.db.session import get_db
from secondbrain.ingestion.loaders import EXTENSION_TO_TYPE
from secondbrain.ingestion.tasks import process_document
from secondbrain.schemas import DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])


def to_out(doc: Document) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        title=doc.title,
        source_type=doc.source_type,
        status=doc.status,
        error=doc.error,
        created_at=doc.created_at,
        document_date=doc.document_date,
        tags=sorted(t.name for t in doc.tags),
    )


@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    file: UploadFile = File(...),
    tags: str = Form(""),
    db: Session = Depends(get_db),
) -> DocumentOut:
    """Upload a .md / .txt / .pdf file. Ingestion runs in a Celery worker."""
    filename = Path(file.filename or "upload").name  # never trust client paths
    ext = Path(filename).suffix.lower()
    source_type = EXTENSION_TO_TYPE.get(ext)
    if source_type is None:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Use: {sorted(EXTENSION_TO_TYPE)}")

    data = file.file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")

    digest = hashlib.sha256(data).hexdigest()
    existing = db.scalar(select(Document).where(Document.content_hash == digest))
    if existing:  # idempotent: same content is never ingested twice
        return to_out(existing)

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir / f"{digest[:16]}{ext}"
    path.write_bytes(data)

    doc = Document(
        title=Path(filename).stem,
        source_type=source_type,
        source_uri=str(path),
        content_hash=digest,
        status="pending",
        meta={"filename": filename},
    )
    doc.tags = get_or_create_tags(db, tags.split(","))
    db.add(doc)
    db.commit()
    process_document.delay(doc.id)
    return to_out(doc)


@router.get("", response_model=list[DocumentOut])
def list_documents(
    tag: str | None = None,
    source_type: str | None = None,
    status_: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[DocumentOut]:
    stmt = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
    if source_type:
        stmt = stmt.where(Document.source_type == source_type)
    if status_:
        stmt = stmt.where(Document.status == status_)
    if tag:
        tagged = (
            select(DocumentTag.c.document_id)
            .join(Tag, Tag.id == DocumentTag.c.tag_id)
            .where(Tag.name == normalize_tag(tag))
        )
        stmt = stmt.where(Document.id.in_(tagged))
    return [to_out(d) for d in db.scalars(stmt)]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, db: Session = Depends(get_db)) -> DocumentOut:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return to_out(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, db: Session = Depends(get_db)) -> None:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    if doc.source_uri and doc.source_type != "web":
        Path(doc.source_uri).unlink(missing_ok=True)
    db.delete(doc)
    db.commit()
