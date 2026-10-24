from datetime import UTC, datetime, time, timedelta

from sqlalchemy import ColumnElement, func, select

from secondbrain.db.models import Document, DocumentTag, Tag
from secondbrain.schemas import FilterSpec


def build_conditions(f: FilterSpec) -> list[ColumnElement[bool]]:
    """Translate a FilterSpec into SQL conditions on Document (pre-filtering)."""
    conds: list[ColumnElement[bool]] = [Document.status == "ready"]
    effective_date = func.coalesce(Document.document_date, Document.created_at)
    if f.source_types:
        conds.append(Document.source_type.in_(f.source_types))
    if f.document_ids:
        conds.append(Document.id.in_(f.document_ids))
    if f.date_from:
        conds.append(effective_date >= datetime.combine(f.date_from, time.min, tzinfo=UTC))
    if f.date_to:
        end = datetime.combine(f.date_to + timedelta(days=1), time.min, tzinfo=UTC)
        conds.append(effective_date < end)  # inclusive of the end day
    if f.tags:
        tagged = (
            select(DocumentTag.c.document_id)
            .join(Tag, Tag.id == DocumentTag.c.tag_id)
            .where(Tag.name.in_([t.strip().lower().lstrip("#") for t in f.tags]))
        )
        conds.append(Document.id.in_(tagged))
    return conds
