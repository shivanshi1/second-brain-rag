from sqlalchemy import select
from sqlalchemy.orm import Session

from secondbrain.db.models import Tag


def normalize_tag(name: str) -> str:
    return name.strip().lower().lstrip("#")


def get_or_create_tags(db: Session, names: list[str]) -> list[Tag]:
    cleaned = sorted({normalize_tag(n) for n in names if n and n.strip()})
    if not cleaned:
        return []
    existing = {t.name: t for t in db.scalars(select(Tag).where(Tag.name.in_(cleaned)))}
    tags = []
    for name in cleaned:
        tag = existing.get(name)
        if tag is None:
            tag = Tag(name=name)
            db.add(tag)
        tags.append(tag)
    db.flush()
    return tags
