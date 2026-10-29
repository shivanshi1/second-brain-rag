import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from secondbrain.db.models import Chunk, Document
from secondbrain.retrieval.filters import build_conditions
from secondbrain.schemas import FilterSpec


def _or_terms(query: str) -> str:
    """Build a safe OR expression from plain word tokens for PostgreSQL tsquery."""
    return " | ".join(re.findall(r"\w+", query.lower()))


def search_keyword(
    db: Session, query: str, filters: FilterSpec, k: int = 30
) -> list[tuple[int, float]]:
    """Postgres full-text search. Returns (chunk_id, ts_rank_cd), best first."""
    terms = _or_terms(query)
    if not terms:
        return []
    tsquery = func.to_tsquery("english", terms)
    rank = func.ts_rank_cd(Chunk.tsv, tsquery)
    stmt = (
        select(Chunk.id, rank.label("rank"))
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.tsv.op("@@")(tsquery), *build_conditions(filters))
        .order_by(rank.desc())
        .limit(k)
    )
    return [(row.id, float(row.rank)) for row in db.execute(stmt)]
