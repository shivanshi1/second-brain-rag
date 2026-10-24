from sqlalchemy import func, select
from sqlalchemy.orm import Session

from secondbrain.db.models import Chunk, Document
from secondbrain.retrieval.filters import build_conditions
from secondbrain.schemas import FilterSpec


def search_keyword(
    db: Session, query: str, filters: FilterSpec, k: int = 30
) -> list[tuple[int, float]]:
    """Postgres full-text search. Returns (chunk_id, ts_rank_cd), best first."""
    tsquery = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(Chunk.tsv, tsquery)
    stmt = (
        select(Chunk.id, rank.label("rank"))
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.tsv.op("@@")(tsquery), *build_conditions(filters))
        .order_by(rank.desc())
        .limit(k)
    )
    return [(row.id, float(row.rank)) for row in db.execute(stmt)]
