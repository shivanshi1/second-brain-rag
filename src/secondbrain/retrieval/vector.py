from sqlalchemy import select
from sqlalchemy.orm import Session

from secondbrain.db.models import Chunk, Document
from secondbrain.retrieval.filters import build_conditions
from secondbrain.schemas import FilterSpec


def search_vector(
    db: Session, query_embedding: list[float], filters: FilterSpec, k: int = 30
) -> list[tuple[int, float]]:
    """Return (chunk_id, cosine_distance), nearest first."""
    distance = Chunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Chunk.id, distance.label("distance"))
        .join(Document, Document.id == Chunk.document_id)
        .where(*build_conditions(filters))
        .order_by(distance)
        .limit(k)
    )
    return [(row.id, float(row.distance)) for row in db.execute(stmt)]
