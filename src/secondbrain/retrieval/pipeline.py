from sqlalchemy import select
from sqlalchemy.orm import Session

from secondbrain.db.models import Chunk, Document
from secondbrain.ingestion.embedder import embed_query
from secondbrain.retrieval.fusion import reciprocal_rank_fusion
from secondbrain.retrieval.keyword import search_keyword
from secondbrain.retrieval.vector import search_vector
from secondbrain.schemas import FilterSpec, SearchHit


def hybrid_search(
    db: Session, query: str, filters: FilterSpec, top_k: int = 8, candidates: int = 30
) -> list[SearchHit]:
    """Vector + keyword search fused with RRF.

    TODO (Phase 3): rerank the fused candidates with a cross-encoder and
    expose the reranker score for the strict-mode relevance gate.
    """
    vector_hits = search_vector(db, embed_query(query), filters, candidates)
    keyword_hits = search_keyword(db, query, filters, candidates)
    fused = reciprocal_rank_fusion(
        [[cid for cid, _ in vector_hits], [cid for cid, _ in keyword_hits]]
    )[:top_k]
    if not fused:
        return []

    rows = db.execute(
        select(Chunk, Document.title)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_([cid for cid, _ in fused]))
    ).all()
    by_id = {chunk.id: (chunk, title) for chunk, title in rows}
    return [
        SearchHit(
            chunk_id=cid,
            document_id=by_id[cid][0].document_id,
            document_title=by_id[cid][1],
            heading_path=by_id[cid][0].heading_path,
            page_number=by_id[cid][0].page_number,
            text=by_id[cid][0].text,
            score=score,
        )
        for cid, score in fused
        if cid in by_id
    ]
