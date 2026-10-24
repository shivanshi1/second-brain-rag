from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from secondbrain.db.session import get_db
from secondbrain.retrieval.pipeline import hybrid_search
from secondbrain.schemas import SearchHit, SearchRequest

router = APIRouter(tags=["search"])


@router.post("/search", response_model=list[SearchHit])
def search(req: SearchRequest, db: Session = Depends(get_db)) -> list[SearchHit]:
    """Retrieval only (no LLM): hybrid vector + keyword search with filters."""
    return hybrid_search(db, req.query, req.filters, req.top_k)
