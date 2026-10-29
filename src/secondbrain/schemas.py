from datetime import date, datetime

from pydantic import BaseModel, Field


class FilterSpec(BaseModel):
    """Filters applied inside SQL (pre-filtering) for both vector and keyword search."""

    tags: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)  # md | pdf | text | web
    document_ids: list[int] = Field(default_factory=list)
    date_from: date | None = None
    date_to: date | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: FilterSpec = Field(default_factory=FilterSpec)
    top_k: int = Field(default=8, ge=1, le=50)


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    heading_path: str
    page_number: int | None
    text: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    filters: FilterSpec = Field(default_factory=FilterSpec)
    strict: bool = True  # "answer only from my documents"
    top_k: int = Field(default=3, ge=1, le=20)


class Citation(BaseModel):
    n: int
    document: str
    heading: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    citations_valid: bool
    supported_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Share of sentences in the final model draft that passed the NLI check, "
            "measured before strict-mode filtering. This is an estimate, not proof."
        ),
    )
    latency_ms: int


class DocumentOut(BaseModel):
    id: int
    title: str
    source_type: str
    status: str
    error: str | None
    created_at: datetime
    document_date: datetime | None
    tags: list[str]
