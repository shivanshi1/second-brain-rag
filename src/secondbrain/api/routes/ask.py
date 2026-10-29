import re
from time import perf_counter

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from secondbrain.config import settings
from secondbrain.db.models import QueryLog
from secondbrain.db.session import get_db
from secondbrain.generation.grounding import (
    AnswerAssessment,
    SupportVerificationError,
    verify_answer_support,
)
from secondbrain.generation.llm import get_llm
from secondbrain.generation.prompts import STRICT_REFUSAL, build_messages, display_heading
from secondbrain.retrieval.pipeline import hybrid_search
from secondbrain.schemas import AskRequest, AskResponse, Citation, SearchHit

router = APIRouter(tags=["ask"])
CITATION_RE = re.compile(r"\[(\d+)\]")


def _record_query(
    db: Session,
    req: AskRequest,
    hits: list[SearchHit],
    answer: str | None,
    latency_ms: int,
) -> None:
    """Persist the retrieval and answer details for later debugging/evaluation."""
    db.add(
        QueryLog(
            question=req.question,
            filters=req.filters.model_dump(mode="json"),
            answer=answer,
            chunk_ids=[hit.chunk_id for hit in hits],
            latency_ms=latency_ms,
            model=(
                settings.ollama_model
                if settings.llm_provider == "ollama"
                else settings.llm_provider
            ),
        )
    )
    db.commit()


def _citations_from_answer(answer: str, hits: list[SearchHit]) -> tuple[list[Citation], bool]:
    """Map valid [n] markers to retrieved hits and validate marker references."""
    numbers = [int(number) for number in CITATION_RE.findall(answer)]
    citations_valid = bool(numbers) and all(1 <= number <= len(hits) for number in numbers)
    citations: list[Citation] = []
    seen: set[int] = set()

    for number in numbers:
        if number < 1 or number > len(hits) or number in seen:
            continue
        seen.add(number)
        hit = hits[number - 1]
        citations.append(
            Citation(
                n=number,
                document=hit.document_title,
                heading=display_heading(hit.document_title, hit.heading_path, hit.page_number),
                snippet=hit.text[:300],
            )
        )

    return citations, citations_valid


def _remove_citation_markers(text: str) -> str:
    clean = CITATION_RE.sub("", text)
    return re.sub(r"\s+([.!?])", r"\1", clean).strip()


def _strict_answer(answer: str, assessment: AnswerAssessment) -> str:
    """Drop sentences that the support verifier did not validate."""
    kept = [sentence.text for sentence in assessment.sentences if sentence.supported]
    return " ".join(kept) if kept else STRICT_REFUSAL


def _open_answer(assessment: AnswerAssessment) -> str:
    """Label unsupported sentences and remove citations that would overstate support."""
    output = []
    for sentence in assessment.sentences:
        if sentence.supported:
            output.append(sentence.text)
        else:
            clean = _remove_citation_markers(sentence.text)
            if clean:
                output.append(f"Not from your notes: {clean}")
    return " ".join(output)


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)) -> AskResponse:
    """Retrieve supporting passages and ask the configured LLM to answer."""
    started = perf_counter()
    hits = hybrid_search(db, req.question, req.filters, req.top_k)

    # Do not ask the model to invent an answer in strict mode when retrieval found
    # no evidence at all. Score-based refusal thresholds belong to the next phase.
    if req.strict and not hits:
        latency_ms = round((perf_counter() - started) * 1000)
        _record_query(db, req, hits, STRICT_REFUSAL, latency_ms)
        return AskResponse(
            answer=STRICT_REFUSAL,
            citations=[],
            citations_valid=False,
            supported_ratio=0.0,
            latency_ms=latency_ms,
        )

    try:
        llm = get_llm()
        answer = llm.generate(build_messages(req.question, hits, req.strict), temperature=0.0)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        latency_ms = round((perf_counter() - started) * 1000)
        _record_query(db, req, hits, None, latency_ms)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The configured language model could not generate an answer.",
        ) from exc

    try:
        assessment = verify_answer_support(answer, hits)
        if req.strict and (assessment.supported_ratio < 1.0 or not assessment.citations_valid):
            unsupported = [
                sentence.text for sentence in assessment.sentences if not sentence.supported
            ]
            answer = llm.generate(
                build_messages(
                    req.question,
                    hits,
                    req.strict,
                    previous_answer=answer,
                    unsupported_sentences=unsupported,
                ),
                temperature=0.0,
            )
            assessment = verify_answer_support(answer, hits)
    except SupportVerificationError as exc:
        latency_ms = round((perf_counter() - started) * 1000)
        _record_query(db, req, hits, None, latency_ms)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The sentence support model is unavailable; no answer was returned.",
        ) from exc
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        latency_ms = round((perf_counter() - started) * 1000)
        _record_query(db, req, hits, None, latency_ms)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The configured language model could not generate an answer.",
        ) from exc

    answer = _strict_answer(answer, assessment) if req.strict else _open_answer(assessment)
    citations, citations_valid = _citations_from_answer(answer, hits)
    latency_ms = round((perf_counter() - started) * 1000)
    _record_query(db, req, hits, answer, latency_ms)
    return AskResponse(
        answer=answer,
        citations=citations,
        citations_valid=citations_valid,
        supported_ratio=assessment.supported_ratio,
        latency_ms=latency_ms,
    )
