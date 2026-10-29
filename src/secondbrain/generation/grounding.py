"""Sentence-level citation and entailment checks for generated answers."""

import re
from dataclasses import dataclass
from functools import lru_cache

from secondbrain.config import settings
from secondbrain.schemas import SearchHit

CITATION_RE = re.compile(r"\[(\d+)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class SupportVerificationError(RuntimeError):
    """Raised when the local NLI model cannot be loaded or used."""


@dataclass(frozen=True)
class SentenceAssessment:
    text: str
    supported: bool
    cited_blocks: tuple[int, ...]
    entailment_score: float | None


@dataclass(frozen=True)
class AnswerAssessment:
    sentences: tuple[SentenceAssessment, ...]
    citations_valid: bool
    supported_ratio: float


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load the NLI model once, on first use, rather than during API startup."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.nli_model)


def _predict_entailment_scores(pairs: list[tuple[str, str]]) -> list[float]:
    """Return entailment probabilities for (source passage, answer sentence) pairs."""
    if not pairs:
        return []
    try:
        model = _get_cross_encoder()
        id_to_label = model.model.config.id2label
        entailment_id = next(
            int(index) for index, label in id_to_label.items() if label.casefold() == "entailment"
        )
        probabilities = model.predict(pairs, apply_softmax=True, show_progress_bar=False)
        return [float(row[entailment_id]) for row in probabilities]
    except Exception as exc:
        raise SupportVerificationError("Could not run the configured NLI model.") from exc


def verify_answer_support(
    answer: str,
    hits: list[SearchHit],
    threshold: float | None = None,
) -> AnswerAssessment:
    """Check citation IDs and score each sentence against its cited passages.

    The entailment score is a model estimate, not a proof of factual support. A
    sentence is supported when at least one cited passage reaches ``threshold``.
    """
    cutoff = settings.nli_entailment_threshold if threshold is None else threshold
    blocks = {index: hit.text for index, hit in enumerate(hits, start=1)}
    pieces = [piece.strip() for piece in SENTENCE_RE.split(answer.strip()) if piece.strip()]
    citation_numbers = [int(number) for number in CITATION_RE.findall(answer)]
    citations_valid = bool(citation_numbers) and all(
        number in blocks for number in citation_numbers
    )

    sentence_data: list[tuple[str, tuple[int, ...]]] = []
    pairs: list[tuple[str, str]] = []
    pair_sentence_indexes: list[int] = []
    for piece in pieces:
        cited = tuple(dict.fromkeys(int(number) for number in CITATION_RE.findall(piece)))
        sentence_index = len(sentence_data)
        sentence_data.append((piece, cited))
        clean = re.sub(r"\s+([.!?])", r"\1", CITATION_RE.sub("", piece)).strip()
        for block_number in cited if clean else ():
            if block_number in blocks:
                pairs.append((blocks[block_number], clean))
                pair_sentence_indexes.append(sentence_index)

    scores = _predict_entailment_scores(pairs) if pairs else []
    best_scores: list[float | None] = [None] * len(sentence_data)
    for sentence_index, score in zip(pair_sentence_indexes, scores, strict=True):
        current = best_scores[sentence_index]
        if current is None or score > current:
            best_scores[sentence_index] = score

    assessments = tuple(
        SentenceAssessment(
            text=piece,
            supported=(
                bool(cited)
                and bool(CITATION_RE.sub("", piece).strip())
                and all(number in blocks for number in cited)
                and (sentence_score := best_scores[index]) is not None
                and sentence_score >= cutoff
            ),
            cited_blocks=cited,
            entailment_score=best_scores[index],
        )
        for index, (piece, cited) in enumerate(sentence_data)
    )
    supported_count = sum(sentence.supported for sentence in assessments)
    ratio = supported_count / len(assessments) if assessments else 0.0
    return AnswerAssessment(
        sentences=assessments,
        citations_valid=citations_valid,
        supported_ratio=ratio,
    )
