from pathlib import Path

import pytest

from secondbrain.generation import grounding
from secondbrain.schemas import SearchHit


@pytest.fixture
def overclaim_answer():
    return (Path(__file__).parent / "fixtures" / "phase3_overclaim.txt").read_text(encoding="utf-8")


@pytest.fixture
def source_hit():
    return SearchHit(
        chunk_id=1,
        document_id=1,
        document_title="Microservices vs Monolith",
        heading_path="Monolith first",
        page_number=None,
        text="A monolith is simpler to build, test and deploy while the team and domain are small.",
        score=0.03,
    )


def test_exact_overclaim_example_only_supports_the_first_sentence(
    monkeypatch, overclaim_answer, source_hit
):
    observed_pairs = []

    def predict(pairs):
        observed_pairs.extend(pairs)
        return [0.98, 0.12]

    monkeypatch.setattr(grounding, "_predict_entailment_scores", predict)

    report = grounding.verify_answer_support(overclaim_answer, [source_hit], threshold=0.7)

    assert len(report.sentences) == 3
    assert [sentence.supported for sentence in report.sentences] == [True, False, False]
    assert report.citations_valid is True
    assert report.supported_ratio == pytest.approx(1 / 3)
    assert len(observed_pairs) == 2
    assert observed_pairs[0][0] == source_hit.text
    assert "You should use a monolith" in observed_pairs[0][1]
    assert "microservices architecture a better choice" in observed_pairs[1][1]


def test_invalid_citation_cannot_be_supported_even_if_nli_score_is_high(monkeypatch, source_hit):
    monkeypatch.setattr(grounding, "_predict_entailment_scores", lambda pairs: [0.99])

    report = grounding.verify_answer_support("A monolith is simple [1][9].", [source_hit])

    assert report.citations_valid is False
    assert report.sentences[0].supported is False
    assert report.supported_ratio == 0.0


def test_uncited_sentence_is_not_supported_and_does_not_load_model(source_hit, monkeypatch):
    monkeypatch.setattr(
        grounding,
        "_predict_entailment_scores",
        lambda pairs: pytest.fail("no cited block pairs should be scored"),
    )

    report = grounding.verify_answer_support(
        "A monolith is simpler for a small team.", [source_hit]
    )

    assert report.citations_valid is False
    assert report.sentences[0].supported is False
    assert report.supported_ratio == 0.0
