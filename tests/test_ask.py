import httpx
import pytest
from fastapi.testclient import TestClient

from secondbrain.api.deps import require_api_key
from secondbrain.api.main import app
from secondbrain.api.routes import ask as ask_route
from secondbrain.db.session import get_db
from secondbrain.generation.grounding import (
    AnswerAssessment,
    SentenceAssessment,
    SupportVerificationError,
)
from secondbrain.schemas import SearchHit


class FakeSession:
    def __init__(self):
        self.added = []
        self.commit_count = 0

    def add(self, item):
        self.added.append(item)

    def commit(self):
        self.commit_count += 1


@pytest.fixture
def fake_db():
    return FakeSession()


@pytest.fixture
def client(fake_db):
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[require_api_key] = lambda: None
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)


@pytest.fixture
def one_hit():
    return SearchHit(
        chunk_id=17,
        document_id=4,
        document_title="Architecture notes",
        heading_path="Monolith first",
        page_number=None,
        text="A monolith is simpler while the team and domain are small.",
        score=0.02,
    )


class FakeLLM:
    def __init__(self, answers):
        self.answers = answers if isinstance(answers, list) else [answers]
        self.messages = []
        self.temperatures = []

    def generate(self, messages, *, temperature=0.1):
        self.messages.append(messages)
        self.temperatures.append(temperature)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def make_assessment(texts, supported, ratio=None, citations_valid=True):
    sentences = tuple(
        SentenceAssessment(
            text=text,
            supported=is_supported,
            cited_blocks=(1,) if "[1]" in text else (),
            entailment_score=0.95 if is_supported else 0.1,
        )
        for text, is_supported in zip(texts, supported, strict=True)
    )
    if ratio is None:
        ratio = sum(supported) / len(supported) if supported else 0.0
    return AnswerAssessment(
        sentences=sentences,
        citations_valid=citations_valid,
        supported_ratio=ratio,
    )


def test_ask_returns_answer_citations_and_logs_query(client, fake_db, monkeypatch, one_hit):
    answer = "A monolith can be simpler for a small team [1]."
    llm = FakeLLM(answer)
    report = make_assessment([answer], [True])
    monkeypatch.setattr(ask_route, "hybrid_search", lambda *args: [one_hit])
    monkeypatch.setattr(ask_route, "get_llm", lambda: llm)
    monkeypatch.setattr(ask_route, "verify_answer_support", lambda *args: report)

    response = client.post(
        "/ask", json={"question": "When should I use a monolith?", "filters": {"tags": ["demo"]}}
    )

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == answer
    assert result["citations"] == [
        {
            "n": 1,
            "document": "Architecture notes",
            "heading": "Monolith first",
            "snippet": one_hit.text,
        }
    ]
    assert result["citations_valid"] is True
    assert result["supported_ratio"] == 1.0
    assert "Question: When should I use a monolith?" in llm.messages[0][1]["content"]
    assert llm.temperatures == [0.0]

    assert fake_db.commit_count == 1
    query_log = fake_db.added[0]
    assert query_log.question == "When should I use a monolith?"
    assert query_log.filters["tags"] == ["demo"]
    assert query_log.chunk_ids == [17]
    assert query_log.answer == answer


def test_strict_ask_with_no_hits_returns_refusal_without_calling_llm(client, fake_db, monkeypatch):
    monkeypatch.setattr(ask_route, "hybrid_search", lambda *args: [])
    monkeypatch.setattr(ask_route, "get_llm", lambda: pytest.fail("LLM should not be called"))

    response = client.post("/ask", json={"question": "A question with no matching notes"})

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "I can't find this in your documents."
    assert result["citations"] == []
    assert result["citations_valid"] is False
    assert result["supported_ratio"] == 0.0
    assert fake_db.added[0].chunk_ids == []
    assert fake_db.added[0].answer == result["answer"]


def test_strict_mode_retries_with_unsupported_sentences_and_returns_supported_text(
    client, monkeypatch, one_hit
):
    first = "A monolith is simpler for small teams [1]. A monolith is always best [1]."
    second = "A monolith is simpler while the team and domain are small [1]."
    llm = FakeLLM([first, second])
    reports = iter(
        [
            make_assessment(
                ["A monolith is simpler for small teams [1].", "A monolith is always best [1]."],
                [True, False],
            ),
            make_assessment([second], [True]),
        ]
    )
    monkeypatch.setattr(ask_route, "hybrid_search", lambda *args: [one_hit])
    monkeypatch.setattr(ask_route, "get_llm", lambda: llm)
    monkeypatch.setattr(ask_route, "verify_answer_support", lambda *args: next(reports))

    response = client.post("/ask", json={"question": "When should I use a monolith?"})

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == second
    assert result["citations_valid"] is True
    assert result["supported_ratio"] == 1.0
    assert len(llm.messages) == 2
    assert "A monolith is always best" in llm.messages[1][1]["content"]
    assert "at most two complete sentences" in llm.messages[1][1]["content"].lower()


def test_strict_mode_drops_unsupported_sentence_if_retry_still_fails(client, monkeypatch, one_hit):
    supported = "A monolith is simpler while the team is small [1]."
    unsupported = "Microservices are always better later [1]."
    llm = FakeLLM([f"{supported} {unsupported}", f"{supported} {unsupported}"])
    report = make_assessment([supported, unsupported], [True, False], ratio=0.5)
    monkeypatch.setattr(ask_route, "hybrid_search", lambda *args: [one_hit])
    monkeypatch.setattr(ask_route, "get_llm", lambda: llm)
    monkeypatch.setattr(ask_route, "verify_answer_support", lambda *args: report)

    response = client.post("/ask", json={"question": "When should I use a monolith?"})

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == supported
    assert result["citations_valid"] is True
    assert result["supported_ratio"] == 0.5


def test_strict_mode_refuses_when_retry_has_no_supported_sentences(client, monkeypatch, one_hit):
    unsupported = "Microservices are always better for every team [1]."
    llm = FakeLLM([unsupported, unsupported])
    report = make_assessment([unsupported], [False], ratio=0.0)
    monkeypatch.setattr(ask_route, "hybrid_search", lambda *args: [one_hit])
    monkeypatch.setattr(ask_route, "get_llm", lambda: llm)
    monkeypatch.setattr(ask_route, "verify_answer_support", lambda *args: report)

    response = client.post("/ask", json={"question": "When should I use a monolith?"})

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "I can't find this in your documents."
    assert result["citations"] == []
    assert result["citations_valid"] is False
    assert result["supported_ratio"] == 0.0


def test_open_mode_labels_unsupported_sentences_and_removes_their_citations(
    client, monkeypatch, one_hit
):
    supported = "A monolith is simpler while the team is small [1]."
    unsupported = "Microservices are always better later [1]."
    answer = f"{supported} {unsupported}"
    llm = FakeLLM(answer)
    report = make_assessment([supported, unsupported], [True, False], ratio=0.5)
    monkeypatch.setattr(ask_route, "hybrid_search", lambda *args: [one_hit])
    monkeypatch.setattr(ask_route, "get_llm", lambda: llm)
    monkeypatch.setattr(ask_route, "verify_answer_support", lambda *args: report)

    response = client.post(
        "/ask", json={"question": "When should I use a monolith?", "strict": False}
    )

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == (
        f"{supported} Not from your notes: Microservices are always better later."
    )
    assert result["citations_valid"] is True
    assert result["supported_ratio"] == 0.5


def test_ask_returns_bad_gateway_and_logs_when_llm_is_unavailable(
    client, fake_db, monkeypatch, one_hit
):
    llm = FakeLLM(httpx.ConnectError("Ollama unavailable"))
    monkeypatch.setattr(ask_route, "hybrid_search", lambda *args: [one_hit])
    monkeypatch.setattr(ask_route, "get_llm", lambda: llm)

    response = client.post("/ask", json={"question": "When should I use a monolith?"})

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The configured language model could not generate an answer."
    )
    assert fake_db.commit_count == 1
    assert fake_db.added[0].answer is None
    assert fake_db.added[0].chunk_ids == [17]


def test_ask_returns_service_unavailable_if_nli_model_cannot_run(
    client, fake_db, monkeypatch, one_hit
):
    answer = "A monolith is simpler while the team is small [1]."
    llm = FakeLLM(answer)
    monkeypatch.setattr(ask_route, "hybrid_search", lambda *args: [one_hit])
    monkeypatch.setattr(ask_route, "get_llm", lambda: llm)
    monkeypatch.setattr(
        ask_route,
        "verify_answer_support",
        lambda *args: (_ for _ in ()).throw(SupportVerificationError("model unavailable")),
    )

    response = client.post("/ask", json={"question": "When should I use a monolith?"})

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The sentence support model is unavailable; no answer was returned."
    )
    assert fake_db.added[0].answer is None
