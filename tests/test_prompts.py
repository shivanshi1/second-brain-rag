from secondbrain.generation.prompts import build_context, build_messages
from secondbrain.schemas import SearchHit

HIT = SearchHit(
    chunk_id=1,
    document_id=1,
    document_title="Celery notes",
    heading_path="Beat > Schedules",
    page_number=None,
    text="Use celery beat for periodic tasks.",
    score=0.5,
)


def test_context_is_numbered_with_source():
    ctx = build_context([HIT])
    assert ctx.startswith("[1] Celery notes > Beat > Schedules")


def test_strict_mode_includes_refusal_rule():
    msgs = build_messages("What is beat?", [HIT], strict=True)
    assert "can't find this in your documents" in msgs[0]["content"]
    assert "Question: What is beat?" in msgs[1]["content"]


def test_prompt_requires_complete_sentence_with_citation():
    msgs = build_messages("What is beat?", [HIT], strict=True)
    assert "Never reply with citations alone" in msgs[0]["content"]
    assert "A monolith is simpler while the team and domain are small [1]." in msgs[0]["content"]
    assert "Answer in at most two complete sentences" in msgs[0]["content"]
    assert "Answer in at most two complete sentences" in msgs[1]["content"]


def test_retry_prompt_labels_previous_draft_untrusted():
    msgs = build_messages("What is beat?", [HIT], strict=True, previous_answer="[1]")
    assert "previous draft" in msgs[1]["content"].lower()
    assert "[1]" in msgs[1]["content"]


def test_retry_prompt_identifies_unsupported_sentences():
    msgs = build_messages(
        "What is beat?",
        [HIT],
        strict=True,
        unsupported_sentences=["This sentence is not in the notes."],
    )
    assert "These sentences were not supported" in msgs[1]["content"]
    assert "This sentence is not in the notes." in msgs[1]["content"]


def test_context_does_not_repeat_document_title_as_heading():
    hit = HIT.model_copy(
        update={
            "document_title": "Celery notes",
            "heading_path": "Celery notes > Beat > Schedules",
        }
    )
    assert build_context([hit]).startswith("[1] Celery notes > Beat > Schedules")
