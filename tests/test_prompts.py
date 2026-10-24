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
