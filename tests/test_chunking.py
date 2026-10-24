import pytest

from secondbrain.ingestion.chunking import chunk_sections, window_words
from secondbrain.ingestion.types import Section


def test_short_text_is_one_chunk():
    assert window_words("a b c", max_words=10, overlap=2) == ["a b c"]


def test_empty_text_has_no_chunks():
    assert window_words("   ", max_words=10, overlap=2) == []


def test_windows_overlap():
    text = " ".join(str(i) for i in range(25))
    chunks = window_words(text, max_words=10, overlap=3)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert chunks[-1].split()[-1] == "24"  # nothing is dropped


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        window_words("a b c", max_words=5, overlap=5)


def test_chunks_keep_heading_and_page_metadata():
    sections = [
        Section(text="word " * 50, heading_path="Intro > Basics", page_number=2),
        Section(text="short", heading_path="Outro"),
    ]
    chunks = chunk_sections(sections, max_words=20, overlap=5)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].heading_path == "Intro > Basics" and chunks[0].page_number == 2
    assert chunks[-1].heading_path == "Outro"
    assert chunks[0].embed_text.startswith("Intro > Basics\n")
