from secondbrain.ingestion.loaders.markdown import load_markdown, split_markdown

SAMPLE = """---
title: Architecture Notes
tags: [architecture, "#backend"]
date: 2025-03-01
---
# Architecture

Intro text.

## Monolith first

Start simple.

```python
# not a heading
print("hi")
```

## Microservices

Scale teams independently.
"""


def test_heading_paths():
    sections = split_markdown("# A\nbody a\n## B\nbody b\n## C\nbody c\n# D\nbody d")
    assert [s.heading_path for s in sections] == ["A", "A > B", "A > C", "D"]


def test_code_fences_are_not_headings():
    sections = split_markdown("# A\n```\n# fake\n```\ntext")
    assert len(sections) == 1 and "# fake" in sections[0].text


def test_frontmatter_title_tags_date(tmp_path):
    f = tmp_path / "n.md"
    f.write_text(SAMPLE)
    doc = load_markdown(f)
    assert doc.title == "Architecture Notes"
    assert doc.tags == ["architecture", "#backend"]
    assert doc.document_date is not None and doc.document_date.year == 2025
    assert any(s.heading_path == "Architecture > Microservices" for s in doc.sections)


def test_title_falls_back_to_filename(tmp_path):
    f = tmp_path / "plain-note.md"
    f.write_text("just text, no headings")
    assert load_markdown(f).title == "plain-note"
