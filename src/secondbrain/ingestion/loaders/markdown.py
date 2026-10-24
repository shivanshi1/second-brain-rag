import re
from datetime import UTC, date, datetime
from pathlib import Path

import frontmatter

from secondbrain.ingestion.types import LoadedDocument, Section

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def split_markdown(text: str) -> list[Section]:
    """Split into sections, tracking the heading path ('A > B > C'). Ignores fenced code."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(text=body, heading_path=" > ".join(h for _, h in stack)))

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        match = None if in_code else HEADING_RE.match(line)
        if match:
            flush()
            buf.clear()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2)))
        else:
            buf.append(line)
    flush()
    return sections


def _to_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str):
        try:
            return _to_datetime(datetime.fromisoformat(value))
        except ValueError:
            return None
    return None


def load_markdown(path: Path) -> LoadedDocument:
    post = frontmatter.loads(path.read_text(encoding="utf-8", errors="replace"))
    sections = split_markdown(post.content)

    raw_tags = post.get("tags", [])
    if isinstance(raw_tags, str):
        raw_tags = [t for t in re.split(r"[,\s]+", raw_tags) if t]
    elif not isinstance(raw_tags, list):
        raw_tags = []

    title = post.get("title") or (sections[0].heading_path.split(" > ")[0] if sections else "")
    return LoadedDocument(
        title=str(title or path.stem),
        sections=sections,
        tags=[str(t) for t in raw_tags],
        document_date=_to_datetime(post.get("date")),
    )
