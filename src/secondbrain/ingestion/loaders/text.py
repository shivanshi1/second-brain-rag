from pathlib import Path

from secondbrain.ingestion.types import LoadedDocument, Section


def load_text(path: Path) -> LoadedDocument:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return LoadedDocument(title=path.stem, sections=[Section(text=text)] if text else [])
