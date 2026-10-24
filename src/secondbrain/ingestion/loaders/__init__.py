from pathlib import Path

from secondbrain.ingestion.loaders.markdown import load_markdown
from secondbrain.ingestion.loaders.pdf import load_pdf
from secondbrain.ingestion.loaders.text import load_text
from secondbrain.ingestion.types import LoadedDocument

EXTENSION_TO_TYPE = {".md": "md", ".markdown": "md", ".txt": "text", ".pdf": "pdf"}


def load_file(path: Path, source_type: str) -> LoadedDocument:
    loaders = {"md": load_markdown, "pdf": load_pdf, "text": load_text}
    if source_type not in loaders:
        raise ValueError(f"Unsupported source type: {source_type}")
    return loaders[source_type](path)
