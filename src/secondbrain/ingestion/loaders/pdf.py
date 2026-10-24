from pathlib import Path

from secondbrain.ingestion.types import LoadedDocument, Section


def load_pdf(path: Path) -> LoadedDocument:
    import pymupdf4llm  # imported lazily: heavy

    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    sections = [
        Section(text=page["text"].strip(), page_number=i)
        for i, page in enumerate(pages, start=1)
        if page["text"].strip()
    ]
    if not sections:
        raise ValueError("No extractable text (scanned PDF? OCR is a stretch goal).")
    return LoadedDocument(title=path.stem, sections=sections, meta={"pages": len(pages)})
