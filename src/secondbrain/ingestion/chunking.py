"""Heading-aware chunking. Sections are already split by heading/page by the loaders;
here we window long sections by words with overlap. Word counts approximate tokens."""

from secondbrain.ingestion.types import ChunkData, Section


def window_words(text: str, max_words: int, overlap: int) -> list[str]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []
    step = max_words - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return out


def chunk_sections(
    sections: list[Section], max_words: int = 300, overlap: int = 40
) -> list[ChunkData]:
    chunks: list[ChunkData] = []
    for section in sections:
        for piece in window_words(section.text, max_words, overlap):
            chunks.append(
                ChunkData(
                    index=len(chunks),
                    text=piece,
                    heading_path=section.heading_path,
                    page_number=section.page_number,
                )
            )
    return chunks
