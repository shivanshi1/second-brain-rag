from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Section:
    text: str
    heading_path: str = ""
    page_number: int | None = None


@dataclass
class LoadedDocument:
    title: str
    sections: list[Section]
    tags: list[str] = field(default_factory=list)
    document_date: datetime | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class ChunkData:
    index: int
    text: str
    heading_path: str
    page_number: int | None

    @property
    def embed_text(self) -> str:
        """Heading path is prepended for embedding so chunks keep their context."""
        return f"{self.heading_path}\n{self.text}" if self.heading_path else self.text
