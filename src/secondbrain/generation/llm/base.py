from collections.abc import Iterator
from typing import Protocol


class LLMClient(Protocol):
    """Provider-agnostic interface. Implement for Ollama, Groq, Gemini, or a test fake."""

    def generate(self, messages: list[dict], *, temperature: float = 0.1) -> str: ...

    def stream(self, messages: list[dict], *, temperature: float = 0.1) -> Iterator[str]: ...
