import json
from collections.abc import Iterator

import httpx

from secondbrain.config import settings


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 120):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout

    def _payload(self, messages: list[dict], temperature: float, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }

    def generate(self, messages: list[dict], *, temperature: float = 0.1) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json=self._payload(messages, temperature, stream=False),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def stream(self, messages: list[dict], *, temperature: float = 0.1) -> Iterator[str]:
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=self._payload(messages, temperature, stream=True),
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if not chunk.get("done"):
                        yield chunk["message"]["content"]
