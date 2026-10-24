from secondbrain.config import settings
from secondbrain.generation.llm.base import LLMClient
from secondbrain.generation.llm.ollama import OllamaClient


def get_llm() -> LLMClient:
    if settings.llm_provider == "ollama":
        return OllamaClient()
    # TODO: add GroqClient / GeminiClient (free tiers) as fallbacks
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
