"""AI provider clients, base interfaces, and fallback manager."""

from functools import lru_cache

from raphael.providers.base import ChatMessage, LLMProvider, LLMResponse, LLMStreamChunk
from raphael.providers.groq import GroqProvider
from raphael.providers.manager import ProviderManager
from raphael.providers.nim import NimProvider
from raphael.providers.ollama import OllamaProvider
from raphael.providers.openrouter import OpenRouterProvider


@lru_cache(maxsize=1)
def get_provider_manager() -> ProviderManager:
    """Return the global cached ProviderManager instance."""
    return ProviderManager()


__all__ = [
    "ChatMessage",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMProvider",
    "NimProvider",
    "OpenRouterProvider",
    "GroqProvider",
    "OllamaProvider",
    "ProviderManager",
    "get_provider_manager",
]
