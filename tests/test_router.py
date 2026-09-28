"""Unit tests for ModelRouter complexity classification and provider routing."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from raphael.providers.base import ChatMessage, LLMResponse
from raphael.providers.manager import ProviderManager
from raphael.providers.router import ComplexityLevel, ModelRouter


@pytest.fixture
def mock_manager():
    """Mocked ProviderManager with simulated provider configs."""
    manager = MagicMock(spec=ProviderManager)
    manager.nim = MagicMock()
    manager.nim.is_configured.return_value = True
    manager.groq = MagicMock()
    manager.groq.is_configured.return_value = True
    manager.openrouter = MagicMock()
    manager.openrouter.is_configured.return_value = False
    manager.ollama = MagicMock()
    manager.ollama.is_configured.return_value = True

    manager.send_with_fallback = AsyncMock(
        return_value=LLMResponse(
            content="Mocked response",
            provider="groq",
            model="llama-3.1-8b-instant",
        )
    )
    return manager


def test_classify_simple_greetings(mock_manager):
    router = ModelRouter(manager=mock_manager)
    level, reason, override = router.classify_complexity("Hello Raphael, what time is it?")
    assert level == ComplexityLevel.SIMPLE
    assert override is None


def test_classify_complex_coding_query(mock_manager):
    router = ModelRouter(manager=mock_manager)
    level, reason, override = router.classify_complexity(
        "Can you refactor this Python function to optimize recursion and fix the bug?"
    )
    assert level == ComplexityLevel.COMPLEX
    assert "code" in reason.lower() or "keyword" in reason.lower()


def test_classify_manual_overrides(mock_manager):
    router = ModelRouter(manager=mock_manager)

    level, _, override = router.classify_complexity("/fast explain quantum mechanics in one word")
    assert level == ComplexityLevel.SIMPLE
    assert override == "fast"

    level, _, override = router.classify_complexity("/strong write a complete microservices plan")
    assert level == ComplexityLevel.COMPLEX
    assert override == "strong"

    level, _, override = router.classify_complexity("/local how much disk space is left?")
    assert level == ComplexityLevel.SIMPLE
    assert override == "local"


def test_routing_decisions(mock_manager):
    router = ModelRouter(manager=mock_manager)

    # Simple with Groq configured -> groq
    decision_simple = router.route("Hello")
    assert decision_simple.complexity == ComplexityLevel.SIMPLE
    assert decision_simple.provider_name == "groq"

    # Complex -> NIM 70B
    decision_complex = router.route("Design a high-concurrency database architecture with sharding")
    assert decision_complex.complexity == ComplexityLevel.COMPLEX
    assert decision_complex.provider_name == "nim"


@pytest.mark.asyncio
async def test_router_send(mock_manager):
    router = ModelRouter(manager=mock_manager)
    messages = [
        ChatMessage(role="system", content="You are RAPHAEL."),
        ChatMessage(role="user", content="/fast What is the capital of France?"),
    ]

    response = await router.send(messages)
    assert response.content == "Mocked response"
    mock_manager.send_with_fallback.assert_called_once()
