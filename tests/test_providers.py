"""Unit tests for AI providers and fallback manager."""

from unittest.mock import MagicMock, patch

import pytest

from raphael.providers.base import ChatMessage, LLMProvider, LLMResponse
from raphael.providers.groq import GroqProvider
from raphael.providers.manager import ProviderManager
from raphael.providers.nim import NimProvider
from raphael.providers.ollama import OllamaProvider
from raphael.providers.openrouter import OpenRouterProvider


def test_chat_message_serialization():
    """Verify ChatMessage dictionary conversion and string normalization."""
    msg = ChatMessage(role="user", content="Hello")
    assert msg.to_dict() == {"role": "user", "content": "Hello"}

    normalized = LLMProvider.normalize_messages("Hello world")
    assert len(normalized) == 1
    assert normalized[0].content == "Hello world"
    assert normalized[0].role == "user"


@patch("httpx.Client.post")
def test_nim_provider_send(mock_post):
    """Verify NIM provider parses chat completions accurately."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "NIM answer"}}],
        "usage": {"total_tokens": 42},
    }
    mock_post.return_value = mock_resp

    provider = NimProvider(api_key="nvapi-test-key-12345")
    assert provider.is_configured() is True

    res = provider.send("Test prompt")
    assert res.content == "NIM answer"
    assert res.provider == "nim"
    assert res.usage["total_tokens"] == 42


@patch("httpx.Client.post")
def test_groq_provider_send(mock_post):
    """Verify Groq provider parses chat completions accurately."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Groq fast answer"}}],
        "usage": {"total_tokens": 15},
    }
    mock_post.return_value = mock_resp

    provider = GroqProvider(api_key="gsk_test-key-12345")
    assert provider.is_configured() is True

    res = provider.send("Quick question")
    assert res.content == "Groq fast answer"
    assert res.provider == "groq"


@patch("httpx.Client.post")
def test_openrouter_provider_send(mock_post):
    """Verify OpenRouter provider parses chat completions accurately."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "OpenRouter answer"}}],
        "usage": {"total_tokens": 30},
    }
    mock_post.return_value = mock_resp

    provider = OpenRouterProvider(api_key="sk-or-v1-test")
    assert provider.is_configured() is True

    res = provider.send("Explain quantum computing")
    assert res.content == "OpenRouter answer"
    assert res.provider == "openrouter"


@patch("httpx.Client.post")
def test_ollama_provider_send(mock_post):
    """Verify local Ollama provider parses chat completions."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": "Ollama local response"},
        "prompt_eval_count": 10,
        "eval_count": 20,
    }
    mock_post.return_value = mock_resp

    provider = OllamaProvider(host="http://localhost:11434")
    assert provider.is_configured() is True

    res = provider.send("Local prompt")
    assert res.content == "Ollama local response"
    assert res.provider == "ollama"


def test_provider_manager_fallback_chain():
    """Verify fallback chain switches to next provider when primary fails."""
    manager = ProviderManager(fallback_chain=["nim", "groq", "ollama"])

    # Mock failing NIM provider
    failing_nim = MagicMock(spec=LLMProvider)
    failing_nim.is_configured.return_value = True
    failing_nim.send.side_effect = Exception("NIM rate limit exceeded")

    # Mock working Groq provider
    working_groq = MagicMock(spec=LLMProvider)
    working_groq.is_configured.return_value = True
    working_groq.send.return_value = LLMResponse(
        content="Fallback response from Groq",
        model="llama-3.1-8b-instant",
        provider="groq",
    )

    manager.register_provider("nim", failing_nim)
    manager.register_provider("groq", working_groq)

    # Should attempt NIM, catch error, and successfully return from Groq
    res = manager.send_with_fallback("Hello assistant", preferred_provider="nim")
    assert res.content == "Fallback response from Groq"
    assert res.provider == "groq"
    failing_nim.send.assert_called_once()
    working_groq.send.assert_called_once()


def test_provider_manager_all_fail():
    """Verify exception is raised when all providers in fallback chain fail."""
    manager = ProviderManager(fallback_chain=["nim"])
    failing_nim = MagicMock(spec=LLMProvider)
    failing_nim.is_configured.return_value = True
    failing_nim.send.side_effect = Exception("API connection timeout")

    manager.register_provider("nim", failing_nim)

    with pytest.raises(RuntimeError, match="All AI providers in fallback chain failed"):
        manager.send_with_fallback("Hello")


def completion(content, **kwargs):
    import httpx

    return httpx.Response(
        200,
        request=httpx.Request("POST", "https://example.test/completions"),
        json={"choices": [{"message": {"content": content, **kwargs}}]},
    )


@patch("httpx.Client.post")
def test_nim_null_reply_tries_backup_without_speaking_reasoning(mock_post):
    mock_post.side_effect = [
        completion(None, reasoning_content="private reasoning"),
        completion("I'm here."),
    ]
    provider = NimProvider(api_key="test-key", model="nvidia/nemotron-3-super-120b-a12b")
    provider.fallback_model = "meta/llama-3.2-90b-vision-instruct"
    response = provider.send("Are you good?", max_tokens=400)
    assert response.content == "I'm here."
    assert response.model == provider.fallback_model
    initial = mock_post.call_args_list[0].kwargs["json"]
    backup = mock_post.call_args_list[1].kwargs["json"]
    assert initial["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in backup
    assert mock_post.call_count == 2


@patch("httpx.Client.post")
@pytest.mark.parametrize("content", [None, "", "<think>private reasoning</think>"])
def test_nim_empty_replies_fail_clearly_after_one_backup(mock_post, content):
    mock_post.side_effect = [completion(content), completion(content)]
    provider = NimProvider(api_key="test-key", model="nvidia/nemotron-3-super-120b-a12b")
    provider.fallback_model = "meta/backup"
    with pytest.raises(RuntimeError, match="no user-visible answer"):
        provider.send("Hello")
    assert mock_post.call_count == 2


@patch("httpx.Client.post")
def test_nim_caller_can_explicitly_enable_reasoning(mock_post):
    mock_post.return_value = completion("visible answer", reasoning_content="private thoughts")
    provider = NimProvider(api_key="test-key", model="nvidia/nemotron-3-super-120b-a12b")
    template = {"enable_thinking": True, "low_effort": True}
    assert (
        provider.send("A complex query", chat_template_kwargs=template).content == "visible answer"
    )
    assert mock_post.call_args.kwargs["json"]["chat_template_kwargs"] == template
    assert template == {"enable_thinking": True, "low_effort": True}


def test_nim_template_options_are_model_specific_for_streams():
    payload = {"model": "nvidia/nemotron-3-super-120b-a12b", "stream": True}
    assert NimProvider._model_payload(payload, payload["model"])["chat_template_kwargs"] == {
        "enable_thinking": False,
    }
    assert "chat_template_kwargs" not in NimProvider._model_payload(payload, "meta/backup")
    assert "chat_template_kwargs" not in payload


@patch("httpx.Client.post")
@patch("raphael.providers.nim.time.sleep")
def test_nim_transient_http_errors_retry_then_use_backup(mock_sleep, mock_post):
    import httpx

    request = httpx.Request("POST", "https://example.test/completions")
    response = httpx.Response(503, request=request)
    errors = [
        httpx.HTTPStatusError("Unavailable", request=request, response=response) for _ in range(3)
    ]
    mock_post.side_effect = [*errors, completion("backup answer")]
    provider = NimProvider(api_key="test-key", model="nvidia/nemotron-3-super-120b-a12b")
    provider.fallback_model = "meta/backup"
    assert provider.send("Hello").content == "backup answer"
    assert mock_post.call_count == 4
    assert mock_sleep.call_count == 2  # No pointless delay after the last failure.
