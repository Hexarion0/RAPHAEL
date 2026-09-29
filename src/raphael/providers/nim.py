"""NVIDIA NIM LLM provider implementation."""

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

from raphael.config import get_settings
from raphael.logging import get_logger
from raphael.providers.base import ChatMessage, LLMProvider, LLMResponse, LLMStreamChunk

logger = get_logger("providers.nim")


class NimProvider(LLMProvider):
    """Client for NVIDIA NIM (Inference Microservice) Cloud API."""

    name: str = "nim"
    default_model: str = "nvidia/nemotron-3-super-120b-a12b"
    base_url: str = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        settings = get_settings()
        if api_key:
            self.api_key = api_key
        elif settings.providers.nim_api_key:
            self.api_key = settings.providers.nim_api_key.get_secret_value()
        else:
            self.api_key = None

        self.default_model = model or settings.providers.nim_model or self.default_model
        self.complex_model = settings.providers.nim_complex_model
        self.fallback_model = settings.providers.nim_fallback_model or "meta/llama-3.2-90b-vision-instruct"
        self.timeout = timeout

    def is_configured(self) -> bool:
        """Return True if NIM API key is configured."""
        return bool(self.api_key and len(self.api_key.strip()) > 0)

    def _get_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("NVIDIA NIM API key is not configured.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def health_check(self) -> bool:
        """Verify API connectivity with a lightweight model list request."""
        if not self.is_configured():
            return False
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/models", headers=self._get_headers())
                return res.status_code == 200
        except Exception as err:
            logger.debug("NIM health check failed: %s", err)
            return False

    RELIABLE_BACKUP_MODEL: str = "meta/llama-3.2-90b-vision-instruct"

    def send(
        self,
        messages: list[ChatMessage] | str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat messages to NVIDIA NIM — retries same model 3× before falling back."""
        msgs = self.normalize_messages(messages)
        target_model = model or self.default_model
        fallback_model = self.fallback_model or self.RELIABLE_BACKUP_MODEL

        payload = {
            "model": target_model,
            "messages": [m.to_dict() for m in msgs],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs,
        }

        RETRYABLE = (429, 500, 502, 503, 504)
        MAX_RETRIES = 3

        start_t = time.time()
        with httpx.Client(timeout=self.timeout) as client:
            last_err: httpx.HTTPStatusError | None = None
            for attempt in range(MAX_RETRIES):
                try:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._get_headers(),
                        json=payload,
                    )
                    resp.raise_for_status()
                    last_err = None
                    break
                except httpx.HTTPStatusError as err:
                    last_err = err
                    if err.response.status_code not in RETRYABLE:
                        raise
                    wait = 0.5 * (2 ** attempt)  # 0.5s → 1s → 2s
                    logger.warning(
                        "NIM '%s' HTTP %d (attempt %d/%d) — retrying in %.1fs...",
                        target_model,
                        err.response.status_code,
                        attempt + 1,
                        MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)

            # All retries exhausted — fall back to secondary model once
            if last_err is not None:
                if target_model != fallback_model:
                    logger.warning(
                        "NIM '%s' failed after %d retries. Falling back to '%s'.",
                        target_model, MAX_RETRIES, fallback_model,
                    )
                    payload["model"] = fallback_model
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._get_headers(),
                        json=payload,
                    )
                    resp.raise_for_status()
                    target_model = fallback_model
                else:
                    raise last_err

            data = resp.json()

        latency = time.time() - start_t
        choice = data["choices"][0]
        content = choice["message"]["content"]
        # Strip internal thinking/reasoning tags if present
        content = self.clean_reasoning(content)
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=target_model,
            provider=self.name,
            usage=usage,
            latency=latency,
        )

    @staticmethod
    def clean_reasoning(text: str) -> str:
        """Strip internal reasoning tags and Nemotron CoT monologue from model responses."""
        import re

        cleaned = text.strip()

        # 1. Remove properly closed XML thinking tags
        cleaned = re.sub(
            r"<(think|thought|reasoning|reflection)>[\s\S]*?</\1>",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # 2. Handle unclosed opening thinking tags — keep prefix before tag, discard rest
        unclosed = re.search(r"<(think|thought|reasoning|reflection)>", cleaned, re.IGNORECASE)
        if unclosed:
            after = cleaned[unclosed.end():].strip()
            close = re.search(r"</(think|thought|reasoning|reflection)>", after, re.IGNORECASE)
            if close:
                cleaned = after[close.end():].strip()
            else:
                cleaned = cleaned[:unclosed.start()].strip()

        cleaned = cleaned.strip()

        # 3. Detect Nemotron-style CoT ending with a "Final X:" marker
        final_marker_match = re.search(
            r"(?:Final\s+(?:decision|call|answer|response)|Decision|Direct\s+response)[:\s]+(.+)$",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if final_marker_match:
            final_part = final_marker_match.group(1).strip()
            quoted_match = re.match(r'^\"(.+)\"$', final_part, re.DOTALL)
            if quoted_match:
                return quoted_match.group(1).strip()
            return final_part

        # 4. Detect raw thinking monologue at start — return last paragraph as reply
        if re.match(
            r"^\s*(Okay[,.]?\s|Hmm[,.]?\s|Alright[,.]?\s|Let me\s|Looking at|The user|I need to|I should|I'll analyze)",
            cleaned,
            re.IGNORECASE,
        ):
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
            if paragraphs:
                return paragraphs[-1]

        return cleaned.strip()

    def stream(
        self,
        messages: list[ChatMessage] | str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Iterator[LLMStreamChunk]:
        """Stream chat completion chunks from NVIDIA NIM — retries same model 3× before falling back."""
        msgs = self.normalize_messages(messages)
        target_model = model or self.default_model
        fallback_model = self.fallback_model or self.RELIABLE_BACKUP_MODEL

        payload = {
            "model": target_model,
            "messages": [m.to_dict() for m in msgs],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }

        RETRYABLE = (429, 500, 502, 503, 504)
        MAX_RETRIES = 3

        def _iter_stream(client: httpx.Client, active_model: str) -> Iterator[LLMStreamChunk]:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        yield LLMStreamChunk(delta="", model=active_model, provider=self.name, is_final=True)
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        delta = chunk_json["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield LLMStreamChunk(delta=delta, model=active_model, provider=self.name, is_final=False)
                    except json.JSONDecodeError:
                        continue

        with httpx.Client(timeout=self.timeout) as client:
            last_err: httpx.HTTPStatusError | None = None
            for attempt in range(MAX_RETRIES):
                try:
                    yield from _iter_stream(client, target_model)
                    return
                except httpx.HTTPStatusError as err:
                    last_err = err
                    if err.response.status_code not in RETRYABLE:
                        raise
                    wait = 0.5 * (2 ** attempt)  # 0.5s → 1s → 2s
                    logger.warning(
                        "NIM stream '%s' HTTP %d (attempt %d/%d) — retrying in %.1fs...",
                        target_model,
                        err.response.status_code,
                        attempt + 1,
                        MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)

            # All retries exhausted — fall back once
            if last_err is not None:
                if target_model != fallback_model:
                    logger.warning(
                        "NIM stream '%s' failed after %d retries. Falling back to '%s'.",
                        target_model, MAX_RETRIES, fallback_model,
                    )
                    payload["model"] = fallback_model
                    yield from _iter_stream(client, fallback_model)
                else:
                    raise last_err

