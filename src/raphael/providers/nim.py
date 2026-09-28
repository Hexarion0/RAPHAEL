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

    RELIABLE_BACKUP_MODEL: str = "meta/llama-3.2-11b-vision-instruct"

    def send(
        self,
        messages: list[ChatMessage] | str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat messages to NVIDIA NIM with automatic model fallback on server errors."""
        msgs = self.normalize_messages(messages)
        target_model = model or self.default_model

        payload = {
            "model": target_model,
            "messages": [m.to_dict() for m in msgs],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs,
        }

        start_t = time.time()
        with httpx.Client(timeout=self.timeout) as client:
            try:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as err:
                # If primary model is overloaded (503/504) or unavailable (404), fallback to 11B
                if target_model != self.RELIABLE_BACKUP_MODEL and err.response.status_code in (
                    404,
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    logger.warning(
                        "NIM model '%s' returned HTTP %d. Retrying with reliable backup '%s'...",
                        target_model,
                        err.response.status_code,
                        self.RELIABLE_BACKUP_MODEL,
                    )
                    payload["model"] = self.RELIABLE_BACKUP_MODEL
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._get_headers(),
                        json=payload,
                    )
                    resp.raise_for_status()
                    target_model = self.RELIABLE_BACKUP_MODEL
                else:
                    raise

            data = resp.json()

        latency = time.time() - start_t
        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=target_model,
            provider=self.name,
            usage=usage,
            latency=latency,
        )

    def stream(
        self,
        messages: list[ChatMessage] | str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Iterator[LLMStreamChunk]:
        """Stream chat completion chunks from NVIDIA NIM."""
        msgs = self.normalize_messages(messages)
        target_model = model or self.default_model

        payload = {
            "model": target_model,
            "messages": [m.to_dict() for m in msgs],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }

        with httpx.Client(timeout=self.timeout) as client:
            try:
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
                        data_str = line[len("data:") :].strip()
                        if data_str == "[DONE]":
                            yield LLMStreamChunk(
                                delta="",
                                model=target_model,
                                provider=self.name,
                                is_final=True,
                            )
                            break
                        try:
                            chunk_json = json.loads(data_str)
                            delta = chunk_json["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield LLMStreamChunk(
                                    delta=delta,
                                    model=target_model,
                                    provider=self.name,
                                    is_final=False,
                                )
                        except json.JSONDecodeError:
                            continue
            except httpx.HTTPStatusError as err:
                if target_model != self.RELIABLE_BACKUP_MODEL and err.response.status_code in (
                    404,
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    logger.warning(
                        "NIM streaming with '%s' failed (HTTP %d). Falling back to '%s'...",
                        target_model,
                        err.response.status_code,
                        self.RELIABLE_BACKUP_MODEL,
                    )
                    payload["model"] = self.RELIABLE_BACKUP_MODEL
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
                            data_str = line[len("data:") :].strip()
                            if data_str == "[DONE]":
                                yield LLMStreamChunk(
                                    delta="",
                                    model=self.RELIABLE_BACKUP_MODEL,
                                    provider=self.name,
                                    is_final=True,
                                )
                                break
                            try:
                                chunk_json = json.loads(data_str)
                                delta = chunk_json["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield LLMStreamChunk(
                                        delta=delta,
                                        model=self.RELIABLE_BACKUP_MODEL,
                                        provider=self.name,
                                        is_final=False,
                                    )
                            except json.JSONDecodeError:
                                continue
                else:
                    raise
