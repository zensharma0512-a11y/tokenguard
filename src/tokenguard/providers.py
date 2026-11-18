"""Provider adapters. Everything is translated to and from the OpenAI chat format.

* ``openai``: any OpenAI-compatible API (OpenAI, Groq, Together, OpenRouter,
  a local vLLM or Ollama server). The request is passed through.
* ``anthropic``: the Anthropic Messages API, translated both ways.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

# http(url, headers, body_bytes, timeout) -> (status, body_bytes). Raises OSError on network failure.
HTTP = Callable[[str, dict, bytes, float], tuple[int, bytes]]


def urllib_http(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class ProviderError(Exception):
    def __init__(self, provider: str, status: int | None, message: str, *, retryable: bool):
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.status = status
        self.retryable = retryable


def _retryable(status: int) -> bool:
    return status in (408, 409, 429) or status >= 500


@dataclass
class Provider:
    name: str
    type: str          # "openai" | "anthropic"
    base_url: str
    api_key: str
    timeout: float = 60.0
    http: HTTP = urllib_http

    def chat(self, model: str, request: dict[str, Any]) -> dict[str, Any]:
        """Send an OpenAI-format chat request; return an OpenAI-format response."""
        if self.type == "openai":
            return self._openai(model, request)
        if self.type == "anthropic":
            return self._anthropic(model, request)
        raise ValueError(f"unknown provider type {self.type!r}")

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        try:
            status, raw = self.http(url, headers, json.dumps(payload).encode(), self.timeout)
        except OSError as exc:  # connection refused, DNS failure, timeout
            raise ProviderError(self.name, None, f"network error: {exc}", retryable=True) from exc
        if not 200 <= status < 300:
            snippet = raw[:300].decode("utf-8", "replace")
            raise ProviderError(self.name, status, f"HTTP {status}: {snippet}", retryable=_retryable(status))
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(self.name, status, "invalid JSON from provider", retryable=True) from exc

    def _openai(self, model: str, request: dict) -> dict:
        payload = {**request, "model": model}
        headers = {"content-type": "application/json", "authorization": f"Bearer {self.api_key}"}
        return self._post(self.base_url.rstrip("/") + "/chat/completions", headers, payload)

    def _anthropic(self, model: str, request: dict) -> dict:
        system_parts, messages = [], []
        for m in request["messages"]:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                messages.append({"role": m["role"], "content": m["content"]})
        payload: dict[str, Any] = {"model": model, "max_tokens": request["max_tokens"], "messages": messages}
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        for k in ("temperature", "top_p"):
            if k in request:
                payload[k] = request[k]
        if "stop" in request:
            stop = request["stop"]
            payload["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        data = self._post(self.base_url.rstrip("/") + "/v1/messages", headers, payload)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        finish = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}.get(data.get("stop_reason"), "stop")
        pt, ct = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        return {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model", model),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }
