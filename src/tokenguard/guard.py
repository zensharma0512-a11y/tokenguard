"""The request pipeline: validate -> rate limit -> cache -> reserve -> call with fallback -> settle."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from .config import Config, Route
from .providers import ProviderError
from .store import BudgetExceeded, RateLimited, Store, User

# Request fields we forward. Anything else is dropped rather than passed blindly to a provider.
ALLOWED_FIELDS = {"model", "messages", "max_tokens", "temperature", "top_p", "stop", "user", "seed"}


class BadRequest(Exception):
    pass


class AllProvidersFailed(Exception):
    def __init__(self, errors: list[ProviderError]):
        super().__init__("; ".join(str(e) for e in errors))
        self.errors = errors


@dataclass
class Result:
    response: dict[str, Any]
    cache_hit: bool = False
    provider: str | None = None
    cost_micros: int = 0
    attempts: list[str] = field(default_factory=list)


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Deliberately pessimistic: ~3 chars per token plus per-message overhead.

    We only need an upper bound for the reservation; the real charge comes
    from the provider's reported usage.
    """
    chars = sum(len(m.get("content", "")) for m in messages)
    return math.ceil(chars / 3) + 8 * len(messages) + 16


def cache_key(model: str, req: dict) -> str:
    material = {k: req[k] for k in sorted(req) if k not in ("user",)}
    material["model"] = model
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Guard:
    def __init__(self, store: Store, config: Config):
        self.store = store
        self.config = config

    def validate(self, body: Any) -> dict:
        if not isinstance(body, dict):
            raise BadRequest("body must be a JSON object")
        if body.get("stream"):
            raise BadRequest("streaming is not supported yet; set stream to false")
        model = body.get("model")
        if model not in self.config.models:
            raise BadRequest(f"unknown model {model!r}; available: {sorted(self.config.models)}")
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise BadRequest("messages must be a non-empty list")
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant"):
                raise BadRequest("each message needs role system, user or assistant")
            if not isinstance(m.get("content"), str):
                raise BadRequest("message content must be a string")
        mt = body.get("max_tokens", self.config.default_max_tokens)
        if not isinstance(mt, int) or isinstance(mt, bool) or mt < 1:
            raise BadRequest("max_tokens must be a positive integer")
        if mt > self.config.max_tokens_limit:
            raise BadRequest(f"max_tokens may not exceed {self.config.max_tokens_limit}")
        req = {k: v for k, v in body.items() if k in ALLOWED_FIELDS and k != "stream"}
        req["max_tokens"] = mt
        return req

    def _cacheable(self, req: dict, force: bool) -> bool:
        return force or req.get("temperature") == 0

    def worst_case(self, routes: list[Route], req: dict) -> int:
        pt = estimate_prompt_tokens(req["messages"])
        return max(r.cost(pt, req["max_tokens"]) for r in routes)

    def handle(self, user: User, body: Any, *, force_cache: bool = False, bypass_cache: bool = False) -> Result:
        req = self.validate(body)
        model = req["model"]
        routes = self.config.models[model]
        self.store.take_rate_token(user)  # raises RateLimited

        key = cache_key(model, req)
        cacheable = self._cacheable(req, force_cache) and not bypass_cache
        if cacheable:
            hit = self.store.cache_get(user, key, self.config.cache_ttl)
            if hit is not None:
                self.store.log(user, model=model, provider=None, cache_hit=True, status="cache_hit")
                return Result(hit, cache_hit=True)

        reserve = self.worst_case(routes, req)
        pk = self.store.reserve(user, reserve)  # raises BudgetExceeded
        errors: list[ProviderError] = []
        settled = False
        try:
            for route in routes:
                provider = self.config.providers[route.provider]
                try:
                    resp = provider.chat(route.model, {k: v for k, v in req.items() if k != "model"})
                except ProviderError as exc:
                    errors.append(exc)
                    if exc.retryable:
                        continue  # fall back to the next route
                    break         # 4xx: the request itself is bad; another provider won't help
                usage = resp.get("usage") or {}
                pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
                cost = route.cost(pt, ct) if usage else reserve  # no usage reported: charge the reservation
                resp["model"] = model
                self.store.settle(user, pk, reserve, cost)
                settled = True
                self.store.log(user, model=model, provider=route.provider, prompt_tokens=pt,
                               completion_tokens=ct, cost_micros=cost)
                if cacheable:
                    self.store.cache_put(user, key, resp)
                return Result(resp, provider=route.provider, cost_micros=cost,
                              attempts=[e.provider for e in errors] + [route.provider])
        except BaseException:
            if not settled:
                self.store.settle(user, pk, reserve, 0)
            raise
        self.store.settle(user, pk, reserve, 0)
        self.store.log(user, model=model, provider=None, status="error")
        raise AllProvidersFailed(errors)


__all__ = ["AllProvidersFailed", "BadRequest", "BudgetExceeded", "Guard", "RateLimited", "Result"]
