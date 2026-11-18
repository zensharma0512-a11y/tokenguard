"""Load the routing and pricing config."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .providers import HTTP, Provider, urllib_http
from .store import to_micros


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Route:
    provider: str
    model: str
    input_micros_per_mtok: int   # micro-dollars per 1M input tokens
    output_micros_per_mtok: int

    def cost(self, prompt_tokens: int, completion_tokens: int) -> int:
        # Round up so many tiny requests can never add up to more than we charged.
        total = prompt_tokens * self.input_micros_per_mtok + completion_tokens * self.output_micros_per_mtok
        return -(-total // 1_000_000)


@dataclass
class Config:
    providers: dict[str, Provider]
    models: dict[str, list[Route]]          # public model name -> ordered fallback routes
    default_max_tokens: int = 1024
    max_tokens_limit: int = 8192
    cache_ttl: float = 3600.0
    extra: dict[str, Any] = field(default_factory=dict)


def load(source: str | Path | dict, *, http: HTTP = urllib_http, env: dict | None = None) -> Config:
    env = os.environ if env is None else env
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text())

    providers: dict[str, Provider] = {}
    for name, p in (data.get("providers") or {}).items():
        ptype = p.get("type")
        if ptype not in ("openai", "anthropic"):
            raise ConfigError(f"provider {name}: type must be 'openai' or 'anthropic'")
        if not p.get("base_url"):
            raise ConfigError(f"provider {name}: base_url is required")
        key = p.get("api_key") or env.get(p.get("api_key_env", ""), "")
        if not key:
            raise ConfigError(f"provider {name}: set api_key_env to an environment variable that holds the key")
        providers[name] = Provider(name, ptype, p["base_url"], key, float(p.get("timeout", 60)), http)

    models: dict[str, list[Route]] = {}
    for public, m in (data.get("models") or {}).items():
        routes = []
        for r in m.get("routes", []):
            if r.get("provider") not in providers:
                raise ConfigError(f"model {public}: unknown provider {r.get('provider')!r}")
            try:
                inp, out = float(r["input_usd_per_mtok"]), float(r["output_usd_per_mtok"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError(f"model {public}: each route needs input_usd_per_mtok and output_usd_per_mtok") from exc
            if inp < 0 or out < 0 or not r.get("model"):
                raise ConfigError(f"model {public}: bad route {r}")
            routes.append(Route(r["provider"], r["model"], to_micros(inp), to_micros(out)))
        if not routes:
            raise ConfigError(f"model {public}: needs at least one route")
        models[public] = routes
    if not models:
        raise ConfigError("config defines no models")

    limits = data.get("limits", {})
    return Config(
        providers=providers,
        models=models,
        default_max_tokens=int(limits.get("default_max_tokens", 1024)),
        max_tokens_limit=int(limits.get("max_tokens_limit", 8192)),
        cache_ttl=float(data.get("cache", {}).get("ttl_seconds", 3600)),
    )
