# tokenguard

[![CI](https://github.com/zensharma0512-a11y/tokenguard/actions/workflows/ci.yml/badge.svg)](https://github.com/zensharma0512-a11y/tokenguard/actions/workflows/ci.yml)

An **OpenAI-compatible LLM proxy that keeps your AI bill under control.** Give each user or team their own API key with a dollar budget and a rate limit. tokenguard checks every request against the budget *before* it reaches the provider, caches repeated answers, and falls back to another provider when one is down.

```
your app ──► tokenguard ──► OpenAI          (primary)
 (any OpenAI     │      └──► Anthropic       (fallback on 5xx / 429 / timeout)
  client lib)    │
                 └── budgets · rate limits · cache · usage log (SQLite)
```

Point any OpenAI client at it by changing one line:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="tg_…")   # a tokenguard key
client.chat.completions.create(model="fast", messages=[{"role": "user", "content": "Hi"}])
```

## What it does

| Feature | Details |
|---|---|
| **Per-user dollar budgets** | Daily or monthly, stored as integer micro-dollars (no float drift). Over budget returns `402` with how much is left |
| **No overspending under load** | Each request *reserves* its worst-case cost up front; the real cost from the provider's `usage` is charged afterwards. Check and reserve happen in one transaction |
| **Rate limits** | Token bucket per user (requests per minute), `429` with `Retry-After` |
| **Provider fallback** | Ordered routes per model. Retries the next provider on network errors, timeouts, `429` and `5xx`; does *not* retry on `4xx` (a bad request fails everywhere) |
| **Anthropic ⇄ OpenAI translation** | Call Anthropic models with the OpenAI request and response format |
| **Response cache** | Exact-match, per user, for `temperature: 0` requests (or force it with `x-tokenguard-cache: force`). Cache hits cost $0 |
| **Usage reporting** | Per-user requests, tokens, cost and cache hits, broken down by model |
| **Model aliases** | Clients ask for `fast` or `smart`; you change the real model behind it in config |
| **Safe forwarding** | Only known request fields are forwarded to providers |

## Why reservations matter

The naive approach is "check `spent < budget`, call the model, then add the cost." With 50 requests in flight at once, all 50 pass the check before any of them is charged, and the budget is blown.

`test_concurrent_requests_cannot_overspend` runs 50 overlapping requests against a budget that only pays for about 13. With reservations, spend stays under budget. Remove the reservation from the check and the test fails, which shows it is actually testing the race.

## Quick start

```bash
pip install "tokenguard[server] @ git+https://github.com/zensharma0512-a11y/tokenguard"
cp tokenguard.example.json tokenguard.json      # edit models, prices, providers
export OPENAI_API_KEY=… ANTHROPIC_API_KEY=… TOKENGUARD_ADMIN_KEY=change-me
tokenguard serve --config tokenguard.json --port 8000
```

Create a user:

```bash
curl -s localhost:8000/admin/users -H "authorization: Bearer $TOKENGUARD_ADMIN_KEY" \
  -d '{"name": "mobile-app", "budget_usd": 25, "period": "month", "rpm": 120}'
# → {"id": "u_…", "api_key": "tg_…", "remaining_usd": 25.0, …}   (key shown once)
```

## Config

```json
{
  "providers": {
    "openai":    { "type": "openai",    "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY" },
    "anthropic": { "type": "anthropic", "base_url": "https://api.anthropic.com", "api_key_env": "ANTHROPIC_API_KEY" }
  },
  "models": {
    "fast": { "routes": [
      { "provider": "openai",    "model": "…", "input_usd_per_mtok": 0.15, "output_usd_per_mtok": 0.60 },
      { "provider": "anthropic", "model": "…", "input_usd_per_mtok": 1.00, "output_usd_per_mtok": 5.00 }
    ]}
  },
  "limits": { "default_max_tokens": 1024, "max_tokens_limit": 8192 },
  "cache":  { "ttl_seconds": 3600 }
}
```

**Prices in the example are placeholders.** Copy current prices from each provider's pricing page. The `openai` type works with any OpenAI-compatible server: Groq, Together, OpenRouter, vLLM, Ollama.

## API

| | Route | Auth |
|---|---|---|
| `POST` | `/v1/chat/completions` | user key |
| `GET` | `/v1/budget` | user key: spent and remaining this period |
| `POST` | `/admin/users` | admin: `{name, budget_usd, period, rpm}` |
| `GET` | `/admin/users` | admin |
| `PATCH` | `/admin/users/{id}` | admin: `{budget_usd?, rpm?, active?}` |
| `GET` | `/admin/users/{id}/usage` | admin |

Response headers: `x-tokenguard-provider`, `x-tokenguard-cache: hit|miss`, `x-tokenguard-cost-usd`. Errors use the OpenAI error shape, so client libraries show them properly.

## Tests

27 tests, no network needed (providers are faked at the HTTP layer): cost maths, fallback order, Anthropic translation, 4xx-no-fallback, budget refusal before any provider call, period reset, the concurrency test above, rate limiting, cache TTL and per-user isolation, input validation, and the full HTTP API.

```bash
pip install -e ".[dev,server]"
pytest -v
```

## Limitations

- **No streaming yet** (`stream: true` returns `400`). Streaming means charging from the final usage chunk; it's the next feature.
- Text-only messages (no images or tool calls yet).
- The prompt estimate (≈3 characters per token) is deliberately pessimistic. It only sizes the reservation; you're charged real usage.
- If a provider reports more tokens than reserved, the real cost is still charged, so a budget can be exceeded by at most one request's difference.
- Single-node SQLite. For several instances, move the budget table to Postgres or Redis.

## License

MIT
