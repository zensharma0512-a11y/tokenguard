import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tokenguard import (
    MICRO, AllProvidersFailed, BadRequest, BudgetExceeded, ConfigError, Guard, RateLimited, Store,
    estimate_prompt_tokens, load,
)
from tokenguard.config import Route
from tokenguard.server import create_app

from asgi import request

OPENAI_URL = "https://oa.test/v1/chat/completions"
ANTHROPIC_URL = "https://an.test/v1/messages"


class Clock:
    def __init__(self, t=1_763_424_000.0):  # 2025-11-18 00:00 UTC
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def openai_ok(text="hi", pt=10, ct=5):
    return 200, json.dumps({
        "id": "chatcmpl-1", "object": "chat.completion", "model": "real-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }).encode()


def anthropic_ok(text="hello from anthropic", pt=12, ct=6):
    return 200, json.dumps({
        "id": "msg_1", "type": "message", "model": "anthropic-model", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": pt, "output_tokens": ct},
    }).encode()


class FakeHTTP:
    def __init__(self):
        self.queues: dict[str, list] = {}
        self.calls: list[tuple[str, dict, dict]] = []
        self.lock = threading.Lock()

    def queue(self, url, *responses):
        self.queues.setdefault(url, []).extend(responses)

    def __call__(self, url, headers, body, timeout):
        with self.lock:
            self.calls.append((url, headers, json.loads(body)))
            q = self.queues.get(url) or []
            r = q.pop(0) if q else openai_ok()
        if isinstance(r, Exception):
            raise r
        return r


CONFIG = {
    "providers": {
        "oa": {"type": "openai", "base_url": "https://oa.test/v1", "api_key": "sk-oa"},
        "an": {"type": "anthropic", "base_url": "https://an.test", "api_key": "sk-an"},
    },
    "models": {
        "smart": {"routes": [
            {"provider": "oa", "model": "gpt-real", "input_usd_per_mtok": 2.0, "output_usd_per_mtok": 8.0},
            {"provider": "an", "model": "anthropic-real", "input_usd_per_mtok": 3.0, "output_usd_per_mtok": 15.0},
        ]},
    },
    "limits": {"default_max_tokens": 100, "max_tokens_limit": 1000},
}


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    http = FakeHTTP()
    store = Store(tmp_path / "tg.db", clock=clock)
    cfg = load(CONFIG, http=http, env={})
    guard = Guard(store, cfg)
    user, key = store.create_user("alice", budget_usd=1.0, period="day", rpm=1000)
    yield clock, http, store, guard, user, key
    store.close()


def chat(content="hello", **kw):
    return {"model": "smart", "messages": [{"role": "user", "content": content}], **kw}


# -- pricing -----------------------------------------------------------------


def test_route_cost_rounds_up_in_micro_dollars():
    r = Route("p", "m", input_micros_per_mtok=2 * MICRO, output_micros_per_mtok=8 * MICRO)
    assert r.cost(1_000_000, 0) == 2 * MICRO
    assert r.cost(10, 5) == 10 * 2 + 5 * 8  # $2/Mtok * 10 tokens = 20 micro-dollars
    tiny = Route("p", "m", input_micros_per_mtok=1, output_micros_per_mtok=0)
    assert tiny.cost(1, 0) == 1  # a fraction of a micro-dollar rounds up, never down to 0


def test_estimate_is_an_upper_bound_for_typical_text():
    msgs = [{"role": "user", "content": "The quick brown fox jumps over the lazy dog. " * 50}]
    # English averages ~4 chars/token; our estimate assumes 3, so it should be well above chars/4.
    assert estimate_prompt_tokens(msgs) > len(msgs[0]["content"]) / 4


# -- happy path and accounting ----------------------------------------------


def test_successful_call_charges_actual_usage(env):
    clock, http, store, guard, user, _ = env
    http.queue(OPENAI_URL, openai_ok(pt=1000, ct=500))
    res = guard.handle(user, chat())
    assert res.provider == "oa" and not res.cache_hit
    assert res.response["model"] == "smart"  # callers see the public alias
    assert res.cost_micros == 1000 * 2 + 500 * 8  # micro-dollars
    _, spent, reserved = store.spend(user)
    assert spent == 6000 and reserved == 0
    url, headers, sent = http.calls[0]
    assert headers["authorization"] == "Bearer sk-oa"
    assert sent["model"] == "gpt-real" and sent["max_tokens"] == 100


def test_unknown_fields_are_not_forwarded(env):
    _, http, _, guard, user, _ = env
    guard.handle(user, chat(temperature=0.2, functions=[{"x": 1}], api_key="steal-me"))
    sent = http.calls[0][2]
    assert "functions" not in sent and "api_key" not in sent and sent["temperature"] == 0.2


# -- fallback ---------------------------------------------------------------


def test_falls_back_on_5xx_and_translates_anthropic(env):
    _, http, store, guard, user, _ = env
    http.queue(OPENAI_URL, (503, b"overloaded"))
    http.queue(ANTHROPIC_URL, anthropic_ok(pt=100, ct=50))
    res = guard.handle(user, chat(content="hi", messages=[
        {"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]))
    assert res.provider == "an" and res.attempts == ["oa", "an"]
    assert res.response["choices"][0]["message"]["content"] == "hello from anthropic"
    assert res.response["usage"] == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    assert res.cost_micros == 100 * 3 + 50 * 15
    url, headers, sent = http.calls[1]
    assert headers["x-api-key"] == "sk-an" and headers["anthropic-version"]
    assert sent["system"] == "be brief" and sent["messages"] == [{"role": "user", "content": "hi"}]


def test_falls_back_on_network_error_and_429(env):
    _, http, _, guard, user, _ = env
    http.queue(OPENAI_URL, ConnectionRefusedError("refused"), (429, b"slow down"))
    http.queue(ANTHROPIC_URL, anthropic_ok(), anthropic_ok())
    assert guard.handle(user, chat("a")).provider == "an"
    assert guard.handle(user, chat("b")).provider == "an"


def test_does_not_fall_back_on_4xx(env):
    _, http, store, guard, user, _ = env
    http.queue(OPENAI_URL, (400, b'{"error": "bad request"}'))
    with pytest.raises(AllProvidersFailed) as ei:
        guard.handle(user, chat())
    assert ei.value.errors[0].status == 400
    assert [c[0] for c in http.calls] == [OPENAI_URL]  # anthropic never called
    assert store.spend(user)[1:] == (0, 0)  # nothing charged, reservation released


def test_all_providers_down_releases_reservation(env):
    _, http, store, guard, user, _ = env
    http.queue(OPENAI_URL, (500, b"x"))
    http.queue(ANTHROPIC_URL, (529, b"overloaded"))
    with pytest.raises(AllProvidersFailed):
        guard.handle(user, chat())
    assert store.spend(user)[1:] == (0, 0)


# -- budgets ---------------------------------------------------------------


def test_budget_blocks_before_calling_provider(env):
    _, http, store, guard, _, _ = env
    poor, _ = store.create_user("bob", budget_usd=0.0001, period="day", rpm=1000)  # 100 micro-dollars
    with pytest.raises(BudgetExceeded) as ei:
        guard.handle(poor, chat(max_tokens=1000))  # worst case 1000 * 15 micros at the pricier route
    assert ei.value.needed_micros > 100
    assert http.calls == []


def test_budget_resets_next_period(env):
    clock, http, store, guard, _, _ = env
    u, _ = store.create_user("carol", budget_usd=0.01, period="day", rpm=1000)  # 10_000 micros
    http.queue(OPENAI_URL, openai_ok(pt=1000, ct=1000))  # costs 10_000 micros
    guard.handle(u, chat(max_tokens=10))
    with pytest.raises(BudgetExceeded):
        guard.handle(u, chat(max_tokens=10))
    clock.advance(86_400)
    guard.handle(u, chat(max_tokens=10))


def test_concurrent_requests_cannot_overspend(env):
    """50 overlapping requests against a budget that pays for ~13 must not overspend it.

    Each response really costs 1200 micro-dollars (200 prompt + 100 completion tokens).
    A naive "check spent < budget, then call, then charge" lets every in-flight request
    pass the check at once. Reservations make the check and the hold one atomic step.
    """
    import time as _time

    _, http, store, guard, _, _ = env
    u, _ = store.create_user("dave", budget_usd=0.016, period="day", rpm=10_000)  # 16_000 micros

    def slow(url, headers, body, timeout):
        _time.sleep(0.02)  # keep many requests in flight at the same time
        return openai_ok(pt=200, ct=100)

    for p in guard.config.providers.values():
        p.http = slow
    ok, refused = [], []

    def one(i):
        try:
            guard.handle(u, chat(f"q{i}"))
            ok.append(i)
        except BudgetExceeded:
            refused.append(i)

    with ThreadPoolExecutor(max_workers=25) as pool:
        list(pool.map(one, range(50)))
    _, spent, reserved = store.spend(u)
    assert reserved == 0
    assert spent == 1200 * len(ok) <= 16_000
    assert len(ok) >= 9 and len(ok) + len(refused) == 50


# -- rate limit and cache -----------------------------------------------------


def test_rate_limit_token_bucket(env):
    clock, _, store, guard, _, _ = env
    u, _ = store.create_user("erin", budget_usd=10, period="day", rpm=2)
    guard.handle(u, chat("1"))
    guard.handle(u, chat("2"))
    with pytest.raises(RateLimited) as ei:
        guard.handle(u, chat("3"))
    assert 29 <= ei.value.retry_after <= 30
    clock.advance(30)
    guard.handle(u, chat("4"))


def test_cache_for_deterministic_requests_only(env):
    clock, http, store, guard, user, _ = env
    a = guard.handle(user, chat("same", temperature=0))
    b = guard.handle(user, chat("same", temperature=0))
    assert not a.cache_hit and b.cache_hit and b.response == a.response
    assert len(http.calls) == 1
    guard.handle(user, chat("same", temperature=0.7))
    guard.handle(user, chat("same", temperature=0.7))
    assert len(http.calls) == 3  # non-zero temperature is never cached
    clock.advance(3601)
    assert not guard.handle(user, chat("same", temperature=0)).cache_hit  # TTL expired


def test_cache_is_per_user(env):
    _, http, store, guard, user, _ = env
    other, _ = store.create_user("frank", budget_usd=1, period="day", rpm=100)
    guard.handle(user, chat("secret question", temperature=0))
    assert not guard.handle(other, chat("secret question", temperature=0)).cache_hit


# -- validation and config -----------------------------------------------------


@pytest.mark.parametrize("body,msg", [
    ([], "JSON object"),
    ({"model": "nope", "messages": [{"role": "user", "content": "x"}]}, "unknown model"),
    ({"model": "smart", "messages": []}, "non-empty"),
    ({"model": "smart", "messages": [{"role": "tool", "content": "x"}]}, "role"),
    ({"model": "smart", "messages": [{"role": "user", "content": 5}]}, "string"),
    ({"model": "smart", "messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, "positive"),
    ({"model": "smart", "messages": [{"role": "user", "content": "x"}], "max_tokens": 5000}, "exceed"),
    ({"model": "smart", "messages": [{"role": "user", "content": "x"}], "stream": True}, "streaming"),
])
def test_validation(env, body, msg):
    *_, guard, user, _ = env
    with pytest.raises(BadRequest, match=msg):
        guard.handle(user, body)


def test_config_errors():
    with pytest.raises(ConfigError):
        load({"providers": {}, "models": {}}, env={})
    with pytest.raises(ConfigError, match="api_key_env"):
        load({"providers": {"x": {"type": "openai", "base_url": "u", "api_key_env": "MISSING"}},
              "models": {"m": {"routes": []}}}, env={})
    with pytest.raises(ConfigError, match="unknown provider"):
        load({**CONFIG, "models": {"m": {"routes": [{"provider": "zzz", "model": "a",
              "input_usd_per_mtok": 1, "output_usd_per_mtok": 1}]}}}, env={})
    with pytest.raises(ConfigError, match="usd_per_mtok"):
        load({**CONFIG, "models": {"m": {"routes": [{"provider": "oa", "model": "a"}]}}}, env={})


def test_example_config_loads():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "tokenguard.example.json"
    cfg = load(path, env={"OPENAI_API_KEY": "a", "ANTHROPIC_API_KEY": "b", "OLLAMA_KEY": "c"})
    assert set(cfg.models) == {"fast", "free"}


# -- HTTP API ------------------------------------------------------------------

ADMIN = {"authorization": "Bearer admin-secret"}


@pytest.fixture
def api(env):
    _, http, store, guard, _, _ = env
    return create_app(store, guard.config, "admin-secret"), http, store


def test_api_end_to_end(api):
    app, http, store = api
    r = request(app, "POST", "/admin/users", json={"name": "zoe", "budget_usd": 0.5, "period": "month", "rpm": 100}, headers=ADMIN)
    assert r.status == 201
    key, uid = r.json()["api_key"], r.json()["id"]
    auth = {"authorization": f"Bearer {key}"}

    http.queue(OPENAI_URL, openai_ok(pt=1000, ct=1000))
    r = request(app, "POST", "/v1/chat/completions", json=chat(), headers=auth)
    assert r.status == 200 and r.json()["choices"][0]["message"]["content"] == "hi"
    assert r.headers["x-tokenguard-provider"] == "oa" and r.headers["x-tokenguard-cache"] == "miss"
    assert r.headers["x-tokenguard-cost-usd"] == "0.010000"

    b = request(app, "GET", "/v1/budget", headers=auth).json()
    assert b["spent_usd"] == 0.01 and b["remaining_usd"] == 0.49

    usage = request(app, "GET", f"/admin/users/{uid}/usage", headers=ADMIN).json()
    assert usage["requests"] == 1 and usage["cost_usd"] == 0.01 and usage["by_model"]["smart"]["requests"] == 1

    assert request(app, "PATCH", f"/admin/users/{uid}", json={"active": False}, headers=ADMIN).status == 200
    assert request(app, "POST", "/v1/chat/completions", json=chat(), headers=auth).status == 401


def test_api_error_mapping(api):
    app, http, store = api
    u, key = store.create_user("y", budget_usd=0.000001, period="day", rpm=1)
    auth = {"authorization": f"Bearer {key}"}
    assert request(app, "POST", "/v1/chat/completions", json=chat()).status == 401
    assert request(app, "POST", "/v1/chat/completions", body=b"{bad", headers=auth).status == 400
    r = request(app, "POST", "/v1/chat/completions", json=chat(), headers=auth)
    assert r.status == 402 and r.json()["error"]["code"] == "budget_exceeded"
    r = request(app, "POST", "/v1/chat/completions", json=chat(), headers=auth)
    assert r.status == 429 and int(r.headers["retry-after"]) >= 1

    rich, rkey = store.create_user("z", budget_usd=5, period="day", rpm=100)
    rauth = {"authorization": f"Bearer {rkey}"}
    http.queue(OPENAI_URL, (500, b"x"))
    http.queue(ANTHROPIC_URL, (500, b"x"))
    assert request(app, "POST", "/v1/chat/completions", json=chat(), headers=rauth).status == 502
    http.queue(OPENAI_URL, (400, b"context too long"))
    assert request(app, "POST", "/v1/chat/completions", json=chat(), headers=rauth).status == 400


def test_admin_routes_need_admin_key(api):
    app, _, _ = api
    assert request(app, "GET", "/admin/users").status == 401
    assert request(app, "GET", "/admin/users", headers={"authorization": "Bearer nope"}).status == 401
    assert request(app, "POST", "/admin/users", json={"name": ""}, headers=ADMIN).status == 400
    assert request(app, "POST", "/admin/users", json={"name": "x", "period": "year"}, headers=ADMIN).status == 400
    assert request(app, "PATCH", "/admin/users/u_nope", json={}, headers=ADMIN).status == 404
