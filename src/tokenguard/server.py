"""OpenAI-compatible proxy with budgets, caching and fallback. Run with ``tokenguard serve``.

Client routes (``Authorization: Bearer tg_...``, a key from the admin API):
    POST /v1/chat/completions
    GET  /v1/budget

Admin routes (``Authorization: Bearer $TOKENGUARD_ADMIN_KEY``):
    POST  /admin/users              {"name", "budget_usd", "period", "rpm"}
    GET   /admin/users
    PATCH /admin/users/{id}         {"budget_usd"?, "rpm"?, "active"?}
    GET   /admin/users/{id}/usage
"""

from __future__ import annotations

import contextlib
import hmac
import json
import math
import os
from dataclasses import asdict

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .config import Config
from .guard import AllProvidersFailed, BadRequest, Guard
from .store import MICRO, BudgetExceeded, RateLimited, Store, User


def _err(status: int, message: str, code: str, headers: dict | None = None) -> JSONResponse:
    # Same error shape as the OpenAI API, so existing client libraries surface it cleanly.
    return JSONResponse({"error": {"message": message, "type": code, "code": code}}, status_code=status, headers=headers)


def _bearer(request: Request) -> str:
    h = request.headers.get("authorization", "")
    return h[7:] if h.startswith("Bearer ") else ""


async def _json(request: Request):
    try:
        return await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise BadRequest("body must be valid JSON") from None


def _user_view(store: Store, u: User) -> dict:
    pk, spent, reserved = store.spend(u)
    return {
        **{k: v for k, v in asdict(u).items() if k != "budget_micros"},
        "budget_usd": u.budget_micros / MICRO,
        "period_key": pk,
        "spent_usd": spent / MICRO,
        "remaining_usd": max(0, u.budget_micros - spent - reserved) / MICRO,
    }


def create_app(store: Store, config: Config, admin_key: str) -> Starlette:
    if not admin_key:
        raise ValueError("admin_key is required")
    guard = Guard(store, config)

    def client(request: Request) -> User | None:
        key = _bearer(request)
        user = store.user_by_key(key) if key else None
        return user if user and user.active else None

    def is_admin(request: Request) -> bool:
        return hmac.compare_digest(_bearer(request).encode(), admin_key.encode())

    async def chat(request: Request):
        user = client(request)
        if user is None:
            return _err(401, "invalid or missing API key", "invalid_api_key")
        try:
            body = await _json(request)
            result = await run_in_threadpool(
                guard.handle, user, body,
                force_cache=request.headers.get("x-tokenguard-cache") == "force",
                bypass_cache=request.headers.get("x-tokenguard-cache") == "bypass",
            )
        except BadRequest as exc:
            return _err(400, str(exc), "invalid_request_error")
        except RateLimited as exc:
            wait = max(1, math.ceil(exc.retry_after))
            return _err(429, f"rate limit exceeded, retry in {wait}s", "rate_limit_exceeded", {"retry-after": str(wait)})
        except BudgetExceeded as exc:
            return _err(
                402,
                f"budget exceeded: request may cost up to ${exc.needed_micros / MICRO:.6f}, "
                f"${exc.remaining_micros / MICRO:.6f} left this period",
                "budget_exceeded",
            )
        except AllProvidersFailed as exc:
            last = exc.errors[-1] if exc.errors else None
            if last is not None and last.status is not None and 400 <= last.status < 500 and not last.retryable:
                return _err(last.status, str(last), "provider_rejected_request")
            return _err(502, f"all providers failed: {exc}", "upstream_unavailable")
        headers = {
            "x-tokenguard-cache": "hit" if result.cache_hit else "miss",
            "x-tokenguard-cost-usd": f"{result.cost_micros / MICRO:.6f}",
        }
        if result.provider:
            headers["x-tokenguard-provider"] = result.provider
        return JSONResponse(result.response, headers=headers)

    async def budget(request: Request):
        user = client(request)
        if user is None:
            return _err(401, "invalid or missing API key", "invalid_api_key")
        return JSONResponse(_user_view(store, user))

    async def admin_create(request: Request):
        if not is_admin(request):
            return _err(401, "admin key required", "invalid_api_key")
        try:
            b = await _json(request)
            if not isinstance(b, dict):
                raise BadRequest("body must be a JSON object")
            user, key = store.create_user(
                b.get("name", ""), budget_usd=float(b.get("budget_usd", 0)),
                period=b.get("period", "month"), rpm=int(b.get("rpm", 60)),
            )
        except (BadRequest, ValueError, TypeError) as exc:
            return _err(400, str(exc), "invalid_request_error")
        return JSONResponse({**_user_view(store, user), "api_key": key}, status_code=201)

    async def admin_list(request: Request):
        if not is_admin(request):
            return _err(401, "admin key required", "invalid_api_key")
        return JSONResponse({"data": [_user_view(store, u) for u in store.users()]})

    async def admin_update(request: Request):
        if not is_admin(request):
            return _err(401, "admin key required", "invalid_api_key")
        uid = request.path_params["id"]
        if store.get_user(uid) is None:
            return _err(404, "user not found", "not_found")
        try:
            b = await _json(request)
            if not isinstance(b, dict):
                raise BadRequest("body must be a JSON object")
            active = b.get("active")
            if active is not None and not isinstance(active, bool):
                raise BadRequest("active must be true or false")
            user = store.update_user(
                uid,
                budget_usd=float(b["budget_usd"]) if "budget_usd" in b else None,
                rpm=int(b["rpm"]) if "rpm" in b else None,
                active=active,
            )
        except (BadRequest, ValueError, TypeError) as exc:
            return _err(400, str(exc), "invalid_request_error")
        return JSONResponse(_user_view(store, user))

    async def admin_usage(request: Request):
        if not is_admin(request):
            return _err(401, "admin key required", "invalid_api_key")
        uid = request.path_params["id"]
        if store.get_user(uid) is None:
            return _err(404, "user not found", "not_found")
        return JSONResponse(store.usage(uid))

    async def healthz(request: Request):
        return JSONResponse({"ok": True, "models": sorted(config.models)})

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        store.close()

    return Starlette(
        routes=[
            Route("/healthz", healthz),
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/v1/budget", budget, methods=["GET"]),
            Route("/admin/users", admin_create, methods=["POST"]),
            Route("/admin/users", admin_list, methods=["GET"]),
            Route("/admin/users/{id}", admin_update, methods=["PATCH"]),
            Route("/admin/users/{id}/usage", admin_usage, methods=["GET"]),
        ],
        lifespan=lifespan,
    )


def main() -> None:
    import argparse

    import uvicorn

    from .config import load

    p = argparse.ArgumentParser(prog="tokenguard")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--config", default=os.environ.get("TOKENGUARD_CONFIG", "tokenguard.json"))
    s.add_argument("--db", default=os.environ.get("TOKENGUARD_DB", "tokenguard.db"))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    admin = os.environ.get("TOKENGUARD_ADMIN_KEY")
    if not admin:
        raise SystemExit("set TOKENGUARD_ADMIN_KEY")
    app = create_app(Store(args.db), load(args.config), admin)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
