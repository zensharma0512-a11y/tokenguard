"""Tiny dependency-free ASGI test client (avoids pulling in httpx just for tests)."""

from __future__ import annotations

import asyncio
import json as _json
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass
class Resp:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self):
        return _json.loads(self.body)


def request(app, method: str, url: str, *, json=None, body: bytes | None = None, headers: dict | None = None) -> Resp:
    parts = urlsplit(url)
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    if json is not None:
        body = _json.dumps(json).encode()
        hdrs.setdefault("content-type", "application/json")
    body = body or b""
    hdrs["content-length"] = str(len(body))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": parts.path,
        "raw_path": parts.path.encode(),
        "query_string": parts.query.encode(),
        "root_path": "",
        "headers": [(k.encode(), v.encode()) for k, v in hdrs.items()],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    sent = {"done": False}
    out = {"status": None, "headers": {}, "body": b""}

    async def receive():
        if not sent["done"]:
            sent["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(3600)

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
            out["headers"] = {k.decode().lower(): v.decode() for k, v in message.get("headers", [])}
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    return Resp(out["status"], out["headers"], out["body"])
