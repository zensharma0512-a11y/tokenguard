"""Budgets, rate limits, cache and usage log, all in one SQLite file.

Money is stored as integer micro-dollars (1 USD = 1_000_000) to avoid
floating-point drift when adding up thousands of small charges.

Budget enforcement uses *reservations*: before calling a provider we
reserve the worst-case cost of the request (prompt estimate + max_tokens
at the most expensive route's price). If that would exceed the budget,
the request is refused up front. When the response arrives we release
the reservation and charge the real cost from the provider's ``usage``.
Because reserve-and-check is one transaction, many concurrent requests
can't overspend a budget together.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

MICRO = 1_000_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    key_hash      TEXT NOT NULL UNIQUE,
    budget_micros INTEGER NOT NULL,
    period        TEXT NOT NULL,      -- 'day' | 'month'
    rpm           INTEGER NOT NULL,   -- requests per minute
    active        INTEGER NOT NULL DEFAULT 1,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS spend (
    user_id         TEXT NOT NULL,
    period_key      TEXT NOT NULL,
    spent_micros    INTEGER NOT NULL DEFAULT 0,
    reserved_micros INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, period_key)
);
CREATE TABLE IF NOT EXISTS buckets (
    user_id  TEXT PRIMARY KEY,
    tokens   REAL NOT NULL,
    updated  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS cache (
    user_id    TEXT NOT NULL,
    key        TEXT NOT NULL,
    response   TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (user_id, key)
);
CREATE TABLE IF NOT EXISTS usage_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           TEXT NOT NULL,
    at                REAL NOT NULL,
    model             TEXT NOT NULL,
    provider          TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_micros       INTEGER NOT NULL DEFAULT 0,
    cache_hit         INTEGER NOT NULL DEFAULT 0,
    status            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_user_at ON usage_log (user_id, at);
"""


class BudgetExceeded(Exception):
    def __init__(self, remaining_micros: int, needed_micros: int):
        super().__init__("budget exceeded")
        self.remaining_micros = remaining_micros
        self.needed_micros = needed_micros


class RateLimited(Exception):
    def __init__(self, retry_after: float):
        super().__init__("rate limited")
        self.retry_after = retry_after


@dataclass(frozen=True)
class User:
    id: str
    name: str
    budget_micros: int
    period: str
    rpm: int
    active: bool
    created_at: float


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def period_key(period: str, ts: float) -> str:
    d = datetime.fromtimestamp(ts, timezone.utc)
    if period == "day":
        return d.strftime("%Y-%m-%d")
    if period == "month":
        return d.strftime("%Y-%m")
    raise ValueError(f"unknown period {period!r}")


def to_micros(usd: float) -> int:
    return int(round(usd * MICRO))


class Store:
    def __init__(self, path: str | Path, *, clock: Callable[[], float] = time.time):
        self.path = str(path)
        self.clock = clock
        self._local = threading.local()
        self._all: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self.conn.executescript(SCHEMA)

    @property
    def conn(self) -> sqlite3.Connection:
        """One connection per thread: SQLite connections must not be shared mid-transaction."""
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
            with self._lock:
                self._all.append(c)
        return c

    def _tx(self):
        conn = self.conn

        class _T:
            def __enter__(s):
                conn.execute("BEGIN IMMEDIATE")
                return conn

            def __exit__(s, et, *_):
                conn.execute("ROLLBACK" if et else "COMMIT")

        return _T()

    # -- users ---------------------------------------------------------------

    def create_user(self, name: str, *, budget_usd: float, period: str = "month", rpm: int = 60) -> tuple[User, str]:
        """Create a user and return (user, api_key). The key is only returned here."""
        if not name:
            raise ValueError("name is required")
        if budget_usd < 0:
            raise ValueError("budget_usd must be >= 0")
        if period not in ("day", "month"):
            raise ValueError("period must be 'day' or 'month'")
        if rpm < 1:
            raise ValueError("rpm must be >= 1")
        key = "tg_" + secrets.token_urlsafe(24)
        user = User("u_" + secrets.token_hex(6), name, to_micros(budget_usd), period, rpm, True, self.clock())
        self.conn.execute(
            "INSERT INTO users (id, name, key_hash, budget_micros, period, rpm, active, created_at) VALUES (?,?,?,?,?,?,1,?)",
            (user.id, name, hash_key(key), user.budget_micros, period, rpm, user.created_at),
        )
        return user, key

    def _user(self, row) -> User:
        id_, name, budget, period, rpm, active, created = row
        return User(id_, name, budget, period, rpm, bool(active), created)

    _UCOLS = "id, name, budget_micros, period, rpm, active, created_at"

    def user_by_key(self, key: str) -> User | None:
        row = self.conn.execute(f"SELECT {self._UCOLS} FROM users WHERE key_hash = ?", (hash_key(key),)).fetchone()
        return self._user(row) if row else None

    def get_user(self, user_id: str) -> User | None:
        row = self.conn.execute(f"SELECT {self._UCOLS} FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._user(row) if row else None

    def users(self) -> list[User]:
        return [self._user(r) for r in self.conn.execute(f"SELECT {self._UCOLS} FROM users ORDER BY created_at")]

    def update_user(self, user_id: str, *, budget_usd: float | None = None, rpm: int | None = None,
                    active: bool | None = None) -> User:
        if budget_usd is not None and budget_usd < 0:
            raise ValueError("budget_usd must be >= 0")
        if rpm is not None and rpm < 1:
            raise ValueError("rpm must be >= 1")
        sets, args = [], []
        if budget_usd is not None:
            sets.append("budget_micros = ?"); args.append(to_micros(budget_usd))
        if rpm is not None:
            sets.append("rpm = ?"); args.append(rpm)
        if active is not None:
            sets.append("active = ?"); args.append(int(active))
        if sets:
            self.conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", (*args, user_id))
        user = self.get_user(user_id)
        if user is None:
            raise KeyError(user_id)
        return user

    # -- rate limit (token bucket) ------------------------------------------

    def take_rate_token(self, user: User) -> None:
        """Token bucket: capacity = rpm, refills at rpm/60 tokens per second."""
        now = self.clock()
        rate = user.rpm / 60.0
        with self._tx() as c:
            row = c.execute("SELECT tokens, updated FROM buckets WHERE user_id = ?", (user.id,)).fetchone()
            tokens = float(user.rpm) if row is None else min(float(user.rpm), row[0] + (now - row[1]) * rate)
            if tokens < 1.0:
                c.execute("INSERT OR REPLACE INTO buckets VALUES (?, ?, ?)", (user.id, tokens, now))
                raise RateLimited((1.0 - tokens) / rate)
            c.execute("INSERT OR REPLACE INTO buckets VALUES (?, ?, ?)", (user.id, tokens - 1.0, now))

    # -- budget --------------------------------------------------------------

    def spend(self, user: User) -> tuple[str, int, int]:
        """Return (period_key, spent_micros, reserved_micros) for the current period."""
        pk = period_key(user.period, self.clock())
        row = self.conn.execute(
            "SELECT spent_micros, reserved_micros FROM spend WHERE user_id = ? AND period_key = ?", (user.id, pk)
        ).fetchone()
        return pk, (row[0] if row else 0), (row[1] if row else 0)

    def reserve(self, user: User, micros: int) -> str:
        """Reserve ``micros`` against the budget, or raise BudgetExceeded. Returns the period key."""
        pk = period_key(user.period, self.clock())
        with self._tx() as c:
            c.execute("INSERT OR IGNORE INTO spend (user_id, period_key) VALUES (?, ?)", (user.id, pk))
            budget = c.execute("SELECT budget_micros FROM users WHERE id = ?", (user.id,)).fetchone()[0]
            spent, reserved = c.execute(
                "SELECT spent_micros, reserved_micros FROM spend WHERE user_id = ? AND period_key = ?", (user.id, pk)
            ).fetchone()
            remaining = budget - spent - reserved
            if micros > remaining:
                raise BudgetExceeded(max(0, remaining), micros)
            c.execute(
                "UPDATE spend SET reserved_micros = reserved_micros + ? WHERE user_id = ? AND period_key = ?",
                (micros, user.id, pk),
            )
        return pk

    def settle(self, user: User, pk: str, reserved: int, actual: int) -> None:
        """Release a reservation and charge the actual cost (0 for a failed request)."""
        with self._tx() as c:
            c.execute(
                "UPDATE spend SET reserved_micros = MAX(0, reserved_micros - ?), spent_micros = spent_micros + ? "
                "WHERE user_id = ? AND period_key = ?",
                (reserved, actual, user.id, pk),
            )

    # -- cache ---------------------------------------------------------------

    def cache_get(self, user: User, key: str, ttl: float) -> dict | None:
        row = self.conn.execute(
            "SELECT response, created_at FROM cache WHERE user_id = ? AND key = ?", (user.id, key)
        ).fetchone()
        if row and self.clock() - row[1] <= ttl:
            return json.loads(row[0])
        return None

    def cache_put(self, user: User, key: str, response: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (user_id, key, response, created_at) VALUES (?, ?, ?, ?)",
            (user.id, key, json.dumps(response), self.clock()),
        )

    def cache_purge(self, ttl: float) -> int:
        return self.conn.execute("DELETE FROM cache WHERE created_at < ?", (self.clock() - ttl,)).rowcount

    # -- usage log -----------------------------------------------------------

    def log(self, user: User, *, model: str, provider: str | None, prompt_tokens: int = 0,
            completion_tokens: int = 0, cost_micros: int = 0, cache_hit: bool = False, status: str = "ok") -> None:
        self.conn.execute(
            "INSERT INTO usage_log (user_id, at, model, provider, prompt_tokens, completion_tokens, cost_micros, "
            "cache_hit, status) VALUES (?,?,?,?,?,?,?,?,?)",
            (user.id, self.clock(), model, provider, prompt_tokens, completion_tokens, cost_micros, int(cache_hit), status),
        )

    def usage(self, user_id: str, *, since: float = 0.0) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
            "COALESCE(SUM(cost_micros),0), COALESCE(SUM(cache_hit),0) FROM usage_log WHERE user_id = ? AND at >= ?",
            (user_id, since),
        ).fetchone()
        by_model = {
            m: {"requests": n, "cost_usd": c / MICRO}
            for m, n, c in self.conn.execute(
                "SELECT model, COUNT(*), SUM(cost_micros) FROM usage_log WHERE user_id = ? AND at >= ? GROUP BY model",
                (user_id, since),
            )
        }
        return {
            "requests": row[0], "prompt_tokens": row[1], "completion_tokens": row[2],
            "cost_usd": row[3] / MICRO, "cache_hits": row[4], "by_model": by_model,
        }

    def close(self) -> None:
        with self._lock:
            for c in self._all:
                c.close()
            self._all.clear()
        self._local = threading.local()
