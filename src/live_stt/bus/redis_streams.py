"""Redis Streams bus (redis-py 8.1.0; Redis 7 server).

* Streams with consumer groups (XREADGROUP/XACK); request streams are XDEL'd after ack so audio
  payloads do not accumulate; ``MAXLEN ~`` caps every stream as a safety net.
* On (re)start a consumer reads its own pending entries (id "0") before new ones (">"), and
  ``claim_stale`` (XAUTOCLAIM) takes over entries left pending by a dead consumer.
* Live fan-out via PUBLISH (pub/sub) for the WebSocket API.
* Every call retries on connection errors with backoff so a Redis restart does not crash
  workers; ``publish`` blocks (bounded by ``max_retry_s``) rather than dropping data.
"""

from __future__ import annotations

import json
import time
from typing import Callable

import redis
import redis.asyncio as aioredis

from live_stt.bus.base import Bus, Message, decode, encode
from live_stt.logging import get_logger

log = get_logger(__name__)

_RETRY_EXC = (redis.ConnectionError, redis.TimeoutError, ConnectionError, OSError)


def _sid(x) -> str:
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


class RedisStreamsBus(Bus):
    def __init__(
        self,
        url: str,
        maxlen: int = 100_000,
        maxlen_overrides: dict[str, int] | None = None,
        delete_on_ack: tuple[str, ...] = (),
        max_retry_s: float = 60.0,
    ) -> None:
        self.url = url
        self.maxlen = maxlen
        self.maxlen_overrides = maxlen_overrides or {}
        self.delete_on_ack = set(delete_on_ack)
        self.max_retry_s = max_retry_s
        self.r = redis.Redis.from_url(
            url, socket_timeout=30, socket_connect_timeout=5, health_check_interval=15
        )
        self._groups: set[tuple[str, str]] = set()
        self._pending_done: set[tuple[str, str, str]] = set()
        self._pending_cursor: dict[tuple[str, str, str], str] = {}

    # ------------------------------------------------------------ resilience
    def _retry(self, fn: Callable, *a, **kw):
        delay, waited = 0.2, 0.0
        while True:
            try:
                return fn(*a, **kw)
            except redis.ResponseError:
                raise
            except _RETRY_EXC as e:
                if waited >= self.max_retry_s:
                    raise
                log.warning("redis_retry", error=str(e), wait_s=round(delay, 2))
                time.sleep(delay)
                waited += delay
                delay = min(delay * 2, 5.0)
                self._groups.clear()  # server may have restarted without persistence

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except _RETRY_EXC:
            return False

    # --------------------------------------------------------------- streams
    def publish(self, stream: str, msg: Message) -> str:
        maxlen = self.maxlen_overrides.get(stream, self.maxlen)
        mid = self._retry(self.r.xadd, stream, encode(msg), maxlen=maxlen, approximate=True)
        return _sid(mid)

    def ensure_group(self, stream: str, group: str) -> None:
        if (stream, group) in self._groups:
            return
        try:
            self._retry(self.r.xgroup_create, stream, group, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        self._groups.add((stream, group))

    def consume(self, stream, group, consumer, count=10, block_ms=0):
        self.ensure_group(stream, group)
        key = (stream, group, consumer)
        if key not in self._pending_done:
            # Our own un-acked deliveries from a previous life first, walking the pending list
            # with a cursor (re-reading from "0" would return the same entries every call).
            cur = self._pending_cursor.get(key, "0")
            res = self._retry(self.r.xreadgroup, group, consumer, {stream: cur}, count=count)
            raw = res[0][1] if res else []
            if raw:
                self._pending_cursor[key] = _sid(raw[-1][0])
            entries = [(i, f) for i, f in raw if f]  # deleted entries come back empty
            if entries:
                return [(_sid(i), decode(f)) for i, f in entries]
            if not raw:
                self._pending_done.add(key)
                self._pending_cursor.pop(key, None)
        try:
            res = self._retry(
                self.r.xreadgroup, group, consumer, {stream: ">"}, count=count,
                block=block_ms if block_ms > 0 else None,
            )
        except redis.ResponseError as e:
            if "NOGROUP" in str(e):  # stream deleted / server flushed
                self._groups.discard((stream, group))
                self._pending_done.discard(key)
                self.ensure_group(stream, group)
                return []
            raise
        out = []
        for _s, entries in res or []:
            for i, f in entries:
                if f:
                    out.append((_sid(i), decode(f)))
        return out

    def claim_stale(self, stream, group, consumer, min_idle_ms=60_000, count=50):
        """Take over entries pending on dead consumers for longer than ``min_idle_ms``."""
        self.ensure_group(stream, group)
        res = self._retry(
            self.r.xautoclaim, stream, group, consumer, min_idle_time=min_idle_ms,
            start_id="0-0", count=count,
        )
        entries = res[1] if res and len(res) > 1 else []
        return [(_sid(i), decode(f)) for i, f in entries if f]

    def ack(self, stream, group, ids):
        if not ids:
            return
        self._retry(self.r.xack, stream, group, *ids)
        if stream in self.delete_on_ack:
            self._retry(self.r.xdel, stream, *ids)

    def depth(self, stream, group):
        try:
            info = self._retry(self.r.xinfo_groups, stream)
        except redis.ResponseError:
            return 0
        for g in info:
            name = _sid(g.get("name", g.get(b"name", b"")))
            if name == group:
                lag = g.get("lag") or 0
                return int(g.get("pending", 0)) + int(lag)
        return int(self._retry(self.r.xlen, stream))

    def oldest_pending_age_s(self, stream: str, group: str) -> float:
        """Age (by stream id timestamp) of the oldest entry not yet acked by the group."""
        try:
            summary = self._retry(self.r.xpending, stream, group)
        except redis.ResponseError:
            return 0.0
        mids = []
        if summary and summary.get("pending"):
            mids.append(_sid(summary["min"]))
        info = self._retry(self.r.xinfo_groups, stream)
        for g in info:
            if _sid(g.get("name")) == group:
                last = _sid(g.get("last-delivered-id"))
                nxt = self._retry(self.r.xrange, stream, min=f"({last}", max="+", count=1)
                if nxt:
                    mids.append(_sid(nxt[0][0]))
        if not mids:
            return 0.0
        ms = min(int(m.split("-")[0]) for m in mids)
        return max(0.0, time.time() - ms / 1000)

    # ------------------------------------------------------------- broadcast
    def broadcast(self, channel, msg):
        m = {k: v for k, v in msg.items() if k != "pcm"}
        self._retry(self.r.publish, channel, json.dumps(m, ensure_ascii=False))

    def set_value(self, key, value, ttl_s=None):
        self._retry(self.r.set, key, json.dumps(value, ensure_ascii=False), ex=ttl_s)

    def get_value(self, key):
        v = self._retry(self.r.get, key)
        return json.loads(v) if v else None

    def close(self):
        try:
            self.r.close()
        except Exception:  # noqa: BLE001
            pass


class AsyncRedisStreams:
    """Minimal asyncio counterpart used by the persister and the API."""

    def __init__(self, url: str) -> None:
        self.r = aioredis.Redis.from_url(url, socket_connect_timeout=5, health_check_interval=15)
        self._groups: set[tuple[str, str]] = set()
        self._pending_done: set[tuple[str, str, str]] = set()
        self._cursor: dict[tuple[str, str, str], str] = {}

    async def ensure_group(self, stream, group):
        if (stream, group) in self._groups:
            return
        try:
            await self.r.xgroup_create(stream, group, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        self._groups.add((stream, group))

    async def consume(self, streams: list[str], group, consumer, count=100, block_ms=200):
        for s in streams:
            await self.ensure_group(s, group)
        out: list[tuple[str, str, Message]] = []
        pend = [s for s in streams if (s, group, consumer) not in self._pending_done]
        if pend:
            cur = {s: self._cursor.get((s, group, consumer), "0") for s in pend}
            res = await self.r.xreadgroup(group, consumer, cur, count=count)
            seen = set()
            for s, entries in res or []:
                ss = _sid(s)
                if entries:
                    seen.add(ss)
                    self._cursor[(ss, group, consumer)] = _sid(entries[-1][0])
                out += [(ss, _sid(i), decode(f)) for i, f in entries if f]
            for s in pend:
                if s not in seen:
                    self._pending_done.add((s, group, consumer))
                    self._cursor.pop((s, group, consumer), None)
            if out:
                return out
        try:
            res = await self.r.xreadgroup(
                group, consumer, {s: ">" for s in streams}, count=count, block=block_ms
            )
        except redis.ResponseError as e:
            if "NOGROUP" in str(e):
                self._groups.clear()
                return []
            raise
        for s, entries in res or []:
            out += [(_sid(s), _sid(i), decode(f)) for i, f in entries if f]
        return out

    async def ack(self, stream, group, ids):
        if ids:
            await self.r.xack(stream, group, *ids)

    def reset_groups(self) -> None:
        self._groups.clear()
        self._pending_done.clear()
        self._cursor.clear()

    async def close(self):
        await self.r.aclose()
