"""Message bus interface + codec + in-memory implementation (tests).

Messages are dicts with JSON-serialisable values, plus an optional binary ``pcm`` payload
(int16 little-endian mono 16 kHz) carried out-of-band (not base64) to keep messages small.

Semantics required from every implementation (Redis Streams now, Kafka later):
* ``publish(stream, msg)`` appends durably and returns an id.
* ``consume(stream, group, consumer, count, block_ms)`` returns undelivered messages for the group;
  on restart a consumer first receives its own **pending** (delivered, un-acked) messages again,
  so crashed work is retried (at-least-once).
* ``ack(stream, group, ids)`` marks done (and may delete the entry to bound memory).
* ``broadcast(channel, msg)`` / live subscription: fire-and-forget fan-out (WebSocket).
"""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np

Message = dict[str, Any]


def encode(msg: Message) -> dict[bytes, bytes]:
    m = dict(msg)
    pcm = m.pop("pcm", None)
    fields = {b"d": json.dumps(m, ensure_ascii=False, separators=(",", ":")).encode()}
    if pcm is not None:
        fields[b"pcm"] = bytes(pcm)
    return fields


def decode(fields: dict) -> Message:
    raw = fields.get(b"d", fields.get("d"))
    msg = json.loads(raw)
    pcm = fields.get(b"pcm", fields.get("pcm"))
    if pcm is not None:
        msg["pcm"] = pcm
    return msg


def f32_to_pcm16(x: np.ndarray) -> bytes:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def pcm16_to_f32(b: bytes) -> np.ndarray:
    if len(b) % 2:
        b = b[:-1]
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


class Bus(ABC):
    @abstractmethod
    def publish(self, stream: str, msg: Message) -> str: ...

    @abstractmethod
    def consume(
        self, stream: str, group: str, consumer: str, count: int = 10, block_ms: int = 0
    ) -> list[tuple[str, Message]]: ...

    @abstractmethod
    def ack(self, stream: str, group: str, ids: list[str]) -> None: ...

    @abstractmethod
    def broadcast(self, channel: str, msg: Message) -> None: ...

    @abstractmethod
    def depth(self, stream: str, group: str) -> int:
        """Messages not yet acked by the group (pending + undelivered), best effort."""

    def set_value(self, key: str, value: Message, ttl_s: int | None = None) -> None:
        raise NotImplementedError

    def get_value(self, key: str) -> Message | None:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class InMemoryBus(Bus):
    """Thread-safe single-process bus with the same delivery semantics (for tests)."""

    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._streams: dict[str, list[tuple[str, Message]]] = defaultdict(list)
        self._cursor: dict[tuple[str, str], int] = defaultdict(int)
        self._pending: dict[tuple[str, str], dict[str, tuple[str, Message]]] = defaultdict(dict)
        self._acked: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._seq = 0
        self.broadcasts: list[tuple[str, Message]] = []
        self._kv: dict[str, Message] = {}

    def publish(self, stream: str, msg: Message) -> str:
        with self._lock:
            self._seq += 1
            mid = f"{int(time.time() * 1000)}-{self._seq}"
            self._streams[stream].append((mid, decode(encode(msg))))
            self._lock.notify_all()
            return mid

    def consume(self, stream, group, consumer, count=10, block_ms=0):
        deadline = time.monotonic() + block_ms / 1000
        with self._lock:
            key = (stream, group)
            # Own pending first (redelivery after restart semantics: pending owned by name).
            pend = [(mid, m) for mid, (c, m) in self._pending[key].items() if c == consumer]
            if pend and getattr(self, "_redeliver", {}).get((stream, group, consumer)):
                self._redeliver[(stream, group, consumer)] = False
                return pend[:count]
            while True:
                items = self._streams[stream]
                cur = self._cursor[key]
                if cur < len(items):
                    batch = items[cur : cur + count]
                    self._cursor[key] = cur + len(batch)
                    for mid, m in batch:
                        self._pending[key][mid] = (consumer, m)
                    return list(batch)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._lock.wait(remaining)

    def simulate_restart(self, stream: str, group: str, consumer: str) -> None:
        """Next consume() by this consumer returns its pending messages again."""
        if not hasattr(self, "_redeliver"):
            self._redeliver = {}
        self._redeliver[(stream, group, consumer)] = True

    def ack(self, stream, group, ids):
        with self._lock:
            for i in ids:
                self._pending[(stream, group)].pop(i, None)
                self._acked[(stream, group)].add(i)

    def broadcast(self, channel, msg):
        with self._lock:
            self.broadcasts.append((channel, msg))

    def depth(self, stream, group):
        with self._lock:
            key = (stream, group)
            return len(self._streams[stream]) - self._cursor[key] + len(self._pending[key])

    def set_value(self, key, value, ttl_s=None):
        self._kv[key] = value

    def get_value(self, key):
        return self._kv.get(key)

    def messages(self, stream: str) -> list[Message]:
        with self._lock:
            return [m for _, m in self._streams[stream]]
