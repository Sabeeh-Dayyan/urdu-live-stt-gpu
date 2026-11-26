"""Priority scheduler for the ASR service (pure logic, thread-safe).

* Two bounded queues: FAST (never dropped by the scheduler) and REFINE (droppable).
* FAST always first. REFINE only when the FAST queue is empty.
* Partials: a newer partial for the same ``segment_key`` supersedes an older queued one, and a
  final supersedes queued partials of its segment (superseded jobs are returned as ``skipped``
  so they can be acked — nothing is silently lost).
* REFINE jobs older than ``refine_max_lag_s`` are skipped (→ ``refine_skipped`` events).
* Optional load shedding: with ``fast_max_lag_s`` > 0, FAST jobs older than that are skipped
  (→ ``fast_skipped`` gap events) so live text stays close to real time on an overloaded host
  instead of falling hours behind. 0 (default) = FAST is never dropped.
* Micro-batching (``batched=True``): take up to ``max_batch`` jobs once that many are queued or
  the oldest has waited ``max_wait_s``; fill the batch with the jobs closest in duration to the
  oldest job (length grouping → less padding waste).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

FAST, REFINE = "fast", "refine"


@dataclass
class Job:
    msg_id: str
    pass_: str  # fast | refine
    duration_s: float
    t_enqueued: float  # wall-clock epoch when the worker enqueued it
    segment_key: str = ""
    is_partial: bool = False
    payload: object = None
    t_received: float = field(default_factory=time.monotonic)


@dataclass
class Decision:
    pass_: str | None
    jobs: list[Job]
    skipped: list[tuple[Job, str]]  # (job, reason)


def _no_expire(j: Job) -> bool:
    return isinstance(j.payload, dict) and bool(j.payload.get("no_expire"))


class Scheduler:
    def __init__(
        self,
        fast_max: int = 256,
        refine_max: int = 64,
        refine_max_lag_s: float = 300.0,
        fast_max_lag_s: float = 0.0,
        max_batch: int = 1,
        max_wait_s: float = 0.05,
        batched: bool = False,
        pack_s: float = 0.0,
        pack_gap_s: float = 0.3,
        wall: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
    ) -> None:
        self.fast: deque[Job] = deque()
        self.refine: deque[Job] = deque()
        self.fast_max = fast_max
        self.refine_max = refine_max
        self.refine_max_lag_s = refine_max_lag_s
        self.fast_max_lag_s = fast_max_lag_s
        self.max_batch = max(1, max_batch if batched else 1)
        self.max_wait_s = max_wait_s
        self.batched = batched
        self.pack_s = pack_s
        self.pack_gap_s = pack_gap_s
        self.wall = wall
        self.mono = mono
        self._skipped: list[tuple[Job, str]] = []
        self._cv = threading.Condition()

    # ----------------------------------------------------------------- input
    def space(self, pass_: str) -> int:
        with self._cv:
            if pass_ == FAST:
                return max(0, self.fast_max - len(self.fast))
            return max(0, self.refine_max - len(self.refine))

    def push(self, job: Job) -> None:
        with self._cv:
            if job.pass_ == FAST:
                if job.segment_key:
                    keep = deque()
                    for q in self.fast:
                        if q.is_partial and q.segment_key == job.segment_key:
                            self._skipped.append((q, "superseded"))
                        else:
                            keep.append(q)
                    self.fast = keep
                self.fast.append(job)
            else:
                if len(self.refine) >= self.refine_max:
                    victim = next((q for q in self.refine if not _no_expire(q)), None)
                    if victim is not None:
                        self.refine.remove(victim)
                        self._skipped.append((victim, "refine_queue_full"))
                self.refine.append(job)
            self._cv.notify_all()

    # ---------------------------------------------------------------- output
    def wait(self, timeout: float) -> None:
        with self._cv:
            if not self.fast and not self.refine and not self._skipped:
                self._cv.wait(timeout)

    def lag_s(self, pass_: str) -> float:
        with self._cv:
            q = self.fast if pass_ == FAST else self.refine
            return max(0.0, self.wall() - q[0].t_enqueued) if q else 0.0

    def depth(self, pass_: str) -> int:
        with self._cv:
            return len(self.fast if pass_ == FAST else self.refine)

    def next(self) -> Decision:
        with self._cv:
            skipped, self._skipped = self._skipped, []
            now = self.wall()
            if self.fast_max_lag_s > 0 and self.fast and \
                    now - self.fast[0].t_enqueued > self.fast_max_lag_s:
                keep = deque()
                for q in self.fast:
                    if now - q.t_enqueued > self.fast_max_lag_s and not _no_expire(q):
                        skipped.append((q, "fast_lag"))
                    else:
                        keep.append(q)
                self.fast = keep
            # Expire stale refinements (uploaded-file jobs carry no_expire: never dropped).
            if self.refine and any(now - q.t_enqueued > self.refine_max_lag_s
                                   and not _no_expire(q) for q in self.refine):
                keep = deque()
                for q in self.refine:
                    if now - q.t_enqueued > self.refine_max_lag_s and not _no_expire(q):
                        skipped.append((q, "refine_lag"))
                    else:
                        keep.append(q)
                self.refine = keep
            for pass_, q in ((FAST, self.fast), (REFINE, self.refine)):
                if not q:
                    continue
                if pass_ == REFINE and self.fast:
                    break
                if self.max_batch > 1 and len(q) < self.max_batch:
                    waited = self.mono() - q[0].t_received
                    if waited < self.max_wait_s:
                        return Decision(None, [], skipped)
                jobs = self._take(q)
                return Decision(pass_, jobs, skipped)
            return Decision(None, [], skipped)

    def _take(self, q: deque[Job]) -> list[Job]:
        if self.pack_s > 0 and not self.batched:
            return self._take_pack(q)
        first = q.popleft()
        if self.max_batch == 1 or not q:
            return [first]
        rest = sorted(q, key=lambda j: abs(j.duration_s - first.duration_s))
        chosen = rest[: self.max_batch - 1]
        ids = {id(j) for j in chosen}
        remaining = deque(j for j in q if id(j) not in ids)
        q.clear()
        q.extend(remaining)
        return [first, *chosen]

    @staticmethod
    def _packable(j: Job) -> bool:
        kind = j.payload.get("kind") if isinstance(j.payload, dict) else None
        return not j.is_partial and kind not in ("stream", "stream_end")

    def _take_pack(self, q: deque[Job]) -> list[Job]:
        """FIFO jobs whose total duration (+ gaps) fits in ``pack_s``. Partials and streaming
        jobs are never packed (latency-critical, superseded quickly)."""
        first = q.popleft()
        out, total = [first], first.duration_s
        if not self._packable(first):
            return out
        while q and self._packable(q[0]):
            if total + self.pack_gap_s + q[0].duration_s > self.pack_s:
                break
            nxt = q.popleft()
            out.append(nxt)
            total += self.pack_gap_s + nxt.duration_s
        return out
