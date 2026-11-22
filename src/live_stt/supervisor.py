"""Supervisor: starts and monitors child processes (spawn start method — macOS safe).

Children: one ASR service, one persister, the API, and one worker per enabled channel.
Channels come from the YAML config and, when PostgreSQL is reachable, the ``channels`` table
(API-managed); the DB wins for ids present in both. The channel set is re-synced every
``sync_every_s``: new/enabled channels are started, removed/disabled ones stopped, changed ones
(URL/track/language) restarted.

Crashed children restart with exponential backoff + jitter (reset after 60 s of stable running).
Health (children, pids, restarts) is written to Redis key ``stt:health:supervisor``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import signal
import time
from dataclasses import dataclass, field

from live_stt.logging import get_logger, setup_logging
from live_stt.settings import ChannelConfig, Settings, load_settings

log = get_logger(__name__)
STABLE_S = 60.0


# --------------------------------------------------------------------------- child entrypoints
def child_main(kind: str, config: list[str] | None, arg: str | None, metrics_port: int) -> None:
    """Top-level (picklable) entrypoint for every child process."""
    s = load_settings(config)
    ctx = {"proc": kind}
    if kind == "worker":
        ch = ChannelConfig.model_validate_json(arg)
        ctx["channel_id"] = ch.id
    setup_logging(s.logging.level, s.logging.json_logs, **ctx)
    from live_stt import metrics

    try:
        metrics.serve(metrics_port, s.metrics.enabled)
    except OSError as e:
        get_logger("child").warning("metrics_port_busy", port=metrics_port, error=str(e))
    if kind == "worker":
        run_worker_child(s, ch)
    elif kind == "asr":
        run_asr_child(s)
    elif kind == "persister":
        from live_stt.store.persister import run_persister

        run_persister(s)
    elif kind == "selftest":  # used by tests: "crash:<seconds>:<exitcode>" or "sleep"
        parts = (arg or "sleep").split(":")
        time.sleep(float(parts[1]) if len(parts) > 1 else 3600)
        if parts[0] == "crash":
            os._exit(int(parts[2]) if len(parts) > 2 else 3)
    elif kind == "api":
        from live_stt.api.app import run_api

        run_api(s)
    else:
        raise ValueError(kind)


def make_bus(s: Settings):
    from live_stt.bus.redis_streams import RedisStreamsBus

    return RedisStreamsBus(
        s.bus.redis_url, maxlen=s.bus.maxlen,
        maxlen_overrides={s.bus.stream_fast: s.bus.request_maxlen,
                          s.bus.stream_refine: s.bus.request_maxlen},
        delete_on_ack=(s.bus.stream_fast, s.bus.stream_refine),
    )


def run_worker_child(s: Settings, ch: ChannelConfig) -> None:
    from live_stt.worker import run_worker

    # Make the channel resolvable by id inside the worker (it may come from the DB).
    s.channels = [c for c in s.channels if c.id != ch.id] + [ch]
    run_worker(s, ch.id, make_bus(s))


def run_asr_child(s: Settings) -> None:
    from live_stt.asr.backends.faster_whisper import FasterWhisperBackend
    from live_stt.asr.service import AsrService

    svc = AsrService(s, make_bus(s), FasterWhisperBackend(s.asr))

    def _term(*_):
        svc.stop()

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    svc.warmup()
    svc.run()


# --------------------------------------------------------------------------- supervisor
@dataclass
class Child:
    name: str
    kind: str
    arg: str | None
    port: int
    proc: mp.Process | None = None
    restarts: int = 0
    backoff: float = 1.0
    next_start: float = 0.0
    started_at: float = 0.0
    last_exit: int | None = None
    meta: dict = field(default_factory=dict)


class Supervisor:
    def __init__(self, settings: Settings, config: list[str] | None, with_api: bool = True,
                 sync_every_s: float = 10.0) -> None:
        self.s = settings
        self.config = config
        self.ctx = mp.get_context("spawn")
        self.children: dict[str, Child] = {}
        self.sync_every_s = sync_every_s
        self._stop = False
        base = settings.metrics.base_port
        self._add(Child("asr", "asr", None, base + 1))
        self._add(Child("persister", "persister", None, base + 2))
        if with_api:
            self._add(Child("api", "api", None, base))
        self._worker_ports: dict[str, int] = {}

    def _add(self, c: Child) -> None:
        self.children[c.name] = c

    def _port_for(self, channel_id: str) -> int:
        if channel_id not in self._worker_ports:
            used = set(self._worker_ports.values())
            p = self.s.metrics.base_port + 10
            while p in used:
                p += 1
            self._worker_ports[channel_id] = p
        return self._worker_ports[channel_id]

    # ------------------------------------------------------------- channels
    def desired_channels(self) -> dict[str, ChannelConfig]:
        chans = {c.id: c for c in self.s.channels}
        try:
            for c in fetch_db_channels(self.s.db.dsn):
                chans[c.id] = c
        except Exception as e:  # noqa: BLE001 - DB optional for workers
            log.debug("channel_sync_db_unavailable", error=str(e))
        return {k: v for k, v in chans.items() if v.enabled}

    def sync_channels(self) -> None:
        want = self.desired_channels()
        for name in [n for n, c in self.children.items() if c.kind == "worker"]:
            cid = name.split(":", 1)[1]
            c = self.children[name]
            if cid not in want:
                log.info("worker_removed", channel_id=cid)
                self._stop_child(c)
                del self.children[name]
            elif want[cid].model_dump_json() != c.arg:
                log.info("worker_config_changed", channel_id=cid)
                self._stop_child(c)
                c.arg = want[cid].model_dump_json()
                c.next_start = 0.0
                c.backoff = 1.0
        for cid, ch in want.items():
            name = f"worker:{cid}"
            if name not in self.children:
                self._add(Child(name, "worker", ch.model_dump_json(), self._port_for(cid)))

    # ------------------------------------------------------------- process mgmt
    def _start(self, c: Child) -> None:
        c.proc = self.ctx.Process(target=child_main, args=(c.kind, self.config, c.arg, c.port),
                                  name=c.name, daemon=False)
        c.proc.start()
        c.started_at = time.time()
        log.info("child_started", child=c.name, pid=c.proc.pid, restarts=c.restarts)

    def _stop_child(self, c: Child, timeout: float = 10.0) -> None:
        p = c.proc
        if p is None:
            return
        if p.is_alive():
            p.terminate()
            p.join(timeout)
            if p.is_alive():
                log.warning("child_kill", child=c.name)
                p.kill()
                p.join(5)
        c.proc = None

    def tick(self) -> None:
        now = time.time()
        for c in list(self.children.values()):
            if c.proc is not None and not c.proc.is_alive():
                c.last_exit = c.proc.exitcode
                ran = now - c.started_at
                c.proc = None
                if ran > STABLE_S:
                    c.backoff = 1.0
                c.restarts += 1
                c.next_start = now + c.backoff * random.uniform(0.8, 1.2)
                log.error("child_exited", child=c.name, exitcode=c.last_exit, ran_s=round(ran, 1),
                          restart_in_s=round(c.next_start - now, 1))
                c.backoff = min(c.backoff * 2, 60.0)
            if c.proc is None and now >= c.next_start and not self._stop:
                self._start(c)

    def health(self) -> dict:
        return {
            "t": time.time(), "pid": os.getpid(),
            "children": {
                n: {"pid": c.proc.pid if c.proc else None,
                    "alive": bool(c.proc and c.proc.is_alive()),
                    "restarts": c.restarts, "last_exit": c.last_exit,
                    "uptime_s": round(time.time() - c.started_at, 1) if c.proc else 0}
                for n, c in self.children.items()
            },
        }

    def run(self) -> None:
        def _term(*_):
            self._stop = True

        signal.signal(signal.SIGTERM, _term)
        signal.signal(signal.SIGINT, _term)
        bus = None
        try:
            bus = make_bus(self.s)
            bus.max_retry_s = 1.0
        except Exception:  # noqa: BLE001
            pass
        last_sync = 0.0
        last_health = 0.0
        while not self._stop:
            now = time.time()
            if now - last_sync >= self.sync_every_s:
                last_sync = now
                self.sync_channels()
            self.tick()
            if bus is not None and now - last_health >= 5:
                last_health = now
                try:
                    bus.set_value("stt:health:supervisor", self.health(), ttl_s=30)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.5)
        log.info("supervisor_stopping")
        # Stop producers first so in-flight work drains into the bus.
        order = sorted(self.children.values(), key=lambda c: {"worker": 0, "api": 1}.get(c.kind, 2))
        for c in order:
            self._stop_child(c)
        log.info("supervisor_stopped")


def fetch_db_channels(dsn: str, timeout: float = 3.0) -> list[ChannelConfig]:
    import asyncio

    import asyncpg

    async def _q():
        conn = await asyncpg.connect(dsn, timeout=timeout)
        try:
            rows = await conn.fetch(
                "SELECT id, name, url, audio_track::text AS t, language, glossary, enabled "
                "FROM channels")
        finally:
            await conn.close()
        return [
            ChannelConfig(id=r["id"], name=r["name"], url=r["url"],
                          audio_track=json.loads(r["t"] or "{}"), language=r["language"],
                          glossary_path=r["glossary"], enabled=r["enabled"])
            for r in rows
        ]

    return asyncio.run(_q())


def supervise(config: list[str] | None, with_api: bool = True) -> None:
    s = load_settings(config)
    setup_logging(s.logging.level, s.logging.json_logs, proc="supervisor")
    Supervisor(s, config, with_api=with_api).run()
