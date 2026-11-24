"""`stt` command-line entrypoint (Typer)."""

from __future__ import annotations

import json
import time

import typer

from live_stt.logging import get_logger, setup_logging
from live_stt.settings import load_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)
ConfigOpt = typer.Option(None, "--config", "-c", help="Overlay YAML (e.g. config/dev-mac-cpu.yaml)")


def _cfg(config: str | None) -> list[str] | None:
    return [c for c in config.split(",") if c] if config else None


def _init(config: str | None, **ctx):
    s = load_settings(_cfg(config))
    setup_logging(s.logging.level, s.logging.json_logs, **ctx)
    return s


@app.command("show-config")
def show_config(config: str = ConfigOpt) -> None:
    """Print the fully merged configuration."""
    typer.echo(load_settings(_cfg(config)).model_dump_json(indent=2))


@app.command()
def supervise(config: str = ConfigOpt, api: bool = typer.Option(True, help="Also run the API")):
    """Run everything: ASR service, persister, API and one worker per enabled channel."""
    from live_stt.supervisor import supervise as _sup

    _sup(_cfg(config), with_api=api)


@app.command()
def worker(channel: str = typer.Option(..., "--channel"), config: str = ConfigOpt,
           seconds: float = typer.Option(0, help="Stop after N seconds (0 = forever)")):
    """Run a single channel worker in the foreground."""
    from live_stt import metrics
    from live_stt.supervisor import make_bus
    from live_stt.worker import run_worker

    s = _init(config, proc="worker", channel_id=channel)
    idx = [c.id for c in s.channels].index(channel)
    metrics.serve(s.metrics.base_port + 10 + idx, s.metrics.enabled)
    run_worker(s, channel, make_bus(s), stop_after_s=seconds or None)


@app.command("asr-service")
def asr_service(config: str = ConfigOpt):
    """Run the ASR service in the foreground."""
    from live_stt import metrics
    from live_stt.supervisor import run_asr_child

    s = _init(config, proc="asr")
    metrics.serve(s.metrics.base_port + 1, s.metrics.enabled)
    run_asr_child(s)


@app.command()
def persister(config: str = ConfigOpt):
    """Run the persister (bus → PostgreSQL) in the foreground."""
    from live_stt import metrics
    from live_stt.store.persister import run_persister

    s = _init(config, proc="persister")
    metrics.serve(s.metrics.base_port + 2, s.metrics.enabled)
    run_persister(s)


@app.command()
def uploads(config: str = ConfigOpt):
    """Run the uploaded-file transcription runner in the foreground."""
    from live_stt.uploads import run_uploads

    s = _init(config, proc="uploads")
    run_uploads(s)


@app.command()
def api(config: str = ConfigOpt):
    """Run the HTTP/WebSocket API in the foreground."""
    from live_stt.api.app import run_api

    run_api(_init(config, proc="api"))


@app.command()
def migrate(config: str = ConfigOpt):
    """Apply database migrations."""
    import asyncio

    from live_stt.store.db import migrate as _m

    s = _init(config)
    typer.echo(f"applied: {asyncio.run(_m(s.db.dsn)) or 'nothing (up to date)'}")


@app.command("download-models")
def download_models(config: str = ConfigOpt,
                    check: bool = typer.Option(False, help="Only report; exit 1 if missing"),
                    verify: bool = typer.Option(True, help="Verify pinned SHA-256 checksums")):
    """Download the models this config needs (pinned revisions, SHA-256 verified)."""
    from live_stt import models

    s = load_settings(_cfg(config))
    items = models.required(s)
    for it in items:
        typer.echo(f"{'ok     ' if it.present else 'MISSING'} {it.name:<16} {it.path}")
    if check:
        raise typer.Exit(0 if all(i.present for i in items) else 1)
    models.download(s, verify=verify, log=typer.echo)
    typer.echo("models ready")


@app.command("record-accuracy")
def record_accuracy(files: list[str] = typer.Argument(..., help="bench / calibration JSON files"),
                    config: str = ConfigOpt):
    """Store measured accuracy results in PostgreSQL (shown on the Grafana dashboard)."""
    import asyncio

    from live_stt.store.accuracy import record
    from live_stt.store.db import migrate as _m

    from pathlib import Path

    paths = []
    for f in files:  # a directory means every *.json in it
        p = Path(f)
        paths += sorted(str(x) for x in p.glob("*.json")) if p.is_dir() else [f]
    if not paths:
        raise typer.BadParameter("no result files found")
    s = _init(config)
    asyncio.run(_m(s.db.dsn))
    n = asyncio.run(record(s.db.dsn, paths))
    typer.echo(f"recorded {n} rows from {len(paths)} file(s) into {s.db.dsn.rsplit('/', 1)[-1]}")


@app.command()
def tail(config: str = ConfigOpt, raw: bool = typer.Option(False, help="Print JSON events")):
    """Print live transcript events (console consumer)."""
    import redis

    s = load_settings(_cfg(config))
    r = redis.Redis.from_url(s.bus.redis_url)
    ps = r.pubsub(ignore_subscribe_messages=True)
    ps.subscribe(s.bus.pubsub_live)
    for m in ps.listen():
        ev = json.loads(m["data"])
        if raw:
            typer.echo(json.dumps(ev, ensure_ascii=False))
            continue
        t = time.strftime("%H:%M:%S", time.gmtime(ev.get("start_utc") or 0))
        lat = (ev.get("timings") or {}).get("latency_s")
        typer.echo(f"[{t}Z] {ev.get('channel_id')} {ev.get('type'):>8} "
                   f"lat={lat}s  {ev.get('text_raw', '')}  {ev.get('flags') or ''}")


@app.command()
def testbed(config: str = ConfigOpt) -> None:
    """Print the commands that start looping UDP senders for every configured channel."""
    s = _init(config)
    log = get_logger("testbed")
    for c in s.channels:
        port = c.url.rsplit(":", 1)[-1]
        log.info("sender", channel=c.id,
                 cmd=f"testbed/send_loop.sh testbed/media/synthetic.ts {port}")


@app.command()
def bench(
    config: str = ConfigOpt,
    dataset: str = typer.Option("testbed/reference/fleurs_ur.json", help="Reference set JSON"),
    matrix: str = typer.Option("default", help="Matrix name in bench/run_bench.py"),
    limit_s: float = typer.Option(0, help="Use only the first N seconds of reference audio"),
    out: str = typer.Option("bench/out", help="Output directory"),
):
    """Offline benchmark: WER/CER/RTF/hallucination per model/config → docs/BENCHMARKS.md."""
    import sys

    from live_stt.settings import PROJECT_ROOT

    sys.path.insert(0, str(PROJECT_ROOT))
    from bench.run_bench import main as bench_main

    s = _init(config, proc="bench")
    bench_main(s, dataset, matrix, limit_s, out)


@app.command("bench-live")
def bench_live(config: str = ConfigOpt, seconds: float = 300,
               out: str = typer.Option("bench/out/live.json")):
    """Measure live latency/RTF percentiles from transcript events on the bus."""
    import sys

    from live_stt.settings import PROJECT_ROOT

    sys.path.insert(0, str(PROJECT_ROOT))
    from bench.live import main as live_main

    live_main(load_settings(_cfg(config)), seconds, out)


if __name__ == "__main__":
    app()
