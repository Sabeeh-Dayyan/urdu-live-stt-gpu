# Runbook — live-stt

Audience: operators/on-call. Commands assume the repo root. On the Mac dev box all Python services
run natively (`make run`); in production they run from the CUDA image with `--network host`.

## 1. Start / stop

| What | Dev (Mac) | Prod (Linux + GPU) |
|---|---|---|
| Infra | `make infra` | managed Postgres/Redis, or `docker compose up -d postgres redis prometheus grafana` |
| Schema | `make migrate` (also done automatically by the persister) | same, from the image: `... live-stt:cuda migrate -c config/prod-gpu.yaml` |
| Everything | `make run` (= `stt supervise -c config/dev-mac-cpu.yaml`) | `docker run --gpus all --network host -v /srv/models:/app/models live-stt:cuda` |
| Stop | Ctrl-C / `kill -TERM <supervisor pid>` | `docker stop` (SIGTERM; children get 10 s, then SIGKILL) |
| Single component | `./stt worker --channel ch1`, `./stt asr-service`, `./stt persister`, `./stt api` | same subcommands |

Shutdown order: workers first (they flush open segments to the bus), then API, then ASR/persister.
Un-acked bus messages are re-delivered on restart (at-least-once; DB upserts make that idempotent).

Health: `curl -s localhost:8000/health | jq` → `status`, `redis`, `db`, and per-component heartbeats
(`supervisor`, `asr:*`, `worker:<channel>`) with pids, restarts, queue depth/lag, drift, decode errors.
Grafana: http://localhost:3000 → "Live STT". Prometheus: http://localhost:9090.

### Detached (dev box)

`make up` / `make status` / `make logs` / `make down` (`scripts/live.sh`): the supervisor runs
under `nohup` with its PID in `run/live.pid` and logs in `var/log/live.log`. It survives closing
the terminal or editor, but not a reboot. Default config: `config/dev-mac-cpu-broadcast.yaml`; another can
be set with `LIVE_CONFIG=config/dev-mac-cpu.yaml make up`.

### Video relay (UI player, Grafana click-through)

With `relay.enabled`, each worker writes a rolling HLS copy of what it receives to
`var/hls/<channel>/`:
- stream copy, no re-encode, roughly 1 % of one core;
- the window is `relay.window_s` (900 s, about 360 MB per channel at 3.2 Mb/s).

The worker is the only UDP receiver, so the video costs no extra packets and shares the
transcript's timeline. Sentences older than the window play without video; the UI says so. The relay
never stops transcription: on any error it logs `relay_mux_failed` / `relay_open_failed` and retries
after 5 s. Each reconnect adds an `EXT-X-DISCONTINUITY`.
**Seek accuracy:** segments carry `PROGRAM-DATE-TIME` from the host clock at muxer start. Transcript
times come from the PTS clock. The two agree to within the clock's startup correction, a few
hundred ms. This is not yet measured on the broadcast feeds.

### Accuracy on the dashboard

Live panels show **confidence**: mean word probability, `stt_asr_segment_confidence`, stored per
row as `asr_segments.confidence`. Measured accuracy only comes from reference transcripts:
`stt bench …` / `make calibration`, then `stt record-accuracy <result.json…> -c <config>` →
table `accuracy_results` → Grafana "Measured accuracy" panels.

## 2. Add a channel

```bash
curl -s -X POST localhost:8000/channels -H 'content-type: application/json' -d '{
  "id": "ptv_news", "name": "PTV News", "url": "udp://@239.10.10.5:5000",
  "audio_track": {"language": "urd"}, "language": "ur",
  "glossary_path": "config/glossary/ur_default.txt"}'
```
The supervisor re-syncs the channel set every 10 s and starts a worker (metrics port 9110+n).
Audio track selection order: `pid` → `language` (ISO-639 tag in the PMT) → `index` among audio
streams. Find PIDs/tags with `ffprobe -v error -show_streams -of compact 'udp://@239.10.10.5:5000'`.
Multicast on a specific NIC: set `ingest.localaddr` (or `?localaddr=10.0.0.5` in the URL).

## 3. Change a stream URL / disable a channel

```bash
curl -s -X PATCH localhost:8000/channels/ptv_news -H 'content-type: application/json' \
     -d '{"url": "udp://@239.10.10.9:5000"}'
curl -s -X PATCH localhost:8000/channels/ptv_news -H 'content-type: application/json' \
     -d '{"enabled": false}'
```
A changed URL/track/language restarts only that worker (open segment is flushed first).
Channels listed in YAML are seeded into the DB once; afterwards the DB is the source of truth.

## 4. Stream problems

| Symptom | Check | Action |
|---|---|---|
| `stt_stream_up == 0`, `stream_down` events | `ffprobe` the URL from the host; `tcpdump -i <nic> udp port 5000` | Worker reconnects by itself (backoff 0.5 s → 30 s with jitter). Fix routing/IGMP; for multicast check `localaddr` and the switch's IGMP snooping |
| Rising `decode_errors` | `stt_ingest_bytes_total` rate; packet loss on NIC (`netstat -su`, `ip -s link`) | Linux: raise socket buffers — `sysctl -w net.core.rmem_max=33554432 net.core.rmem_default=33554432` (persist in `/etc/sysctl.d/`), keep `ingest.buffer_size` ≤ rmem_max |
| Many `discontinuity` events | expected on playout switches / encoder restarts / looped test files (RESET=1) | Each one re-anchors the clock and closes the open segment. Investigate only if continuous |
| Wrong audio language | `/channels/{id}` audio_track vs `ffprobe` | PATCH `audio_track` with the right `pid` |

## 5. Clocks and timestamps

* Hosts must be NTP-synced: `chronyc tracking` (chrony) or `timedatectl timesync-status`
  (systemd-timesyncd). UTC timestamps = host clock at (re)connect + stream PTS.
* `stt_clock_drift_ms` > 500 ms sustained → `drift` event and gradual slew (20 ms/s). If drift
  keeps growing the encoder clock is off-rate or the host clock is unsynced.
* **Re-anchor a channel manually**: restart its worker (PATCH the channel with the same URL, or
  kill the worker pid from `/health`; the supervisor restarts it). The clock re-anchors on the
  first frame after connect, then removes the probe-buffer offset after `clock.warmup_s`.

## 6. ASR service

| Symptom | Meaning | Action |
|---|---|---|
| `stt_asr_fast_lag_alert == 1` | FAST queue older than `asr.fast_lag_alert_s` — oversubscribed | Reduce channels per box, use a faster FAST config (see BENCHMARKS.md), or add a GPU / ASR service. In local_agreement mode workers automatically double `step_s` |
| `refine_skipped` events | REFINE older than `asr.refine_max_lag_s` is skipped (flagged in DB, never silent) | Expected under load; FAST is unaffected. Add capacity if continuous |
| `error` events with `asr_error` | a segment failed inference 3 times (poison message isolated and acked) | Look at the log line `job_failed_permanently` |
| ASR process restarting | `/health` → `supervisor.children.asr.restarts` | Pending requests are re-delivered to the restarted service (same consumer name); requests left on a dead host are claimed after 2 min |

**Recover from GPU OOM**: CTranslate2 raises → the job is retried, then the batch is split per job.
If the process dies the supervisor restarts it with backoff. Persistent OOM: lower `asr.max_batch`
(e.g. 16 → 8), switch `asr.compute_type` to `int8_float16`, or give REFINE its own GPU by running a
second ASR service with `STT_ASR__DEVICE_INDEX=1 STT_ASR__SERVE_FAST=false` (REFINE-only) and set `STT_ASR__REFINE_ENABLED=false` on the first. Check
`nvidia-smi` for other processes holding memory.

## 7. Data

* Transcripts: `asr_segments` (monthly partitions on `start_utc`, created automatically one month
  ahead; a DEFAULT partition catches anything else). `pass='refined'` rows supersede `fast`.
* Retention: drop old partitions — `DROP TABLE asr_segments_2025_01;` (instant, no vacuum).
* Search: `GET /search?q=…` (Urdu-normalised, trigram index).
* Redis: request streams are deleted after ack; other streams capped with `MAXLEN ~`.
  `appendonly yes` keeps un-acked work across a Redis restart.

## 8. Capacity

Measured numbers and the per-machine estimate are in `docs/BENCHMARKS.md` (derived from measured
RTF; re-run `make bench` on new hardware). Rule: keep FAST+REFINE utilisation ≤ 70 %.

## 9. Soak test

```bash
make infra && make run            # terminal 1
N=2 RESET=1 make testbed          # terminal 2 (looping Urdu programme, PTS reset per loop)
HOURS=24 make soak                # terminal 3 → bench/out/soak.csv, soak_summary.json
```
Pass criteria: RSS slope ≈ 0 MB/h per process, |drift| < 500 ms, `fast_unaccounted == 0`.
