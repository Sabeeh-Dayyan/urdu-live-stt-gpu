# Implementation Plan — status

Legend: [x] done and verified (command/test named) · [~] done with a documented limitation · [ ] not done

## Phase 1 — Foundations
- [x] Repo layout, `pyproject.toml` (pinned, wheel-verified versions), `uv.lock`
- [x] Configs (`default`, `dev-mac-cpu`, `prod-gpu`) + typed `settings.py` (unknown keys rejected)
- [x] `logging.py` (structlog JSON/console), `metrics.py`
- [x] `docker-compose.yml` (postgres 16, redis 7, prometheus, grafana + provisioned dashboard)
- [x] Test bed: `make_ts.sh`, `send_loop.sh` (N channels, RESET=1), `make_synthetic.sh`, `faults.md`
- [x] `clock.py`, `ringbuffer.py`, `urdu_normalize.py` + unit tests
- [x] ✅ pytest green; looping UDP stream; `ffprobe` shows the audio track

## Phase 2 — Single-channel pipeline (CPU, segment mode)
- [x] `ingest/source.py` (PyAV, reconnect w/ backoff+jitter, track by PID/language/index, format changes, monotonic timeline)
- [x] `vad/silero_onnx.py` (Silero VAD v6.2 ONNX, no PyTorch)
- [x] `streaming/segment_policy.py` (+ property tests), `worker.py`
- [x] `bus/` Redis Streams (consumer groups, pending recovery, claim, XDEL after ack) + in-memory bus
- [x] `asr/` backend adapter (faster-whisper), service, scheduler
- [x] `store/` SQL migrations (monthly partitions, BRIN, trigram), persister (idempotent upserts)
- [x] ✅ Full system test (`tests/integration/test_system.py`): UDP → … → PostgreSQL via API; UTC anchored to now
- [x] ✅ Reconnect after sender kill/restart, discontinuity per loop (`test_udp_live_loss_reconnect_and_loop_discontinuity`)
- [x] ✅ Word timestamps vs true speech onsets: 0.13–0.21 s (`test_real_urdu_transcription_quality_and_timestamps`)
- [x] ✅ "Runs 30 min": 4.5 h soak, 0 restarts, 0 lost segments (BENCHMARKS.md → Soak)

## Phase 3 — Benchmark harness + model decision
- [x] Reference set: 40 real Urdu FLEURS utterances + inserted music/noise/silence, VAD-annotated onsets
- [x] `stt bench` (event-driven queue simulation with measured inference), `bench/wer.py`, `report.py`, `bench-live`
- [x] ✅ BENCHMARKS.md table + FAST/REFINE decision (see that file)

## Phase 4 — Two-pass refinement + scheduler + guards
- [x] REFINE queue with ±0.5 s context, priority scheduler, backpressure, explicit `refine_skipped`
- [x] Guards (no-speech/logprob, compression, repetition, blocklist, word-range), prompts, hotwords
- [x] ✅ REFINE never delays FAST (`test_refine_never_delays_fast_under_load`, system test)
- [x] ✅ Refined vs FAST WER and hallucination with/without guards measured (BENCHMARKS.md)

## Phase 5 — API, WebSocket, UI, dashboards
- [x] FastAPI REST + `/ws/live` + `/health` + `/metrics`; vanilla-JS RTL UI; Grafana dashboard JSON
- [x] ✅ API/WS tests incl. SQL-injection-ish inputs, DB outage → 503 (`test_api.py`)
- [~] Browser rendering checked by HTTP/WS tests, not by a human in a browser

## Phase 6 — Streaming mode + GPU readiness
- [x] `local_agreement` (LocalAgreement-2) + tests (prefix commit, trim, absolute timestamps, no lost/duplicated words)
- [x] Cross-channel micro-batching (scheduler + `clip_timestamps` batched backend), adaptive packing on CPU
- [x] `prod-gpu.yaml`, `Dockerfile.cuda`, `Dockerfile` (CPU), Kafka adapter stub + shared bus contract tests
- [x] ✅ `docker build -f Dockerfile.cuda` succeeds (6.8 GB, CUDA 12.4 + cuDNN runtime); image imports
      all modules, loads `prod-gpu.yaml`; `device: auto` falls back to CPU when no GPU is visible
- [x] ✅ CPU image (`Dockerfile`, 1.5 GB) builds and imports on Linux
- [~] Not run on an actual NVIDIA GPU (none on the dev box): GPU RTF / batching / local_agreement
      latency are **TBD — measure** on the production host with `stt bench` + `bench/live_run.sh`
- [x] ✅ local_agreement live on CPU, 1 channel: latency documented (falls behind, as expected)

## Viewer features (requested 2026-11-21)
- [x] Video in the UI: worker-side HLS relay, click a sentence → video from that moment
      (`test_relay_*`, relay output checked with ffprobe; live CH1 relay checked)
- [x] Sentence-level timestamps + export SRT/VTT/CSV/TXT/JSON with confidence filter and time zone
      (`test_sentences_export.py`, `test_export_formats_filters_and_validation`)
- [x] Confidence stored, shown live (UI, Prometheus, Grafana) and measured against references
      (`bench/calibration.py` → BENCHMARKS.md, `accuracy_results`, Grafana panels)
- [x] Grafana: embedded player, transcript table with click-through to video, measured-accuracy panels
      (every panel SQL executed through Grafana's query API)
- [x] Detached run: `make up/down/status/logs`
- [~] "Accuracy above 90 %": **not achievable by filtering**. Measured on FLEURS: words ≥ 0.95
      confidence are 91.3 % correct but are only 78 % of all words; sentence filtering barely
      helps. Needs a better model (e.g. an Urdu fine-tune), measured with `stt bench`
- [~] UI checked by HTTP/WS tests and a JS syntax and parity test, not yet by a person in a browser

## Publishing / server deployment (2026-11-23)
- [x] Server stack `deploy/docker-compose.yml` (CPU and GPU variants), `.env` secrets, API token for writes,
      pinned + verified model downloads, accuracy results seeded for the dashboard
- [x] ✅ CPU stack deployed from the exported repo in Docker's Linux VM: healthy in 15 s, UDP feed →
      transcripts (fast + refined), video, export, Grafana (auth on, datasources OK), Prometheus targets up
- [x] ✅ Full test suite inside the Linux CPU image (CI equivalent)
- [x] Docs: README with architecture diagram + measured metrics, DEPLOYMENT.md, LICENSE (MIT),
      THIRD_PARTY_NOTICES, CHANGELOG, GitHub Actions CI
- [~] GPU image: builds and runs (CPU fallback); never run on an NVIDIA GPU

## Phase 7 — Hardening
- [x] RUNBOOK.md, soak monitor (`make soak`), capacity estimate from measured RTF
- [~] Soak: **4.5 h** run (24 h didn't fit in one night): 0 crashes/restarts, 0 lost FAST segments,
      worker/persister/API memory flat. **Open:** ASR RSS grows ~130 MB/h after warmup (cause not
      proven; likely native allocator). Needs investigation or an RSS-cap restart before 24/7 (BENCHMARKS.md)
