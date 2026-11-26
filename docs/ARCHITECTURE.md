# Architecture — Live UDP Television Speech-to-Text

Status: living document. Every performance figure here is **TBD — measure** until produced by
`stt bench` (see `docs/BENCHMARKS.md`). No number in this file is assumed.

## 1. Purpose

Real-time Urdu-first (Urdu–English code-switched) transcription of live television delivered as
MPEG-TS over UDP (unicast or multicast). Outputs live captions (partial → final), a higher-accuracy
refined pass that replaces live text per segment, absolute UTC word timestamps derived from stream
PTS, persistence + search, and bus events for downstream consumers.

## 2. Shape

```
                 ┌──────────────── Channel Worker (1 OS process per channel, CPU) ────────────────┐
 UDP MPEG-TS ───▶│ Ingest (PyAV demux) → decode → resample 16 kHz mono f32 → PTS→UTC clock      │
                 │ → ring buffer → Silero VAD (ONNX) → Segmenter / Streaming policy               │
                 └───────────────┬────────────────────────────────────────────────────────────────┘
                                 │ ASR requests (int16 PCM + utc_start + channel_id + seq + pass)
                                 ▼  Redis Stream  asr.requests.fast / asr.requests.refine
                 ┌──────────────── ASR Service (1 process per CPU box / per GPU) ─────────────────┐
                 │ Priority scheduler: FAST queue ≫ REFINE queue                                  │
                 │ Dynamic micro-batching across channels (GPU)                                   │
                 │ Backend adapter: faster-whisper (CTranslate2)                                  │
                 │ FAST model: large-v3-turbo    REFINE model: large-v3 (or Urdu fine-tune)      │
                 │ Hallucination guards, Urdu normalisation, glossary/hotwords                   │
                 └───────────────┬────────────────────────────────────────────────────────────────┘
                                 │ transcript events (partial | final | refined)
                                 ▼  Redis Stream transcripts.events  (+ pub/sub fan-out)
                 ┌───────────────┼──────────────────────┬──────────────────────────┐
                 ▼               ▼                      ▼                          ▼
          Persister →      API (FastAPI)          Metrics (Prometheus)      Future consumers
          PostgreSQL       REST + WebSocket       → Grafana                 (keyword alerts, RAG)
```

Process model: `stt supervise` spawns (multiprocessing, `spawn` start method) one worker per
enabled channel, one ASR service per compute device, the persister and the API. Crashed children
restart with exponential backoff + jitter. CPU-heavy work never runs in the API process.

### Data contracts (bus messages)

`AsrRequest`: `channel_id, segment_id, seq, pass (fast|refine), utc_start (float epoch s),
sample_rate, pcm (int16 LE bytes), context_pre_s, context_post_s, is_partial, t_speech_start,
t_speech_end, t_enqueued, prompt`.

`TranscriptEvent`: `type (partial|final|refined|refine_skipped), channel_id, segment_id, seq,
start_utc, end_utc, model, text_raw, text_norm, words[{w,start_utc,end_utc,p}], avg_logprob,
no_speech_prob, compression_ratio, flags[], timings{...}`.

`StreamEvent`: `type (stream_up|stream_down|discontinuity|drift), channel_id, at_utc, details`.

## 3. Decision log

Each decision: **what**, **why**, **rejected alternative**.

### D1 — Python 3.11; asyncio for I/O, processes for CPU
- Why: CTranslate2, onnxruntime, PyAV are Python-first. GIL ⇒ one process per channel for
  ingest/VAD and one ASR process per compute device.
- Version note: some current releases (PyAV ≥ 17, NumPy ≥ 2.4) require Python ≥ 3.12, and several
  ML libs have dropped Intel-macOS wheels. We pin the newest versions that ship
  `cp311 macosx x86_64` wheels **and** Linux x86_64 wheels (see `pyproject.toml` comments).
- Rejected: Node.js (weak ML bindings); Go/Rust (slower iteration — revisit if ingest bottlenecks).

### D2 — Ingest via PyAV, not an ffmpeg subprocess pipe
- Why: PyAV gives per-frame `pts` and `time_base`; a PCM pipe loses them.
- Options: URL `fifo_size=1000000&overrun_nonfatal=1&timeout=5000000[&localaddr=…]&buffer_size=…`;
  format `fflags=nobuffer`, configurable `probesize`/`analyzeduration`; explicit audio track by
  language tag / PID / index; `av.AudioResampler(format="flt", layout="mono", rate=16000)`.
- Reconnect loop with exponential backoff + jitter; emits `stream_down` / `stream_up`.
- Prod: raise `net.core.rmem_max`/`rmem_default` (see RUNBOOK).
- Rejected: ffmpeg CLI → stdout PCM (no PTS); GStreamer (heavier, harder to package).

### D3 — PTS-anchored UTC clock (`ingest/clock.py`)
- `utc = anchor_utc + (pts_unwrapped − pts0) × time_base`.
- Anchor = host clock (NTP/chrony) at first frame after (re)connect.
- 33-bit wraparound unwrapping (2^33 ticks @ 90 kHz ≈ 26.5 h).
- Discontinuity: backward jump, or forward jump > `discontinuity_s` (default 1 s) vs expected ⇒
  hard re-anchor + `discontinuity` event.
- Drift: compare PTS-derived time with wall clock; if |drift| > `drift_threshold_ms` (500)
  sustained for `drift_sustain_s`, emit metric/event and slew the anchor gradually
  (`drift_slew_ms_per_s`) instead of jumping, so word timestamps stay monotonic.
- The ring buffer records, per write, (sample index → UTC) anchor points so any absolute sample
  index maps to UTC with sample accuracy, including across re-anchors.
- Rejected: wall-clock-at-receipt timestamps (jitter from network/decoder buffering, ~100s ms).

### D4 — VAD: Silero VAD via ONNX Runtime
- 512-sample (32 ms) frames at 16 kHz, per-channel recurrent state, configurable threshold,
  min speech, min silence, speech padding. No PyTorch dependency.
- Rejected: WebRTC VAD (poor on music/broadcast); PyTorch Silero (no Intel-mac wheels).

### D5 — ASR: faster-whisper behind `ASRBackend`
- `transcribe(batch: list[AudioJob], opts) -> list[AsrResult]` with word timestamps and segment
  `avg_logprob`, `no_speech_prob`, `compression_ratio`.
- CPU `int8` dev; CUDA `float16`/`int8_float16` prod; `BatchedInferencePipeline` on GPU.
- Rejected: openai-whisper (slower, PyTorch), distil-whisper (English), small models (Urdu
  accuracy), cloud APIs (data sovereignty).

### D6 — Two-pass models
| Pass | Model | Decoding | Goal |
|---|---|---|---|
| FAST | `large-v3-turbo` | beam 1 (GPU) / 1–2 (CPU) | live captions |
| REFINE | `large-v3` (or Urdu fine-tune via `ct2-transformers-converter`) | beam 5, best_of 5 | archive accuracy |

REFINE re-transcribes each finished segment with ~0.5 s context each side; `refined` replaces FAST
for the same `segment_id`. If bench shows turbo WER outside `fast_wer_margin` of large-v3 on our
reference set, FAST switches to large-v3. Decision: **TBD — measure** (Phase 3).

### D7 — Pluggable streaming policy
- `segment` (CPU default): VAD segments, force-cut at `max_segment_s` (8) at lowest-energy point in
  last 1.5 s; one FAST pass per segment, then REFINE. Optional partials every
  `partial_interval_s` only if RTF headroom.
- `local_agreement` (GPU default): LocalAgreement-2 (Macháček et al., 2023) every `step_s`.
- `auto`: `local_agreement` iff backend is CUDA and measured RTF < 0.3.
- Rejected: always-streaming on CPU (encoder processes a 30 s window ⇒ re-runs fall behind).

### D8 — Accuracy levers
Fixed `language` per channel (`ur`), glossary + last sentence as `initial_prompt` (≤ 200 chars),
`hotwords` where supported, `condition_on_previous_text=False`, hallucination guards (no-speech/
logprob, compression ratio, blocklist, repeated n-grams), word timestamps clamped to segment, Urdu
normalisation (raw + normalised both stored).

### D9 — Scheduling & backpressure
Two bounded queues; FAST never dropped; REFINE droppable with explicit `refine_skipped` events.
GPU micro-batching (`max_batch`, `max_wait_ms`, length grouping). CPU: `cpu_threads` = physical
cores, `num_workers=1`, REFINE only when FAST empty.

### D10 — Bus: Redis Streams (v1), Kafka adapter later
Consumer groups (durable, replayable), `MAXLEN ~` trimming, pub/sub for WS fan-out. `Bus` interface
so Kafka/Redpanda (key = `channel_id`) can replace it. Audio as int16 PCM.
- Rejected for v1: Kafka (heavy on a laptop), ZeroMQ (no durability/replay).

### D11 — Storage: PostgreSQL 16
Schema in `store/migrations/*.sql`. `asr_segments` range-partitioned monthly on `start_utc`;
indexes `(channel_id, start_utc)`, BRIN `start_utc`, GIN trigram on `text_norm`.
Upsert key `(channel_id, segment_id, start_utc)` (partition key must be in unique constraints);
`refined` supersedes `fast`.
- **Migrations: plain versioned SQL files** applied by a ~40-line runner with a
  `schema_migrations` table. Rejected: Alembic — it pulls SQLAlchemy, which we otherwise do not use
  (all access is raw `asyncpg`), and our DDL (partitioning, BRIN, pg_trgm) is raw SQL anyway.

### D12 — API & UI: FastAPI + single static HTML
REST + `WS /ws/live`; vanilla JS UI with RTL Urdu (`dir="auto"`, Noto Nastaliq Urdu).

### D13 — Observability
prometheus_client metrics (per-channel ingest, VAD, ASR queue lag, RTF, guard drops, latency
histograms); structlog JSON logs; Prometheus + Grafana in docker-compose with a provisioned
dashboard. Multi-process: each process exposes its own `/metrics` port; Prometheus scrapes all
(simpler and more robust than `prometheus_client` multiprocess mode).

### D14 — Config & packaging
`config/default.yaml` + env overlay files, loaded with pydantic-settings (env `STT__…` overrides).
`uv` lockfile; Typer CLI `stt`. Infra in docker-compose; Python services native on the Mac.

## 4. Decisions changed or added during implementation (with evidence)

| # | Change | Why (evidence) |
|---|---|---|
| C1 | **Adaptive packing** (`asr.pack_s`, CPU): when several jobs are queued, they are concatenated (0.3 s gaps) into one ≤ 28 s window and decoded in a *single* encoder pass (BatchedInferencePipeline, one clip, `without_timestamps=True`); words are mapped back to jobs by midpoint. A lone job is never delayed. | Measured on the i9: the 30 s encoder pass costs 6.9 s of the 8.7 s needed for one 8 s segment, so one-at-a-time turbo FAST runs at RTF 1.1–1.5 (cannot keep up with one channel). One packed 29 s window took 12.8 s (RTF 0.44). Plain long-form decoding of the same window took 21 s because faster-whisper re-encodes after every timestamp ("seek"). See BENCHMARKS.md. |
| C2 | **Drift correction threshold** `clock.drift_correct_ms` = 100 ms (slew), separate from the 500 ms *alert* threshold in the spec. Slew rate is defined per second of media time. | Trace on a live UDP stream: ffmpeg `-re` sends ahead gradually for ~12 s, so after the 3 s warmup a steady −230…−280 ms bias remained (inside the 500 ms threshold, so never corrected) — outside the 300 ms timestamp budget. With the 100 ms correction threshold the offset converges to ≈ +20 ms within ~35 s (regression test `test_udp_timestamp_accuracy_converges`). |
| C3 | Source guarantees a **monotonic output timeline** (strictly increasing chunk starts, ≤ 2 ms overlaps allowed for slew; the startup correction shifts the guard). | Corrupt PTS and resampler resets produced backward chunk timestamps (`test_corrupt_stream_survives`); the ring buffer's UTC→index bisection requires monotonic anchors. |
| C4 | **Startup correction is scoped to the current connection** (`RingBuffer.shift_utc(delta, from_idx)`). | After a reconnect the new clock's correction must not shift UTC of audio received before the reconnect. |
| C5 | **Empty REFINE never blanks FAST** unless guards gave a hallucination verdict; otherwise `refine_skipped:refine_empty`. | Full-system run: a 0.7 s tail segment's words all fell inside the REFINE context margins → refined text "" overwrote correct FAST text. |
| C6 | **`refine_max_lag_s` = 1800 s on the CPU profile** (300 s default kept for GPU). | Full-system run on CPU: continuous speech keeps FAST busy (REFINE has strictly lower priority), so all REFINE jobs aged past 300 s and were skipped (correctly flagged). With a longer window REFINE catches up during non-speech. |
| C7 | **local_agreement runs inside the ASR service** (per-channel `StreamingSessions`), workers stream audio deltas (`kind=stream`) and a `stream_end` per segment; REFINE still segment-based. | LocalAgreement needs the previous hypothesis → state must live with the ASR consumer. Rounds of ≤ 1 job per channel give one backend call per round (cross-channel batching on GPU). Kafka partition key = channel_id provides the needed affinity at scale. |
| C8 | Redis consumers walk their **pending list with a cursor** on restart. | Reading pending from id "0" repeatedly returned the same un-acked entries → duplicate jobs in the scheduler after a restart (`test_pending_redelivered_after_restart`). |
| C9 | Persister: **bad rows never wedge the pipeline** (batch → per-row fallback, NUL/NaN/out-of-range sanitising). | Poison-event test: one malformed event made the whole batch retry forever. |
| C10 | Looping test streams: plain `ffmpeg -stream_loop` keeps PTS **continuous** across loops (ffmpeg offsets timestamps); `RESET=1` restarts ffmpeg per loop to emulate encoder resets / discontinuities. | Observed in `test_udp_live_loss_reconnect_and_loop_discontinuity`; the prompt's assumption that a looping file creates a discontinuity per loop does not hold for ffmpeg ≥ 4. |
| C11 | Package is run **from source via `PYTHONPATH`** (`./stt` wrapper, Makefile) and installed non-editable. | macOS on this machine sets the `hidden` flag on files under `.venv`; Python ≥ 3.11.x skips hidden `.pth` files, silently breaking editable installs. |
| C12 | Pinned versions are the newest with **cp311 macOS-x86_64 + Linux wheels**: onnxruntime 1.23.2 (last Intel-mac wheel), PyAV 18.1.0, NumPy 2.4.6 (newer need Python ≥ 3.12). | PyPI wheel inspection 2026-11-20. |
| C13 | API kept in one module (`api/app.py`) instead of `api/ws.py` + `routes/`. | ~300 lines; splitting added indirection without benefit. |
| C14 | Dev UDP ports 5101/5102. | Another test bed on this machine streams to 5001/5002. |
| C15 | **Video relay inside the worker** (`ingest/relay.py`, `relay.*`): the worker demuxes (not decodes) the first video stream too and stream-copies video+audio into rolling HLS with `PROGRAM-DATE-TIME`; the API serves it, the UI seeks by sentence UTC. Timestamps are re-based to stay monotonic; on every (re)open unreferenced segments are swept and a stale playlist is discarded. | Browsers cannot play UDP/MPEG-TS. A second receiver on the same unicast port could steal datagrams from the worker. Tests: relay does not change one audio sample fed to ASR (`test_relay_does_not_change_transcription_audio`); without the sweep every reconnect leaked one segment file (found by `test_relay_rolling_window…`). |
| C16 | **Confidence ≠ accuracy.** `confidence` (mean word probability) is stored per segment and exported as a Prometheus histogram. Measured accuracy lives in `accuracy_results`, loaded only from reference-based results (`stt record-accuracy`). `bench/calibration.py` measures what a threshold means. | Calibration on FLEURS: words ≥ 0.9 confidence are 88.8 % correct (words < 0.9: 41 %), but segment-level confidence barely separates good from bad segments (80.4 % → 80.5 % accuracy at ≥ 0.9). Presenting confidence as "accuracy" would be wrong. |
| C17 | **Sentence-level timestamps** are derived at read time from word timestamps (`text/sentences.py`: split on `۔ ؟ ? ! .` or a ≥ 1 s pause; same rule in the UI JS, parity-tested). Export: SRT/VTT (relative cues + absolute origin note), CSV (UTF-8 BOM for Excel), TXT, JSON, in a chosen time zone. | Segments are VAD chunks of ≤ 10 s, not sentences; storing sentences would duplicate data and need re-splitting after REFINE replaces the text. |
| C17b | **Sentences are assembled across segment cuts** (`text/sentences.py:assemble`, UI `assemble()`): words of consecutive segments are joined; a sentence ends at punctuation, a ≥ 1 s pause, or past 20 s at its longest internal pause. Used by the UI and the export. | Live Urdu TV news recording (CH1): 91 of 136 segment boundaries (67 %) were < 0.35 s apart (forced cuts in continuous speech), and only 2 of 137 segments contained any punctuation, so per-segment display showed sentences cut in half. Boundaries without punctuation are pause-based estimates. |
| C19 | **Server deployment = `deploy/docker-compose.yml`** (project `live-stt-server`): app, Prometheus and Grafana on the **host network**; Postgres/Redis published on 127.0.0.1 only; one-shot `models` (pinned revisions + SHA-256, `src/live_stt/models.py`) and `accuracy` (loads `bench/results/`) services; secrets only from `deploy/.env`; write endpoints behind `STT_API__TOKEN`. | Multicast feeds and per-process metrics ports need host networking; a bridged container cannot join multicast groups on the host NIC. Verified by deploying the exported repo in Docker's Linux VM, streaming the Urdu test programme over UDP, adding the channel through the API (401 without token, 201 with) and checking transcripts, video, export, Grafana datasources/queries and Prometheus targets. A first run with project name `live-stt` collided with the dev stack's default project name and re-used its Postgres volume (old password) — hence the distinct name. |
| C18 | **Broadcast profile isolation**: `dev-mac-cpu-broadcast.yaml` uses its own database `stt_broadcast`, Redis DB 1 **and its own pub/sub channel** (`transcripts.live.broadcast`). Redis pub/sub is not scoped by DB number: with a dev-profile run active at the same time, its FLEURS captions appeared in the broadcast UI (regression test `test_profiles_sharing_a_redis_server_do_not_share_live_pubsub`). `scripts/live.sh` refuses to start next to another supervisor. | The supervisor overlays YAML channels with the `channels` table (DB wins); the dev DB pins `ch1 → udp://…:5101`, so pointing `ch1` at 5001 in YAML alone would have been silently ignored. |
| C20 | **Video archive = the relay's own segments, kept** (`ingest/archive.py`, `relay.archive_hours`): with an archive the HLS muxer writes unique strftime names (`a/YYYYMMDD/seg_<time>_<session>_<n>.ts`, the session token because ffmpeg's counter restarts on every reconnect) without `delete_segments`; the relay appends each segment ffmpeg lists to a per-day index (`archive/YYYYMMDD.idx`: start, duration, path, discontinuity); a sweep deletes segments older than the retention every 10 min. The API builds VOD playlists with `PROGRAM-DATE-TIME` for any range and stream-copies clips (MP4 via PyAV, TS by concatenation). | No re-encoding and no second receiver. Index lookups for a 1 h range touch two small files instead of listing thousands of segments. ffmpeg writes strftime segment paths into the playlist as given (absolute), so the API rewrites them to relative URIs. Found in testing: the first retention sweep removed the just-created, still-empty day directory before ffmpeg opened the next segment (ENOENT) — directories are now removed only when older than the cutoff. |
| C21 | **Uploads reuse the live pipeline** (`uploads.py`, supervisor child `uploads`): the file is decoded by `AudioSource` (virtual clock = `base_utc` + media time), segmented by the same VAD/`SegmentPolicy` (20 s segments), and sent to the **REFINE** stream with `no_expire`; at most `uploads.max_inflight` (2) segments per upload are queued, tracked through the transcript fan-out. Transcripts are stored under `channel_id = up-…`, so export, search and reports work unchanged. | Live keeps priority without a new queue: REFINE runs only when FAST is idle, and the scheduler never expires or evicts `no_expire` jobs (live refines can still be dropped under load). Same VAD and guards as live, so file and live quality are comparable. Re-processing after a crash reuses `base_utc`, so segment ids repeat and the idempotent upsert avoids duplicates. |
| C22 | **Topic search by words, grouped in time** (`reports.py`, `config/topics/topics.yaml`): a topic is a list of spellings (Whisper writes most English words in Urdu script); `/search` matches any of them (`ILIKE ANY`, trigram index) and groups hits of a channel < `gap_s` (180 s) apart into occurrences. | "The Prime Minister's speech" usually means a stretch of news in which the title keeps recurring. Speaker identification would need a voice model and enrolment, which is not implemented. |
| C23 | **Reports are computed in SQL at request time** (`reports.py`): aggregates per channel and hour, guard flags, stream events, top words (`regexp_split_to_table` + stopwords), topic counts, archive coverage and the recorded offline accuracy; HTML (printable), Markdown, JSON. | No precomputed tables to keep consistent. Measured on the 2-day broadcast test database (5,712 segments, 2 channels) while the live system ran on the same 8-core laptop: overall JSON report 1.8 s, HTML with the full transcript (1.6 MB) 7.0 s. Rows refined later keep only the refined latency, which the report states. |
| C24 | **Browser-managed channels and one hub page**: `/ui/channels` edits the `channels` table through the token-protected API (URL schemes restricted to network protocols — local files only via YAML); `/ui/system` lists every endpoint from `/openapi.json` (every route has a docstring, tested) plus Grafana/Prometheus links from `api.links` with `{scheme}`/`{host}` resolved in the browser. | Operators should not edit YAML or URLs to reach a tool; links resolve to whatever address the UI was opened on, so they work behind any hostname. |
| C25 | **Optional FAST load shedding** (`asr.fast_max_lag_s`, CPU profiles 120 s, default 0): FAST jobs older than the threshold are skipped, acked and recorded as `fast_skipped` system events (UTC, duration) plus `stt_asr_fast_skipped_*` counters. Upload jobs (`no_expire`) are never shed. | 2026-11-24 broadcast run: the laptop was shared with an Android emulator (~2 cores) and the Docker VM (~1.5 cores); FAST never drops by design, so the live text fell 3 h 18 min behind (Redis backlog + 256-job queue). After enabling shedding and restarting, FAST lag was 10–40 s. A visible, logged gap is preferable to hours-old "live" captions; servers sized correctly never hit it. |
| C26 | **Repeat-run trimming** (`guards.min_repeat_run` = 3): a word or 2–3-word phrase repeated ≥ 3 times in a row is cut to one copy (flag `repeat_trimmed`), instead of dropping the whole segment. | Live broadcast DB: segments ending «…اسمبلی وزیراعظم وزیراعظم وزیراعظم وزیراعظم وزیراعظم» passed the n-gram guard (trigram count 3 ≤ 4, unigram rule needs > 8) but are Whisper looping; trimming keeps the real speech before the loop. Two repeats («بہت بہت») are normal Urdu and are kept. |
| C27 | **Single-channel UI, light by default** (`ui/common.css`, `ui/index.html`): one channel per page (first enabled or `?ch=`; a picker appears only with several enabled channels), video and transcript side by side with the transcript sized to the video (`contain: size`), metric cards in a row below, Go-to-time box, dark theme toggle remembered per browser, adjustable Urdu text size; Grafana channel variable single-select, its embedded live page full width, Grafana default theme light (`GF_USERS_DEFAULT_THEME`). The live page's delay card uses live events only (history rows would report the delay they had when written). | User feedback 2026-11-24 on the multi-channel grid: transcript below the player needed scrolling, too many controls, dark-only look. |
| C28 | **Video date-time re-anchoring** (`HlsRelay.resync`): the relay starts a new HLS session (same playlist, `EXT-X-DISCONTINUITY`) at every input timestamp jump, every audio-clock `discontinuity`, and after 10 s without packets; the archive is indexed by segment URI (not "newer than the last date-time"); the per-session file token is unique even within one second. | 2026-11-24 broadcast run: ffmpeg hlsenc writes PROGRAM-DATE-TIME = muxer start + Σ segment durations, and the rebaser collapsed input gaps, so after the test feed's outages the video's date-time fell **3 h 0 min 38 s** behind the wall clock (last segment written 23:10:27, labelled 20:09:48). Clicking a sentence would have played the wrong moment, and every segment written after the first outage was indexed under a date-time that was too early. Regression tests: `test_relay_resyncs_on_timestamp_jump_and_indexes_every_session`, `test_relay_resyncs_after_arrival_gap`. |

## 5. Measured numbers

See `docs/BENCHMARKS.md` (produced by `stt bench` / `stt bench-live` / `make soak`).
