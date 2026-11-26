# Changelog

## 0.2.5 — 2026-11-26
- UI: new broadcast-desk colour scheme (warm paper, graphite, deep teal actions, amber "now"
  highlight; matching dark theme), larger and higher-contrast buttons, tabs and pills, visible
  keyboard focus, animated equaliser logo, card entrance and pointer-tilt depth effects
  (disabled for `prefers-reduced-motion` and touch). Grafana header and diagrams follow the palette.
- Bench: `latency6` matrix. 6 s segments did not beat 8 s on the laptop CPU (pack 12: delay p50
  31.0 s, WER 22.4 %; pack 18: 13.6 s, 22.9 %; vs 12.9 s, 20.7 % at 8 s / 16 s), so the CPU
  profiles keep 8 s / 16 s.

## 0.2.4 — 2026-11-26
- **One-step stream entry**: the live page shows the stream link with **✎ Change** and opens a
  large "Enter the stream link" box by itself when nothing arrives; `make stream URL=…` /
  `scripts/set_stream.sh` from a terminal; `make start` starts Docker services and the system.
- CPU profiles: 8 s segments, 16 s packing (measured delay p50 35.2 → 12.9 s, WER 26.4 → 20.7 %).
- README: "Transcribe a UDP stream in 3 steps", which link format to enter, architecture and
  workflow diagrams (`docs/images/`).

## 0.2.3 — 2026-11-25
- Grafana: live video and transcript first, live stat cards directly below; the embedded page
  follows the computer's light/dark setting (Grafana too, `GF_USERS_DEFAULT_THEME=system`) and has
  its own ☾/☀ button; the header's Light/Dark links switch both.
- UI files are served with `Cache-Control: no-cache` (ETag revalidation), so browsers never show a
  stale page after an update.

## 0.2.2 — 2026-11-25
- **Transcript follows the video**: the sentence (and word) being played is highlighted and kept
  centred; scrolling pauses following for 15 s (⦿ Follow video resumes). **Sync video with text**
  (on by default) plays live video behind the live edge by the measured text delay (90th
  percentile + 8 s, 12–120 s), so the highlighted line is what you hear.
- **Channel setup asks only for the stream link** (`239.1.1.1:5000` is accepted; multicast gets `@`);
  ID, name, Urdu audio track and language are automatic, the rest is under "Advanced options".
- **Grafana redesigned**: header with app links and a Light/Dark switch clear of Grafana's hover
  menu (the embedded page follows the theme); "Live now" cards with sparklines (stream, text delay,
  behind live, confidence, speech share, segments/min); live trend charts; accuracy and stored-range
  sections; pipeline details collapsed.

## 0.2.1 — 2026-11-25
- **New look**: light theme by default with a dark switch; one channel per page; video and transcript
  **side by side** (no scrolling), metric cards in a row below, larger adjustable Urdu text, plain
  labels and a how-to strip. **Go to** any date and time from the live page; search opens Find a moment.
- **Grafana**: single-channel, light theme by default, live page embedded full width, two rows of
  stat cards (new: text delay, audio skipped under load).
- **Load shedding** (`asr.fast_max_lag_s`, CPU profiles 120 s): an overloaded machine skips stale
  live audio (recorded as a gap) instead of drifting hours behind.
- **Fix: video stayed in sync with the transcript only until the first feed outage.** The HLS
  date-time is now re-anchored to the clock after every outage or timestamp jump (it had drifted
  3 h behind in a day-long test), and every archived segment is indexed.
- **Repeat trimming**: Whisper's looping tails («وزیراعظم وزیراعظم وزیراعظم …») are cut to one copy.

## 0.2.0 — 2026-11-24
- **Video archive and time-range replay**: segments are kept `relay.archive_hours` (servers: 24 h)
  with a per-day index; watch any range with its transcript, download it as MP4/TS, see coverage.
- **Search and topics**: several terms at once, saved topics (`config/topics/topics.yaml`), hits
  grouped into occurrences with a Watch button.
- **Reports** for a custom range or overall: HTML (print to PDF), Markdown, JSON; optional transcript.
- **Uploads**: transcribe a video/audio file in the background (REFINE decoder, live keeps
  priority), viewer with clickable transcript, exports and report.
- **Channels in the browser**: add/edit/enable/disable/delete stream URLs (`DELETE /channels/{id}`
  added; URL schemes validated).
- **System & links page** listing every endpoint, Grafana, Prometheus and the API docs; shared
  navigation on every page; API token entered once in the browser.
- **Model selection** (`bench/eval_models.py`, docs/MODELS.md): per-utterance WER/CER/SER on FLEURS.
  GPU profile: REFINE and uploads use large-v3 beam 5, float16 (19.95 % WER / 7.70 % CER, compared with turbo's
  21.14 % / 8.29 %). The Urdu fine-tune tested was within noise of large-v3 and was not adopted.
  `scripts/convert_whisper.py` converts other Hugging Face Whisper checkpoints.

## 0.1.0 — 2026-11-23
First public release.
- Live Urdu speech-to-text from MPEG-TS/UDP feeds: per-channel workers (PyAV, PTS→UTC clock,
  Silero VAD), faster-whisper ASR service (FAST + REFINE passes, hallucination guards), Redis
  Streams, PostgreSQL, FastAPI REST/WebSocket, web UI, Prometheus/Grafana.
- Web UI: live video (HLS relay), sentence-level timestamps across segment cuts, click-to-seek,
  confidence filter and per-word confidence, export SRT/VTT/CSV/TXT/JSON.
- Grafana dashboard: embedded player, transcript table with click-through, live confidence,
  measured accuracy from reference transcripts, pipeline health.
- Server deployment: `deploy/docker-compose.yml` (host networking, pinned and verified model
  downloads, API token for write endpoints, secrets from `deploy/.env`).
- Benchmarks and calibration on CPU: see `docs/BENCHMARKS.md`.
