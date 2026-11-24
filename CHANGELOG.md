# Changelog

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
