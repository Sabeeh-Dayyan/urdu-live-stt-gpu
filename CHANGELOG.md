# Changelog

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
