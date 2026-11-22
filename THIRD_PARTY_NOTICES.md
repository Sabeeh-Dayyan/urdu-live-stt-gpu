# Third-party components

This repository's own code is MIT-licensed (see `LICENSE`). It uses, downloads or bundles:

| Component | Use | License |
|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [CTranslate2](https://github.com/OpenNMT/CTranslate2) | speech recognition engine | MIT |
| OpenAI Whisper `large-v3` / `large-v3-turbo` weights (CTranslate2 conversions, downloaded at deploy time, pinned in `src/live_stt/models.py`) | ASR models | MIT |
| [Silero VAD](https://github.com/snakers4/silero-vad) v6.2 ONNX (downloaded at deploy time) | voice activity detection | MIT |
| [hls.js](https://github.com/video-dev/hls.js) 1.5.20 (vendored: `ui/vendor/hls.min.js`) | browser video player | Apache-2.0 |
| [PyAV](https://github.com/PyAV-Org/PyAV) / FFmpeg | stream demuxing and decoding | BSD-3-Clause / LGPL-2.1+ (Docker images install Debian/Ubuntu `ffmpeg`) |
| ONNX Runtime, FastAPI, asyncpg, redis-py, structlog, prometheus_client, pydantic | runtime libraries | MIT / BSD / Apache-2.0 |
| PostgreSQL, Redis, Prometheus, Grafana (Docker images) | infrastructure | PostgreSQL / BSD-3 (Redis 7) / Apache-2.0 / AGPL-3.0 (Grafana, run unmodified) |
| Google FLEURS `ur_pk` test utterances (`testbed/reference/fleurs_ur.*`) | accuracy reference set | CC-BY-4.0 — Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*, 2022 |

No broadcast footage is included. Broadcast test recordings used during development are local only
and excluded by `.gitignore`.
