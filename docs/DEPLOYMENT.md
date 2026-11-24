# Deployment guide: live server

Audience: whoever installs and operates live-stt on a physical Linux server that receives TV
feeds over UDP. Day-to-day operations (channels, alerts, troubleshooting) are in
[RUNBOOK.md](RUNBOOK.md).

## 1. Server requirements

| | Minimum | Notes |
|---|---|---|
| OS | Linux x86_64 (Ubuntu 22.04/24.04) | host networking is required for multicast feeds |
| GPU | NVIDIA with ≥ 16 GB VRAM (L4, A10, A100, RTX 4090 class) | driver ≥ 550 (CUDA 12.4), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| CPU | 8+ cores | ingest + VAD run on CPU: ≈ 1 % of a core per channel for the video relay, plus decoding/VAD |
| RAM | 32 GB | each channel worker ≈ 350 MB |
| Disk | 30 GB + transcripts + video archive | models ≈ 4.7 GB, image 6.8 GB; **video archive ≈ 1.1 GB per channel-hour** at 2.5 Mb/s (default 24 h ≈ 27 GB per channel, `STT_ARCHIVE_HOURS`); uploaded files |
| Network | NIC on the feed VLAN; NTP | timestamps come from the host clock + stream PTS |

**GPU throughput and latency have not been measured yet** (development hardware had no NVIDIA
GPU). Measure on your server before planning channel counts (section 8).

## 2. Prepare the host (once)

```bash
# Docker Engine + Compose plugin
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # log out/in afterwards
# NVIDIA driver + container toolkit (Ubuntu); check with: nvidia-smi
sudo apt-get install -y nvidia-driver-550
# then follow the NVIDIA Container Toolkit install guide and run:
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # must list the GPU

# UDP receive buffers: without this Linux silently caps the socket buffer at ~208 KB and
# bursts from the feed are dropped.
echo 'net.core.rmem_max=33554432
net.core.rmem_default=33554432' | sudo tee /etc/sysctl.d/90-live-stt.conf
sudo sysctl --system

# Time sync (timestamps are only as good as the host clock)
timedatectl status | grep -i synchronized    # should say "yes" (systemd-timesyncd or chrony)
```

## 3. Install and start

```bash
git clone https://github.com/Sabeeh-Dayyan/urdu-live-stt-gpu.git
cd urdu-live-stt-gpu/deploy
cp .env.example .env
nano .env            # set POSTGRES_PASSWORD, GRAFANA_ADMIN_PASSWORD, STT_API_TOKEN (openssl rand -hex 32)
docker compose up -d --build
```

The first start builds the image, then the one-shot `models` service downloads the pinned models
(4.7 GB, SHA-256 verified) into `../models`. The
`accuracy` service loads the measured accuracy results for the dashboard, and the `stt` service starts.
Follow it with:

```bash
docker compose ps                       # stt should become "healthy" (≈ 1–2 min after models)
docker compose logs -f stt              # JSON logs
curl -s localhost:8000/health | python3 -m json.tool
```

## 4. Add channels

Channels live in PostgreSQL. The supervisor picks up changes within 10 s. The easiest way is the
web UI: open `http://SERVER:8000/ui/channels`, press **🔑** in the top bar and paste
`STT_API_TOKEN` once (stored in that browser only), then **Add channel** with the stream URL.
Supported URLs: `udp://` (multicast/unicast MPEG-TS), `rtp://`, `srt://`, `rtsp://`, `rtmp(s)://`,
`http(s)://` (HLS or progressive). Local file paths are refused through the API.

The same through the API. Find the audio track first:

```bash
ffprobe -v error -show_programs -show_streams -of compact 'udp://@239.10.10.5:5000'
```

```bash
TOKEN=$(grep STT_API_TOKEN .env | cut -d= -f2)
curl -s -X POST localhost:8000/channels \
  -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"id":"news1","name":"News 1","url":"udp://@239.10.10.5:5000",
       "audio_track":{"language":"urd"},"language":"ur",
       "glossary_path":"config/glossary/ur_default.txt"}'
```

* Multicast on a specific NIC: add `?localaddr=10.0.0.5` to the URL.
* Unicast: `udp://@0.0.0.0:5001` (or the server's address).
* Change or disable: `PATCH /channels/news1` with `{"url": …}` or `{"enabled": false}`; remove:
  `DELETE /channels/news1` (transcripts are kept).
* Channel ids: letters, digits, `_ . -` (used in file paths for the video relay).

## 5. Use it

| What | Where |
|---|---|
| **Start page for everything** | `http://SERVER:8000/ui/system`: status, links to every page, Grafana, Prometheus, API docs and a clickable list of all endpoints |
| Live text + video, export | `http://SERVER:8000` |
| Watch a past time range, download video, search, topics | `http://SERVER:8000/ui/archive` |
| Reports (custom range / overall) | `http://SERVER:8000/ui/reports` |
| Transcribe an uploaded file | `http://SERVER:8000/ui/uploads` |
| Manage channels | `http://SERVER:8000/ui/channels` |
| API docs (try every endpoint) | `http://SERVER:8000/docs` |
| Dashboards | `http://SERVER:3000` (user `admin`, password from `.env`) → *Live STT*. Set the **Player URL** variable at the top to `http://SERVER:8000` once and save the dashboard |
| REST API | `GET /channels/{id}/transcripts`, `/channels/{id}/export?format=srt\|vtt\|csv\|txt\|json`, `/search?q=`, `WS /ws/live` (see README) |

## 6. Security checklist

* The API's **write** endpoints (channels, uploads, deletions) require `STT_API_TOKEN`. Read
  endpoints, reports, the UI and the video (live and archive) are open to anyone who can reach
  port 8000. Put the server behind a firewall or a TLS reverse proxy
  with authentication if transcripts are not public, for example Caddy:
  ```
  stt.example.com {
      basicauth { admin <bcrypt-hash> }
      reverse_proxy 127.0.0.1:8000
  }
  ```
* Postgres, Redis and Prometheus listen on 127.0.0.1 only. Expose only 8000 (or the proxy) and 3000.
  With `ufw`: `ufw allow 8000/tcp; ufw allow 3000/tcp; ufw allow from <feed-subnet> to any proto udp`.
* Grafana: anonymous access is off. The dashboard embeds the player, which needs
  `GF_PANELS_DISABLE_SANITIZE_HTML=true`, so only trusted users should get the Editor role.
* `deploy/.env` holds the secrets: `chmod 600 deploy/.env`. It is git-ignored.
* Uploads are limited to `STT_UPLOAD_MAX_MB` (default 4096) and to media files with an audio track;
  a reverse proxy in front needs a matching body limit (Caddy: `request_body { max_size 4GB }`,
  nginx: `client_max_body_size 4g;`).

## 7. Operations

| Task | Command (in `deploy/`) |
|---|---|
| Status / logs | `docker compose ps`, `docker compose logs -f stt` |
| Restart the app only | `docker compose restart stt` |
| Upgrade | `git pull && docker compose up -d --build` (migrations run automatically) |
| Stop everything | `docker compose down` (data stays in Docker volumes) |
| Backup transcripts | `docker compose exec -T postgres pg_dump -U stt stt \| gzip > stt-$(date +%F).sql.gz` |
| Restore | `gunzip -c backup.sql.gz \| docker compose exec -T postgres psql -U stt stt` |
| Disk used by video | `du -sh ../var/hls/*` — the archive keeps `STT_ARCHIVE_HOURS` (default 24); older segments are deleted automatically every 10 min |
| Disk used by uploads | `du -sh ../var/uploads` — kept until deleted in the Upload page or `DELETE /uploads/{id}` |
| Transcript retention | drop old monthly partitions: `docker compose exec postgres psql -U stt -c 'DROP TABLE asr_segments_2026_01'` |

Containers restart automatically (`restart: unless-stopped`), including after a reboot once Docker
is enabled (`sudo systemctl enable docker`).

**Known issue (CPU soak, 4.5 h):** the ASR process's memory grew ≈ 130 MB/h after warm-up. If you
see the same, `docker compose restart stt` during a quiet period. Unfinished work stays queued in
Redis and is picked up again. See BENCHMARKS.md → Soak.

## 8. Measure on your hardware

```bash
docker compose exec stt python -m live_stt.cli bench --matrix quick       # offline WER/RTF on the reference set
docker compose exec stt python -m live_stt.cli bench-live --seconds 600   # live latency on your channels
```
Then record the results so the dashboard shows them:
`docker compose exec stt python -m live_stt.cli record-accuracy bench/out -c <config>`.
The offline benchmark needs the reference set (`testbed/reference/fleurs_ur.wav`, included).
Compare `streaming.mode: segment` and `local_agreement` latency with `bench-live`.
`mode: auto` picks streaming automatically when the measured FAST real-time factor is below 0.3.

## 9. Without Docker (systemd)

```bash
sudo apt-get install -y ffmpeg git make
curl -LsSf https://astral.sh/uv/install.sh | sh        # uv; it also installs Python 3.11
git clone https://github.com/Sabeeh-Dayyan/urdu-live-stt-gpu.git /opt/live-stt && cd /opt/live-stt
uv python install 3.11 && uv venv --python 3.11 .venv && uv sync --frozen --no-editable --extra bench
PYTHONPATH=src .venv/bin/python -m live_stt.cli download-models -c config/prod-gpu.yaml
```
You need PostgreSQL 16 and Redis 7 (packages or the `postgres`/`redis` services of
`deploy/docker-compose.yml`). Point the app at them with `STT_DB__DSN` and `STT_BUS__REDIS_URL`.
Then install the unit:

```ini
# /etc/systemd/system/live-stt.service
[Unit]
Description=live-stt
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/opt/live-stt
Environment=PYTHONPATH=/opt/live-stt/src
Environment=STT_API__TOKEN=change-me
Environment=STT_DB__DSN=postgresql://stt:change-me@127.0.0.1:5432/stt
ExecStart=/opt/live-stt/.venv/bin/python -m live_stt.cli supervise -c config/prod-gpu.yaml
Restart=always
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```
`sudo systemctl daemon-reload && sudo systemctl enable --now live-stt`
