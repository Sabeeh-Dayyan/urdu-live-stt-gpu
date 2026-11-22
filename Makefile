SHELL := /bin/bash
PY    := .venv/bin/python
UV    := .venv/bin/uv
CONFIG ?= config/dev-mac-cpu.yaml
export UV_PROJECT_ENVIRONMENT := .venv
# Run from source (macOS can flag .venv files "hidden"; Python then skips editable-install .pth files).
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)

.PHONY: env test test-unit models infra infra-down reference testbed testbed-stop run migrate \
        tail bench bench-quick bench-live soak

## Create/refresh the project environment (Python 3.11 venv + uv lockfile).
env:
	@test -x $(PY) || python3.11 -m venv .venv
	@test -x $(UV) || $(PY) -m pip install -q uv
	$(UV) sync --all-extras --no-editable --python $(PY)

test:            ## all tests (integration tests need `make infra`; they skip otherwise)
	$(PY) -m pytest -q

test-unit:
	$(PY) -m pytest -q tests/unit

## Silero VAD (~2 MB) + Whisper models (turbo ~1.6 GB, large-v3 ~3 GB) into models/
models:          ## download the models CONFIG needs (pinned revisions, SHA-256 verified)
	$(PY) -m live_stt.cli download-models -c $(CONFIG)
	$(PY) -m live_stt.cli download-models -c config/default.yaml

## Postgres, Redis, Prometheus (:9090), Grafana (:3000)
infra:
	docker compose up -d --wait postgres redis
	docker compose up -d prometheus grafana

infra-down:
	docker compose down

## Real Urdu reference programme from FLEURS ur_pk test (downloads ~150 MB once)
reference:
	@test -f testbed/data/fleurs_ur/test.tsv || ( mkdir -p testbed/data/fleurs_ur/audio && \
	  curl -L -o testbed/data/fleurs_ur/test.tsv https://huggingface.co/datasets/google/fleurs/resolve/main/data/ur_pk/test.tsv && \
	  curl -L -o testbed/data/fleurs_ur/test.tar.gz https://huggingface.co/datasets/google/fleurs/resolve/main/data/ur_pk/audio/test.tar.gz && \
	  tar -xzf testbed/data/fleurs_ur/test.tar.gz -C testbed/data/fleurs_ur/audio )
	$(PY) -m bench.make_reference --n 40 --out testbed/reference/fleurs_ur

testbed/media/fleurs_ur.ts: testbed/reference/fleurs_ur.wav
	@mkdir -p testbed/media
	testbed/make_ts.sh $< $@

testbed/media/synthetic.ts:
	testbed/make_synthetic.sh $@

## Loop the Urdu reference programme to UDP ports 5101.. (N channels; RESET=1 → PTS reset per loop)
## (5101+ rather than 5001+: another test bed on this machine uses 5001/5002.)
testbed: testbed/media/fleurs_ur.ts
	testbed/send_loop.sh $${TS:-testbed/media/fleurs_ur.ts} 5101 - $${N:-2}

testbed-stop:
	-pkill -f "ffmpeg.*udp://127.0.0.1:51"

run:
	$(PY) -m live_stt.cli supervise -c $(CONFIG)

up:              ## run detached (default: broadcast config) — UI :8000, Grafana :3000
	scripts/live.sh start $${LIVE_CONFIG:-config/dev-mac-cpu-broadcast.yaml}

down:            ## stop the detached system
	scripts/live.sh stop

status:          ## detached system status
	scripts/live.sh status

logs:            ## follow the detached system log
	scripts/live.sh logs

calibration:     ## confidence → measured accuracy on the reference set, stored for the dashboard
	$(PY) -m bench.calibration --config $(CONFIG)
	$(PY) -m live_stt.cli record-accuracy bench/out/results_default.json bench/out/calibration_fleurs.json -c $(CONFIG)

dashboard:       ## regenerate the Grafana dashboard JSON
	$(PY) infra/grafana/build_dashboard.py

migrate:
	$(PY) -m live_stt.cli migrate -c $(CONFIG)

tail:
	$(PY) -m live_stt.cli tail -c $(CONFIG)

bench:           ## offline matrix on the reference set → docs/BENCHMARKS.md
	$(PY) -m live_stt.cli bench -c $(CONFIG) --matrix $${MATRIX:-default}

bench-quick:
	$(PY) -m live_stt.cli bench -c $(CONFIG) --matrix quick --limit-s 120

bench-live:      ## live latency percentiles while `make run` + `make testbed` are running
	$(PY) -m live_stt.cli bench-live -c $(CONFIG) --seconds $${SECS:-300}

soak:            ## monitor a running system for HOURS (default 24)
	$(PY) -m bench.soak --config $(CONFIG) --hours $${HOURS:-24}
