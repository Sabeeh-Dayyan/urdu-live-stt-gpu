"""Typed configuration.

Load order (later wins): config/default.yaml → overlay file(s) → environment variables.
Overlay selected with env ``STT_CONFIG`` (comma-separated paths) or ``load_settings(overlay=...)``.
Env override example: ``STT_ASR__DEVICE=cuda``  (prefix ``STT_``, nested delimiter ``__``).

Verified against pydantic-settings 2.16.3: ``YamlConfigSettingsSource(settings_cls, yaml_file=[...],
deep_merge=True)`` merges multiple YAML files recursively.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

class _Model(BaseModel):
    """Config sections reject unknown keys so typos in YAML/env fail loudly."""

    model_config = {"extra": "forbid"}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


class AudioTrack(_Model):
    """Audio stream selection. First non-null criterion wins: pid → language → index."""

    pid: int | None = None
    language: str | None = None  # ISO-639 tag in the PMT, e.g. "urd"
    index: int = 0  # index among audio streams


class ChannelConfig(_Model):
    id: str
    name: str = ""
    url: str
    audio_track: AudioTrack = AudioTrack()
    language: str = "ur"
    glossary_path: str | None = None
    multilingual: bool = False
    enabled: bool = True


class IngestConfig(_Model):
    fifo_size: int = 1_000_000  # in 188-byte packets (ffmpeg udp option)
    buffer_size: int = 8 * 1024 * 1024  # socket receive buffer, bytes
    timeout_us: int = 5_000_000
    localaddr: str | None = None  # NIC address for multicast join
    probesize: int = 500_000
    analyzeduration_us: int = 1_000_000
    reconnect_min_s: float = 0.5
    reconnect_max_s: float = 30.0
    ring_buffer_s: float = 120.0
    sample_rate: int = 16_000


class RelayConfig(_Model):
    """Rolling HLS copy of each channel (video+audio, stream copy) for the UI player."""

    enabled: bool = False
    dir: str = "var/hls"  # <dir>/<channel_id>/index.m3u8
    window_s: float = 900.0  # how far back a sentence can be replayed with video
    segment_s: float = 2.0  # should match the stream's GOP (broadcast test feeds: GOP 50 @ 25 fps = 2 s)


class ClockConfig(_Model):
    discontinuity_s: float = 1.0  # forward jump vs expected that triggers re-anchor
    backward_tolerance_s: float = 0.05  # backward jumps larger than this re-anchor
    warmup_s: float = 3.0  # stream time used to measure decode/probe buffering offset
    drift_threshold_ms: float = 500.0  # alert threshold (drift event flagged alert=true)
    drift_correct_ms: float = 100.0  # sustained |drift| above this is slewed away
    drift_sustain_s: float = 10.0
    drift_target_ms: float = 50.0  # stop slewing once within this
    drift_slew_ms_per_s: float = 20.0  # max slew rate (keeps timestamps monotonic)
    drift_ema_tau_s: float = 5.0


class VadConfig(_Model):
    model_path: str = "models/silero_vad.onnx"
    threshold: float = 0.5
    neg_threshold: float | None = None  # default threshold - 0.15
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    speech_pad_ms: int = 200


class StreamingConfig(_Model):
    mode: Literal["segment", "local_agreement", "auto"] = "segment"
    max_segment_s: float = 8.0
    cut_search_s: float = 1.5
    partials_enabled: bool = False
    partial_interval_s: float = 2.0
    step_s: float = 1.0
    max_buffer_s: float = 15.0
    auto_rtf_threshold: float = 0.3


class DecodeConfig(_Model):
    model: str
    beam_size: int = 1
    best_of: int = 1
    temperature: float = 0.0


class AsrConfig(_Model):
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    device_index: int = 0
    compute_type: str = "int8"
    cpu_threads: int = 8
    num_workers: int = 1
    download_root: str = "models/whisper"
    fast: DecodeConfig = DecodeConfig(model="large-v3-turbo", beam_size=1)
    refine: DecodeConfig = DecodeConfig(model="large-v3", beam_size=5, best_of=5)
    refine_enabled: bool = True
    serve_fast: bool = True  # False → REFINE-only service (e.g. second GPU)
    refine_context_s: float = 0.5
    batched: bool = False  # BatchedInferencePipeline (GPU)
    # CPU throughput lever: when several jobs are queued, pack them (any channel) into one
    # ≤ pack_s window decoded in a single encoder pass (the 30 s encoder dominates CPU cost).
    # 0 disables. A lone job is never delayed to wait for a pack.
    pack_s: float = 0.0
    max_batch: int = 8
    max_wait_ms: int = 50
    # In-RAM bounds inside the ASR service (the rest waits durably in the bus).
    fast_queue_max: int = 256
    refine_queue_max: int = 64
    fast_lag_alert_s: float = 5.0
    refine_max_lag_s: float = 300.0
    fast_wer_margin: float = 0.03  # absolute WER margin turbo vs large-v3
    use_hotwords: bool = True
    prompt_max_chars: int = 200
    warmup: bool = True


class GuardsConfig(_Model):
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    max_ngram_repeats: int = 4
    ngram_n: int = 3
    blocklist_path: str = "config/hallucination_blocklist.txt"


class BusConfig(_Model):
    redis_url: str = "redis://localhost:6379/0"
    maxlen: int = 100_000
    # Request streams carry audio; entries are deleted after ack, this is only a safety cap.
    request_maxlen: int = 5_000
    stream_fast: str = "asr.requests.fast"
    stream_refine: str = "asr.requests.refine"
    stream_transcripts: str = "transcripts.events"
    stream_system: str = "system.events"
    pubsub_live: str = "transcripts.live"


class DbConfig(_Model):
    dsn: str = "postgresql://stt:stt@localhost:5432/stt"
    flush_ms: int = 200
    flush_rows: int = 100


class ApiConfig(_Model):
    host: str = "0.0.0.0"
    port: int = 8000
    # When set, write endpoints (POST/PATCH /channels) require "Authorization: Bearer <token>".
    # Set it on any server reachable by others: STT_API__TOKEN=... (read endpoints stay open).
    token: str | None = None


class MetricsConfig(_Model):
    enabled: bool = True
    # Each process gets its own port: api=base, asr=base+1, persister=base+2, workers=base+10+i
    base_port: int = 9100


class LoggingConfig(_Model):
    level: str = "INFO"
    json_logs: bool = Field(default=True, alias="json")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="STT_", env_nested_delimiter="__", extra="forbid"
    )

    environment: str = "dev"
    ingest: IngestConfig = IngestConfig()
    relay: RelayConfig = RelayConfig()
    clock: ClockConfig = ClockConfig()
    vad: VadConfig = VadConfig()
    streaming: StreamingConfig = StreamingConfig()
    asr: AsrConfig = AsrConfig()
    guards: GuardsConfig = GuardsConfig()
    bus: BusConfig = BusConfig()
    db: DbConfig = DbConfig()
    api: ApiConfig = ApiConfig()
    metrics: MetricsConfig = MetricsConfig()
    logging: LoggingConfig = LoggingConfig()
    channels: list[ChannelConfig] = []

    # Set by load_settings before instantiation (class-level; one config per process).
    _yaml_files: ClassVar[list[Path]] = []

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_src = YamlConfigSettingsSource(
            settings_cls, yaml_file=list(cls._yaml_files), deep_merge=True
        )
        # Priority: init kwargs > env > yaml.
        return (init_settings, env_settings, yaml_src)

    def channel(self, channel_id: str) -> ChannelConfig:
        for c in self.channels:
            if c.id == channel_id:
                return c
        raise KeyError(f"unknown channel {channel_id!r}")


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(overlay: str | Path | list[str | Path] | None = None, **overrides) -> Settings:
    files: list[Path] = [DEFAULT_CONFIG]
    if overlay is None:
        env = os.environ.get("STT_CONFIG")
        overlay = [s for s in env.split(",") if s] if env else []
    if isinstance(overlay, (str, Path)):
        overlay = [overlay]
    for o in overlay:
        p = _resolve(o)
        if not p.exists():
            raise FileNotFoundError(p)
        files.append(p)
    Settings._yaml_files = files
    return Settings(**overrides)


def resolve_path(p: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    return _resolve(p)
