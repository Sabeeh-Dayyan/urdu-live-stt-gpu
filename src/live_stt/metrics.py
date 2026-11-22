"""Prometheus metrics (prometheus_client 0.26.0).

Each OS process serves its own registry on its own port (see MetricsConfig.base_port); Prometheus
scrapes all targets. This avoids prometheus_client multiprocess mode (shared mmap files), which is
fragile with spawn-started children and restarts.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

_LAT_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 60, 120, 300)

# ---------------------------------------------------------------- ingest / worker
PACKETS_IN = Counter("stt_ingest_packets_total", "Demuxed audio packets", ["channel"])
BYTES_IN = Counter("stt_ingest_bytes_total", "Demuxed audio bytes", ["channel"])
AUDIO_SECONDS = Counter("stt_ingest_audio_seconds_total", "Decoded audio seconds", ["channel"])
STREAM_UP = Gauge("stt_stream_up", "1 if stream is receiving", ["channel"])
STREAM_EVENTS = Counter("stt_stream_events_total", "Stream events", ["channel", "type"])
DRIFT_MS = Gauge("stt_clock_drift_ms", "PTS clock drift vs wall clock (ms)", ["channel"])
VAD_SPEECH_RATIO = Gauge("stt_vad_speech_ratio", "Speech fraction over last minute", ["channel"])
SEGMENTS_EMITTED = Counter("stt_segments_emitted_total", "Segments sent to ASR", ["channel", "cut"])

# ---------------------------------------------------------------- ASR service
QUEUE_DEPTH = Gauge("stt_asr_queue_depth", "Pending ASR jobs", ["pass"])
QUEUE_LAG = Gauge("stt_asr_queue_lag_seconds", "Age of oldest pending job", ["pass"])
BATCH_SIZE = Histogram("stt_asr_batch_size", "Jobs per inference batch", ["pass"],
                       buckets=(1, 2, 4, 8, 16, 32))
INFER_SECONDS = Histogram("stt_asr_inference_seconds", "Inference wall time per batch", ["pass"],
                          buckets=_LAT_BUCKETS)
RTF = Histogram("stt_asr_rtf", "Real-time factor per job", ["pass"],
                buckets=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 4))
GUARD_DROPS = Counter("stt_asr_guard_drops_total", "Segments dropped/flagged by guards",
                      ["pass", "reason"])
REFINE_SKIPPED = Counter("stt_asr_refine_skipped_total", "Segments whose refinement was skipped")
FAST_LAG_ALERT = Gauge("stt_asr_fast_lag_alert", "1 if FAST lag exceeds threshold")

# ---------------------------------------------------------------- end-to-end latency
LAT_SPEECH_PARTIAL = Histogram("stt_latency_speech_to_partial_seconds",
                               "Speech start → first partial", ["channel"], buckets=_LAT_BUCKETS)
LAT_END_FINAL = Histogram("stt_latency_speech_end_to_final_seconds",
                          "End of speech → final text", ["channel"], buckets=_LAT_BUCKETS)
LAT_END_REFINED = Histogram("stt_latency_speech_end_to_refined_seconds",
                            "End of speech → refined text", ["channel"], buckets=_LAT_BUCKETS)

# ---------------------------------------------------------------- model confidence
# Mean word probability per published final/refined segment — the model's own confidence, NOT
# measured accuracy (see bench/calibration.py for what a threshold means against references).
CONF_BUCKETS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0)
SEG_CONFIDENCE = Histogram("stt_asr_segment_confidence", "Mean word probability per segment",
                           ["channel", "pass"], buckets=CONF_BUCKETS)

# ---------------------------------------------------------------- persister
ROWS_WRITTEN = Counter("stt_db_rows_written_total", "Rows written", ["table"])


def serve(port: int, enabled: bool = True) -> None:
    if enabled:
        start_http_server(port)
