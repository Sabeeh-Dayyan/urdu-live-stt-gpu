"""Shipped config profiles stay consistent with how the scheduler and the benchmarks work."""

from pathlib import Path

import pytest

from bench.run_bench import Cfg
from live_stt.api.app import bench_label
from live_stt.settings import load_settings

ROOT = Path(__file__).resolve().parents[2]
CPU_PROFILES = ["config/dev-mac-cpu.yaml", "config/dev-mac-cpu-broadcast.yaml", "config/server-cpu.yaml"]


@pytest.mark.parametrize("profile", CPU_PROFILES)
def test_bench_label_matches_benchmark_format(profile):
    """The live page and Grafana look up measured WER by this label, so it must be byte-identical
    to the label `stt bench` records for the same settings."""
    s = load_settings(str(ROOT / profile))
    want = Cfg(s.asr.fast.model, s.asr.fast.beam_size, s.streaming.max_segment_s, s.asr.cpu_threads,
               s.asr.compute_type, s.asr.pack_s).label
    assert bench_label(s) == want
