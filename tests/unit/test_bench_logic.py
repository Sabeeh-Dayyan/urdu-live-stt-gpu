from pathlib import Path

from bench.report import decide, write_report
from bench.soak import summarize
from bench.wer import wer_cer


def test_wer_uses_urdu_normalisation():
    r = wer_cer(["علي نے كتاب پڑھی۔"], ["علی نے کتاب پڑھی"])
    assert r["wer"] == 0.0 and r["cer"] == 0.0
    assert wer_cer(["ایک دو"], [""])["wer"] == 1.0
    assert wer_cer([""], ["x"])["wer"] is None


def run(model, beam, wer, rtf, pack=28):
    return {"label": f"{model} b{beam}", "config": {"model": model, "beam": beam, "pack": pack},
            "wer": wer, "rtf": rtf}


def test_decide_keeps_turbo_within_margin_and_switches_otherwise():
    runs = [run("large-v3-turbo", 1, 0.20, 0.4), run("large-v3-turbo", 5, 0.19, 0.9),
            run("large-v3", 5, 0.18, 2.0)]
    d = decide(runs, 0.03)
    assert d["fast"]["label"] == "large-v3-turbo b1" and not d["switched"]
    assert d["refine"]["label"] == "large-v3 b5"
    runs[0]["wer"] = 0.30
    runs.append(run("large-v3", 1, 0.19, 0.6))
    d = decide(runs, 0.03)
    assert d["switched"] and d["fast"]["label"] == "large-v3 b1"


def test_decide_ignores_errors_and_empty():
    assert decide([], 0.03) == {}
    assert decide([{"label": "x", "error": "boom", "config": {}}], 0.03) == {}


def test_report_renders(tmp_path):
    import json

    f = tmp_path / "r.json"
    f.write_text(json.dumps({"runs": [dict(run("large-v3-turbo", 1, 0.2, 0.4), cer=0.1,
                                           wer_noguard=0.21, rtf_speech=0.5,
                                           latency_sim_p50=3, latency_sim_p95=5, halluc_rate=0,
                                           halluc_rate_noguard=0.2, ts_err_p50=0.2,
                                           ts_err_p95=0.3, segments=10, peak_rss_mb=3000),
                                      {"label": "bad", "config": {}, "error": "x"}]}))
    out = tmp_path / "B.md"
    write_report([f], out, live_file=tmp_path / "none.json")
    txt = out.read_text()
    assert "20.0%" in txt and "FAST" in txt and "error: x" in txt


def test_soak_summary_slope_and_accounting(tmp_path):
    rows = [{"t_h": i / 10, "rss_asr": 1000 + 5 * i, "rss_worker:ch1": 300.0,
             "drift_ms_ch1": (-1) ** i * 20, "fast_emitted": 100 + i, "fast_done": 99 + i,
             "fast_depth": 1} for i in range(20)]
    s = summarize(rows, tmp_path, 2)
    assert abs(s["rss_slope_mb_per_h"]["asr"] - 50) < 1e-6
    assert s["rss_slope_mb_per_h"]["worker:ch1"] == 0
    assert s["fast_unaccounted"] == 0 and s["max_abs_drift_ms"]["ch1"] == 20
    assert summarize([], tmp_path, 1)["samples"] == 0
