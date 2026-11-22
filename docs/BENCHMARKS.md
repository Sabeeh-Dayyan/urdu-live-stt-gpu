# Benchmarks

All numbers below were produced by `stt bench` / `stt bench-live` on the hardware listed.
Nothing is estimated by hand.

## Environment

| | |
|---|---|
| cpu | Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz |
| cores_logical | 16 |
| os | macOS-26.6.2-x86_64-i386-64bit |
| python | 3.11.14 |
| faster_whisper | 1.2.1 |
| ctranslate2 | 4.9.0 |
| onnxruntime | 1.23.2 |
| date_utc | 2026-11-20 20:06 |

## Reference set

`fleurs_ur` — 300 s programme, 24 real Urdu utterances (google/fleurs ur_pk test (CC-BY-4.0)), 3 inserted non-speech sections (music-like jingle / noise / silence) for hallucination measurement. Built by `bench/make_reference.py`.

## Offline results (CPU)

RTF = inference seconds ÷ programme seconds (VAD-gated). Simulated latency = speech_end→final from an event-driven single-channel queue simulation driven by the *measured* inference time of each call (includes VAD silence wait, queueing and packing). `packN` = adaptive packing up to N s per call. WER varies by a few points between runs because Whisper's temperature fallback samples.

| Config | WER | CER | WER no-guard | RTF | RTF/speech | sim. lat p50/p95 (s) | halluc. (guards / none) | ts err p50/p95 (s) | segs | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|
| large-v3-turbo int8 b1 seg8 t8 | 24.3% | 12.9% | 24.3% | 1.162 | 1.450 | 26.78 / 52.93 | 0.0% / 0.0% | 0.20 / 0.51 | 40 | 3369 |
| large-v3-turbo int8 b1 seg8 t8 pack28 | 23.3% | 12.8% | 23.3% | 0.953 | 1.189 | 12.53 / 19.85 | 0.0% / 0.0% | 0.20 / 0.51 | 40 | 5228 |
| large-v3-turbo int8 b2 seg8 t8 pack28 | 25.3% | 14.9% | 25.3% | 0.996 | 1.242 | 14.66 / 20.10 | 0.0% / 0.0% | 0.20 / 0.51 | 40 | 5266 |
| large-v3-turbo int8 b5 seg8 t8 pack28 | 23.7% | 12.9% | 23.7% | 1.003 | 1.251 | 13.23 / 17.75 | 0.0% / 0.0% | 0.20 / 0.51 | 40 | 5270 |
| large-v3-turbo int8 b1 seg6 t8 pack28 | 27.2% | 17.1% | 27.2% | 1.010 | 1.260 | 14.99 / 20.37 | 0.0% / 0.0% | 0.20 / 0.71 | 52 | 5271 |
| large-v3-turbo int8 b1 seg10 t8 pack28 | 23.8% | 12.9% | 23.8% | 0.898 | 1.121 | 11.69 / 15.24 | 0.0% / 0.0% | 0.20 / 0.51 | 33 | 5272 |
| large-v3-turbo int8 b1 seg8 t16 pack28 | 26.4% | 15.3% | 26.4% | 1.034 | 1.290 | 18.99 / 26.13 | 0.0% / 0.0% | 0.20 / 0.51 | 40 | 6955 |
| large-v3 int8 b1 seg8 t8 pack28 | 24.3% | 16.1% | 24.3% | 1.064 | 1.328 | 28.67 / 38.77 | 0.0% / 0.0% | 0.21 / 0.47 | 40 | 10101 |
| large-v3 int8 b5 seg8 t8 pack28 | 23.8% | 16.2% | 23.8% | 1.272 | 1.588 | 76.44 / 104.37 | 0.0% / 0.0% | 0.21 / 0.47 | 40 | 10602 |

## Model decision (from the table above)

* **FAST** = `large-v3-turbo int8 b1 seg8 t8 pack28` — WER 23.3%, RTF 0.953. Within the configured 3-pt margin of large-v3 or the fastest real-time option.
* **REFINE** (best large-v3 config) = `large-v3 int8 b5 seg8 t8 pack28` — WER 23.8%, CER 16.2%, RTF 1.272.
* **Finding:** on this reference set large-v3 gives **no measurable accuracy gain** over turbo (WER within run-to-run noise of ~±2 pts — Whisper's temperature fallback samples), while costing more compute. Configured accordingly: CPU profile REFINE = turbo beam 5 (+ ±0.5 s context, no extra RAM); GPU profile keeps large-v3 pending a broadcast-speech reference set (read speech is the easy case; spontaneous code-switched TV speech is where the larger decoder is expected to matter — must be measured, not assumed).
* Capacity estimate (this CPU, segment mode): FAST+REFINE RTF = 2.225 → ≈ 0.3 channels at 70 % utilisation with refinement, ≈ 0.7 channels FAST-only (REFINE skipped under load). Derived from measured RTF on this programme's speech density.

## Hallucination guards on non-speech (`python -m bench.guard_eval`)

23 non-speech clips (reference jingle/noise/silence sections + synthetic jingles, pink noise, mains hum) fed **directly** to `large-v3-turbo` (bypassing VAD, i.e. the case where VAD is fooled):

* hallucinated text without guards: **23/23**
* after guards: **0/23**

The model emits `موسیقی` ("music") with `no_speech_prob = 0.0` and avg_logprob ≈ −0.3…−0.6, so statistical guards cannot catch it; the boilerplate blocklist (phrase must be ≥ 50 % of the segment) does, without touching real sentences that mention music. In the end-to-end runs the VAD already rejected all inserted non-speech sections (hallucination 0 % with or without guards).

## Live measurements (`bench/live_run.sh` → `stt bench-live`)

One channel, looping Urdu reference programme over UDP through the full system (worker → Redis → ASR → Redis → persister/API). Latency = event publish time − UTC of the end of speech (incl. VAD silence wait, queueing, packing, inference).

### `local_agreement` — 300 s, 37 events

local_agreement mode (step 2 s) on CPU — expected to fall behind: every step re-transcribes up to 15 s of buffer at ~9 s/call (justifies D7: segment mode on CPU).

| Metric | n | p50 | p95 | p99 |
|---|---|---|---|---|
| speech_end→final | 17 | 108.43 | 181.24 | 184.17 |
| speech→partial | 20 | 103.04 | 167.46 | 171.23 |
| queue_wait_fast | 37 | 100.71 | 164.75 | 174.21 |
| word_ts_error_s (signed) | 12 | -0.38 | -0.22 | -0.20 |

* timestamp: median signed error -0.380s; includes sender start-up latency and the TS start offset

### `segment` — 420 s, 72 events

CPU profile (turbo FAST b1 + packing, seg 10, REFINE turbo b5), ffmpeg default sender.

| Metric | n | p50 | p95 | p99 |
|---|---|---|---|---|
| speech_end→final | 49 | 13.58 | 22.16 | 25.53 |
| speech_end→refined | 21 | 110.59 | 178.11 | 180.34 |
| queue_wait_fast | 49 | 4.16 | 12.89 | 15.03 |
| queue_wait_refine | 21 | 96.45 | 161.79 | 164.13 |
| rtf_fast | 49 | 0.88 | 1.56 | 3.55 |
| rtf_refine | 21 | 0.55 | 0.64 | 0.64 |
| word_ts_error_s (signed) | 34 | -0.39 | -0.29 | 0.16 |

* timestamp: median signed error -0.393s vs VAD speech onsets (approximate: includes ffmpeg sender start-up/initial-burst timing; precise accuracy evidence is in the e2e test)

### `segment_burst0` — 300 s, 50 events

Same, sender with `-readrate_initial_burst 0` (encoder-like pacing).

| Metric | n | p50 | p95 | p99 |
|---|---|---|---|---|
| speech_end→final | 34 | 13.07 | 21.95 | 25.13 |
| speech_end→refined | 15 | 109.92 | 169.35 | 174.93 |
| queue_wait_fast | 34 | 3.55 | 11.20 | 14.30 |
| queue_wait_refine | 15 | 95.51 | 152.07 | 157.71 |
| rtf_fast | 34 | 0.90 | 2.00 | 2.90 |
| rtf_refine | 15 | 0.56 | 0.61 | 0.61 |
| word_ts_error_s (signed) | 24 | -0.38 | -0.03 | 0.15 |

* timestamp: median signed error -0.385s vs VAD speech onsets (approximate: includes ffmpeg sender start-up/initial-burst timing; precise accuracy evidence is in the e2e test)

## Confidence vs measured accuracy (`python -m bench.calibration`)

40 reference utterances of `fleurs_ur` transcribed one by one with the FAST decoder (large-v3-turbo int8 b1). Confidence = mean Whisper word probability. A word is correct when the WER alignment matches it to the reference.

| Word confidence ≥ | words kept | correct among kept | correct among the rest |
|---|---|---|---|
| 0.00 | 100.0% (1281) | 82.2% | — |
| 0.50 | 99.5% (1274) | 82.3% | 57.1% |
| 0.70 | 97.0% (1242) | 83.7% | 33.3% |
| 0.80 | 94.1% (1205) | 85.4% | 31.6% |
| 0.90 | 86.1% (1103) | 88.8% | 41.0% |
| 0.95 | 77.7% (995) | 91.3% | 50.7% |

| Segment confidence ≥ | segments kept | accuracy of kept (1 − WER) |
|---|---|---|
| 0.00 | 100.0% (40) | 80.4% |
| 0.50 | 100.0% (40) | 80.4% |
| 0.70 | 100.0% (40) | 80.4% |
| 0.80 | 100.0% (40) | 80.4% |
| 0.90 | 97.5% (39) | 80.5% |
| 0.95 | 75.0% (30) | 80.9% |

* **Finding:** whole-segment confidence barely separates good from bad segments (Whisper is over-confident at segment level), so filtering sentences by confidence hides text without making the rest much more accurate. Word confidence *is* informative: low-confidence words are wrong most of the time, which is why the UI underlines words below 90 %. Read speech only — re-measure on broadcast references.

## Soak (`bench/soak_run.sh`)

Ran **4.50 h** (requested 4.5 h; the spec's 24 h could not fit in one night) — 1 channel, looping Urdu programme with a PTS reset every loop (~8.4 min), CPU profile, samples every 60 s.

| Process | RSS first (MB) | RSS last (MB) | slope 2nd half (MB/h) |
|---|---|---|---|
| api | 62.7 | 62.7 | -0.0 |
| asr | 3415.4 | 4572.0 | 107.58 |
| persister | 60.4 | 61.0 | 0.03 |
| worker:ch1 | 161.4 | 351.2 | 0.6 |

* max |clock drift| per channel (ms): {'ch1': 508.7}
* FAST segments emitted by workers: 1868, FAST results published: 1867, in flight at end: 2, **unaccounted: 0**

**Interpretation (from `soak.csv` and `soak_supervisor.log`):**

* **No crashes or restarts**: all four processes had restart count 0 across 270 samples, and the
  supervisor log contains no tracebacks or exceptions.
* **No lost segments**: every FAST segment a worker emitted was transcribed and published; the
  2 still in flight at the end are the last seconds before shutdown.
* **Worker, persister and API memory is flat** after warmup. The worker grows 161→342 MB in the first
  20 min while its ring buffer and model buffers fill, then stays at about 350 MB.
* **ASR memory grows ~130 MB/h after warmup (4.02 GB at 0.35 h → 4.57 GB at 4.5 h, roughly
  linear).** Python-side state is bounded: the REFINE queue sat at its cap of 64 for the whole
  run, and per-job `attempts` entries are removed. The suspected cause is native allocator growth in
  CTranslate2/MKL from variable-length packed encoder inputs. This is **not proven**. Over 24 h it
  would reach about 7 GB. **Open item:** for 24/7 operation, either find the cause (e.g. run with
  fixed pack lengths, or use jemalloc on Linux) or have the supervisor restart ASR above an RSS cap.
  A restart is safe because unacked requests stay pending in Redis and are reclaimed.
* **Clock drift**: median -50 ms. The 12 samples above 300 ms (max -509 ms) each follow a
  `discontinuity` event, i.e. the test bed restarting ffmpeg each loop (RESET=1). Each restart sends
  ffmpeg's default 0.5 s `readrate_initial_burst` instantly, so the re-anchor lands about 0.5 s early,
  and drift correction then slews it back. This is a test-bed artifact: a real encoder doesn't burst.
  `BURST=0` avoids it.
* **REFINE on CPU can't keep up with continuous speech**: 633 refined vs 872 `refine_skipped` at
  4.0 h. This is the designed degradation; FAST was never delayed (FAST lag stayed under 15 s).
* After the run, the test bed's ffmpeg survived the shutdown (the RESET loop's grandchild was
  re-parented). Fixed in `testbed/send_loop.sh` (cleanup collects grandchildren first) and
  verified.
