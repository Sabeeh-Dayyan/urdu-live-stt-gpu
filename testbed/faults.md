# Fault injection

| Fault | How | Expected behaviour |
|---|---|---|
| Stream loss | `make testbed-stop`, wait 10 s, `make testbed` | Worker logs `stream_down`, reconnects with exponential backoff + jitter, emits `stream_up`; clock re-anchors from wall clock; no crash |
| Loop discontinuity | `RESET=1 make testbed` (restarts ffmpeg each loop; plain `-stream_loop` keeps PTS continuous) | One `discontinuity` event per loop (PTS jumps back); UTC stays monotonic |
| Music/jingle | `make_synthetic.sh` includes 6 s of modulated tone; or splice music in with `ffmpeg -f concat` | VAD rejects most of it; anything reaching ASR is caught by hallucination guards (`stt_asr_guard_drops_total`) |
| Packet loss / jitter | macOS has no `tc`. Run the sender in a Linux container with `--cap-add NET_ADMIN`, then `tc qdisc add dev eth0 root netem loss 1% delay 20ms 10ms` | Decoder errors logged/counted, audio continues; PTS gaps < `discontinuity_s` do not re-anchor |
| Oversubscription | `N=6 make testbed` with 6 enabled channels on CPU | FAST lag grows → `stt_asr_fast_lag_alert`=1; REFINE skipped with explicit `refine_skipped` events |
| Clock drift | Replace `-re` in the sender with `-readrate 1.01` | `drift` event after `drift_sustain_s`, gradual slew, monotonic UTC |
