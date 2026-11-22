# Reference sets (ground truth)

`fleurs_ur.json` + `fleurs_ur.wav` — built by `make reference` (`python -m bench.make_reference`):
40 real Urdu read-speech utterances from Google FLEURS (`ur_pk`, test split, CC-BY-4.0) with their
human transcripts, joined by 0.6–1.6 s pauses, plus 5 inserted non-speech sections (music-like
jingle, pink noise, silence) for hallucination measurement. 16 kHz mono, ~500 s.
`testbed/media/fleurs_ur.ts` is the same programme as broadcast-like MPEG-TS for live tests.

```json
{
  "name": "fleurs_ur", "audio": "testbed/reference/fleurs_ur.wav", "sample_rate": 16000,
  "duration_s": 500.2,
  "segments":  [{"id": "1857", "file": "…wav", "start": 1.0, "end": 10.9,
                 "speech_start": 2.25, "speech_end": 10.39, "text": "اس نے مزید کہا کہ …"}],
  "nonspeech": [{"kind": "jingle", "start": 51.2, "end": 57.9}]
}
```

`start`/`end` delimit the source recording; `speech_start`/`speech_end` are VAD-detected speech
onsets/offsets (FLEURS recordings begin with 0.3–1.3 s of silence) and are used for timestamp
accuracy.

## Your own broadcast footage

1. `testbed/make_ts.sh clip.mp4 testbed/media/clip.ts`
2. `python -m bench.make_reference --draft testbed/media/clip.ts` → `clip.draft.json`
   (offline large-v3, beam 5, VAD on).
3. A human corrects the `text` of every segment (and adds `nonspeech` ranges for music/ads),
   removes `"draft": true`, saves as `testbed/reference/clip.json`.
4. `./stt bench --dataset testbed/reference/clip.json`
