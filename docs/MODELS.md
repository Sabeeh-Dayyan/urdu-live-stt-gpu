# Choosing the speech model

## Candidates (research, 2026-11-24)

No open model has a clean, independently measured Urdu result that clearly beats Whisper
large-v3. Most lower published numbers come from fine-tunes that probably trained on the test
corpus (FLEURS or Common Voice) or were scored on very small sets. Self-reported unless noted:

| Model | Published Urdu result | Trained on | Usable here | Notes |
|---|---|---|---|---|
| `openai/whisper-large-v3` (Systran CT2 build) | measured here: 19.95 % WER / 7.70 % CER (FLEURS) | multilingual | **yes: chosen for REFINE + uploads on GPU** | |
| `openai/whisper-large-v3-turbo` | 24.07 % WER FLEURS (zero-shot, per the awaaz card); measured here per utterance: 21.14 % WER / 8.29 % CER | multilingual | yes (current FAST) | 4 decoder layers: fast |
| [`kingabzpro/whisper-large-v3-urdu`](https://huggingface.co/kingabzpro/whisper-large-v3-urdu) | 26.0 % WER / 9.4 % CER, Common Voice 17 test | Common Voice 17 only | yes, after conversion (Apache-2.0) | never saw FLEURS → **FLEURS is a fair test for it**. Measured here: 19.45 % WER / 7.97 % CER, within noise of large-v3 and slower → not adopted |
| [`awaaz-se-alfaaz/whisper-turbo-v3-urdu`](https://huggingface.co/awaaz-se-alfaaz/whisper-turbo-v3-urdu) (LoRA) | FLEURS 24.1 → 16.9 % WER; own YouTube set only 25.3 → 24.2 % | not stated (FLEURS likely) | not evaluated: FLEURS numbers would be contaminated | |
| [`BuzzASR/urdu`](https://arxiv.org/html/2609.09554v2) | worse than zero-shot large-v3 in its own paper (WER 26.8 vs 22.8) | FLEURS + CV | skipped | |
| [Meta Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) 7B / CTC-1B | 3.5 % CER (Meta's own mix); a third-party paper found the 1B ≈ large-v3 with hallucinations on code-switching | 237 h Urdu incl. FLEURS | would need a new backend (fairseq2) and ~17 GB VRAM for 7B | conflicting numbers; candidate for later |
| SeamlessM4T v2 / MMS | read speech better than Whisper, conversational much worse ("WER We Stand", COLING 2025) | — | **CC-BY-NC: not for commercial use** | |

## Measured here

**Result:** stock `large-v3` is the GPU REFINE model. The Urdu fine-tune is not measurably better
(WER 95 % CI −1.5…+0.6 pts against large-v3, CER slightly worse), and turbo is about 1.2 WER points worse.

See [BENCHMARKS.md → Model comparison](BENCHMARKS.md#model-comparison-fleurs-urdu-per-utterance) and
`bench/results/eval_fleurs_models.json` (per-utterance hypotheses and references included).
Reproduce: `python -m bench.eval_models --model large-v3-urdu --model large-v3 --model large-v3-turbo --every 3`.

Caveats that apply to every number:
* FLEURS is **read** speech. Pakistani TV news (music beds, crosstalk, English words) is harder;
  only a hand-corrected sample of your own channels gives the real figure.
* SER (sentence error rate: any difference after normalisation) is close to 100 % for every model
  on FLEURS, because its references have their own spelling and spacing habits (e.g. «مردو» vs
  «مرد و»). WER/CER are the useful comparison; SER is reported because it was asked for.
* Speed numbers were measured on an 8-core laptop CPU (int8). GPU speed has not been measured.

## Using another model

`asr.fast.model` / `asr.refine.model` take a name under `models/whisper/` or a path to any
CTranslate2 Whisper directory. To convert a Transformers checkpoint from Hugging Face:

```bash
python3 scripts/convert_whisper.py <hf_repo> <commit_sha> <name> --quantization float16 --sha256 <model.safetensors sha256>
```

Then point the config at `<name>` and measure it with `python -m bench.eval_models --model <name>`
(read speech) and `stt bench` (live pipeline) before switching production.
