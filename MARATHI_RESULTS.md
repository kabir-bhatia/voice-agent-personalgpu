# Marathi — second-language validation

Marathi (`mr`) was brought up on the **same pipeline** as Hindi to validate multi-language support.
Marathi uses the **same Devanagari script** as Hindi, so the prompt structure, tokenizer (danda `।`),
and TTS all work unchanged — only the language config flips.

## What changed to enable Marathi
- `agent.py::_instructions_for(lang)` — per-language system prompt (Hindi/Marathi entries: language
  name, script, number/proper-noun transliteration examples). `VoiceAgent(lang=STT_LANGUAGE[:2])`.
- `STT_LANGUAGE=mr` drives `openai.STT(language=...)`; greeting map already had `mr`
  ("नमस्कार! मी तुमची कशी मदत करू शकतो?").
- `tts_server.py::F5_NUMBER_LANG` — the digit→word backstop emits **Hindi** number words, so it is
  bypassed for non-Hindi; Marathi numbers are spelled by the LLM instead. Flip the whole pipeline
  hi↔mr with `STT_LANGUAGE` + `F5_NUMBER_LANG` and a service restart.

## 1. Latency: Hindi vs Marathi
The models and NFE are language-independent, so latency is unchanged.

| Stage | Hindi | Marathi |
|---|---|---|
| LLM (≈40-token reply) | ~0.17 s | ~0.17 s |
| TTS synth (same-length sentence) | 1.68 s | 1.72 s (within noise) |

**→ No measurable latency delta** switching Hindi → Marathi.

## 2. Accuracy: pronunciation and flow
- **LLM (Gemma-3-12B): excellent.** Fluent, grammatical Marathi in Devanagari, numbers spelled as
  Marathi words (e.g. "आठ लाख ते दहा लाख"). Handles Marathi natively with no prompt changes beyond the
  language name. (One minor factual slip observed on a population figure — a knowledge issue, not
  language quality.)
- **TTS (IndicF5): intelligible**, synthesized from the same model. Note: the reference voice is the
  **Hindi** clip (`hindi_tts_out.wav`), so Marathi is spoken cross-lingually and may carry a slight
  Hindi accent. A dedicated Marathi reference clip is the fix if the accent is judged off.
- **STT (IndicConformer): preliminary.** IndicConformer is multilingual and accepts `mr`; a full live
  spoken-accuracy pass was not completed this round. If mis-transcription appears, the `rnnt` decoder
  (`DECODER=rnnt`) is the accuracy lever.

## 3. Problems found & fixes
| Problem | Fix |
|---|---|
| Number backstop spoke **Hindi** number words inside Marathi | `F5_NUMBER_LANG` bypasses the backstop for non-Hindi; LLM spells numbers in-language |
| System prompt was hardcoded Hindi | refactored to `_instructions_for(lang)` (hi/mr), one env var flips the pipeline |
| TTS accent (cross-lingual, Hindi reference) | **open item** — add a Marathi reference clip if needed |

## Status
Latency parity and LLM quality are **confirmed**. Full live STT accuracy and a Marathi reference voice
are the remaining validation/polish items before Marathi is considered production-ready.
