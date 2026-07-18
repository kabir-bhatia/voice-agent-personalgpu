# voice-agent-personalgpu

Fully **self-hosted, low-latency Hindi voice agent** (speech-to-speech) on a single NVIDIA L4 GPU.
Real-time spoken conversation over WebRTC: **IndicConformer** STT → **Gemma-3-12B** LLM (vLLM) →
**IndicF5** TTS, fronted by a self-hosted **LiveKit** SFU. No cloud AI APIs on the hot path.
Also includes a **lip-synced cartoon avatar** front-end and **Marathi** as a second language.

This repo is a **reproducible snapshot** of the running pipeline — code, exact configs, dependency
lockfiles, systemd units, and deploy scripts. It does **not** contain secrets, model weights, or venvs.

## Layout

| Path | What |
|---|---|
| `agent/` | LiveKit agent (`agent.py`), chat template, `pyproject.toml`/`uv.lock`, `requirements.txt`, `.env.example` |
| `stt/` | IndicConformer STT server + launch script + `requirements.txt` |
| `tts/` | IndicF5 TTS server (+ `/v1/visemes`), `visemes.py`, launch script + reference voice clip + `requirements.txt` |
| `avatar-client/` | Browser **lip-synced cartoon avatar** front-end (static `index.html` + `avatar.js` + vendored LiveKit SDK) |
| `livekit/` | Self-hosted LiveKit SFU (docker compose, Caddy/LiveKit templates, token generator) |
| `systemd/` | `--user` service units for stt / tts / agent |
| `deploy/` | vLLM docker run script + captured environment snapshot |
| `ARCHITECTURE.md` | **Full architecture + every configuration value + step-by-step reproduction** (avatar = §9) |
| `MARATHI_RESULTS.md` | Second-language (Marathi) validation: latency delta + accuracy vs Hindi |

## Quick start

See **[ARCHITECTURE.md](ARCHITECTURE.md) §6** for the full from-scratch reproduction. In short: enable
systemd linger, run `deploy/vllm-run.sh` (LLM), create the three venvs from the `requirements.txt`
files, render + `up.sh` the LiveKit SFU, copy the systemd units and `systemctl --user enable --now stt
tts agent`, then mint a join token with `livekit/gen_token.py`.

## Current tuned config (highlights)

- **LLM:** `gaunernst/gemma-3-12b-it-int4-awq`, awq_marlin, gpu-util 0.55, served as `gemma2-9b`.
- **TTS:** IndicF5 fp32, **nfe 10 (flat)**, cfg 2.0, speed 0.67, sentence-merge 24 — tuned for steady,
  natural pace over raw speed.
- **STT:** IndicConformer 600M multilingual, CTC decoder, TF32.
- **Agent:** silero VAD (0.40s), dynamic endpointing (min 0.1 / max 2.0) with the local Hindi
  turn-detector for grammar-aware waiting; barge-in on; greets once; one-env-var language switch (hi/mr).
- **Avatar:** browser-rendered viseme-rigged cartoon, audio-driven lip-sync, **0 extra GPU**
  (`avatar-client/`; see ARCHITECTURE §9).

> ⚠️ Secrets (LiveKit key/secret, HF token) go only in gitignored `.env` / `.env.local`. Never commit them.
