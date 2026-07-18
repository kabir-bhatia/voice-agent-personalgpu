# Self-Hosted Hindi Voice Agent — Architecture & Reproduction Guide

A fully self-hosted, low-latency spoken-conversation agent (STT → LLM → TTS) running on a **single
NVIDIA L4 (23 GB)** GCP VM, fronted by a self-hosted **LiveKit** SFU for WebRTC audio. This document
is the authoritative, reproduce-from-scratch reference: every service, port, model, and configuration
value reflects the **currently running** pipeline.

> Language: Hindi (`hi`) on the hot path. Marathi is validated as a second language (same models; see
> "Multi-language" below).
>
> Front-end: either the generic **meet.livekit.io** (audio only) or the bundled **`avatar-client/`** —
> a lightweight browser page that renders a **lip-synced cartoon avatar** (see §9). The avatar adds
> **no GPU load** (it runs entirely in the browser).

---

## 1. Topology

```
 Browser front-end:  meet.livekit.io (audio)   OR   avatar-client/ (lip-synced cartoon)
        │  WebRTC (WSS/TURN, TLS via Caddy on :443) — audio track ▲ + viseme data channel ▲
        ▼
 ┌─────────────────────────────  GCP L4 VM (asia-south1-b, Mumbai)  ─────────────────────────────┐
 │  LiveKit SFU (docker)            livekit/livekit-server + caddyl4                              │
 │     :7880 ws · :7881 tcp · :50000-60000 udp · :3478/:5349 turn · :443 tls                      │
 │        │ loopback audio frames                                                                 │
 │        ▼                                                                                        │
 │  Agent (systemd --user: agent.service)   livekit-agents 1.6.2, agent.py                        │
 │     VAD (silero) → endpointing (dynamic + turn-detector) → STT → LLM → TTS (sentence-streamed) │
 │        │              │                 │            └─ VisemeTTS also publishes a viseme       │
 │        ▼              ▼                 ▼               timeline on the `lk.viseme` data channel │
 │  STT :8001        LLM :8000         TTS :8002                                                   │
 │  IndicConformer   Gemma-3-12B AWQ   IndicF5 (+ /v1/visemes)                                     │
 │  (systemd)        (docker, vLLM)    (systemd)                                                   │
 └────────────────────────────────────────────────────────────────────────────────────────────────┘
```

All three model servers expose **OpenAI-compatible** HTTP APIs on loopback, so the LiveKit `openai`
plugin drives them unmodified. The avatar rides a parallel WebRTC **data channel** and never touches
the tuned audio path.

---

## 2. Host / environment

| Item | Value |
|---|---|
| Cloud | GCP Compute Engine, zone `asia-south1-b` (Mumbai) |
| Instance | `instance-20260713-005743` |
| GPU | 1× NVIDIA L4, 23 GB |
| Public IP | static, e.g. `34.100.253.119` |
| Domain | `34-100-253-119.sslip.io` (IP-dashed sslip.io → no DNS needed) |
| OS | Ubuntu (systemd, docker + nvidia-container-toolkit, NVIDIA driver ≥ 535) |
| Python mgr | [`uv`](https://github.com/astral-sh/uv) per-service venvs |
| Process mgr | **systemd --user** services (`loginctl enable-linger` REQUIRED) + one docker container for vLLM |

**GPU memory budget (all co-resident):** vLLM 12B ≈ 9.3 GB weights (gpu-util 0.55 ≈ 12.6 GB reserved),
STT ≈ 1–2 GB, TTS ≈ 2–3 GB. Total ≈ 19–20 GB of 23 GB. The **avatar adds 0 GB** — visemes are a few ms
of CPU and the cartoon is drawn in the browser.

---

## 3. Services & exact configuration

### 3.1 LLM — Gemma-3-12B (vLLM, docker)
- Image `vllm/vllm-openai:latest`; container name `vllm`; `--restart unless-stopped`; `--gpus all`.
- Model `gaunernst/gemma-3-12b-it-int4-awq`, `--quantization awq_marlin`.
- `--max-model-len 4096`, `--gpu-memory-utilization 0.55`.
- `--served-model-name gemma2-9b` (kept for backward-compat with the agent's `LLM_MODEL`).
- `--chat-template /chat-templates/gemma_system.jinja` (mounted from `agent/gemma_system.jinja`).
- Listens `127.0.0.1:8000`. Launch: `deploy/vllm-run.sh` (needs `HF_TOKEN`).
- **Why 12B, not 4B:** Gemma-3-4B emitted Latin words, raw digits, wrong-script (Bengali) chars and
  repetition that IndicF5 can't speak → garbled audio. 12B follows the Devanagari-only / numbers-as-words
  prompt reliably. Trade-off: ~25 tok/s decode vs ~60 for 4B (fine for short conversational replies).

### 3.2 STT — IndicConformer 600M (systemd: `stt.service`)
- `stt/indic_server.py`, FastAPI, OpenAI `/v1/audio/transcriptions`, `127.0.0.1:8001`.
- Model `ai4bharat/indic-conformer-600m-multilingual` (multilingual; language passed per request).
- Decoder **CTC** (fast; `rnnt` available via `DECODER` env for max accuracy).
- Default language `hi` (`INDIC_LANG`); TF32 matmul + cuDNN autotune on; startup warmup decode.
- Requires `onnxruntime-gpu==1.20.1` (CUDA 12) and `LD_LIBRARY_PATH` → the venv's `nvidia/*/lib` dirs
  (set in `stt/start_stt.sh`). torch built for cu124.

### 3.3 TTS — IndicF5 (systemd: `tts.service`)
- `tts/tts_server.py`, FastAPI, OpenAI `/v1/audio/speech` (streams raw s16le PCM @ 24 kHz), `:8002`.
- Model `ai4bharat/IndicF5` (flow-matching, zero-shot voice clone). Needs **AI4Bharat's f5_tts fork**
  (`pip install git+https://github.com/ai4bharat/IndicF5.git`) and **`transformers==4.49.0`**
  (5.x breaks with a meta-device error; load with `low_cpu_mem_usage=False`).
- **Voice** = single cloned reference: `tts/hindi_tts_out.wav` + `F5_REF_TEXT`
  ("नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?").
- **Current tuned config (env in `tts/start_tts.sh` / defaults in code):**

  | Knob | Value | Meaning |
  |---|---|---|
  | `F5_NFE` / `F5_NFE_SHORT` | **10 / 10** | flow-matching steps, **flat** (same for every sentence → consistent pace). Raised from 6 after 6 sounded rushed/child-like. |
  | `F5_CFG` | **2.0** | classifier-free guidance (stock). `0.0` would ~2× speed but was rejected on quality. |
  | `F5_DTYPE` | **fp32** | bf16 was ~2× faster but rejected on quality. |
  | `F5_SPEED` | **0.67** | <1 = slower speech; natural pace. |
  | `F5_SENT_MERGE_MIN` | **24** | merge sentence fragments shorter than this into the previous chunk so short replies are spoken as one flowing utterance (fixes choppy "word-salad" feel). |
  | `F5_EXPAND_ACRONYMS` | 1 | "IIT" → "आई आई टी". Plus a Hindi number-word normalizer backstop. |

- Non-streaming per utterance, but the reply is split into sentences and each is synthesized+returned
  as soon as ready → low first-audio latency. `nfe=10` gives TTS first-byte ≈ 0.8 s (short chunk).
- **Viseme endpoint (`/v1/visemes`, for the avatar):** `tts/visemes.py` derives a mouth-shape timeline
  from Devanagari text (rule-based grapheme→phoneme → 12-viseme set, timed over the real audio duration
  with an energy-weighted envelope). During synthesis the timeline is cached per sentence; `/v1/visemes`
  serves it (cache hit = audio-accurate, miss = text-only estimate). See §9.
- **`F5_NUMBER_LANG`:** the Hindi number-word backstop only applies when `hi`; other languages rely on
  the LLM spelling numbers in-language (avoids speaking Hindi number words inside Marathi).

### 3.4 Agent (systemd: `agent.service`)
- `agent/agent.py`, `livekit-agents==1.6.2`, `openai` + `silero` plugins.
- **VAD:** silero, `min_silence_duration=0.40`.
- **Endpointing:** `turn_handling={"endpointing": {"mode":"dynamic","min_delay":0.1,"max_delay":2.0}}`.
  Dynamic mode uses the local **`turn-detector-v1-mini`** (supports `hi`) to decide if the utterance is
  semantically complete: complete → commit near `min_delay`; incomplete (e.g. "मुझे ये जानना है कि…")
  → hold up to `max_delay` so the user isn't cut off. (Grammar-aware waiting.)
- **STT/TTS wiring:** `stt.StreamAdapter(openai.STT(...))` and `tts.StreamAdapter(openai.TTS(...),
  sentence_tokenizer=HindiSentenceTokenizer())`. The custom tokenizer maps the Hindi danda `।` → `.`
  so long replies stream sentence-by-sentence instead of after the whole generation.
- **Barge-in:** on by SDK default (`allow_interruptions`) — user speech stops the agent immediately.
- **Not enabled (by design):** preemptive generation (measured no benefit + GPU contention); native
  backchannel ("go on, I'm listening") — cloud-only in the SDK, unavailable with the offline detector.
- **System prompt:** concise-by-default, Devanagari-only, spell numbers as Hindi words, transliterate
  English → Devanagari (TTS can't voice Latin/digits). **Greet only once** (rule 6 — no re-greeting each
  turn). Per-language greeting map for hi/mr/gu/pa/bn.
- **Language switch:** `_instructions_for(lang)` builds per-language instructions (hi/mr today);
  `VoiceAgent(lang=STT_LANGUAGE[:2])` and `openai.STT(language=STT_LANGUAGE)` are driven by one env var,
  so the whole pipeline flips hi↔mr by setting `STT_LANGUAGE` (+ `F5_NUMBER_LANG`) and restarting.
- **Avatar hook:** `VisemeTTS(openai.TTS)` wraps the TTS — for each sentence it publishes a viseme
  timeline on the `lk.viseme` data channel just as the audio is produced. Purely additive; the audio
  path is unchanged. See §9.

### 3.5 LiveKit SFU (docker compose in `livekit/`)
- `livekit/livekit-server` + `livekit/caddyl4`. Config rendered from `*.template` by `up.sh` (envsubst
  from `.env`). Key flag `rtc.use_external_ip: true` (GCP 1:1 NAT). TURN on 3478/5349, TLS via Caddy
  auto-cert (Let's Encrypt) on the sslip.io domain. `gen_token.py` mints join tokens (pure-stdlib JWT).
- **Firewall:** open tcp 443/7880/7881/5349, udp 3478 and 50000-60000 to the VM.

---

## 4. Ports

| Port | Service | Exposure |
|---|---|---|
| 8000 | vLLM (LLM) | loopback |
| 8001 | IndicConformer (STT) | loopback |
| 8002 | IndicF5 (TTS) | loopback |
| 7880 | LiveKit ws signaling | public |
| 7881 | LiveKit tcp fallback | public |
| 50000-60000/udp | WebRTC media | public |
| 3478/udp, 5349/tcp | TURN | public |
| 443 | Caddy TLS (WSS + TURN-TLS) | public |

---

## 5. Measured latency (live Hindi call, steady state)

| Stage | Time |
|---|---|
| Endpointing (silence → commit) | ~0.54 s |
| STT | ~0.12 s |
| LLM (TTFT + first sentence) | ~0.15 s (TTFT ~0.07 s, decode ~25 tok/s) |
| TTS first-audio (nfe=10) | ~0.83 s |
| **End → first audio** | **~1.5–1.6 s** |

Dominant costs are endpointing and TTS first-byte. The floor on TTS is set by IndicF5 being a diffusion
model; a single-pass engine is the next research step if lower latency is needed.

---

## 6. Reproduce from scratch

**Prereqs:** Ubuntu VM w/ NVIDIA L4 + driver ≥535, `docker` + `nvidia-container-toolkit`, `uv`, an HF
token with access to the gated models (IndicConformer, IndicF5, the Gemma-3 AWQ repo).

```bash
# 0. enable lingering so systemd --user services survive logout (REQUIRED)
sudo loginctl enable-linger "$USER"

# 1. LLM (docker)
export HF_TOKEN=...            # gated repos
cd deploy && ./vllm-run.sh     # serves gemma-3-12b AWQ on :8000

# 2. STT venv
cd ../stt && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
#   torch must be cu124: uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
#   start_stt.sh sets LD_LIBRARY_PATH to the venv nvidia cu12 libs.

# 3. TTS venv  (IndicF5 needs the AI4Bharat fork + transformers 4.49.0)
cd ../tts && uv venv .venv-f5 && uv pip install --python .venv-f5/bin/python -r requirements.txt
#   if requirements are incomplete: uv pip install "f5_tts @ git+https://github.com/ai4bharat/IndicF5.git" "transformers==4.49.0"

# 4. Agent venv
cd ../agent && uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env.local     # fill LiveKit key/secret (must match livekit/.env)

# 5. LiveKit SFU
cd ../livekit && cp .env.example .env   # set DOMAIN + a key/secret pair
./up.sh                                  # renders config + docker compose up

# 6. systemd --user services (edit paths in start_*.sh if home differs)
cp ../systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now stt tts agent

# 7. join token
python livekit/gen_token.py --room test --identity me --minutes 120
```

**Model weights** are pulled from Hugging Face into `~/.cache/huggingface` on first run (mounted into
the vLLM container). They are NOT in this repo.

---

## 7. Operational notes (hard-won)

- **systemd --user + linger is mandatory.** Running the servers in `tmux`/`nohup` over one-shot SSH
  killed them on disconnect; systemd user services with linger persist and auto-restart (`Restart=always`).
- **STT CUDA mismatch:** onnxruntime-gpu 1.20.1 needs CUDA 12 → install torch from the cu124 index, and
  point `LD_LIBRARY_PATH` at the venv's `nvidia/*/lib`.
- **TTS transformers pin:** 5.x breaks IndicF5 (meta-device). Pin `transformers==4.49.0`,
  `low_cpu_mem_usage=False`.
- **Reference audio path:** `F5_REF_AUDIO` defaults to `~/v6/hindi_tts_out.wav`; keep the ref clip next
  to the server (here `tts/hindi_tts_out.wav`).
- **Secrets** (LiveKit key/secret, HF token) live only in `.env` / `.env.local` (gitignored). The
  rendered `livekit.yaml` / `caddy.yaml` and `caddy_data/` (TLS certs) are also gitignored.

---

## 8. Multi-language

STT (IndicConformer) and TTS (IndicF5) are both multilingual Indic; the LLM (Gemma-3-12B) handles all
major Indian languages. Marathi (`mr`) uses the **same Devanagari script** as Hindi, so it works with the
existing prompt/tokenizer by switching `STT_LANGUAGE=mr`, the greeting, and the system-prompt language.
See `MARATHI_RESULTS.md` for the latency delta and accuracy findings vs Hindi. Gujarati / Punjabi /
Bengali need per-script prompt + (likely) per-language TTS reference clips — future work.

---

## 9. Lip-sync avatar (`avatar-client/`)

An optional browser front-end that renders a **lip-synced cartoon face** (Animoji-style) instead of the
generic meet.livekit.io audio call. **Design goal: fast + fully self-hosted + zero extra GPU.** A
photoreal approach (MuseTalk/Tavus-class) was rejected — it needs 3–4 GB VRAM and per-frame GPU that
would contend with the voice pipeline, and it can't render a cartoon. Instead the face is a
**viseme-rigged 2D SVG** drawn entirely in the browser.

**Pieces**
- `tts/visemes.py` — Devanagari grapheme→phoneme → **12-viseme set** (Oculus/Preston-Blair:
  `sil PP FF TH DD kk CH SS nn RR aa E ih oh ou`), timed over the real audio with an energy-weighted
  envelope. Shared by the TTS server (`/v1/visemes`, cached during synthesis) and the agent.
- `agent.py::VisemeTTS` — publishes each sentence's timeline on the `lk.viseme` LiveKit data channel.
- `avatar-client/` — static page (`index.html` + `avatar.js`) using the vendored `livekit-client` UMD
  bundle (no CDN). Connects with a join token, plays the agent audio, and animates the mouth.

**Sync approach (current).** The client is **audio-driven**: it routes the agent's audio *through* Web
Audio so the analysed samples are exactly what's played (no lead/lag), then drives the mouth from that
audio in real time — RMS envelope → jaw openness (hard noise-gate, so the mouth shuts the instant audio
stops), spectral bands → vowel shape (rounded oh/ou vs wide E/ih vs sibilant SS/CH), and a sharp
**energy-dip detector** → bilabial/stop **lip closures** (म/ब/प). This gives inherently tight timing
because everything is derived from the played audio. The server viseme timeline (`lk.viseme`) is still
published and is the better source for a future **3D-mesh rig**; it is not required by the current 2D
client. `?debug=1` shows a live RMS/viseme/closure overlay for threshold tuning.

**Run it**
```bash
# serve the static client (any static host; localhost is a secure context so the mic works)
cd avatar-client && python -m http.server 8777 --bind 127.0.0.1
# open in Chrome (mint the token with livekit/gen_token.py --room <room> --identity me):
#   http://localhost:8777/index.html?url=wss://<DOMAIN>&token=<TOKEN>
```

**Known limitations / next steps.** Lip-sync shape accuracy for non-bilabial consonants is approximate
(spectral heuristic); an adaptive noise floor and blending the server's energy-accurate phoneme timeline
are the next refinements. Hosting the page on the VM over HTTPS (so any device can use it without a local
server) is a separate, pending task.
