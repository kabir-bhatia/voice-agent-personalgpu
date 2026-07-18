# Presentation Content — Self-Hosted Voice Agent
*Final Internship Presentation · GlobalLogic · Kabir Bhatia*
*(Content only — design it however you like. Each slide has: title, subtitle/body, and a **Visual/artifact** instruction telling you what graphic to place and where.)*

> **Chart precision note (applies to every graph):** show **2 decimal places** on data labels and axes
> (e.g. `0.54 s`, not `0.5 s`). In the earlier draft the tool rounded to 1 decimal — force 2-decimal
> formatting on the value labels and the axis so the numbers read precisely.

---

## Slide 1 — Title
**Title:** A Self-Hosted, Low-Latency Indian-Language Voice Agent with a Lip-Synced Avatar
**Subtitle:** Real-time spoken conversation — speech in, speech out — on a single GPU. No cloud AI on the hot path. Hindi & Marathi, plus a browser-rendered talking avatar.
**Presenter line (bottom or under title):** **Kabir Bhatia** · Final Internship Presentation · **GlobalLogic**
**Tag chips:** IndicConformer STT · Gemma-3-12B (vLLM) · IndicF5 TTS · LiveKit WebRTC · Viseme Avatar
**🎨 Visual/artifact:** Clean title layout. Put the presenter/GlobalLogic line small under the title or in the bottom-left; if you have a GlobalLogic logo, top-right corner. Row of 5 tag "pills" at the bottom. No chart on this slide.

---

## Slide 2 — The Problem / Goals
**Title:** Real-time Indian-language voice, without the cloud
**Body (bullets):**
- Latency must feel conversational — a real back-and-forth, not walkie-talkie turns.
- Indian languages first — Hindi & Marathi, natural Devanagari pronunciation.
- Fully self-hosted — no per-minute cloud speech/LLM APIs on the hot path; data stays on our box.
- One commodity GPU — the whole stack co-resident on a single NVIDIA L4 (23 GB).
- Extensible — add a lip-synced avatar without a second GPU.

**Three callouts:**
- 🎯 **Target** — end of speech → first audio back in ~1.5 s, steady-state.
- 🔒 **Constraint** — no OpenAI / Deepgram / ElevenLabs on the path; open models only.
- 💽 **Budget** — ~19–20 GB of 23 GB VRAM for STT + LLM + TTS, all at once.

**🎨 Visual/artifact:** Two-column layout — bullets on the left, the 3 callouts as small icon cards on the right (or a bottom row). No chart.

---

## Slide 3 — System Architecture
**Title:** One VM, four services, OpenAI-compatible on loopback
**Diagram (flow):**
Browser (meet.livekit.io *or* avatar-client/) — WebRTC (WSS/TURN, TLS) → **LiveKit SFU** (livekit-server + caddyl4) → **Agent** (livekit-agents 1.6.2: VAD → dynamic endpointing → STT → LLM → TTS) → three loopback model servers:
- **STT :8001** — IndicConformer 600M (CTC, multilingual) — systemd
- **LLM :8000** — Gemma-3-12B AWQ (vLLM, awq_marlin) — docker
- **TTS :8002** — IndicF5 + `/v1/visemes` (flow-matching clone) — systemd

**Note:** All model servers are OpenAI-compatible on loopback → the LiveKit `openai` plugin drives them unmodified. The agent also publishes a viseme timeline on the `lk.viseme` data channel for the avatar (parallel to audio; never touches the tuned audio path).

**🎨 Visual/artifact:** **This is the centerpiece diagram.** Draw the box-and-arrow architecture: Browser box on the left → arrow → a large "GCP L4 VM" container box holding the LiveKit SFU (top), the Agent (middle, full width), and the 3 model-server boxes (STT/LLM/TTS) in a row at the bottom, each with a downward arrow from the Agent. Label the WebRTC link and the loopback ports. (Reference: `ARCHITECTURE.md` §1 ASCII diagram — redraw it as clean vector shapes.)

---

## Slide 4 — Hardware & Hosting
**Title:** Everything co-resident on one L4
**Table:**
| Item | Value |
|---|---|
| Cloud / zone | GCP · asia-south1-b (Mumbai) |
| GPU | 1× NVIDIA L4 · 23 GB |
| Domain | `<ip>.sslip.io` (no DNS needed) |
| Process manager | systemd --user + linger |
| LLM runtime | docker (vLLM) |
| Python | uv, per-service venvs |

**GPU memory budget (~19–20 GB / 23 GB):**
| Component | VRAM |
|---|---|
| LLM (12B AWQ) | 12.60 GB (gpu-util 0.55) |
| TTS (IndicF5) | 2–3 GB |
| STT (Conformer) | 1–2 GB |
| Avatar | ~0 GB |

**Stats:** 4 model servers, 1 GPU · 0 cloud AI calls on the hot path
**Key ops lesson:** tmux/nohup died on SSH disconnect — **systemd --user + `loginctl enable-linger`** is what made the stack survive and auto-restart.

**🎨 Visual/artifact:** Left = the host table. Right = **a horizontal bar chart of the GPU memory budget** (bars: LLM 12.60, TTS 2.50, STT 1.50, Avatar 0.10 GB; label each bar with its value to 2 decimals, cap axis ~15 GB). Put the "4 / 1 / 0" stats as big numbers below the chart, and the ops-lesson as a highlighted note under the table.

---

## Slide 5 — Speech-to-Text
**Title:** IndicConformer 600M — multilingual, on-GPU
**Body (bullets):**
- Model: AI4Bharat IndicConformer 600M, multilingual Indic — language passed per request.
- Decoder: CTC for speed; `rnnt` available as an accuracy lever.
- Serving: FastAPI, OpenAI `/v1/audio/transcriptions` on `:8001`, TF32 + startup warmup.
- Gotcha solved: onnxruntime-gpu 1.20.1 needs CUDA 12 → torch from the cu124 index + `LD_LIBRARY_PATH` to the venv nvidia libs.

**Callouts:** ⚡ ~0.12 s transcription in the steady-state loop · 🌐 one model serves Hindi, Marathi & other Indic languages.

**🎨 Visual/artifact:** Bullets on the left; two stat/callout cards on the right (the "~0.12 s" and "Multilingual"). No chart. Optional: a small waveform → text icon graphic.

---

## Slide 6 — Language Model
**Title:** Gemma-3-12B (AWQ, vLLM) — why 12B, not 4B
**Table:**
| Config | Value |
|---|---|
| Model | `gaunernst/gemma-3-12b-it-int4-awq` |
| Quantization | awq_marlin |
| Max len / util | 4096 · gpu-util 0.55 |
| Served name | gemma2-9b |
| Decode | ~25 tok/s |

**The decision (the key story of this slide):**
- ❌ **4B failed:** Gemma-3-4B emitted Latin words, raw digits and wrong-script (Bengali) characters + repetition — which the TTS **cannot speak** → garbled audio.
- ✅ **12B fixed it:** reliably follows the "Devanagari-only, numbers-as-words, transliterate English" prompt. Trade-off: ~25 tok/s vs ~60 — fine for short conversational replies.

**🎨 Visual/artifact:** Left = config table. Right = **two contrasting callout boxes** (a red-tinted "4B failed" and a green/teal "12B fixed it"). This is a decision-story slide — make the ❌ / ✅ contrast the visual. No chart.

---

## Slide 7 — Text-to-Speech
**Title:** IndicF5 — natural voice, tuned for steady pace
**Tuned-config table:**
| Knob | Value | Why |
|---|---|---|
| NFE | 10 (flat) | consistent pace (6 sounded rushed/child-like) |
| CFG | 2.0 | 0.0 = ~2× faster but rejected on quality |
| dtype | fp32 | bf16 faster but rejected on quality |
| speed | 0.67 | natural, un-rushed |
| sentence-merge | 24 | short replies spoken as one flowing utterance |

**Body (bullets):**
- Flow-matching, zero-shot voice clone from a single reference clip + text.
- Streaming trick: non-streaming model, but the reply is split into sentences and each is returned as soon as ready → low first-audio latency.
- Pinned `transformers==4.49.0` + AI4Bharat f5_tts fork (5.x breaks with a meta-device error).
- Also serves `/v1/visemes` for the avatar.

**🎨 Visual/artifact:** Left = the 3-column config table (highlight the Value column). Right = the 4 bullets. No chart. Optional: emphasize NFE 10 / CFG 2.0 as small "chips".

---

## Slide 8 — The Agent / Turn-Taking
**Title:** Feeling human: grammar-aware endpointing + barge-in
**Six cards:**
- 🕒 **Dynamic endpointing** — local `turn-detector-v1-mini` (hi) scores if you're *semantically* done: complete → commit at min 0.1s; incomplete ("मुझे ये जानना है कि…") → hold to max 2.0s.
- ✂️ **Sentence streaming** — custom Hindi tokenizer maps the danda `।→.` so long replies stream sentence-by-sentence to TTS.
- 🙋 **Barge-in** — user speech interrupts the agent immediately; pipeline stops and restarts on the new turn.
- 👋 **Greets once** — greeting only on the first turn (no robotic नमस्ते every reply — a real bug we fixed).
- 🗣️ **Devanagari-only prompt** — numbers as words, English transliterated; TTS never sees Latin/digits it can't voice.
- 🚫 **Deliberately off** — preemptive generation (no benefit + GPU contention) and cloud-only backchannel; measured, not guessed.

**🎨 Visual/artifact:** A **2×3 grid of icon cards** (one per item above). Keep each card short: icon + bold header + one line. No chart.

---

## Slide 9 — Latency (the headline)
**Title:** End of speech → first audio: ~1.5–1.6 s
**Breakdown chart (horizontal bars, seconds — label to 2 decimals):**
| Stage | Time (s) |
|---|---|
| Endpointing (silence → commit) | 0.54 |
| STT | 0.12 |
| LLM (TTFT + first sentence) | 0.15 |
| TTS first-audio (nfe=10) | 0.83 |
| **End → first audio (total)** | **1.55** |

**Takeaway:** Dominant costs are endpointing and TTS first-byte (IndicF5 is a diffusion model — a single-pass engine is the next lever).

**🎨 Visual/artifact:** **This is the headline chart.** Horizontal bar chart of the 4 stages (0.54, 0.12, 0.15, 0.83). **Force 2-decimal data labels** (`0.54 s`, `0.12 s`, `0.15 s`, `0.83 s`) — do NOT let it round to `0.5`/`0.1`. Put the **total `1.55 s`** as a big number in a highlighted box beside the chart. Takeaway line underneath. Optional: color the two biggest bars (endpointing, TTS) in the accent color to show they're the bottlenecks.

---

## Slide 10 — The Optimization Journey
**Title:** Tuned by measurement, not by vibes
**Body (bullets):**
- NFE sweep: 32 → 8 nearly halved TTS first-byte; landed on **10 flat** after 6 sounded rushed/child-like.
- Sentence streaming: return each sentence's audio as it's synthesized → far lower first-audio on long replies.
- fp32 vs bf16 & cfg 2.0 vs 0.0: both ~2× faster — **rejected on quality** after A/B listening.
- Endpointing: shorter silence wait + grammar-aware max-delay so users aren't cut off mid-sentence.
- LLM 4B → 12B: the single biggest quality fix (script fidelity).

**Small callouts:** 32 → 10 NFE steps · every speed lever A/B-gated · BENCH markers in logs → trustworthy per-turn latency.

**🎨 Visual/artifact:** Bullets on the left; 3 small stat cards on the right. **Optional mini-chart:** a "TTS first-byte vs NFE" line/bar (NFE 32 → 16 → 8 → falling latency) to visualize the sweep — if you add it, label values to 2 decimals.

---

## Slide 11 — Multi-Language (Marathi)
**Title:** Marathi as a second language — zero latency cost
**Comparison table:**
| Stage | Hindi (s) | Marathi (s) |
|---|---|---|
| LLM (~40-tok reply) | 0.17 | 0.17 |
| TTS synth (same-length sentence) | 1.68 | 1.72 |

**Body (bullets):**
- Same Devanagari script → same prompt/tokenizer/TTS; flip the whole pipeline with **one env var** (`STT_LANGUAGE` + `F5_NUMBER_LANG`).
- LLM quality: excellent — fluent Marathi, numbers spelled as Marathi words.
- Per-language prompt: `_instructions_for(lang)` (hi/mr).
- Number fix: Hindi number-word backstop bypassed for non-Hindi.
- Open item: a Marathi reference voice (TTS currently cross-lingual from the Hindi clip).

**🎨 Visual/artifact:** Left = the Hindi-vs-Marathi comparison (a small **grouped bar chart** works well here — 2 groups, Hindi vs Marathi, values to 2 decimals — or just the table). Right = the bullets. The story is "the bars are basically equal" → zero latency delta.

---

## Slide 12 — Avatar: Motivation
**Title:** A talking face — for 0 extra GPU
**Body (bullets):**
- Goal: an Animoji-style cartoon that lip-syncs the agent — fast, self-hosted, no new hardware.
- Rejected: photoreal (MuseTalk / Tavus-class) — needs 3–4 GB VRAM + per-frame GPU that would contend with the voice pipeline, and can't render a cartoon.
- Chosen: a viseme-rigged 2D SVG drawn entirely in the browser — the face costs the GPU nothing.

**Compare (2 cards):**
- ❌ Server-side video: photoreal, frame-perfect — but +3–4 GB VRAM, added latency, GPU contention. Wrong tool for a cartoon.
- ✅ Browser viseme rig: cartoon, ~0 GPU, sub-frame audio-aligned. Ships as static files.

**🎨 Visual/artifact:** Bullets on the left; a **side-by-side compare** (❌ vs ✅ cards) on the right. Optional: a screenshot/mockup of the cartoon avatar face. No chart.

---

## Slide 13 — Avatar: How It Works
**Title:** Audio-driven, sample-aligned lip-sync
**Flow:** LLM sentence → TTS + `visemes.py` → `lk.viseme` data channel → Browser → Cartoon (SVG)
**Three cards:**
- 🔡 **Viseme engine** — Devanagari → grapheme→phoneme → a 12-viseme set, timed over the real audio (energy-weighted). Shared by server & client.
- 🎚️ **Audio-driven** — audio played *through* Web Audio → analysed samples = what you hear. RMS → jaw openness (hard noise-gate); spectrum → vowel shape.
- 👄 **Bilabials** — a sharp energy *dip* = म/ब/प closure → snap the lips shut, at the true audio time.

**Honest status:** a working prototype — timing is tight; non-bilabial shape accuracy is approximate. Next: adaptive noise floor, blend the server's energy-accurate phoneme timeline, and a future 3D-mesh rig (the `lk.viseme` channel already feeds it).

**🎨 Visual/artifact:** Top = a **horizontal flow strip** (5 chained boxes for the pipeline above). Middle = 3 explainer cards. Bottom = the "honest status" as a highlighted note. Optional: a screenshot of the avatar with the mouth open + a debug-overlay snapshot. No chart.

---

## Slide 14 — Full Stack & Versions
**Title:** Everything on the box
**Table:**
| Layer | Component |
|---|---|
| Transport | LiveKit SFU + caddyl4 (auto-TLS) |
| Agent | livekit-agents 1.6.2 (openai + silero plugins) |
| LLM | Gemma-3-12B AWQ · vLLM |
| STT | IndicConformer 600M · CTC |
| TTS | IndicF5 · transformers 4.49.0 |
| Avatar | viseme SVG + livekit-client (JS) |
| Turn-taking | silero VAD + turn-detector-v1-mini |
| Runtime | docker (vLLM) + systemd --user |
| Env / deps | uv venvs · CUDA 12 / cu124 |
| Networking | sslip.io · TURN · GCP 1:1 NAT |

**Footnote:** Reproducible snapshot in the repo — code, exact configs, lockfiles, systemd units, deploy scripts. No secrets, no weights.

**🎨 Visual/artifact:** A clean two-column reference table (split the 10 rows into two tables side by side). No chart. Optional: small tech logos next to each component.

---

## Slide 15 — Challenges & Lessons
**Title:** Where the time actually went
**Six cards:**
- 🧟 **Process durability** — tmux/nohup died on SSH disconnect. Fix: systemd --user + `enable-linger` — persistent, auto-restarting.
- 🧩 **CUDA / onnxruntime** — onnxruntime-gpu 1.20.1 ↔ torch mismatch → cu124 wheels + explicit `LD_LIBRARY_PATH`.
- 📌 **Dependency pins** — IndicF5 breaks on transformers 5.x (meta-device) → pin 4.49.0, `low_cpu_mem_usage=False`.
- 🔤 **Script fidelity** — only the 12B reliably stays Devanagari-only; a model-size decision driven by TTS constraints.
- 🎧 **WebRTC sync** — lip-sync must analyse the *played* audio (through Web Audio), not the raw track, to avoid lead/lag.
- 🔒 **Self-hosting LiveKit** — layer4 Caddy TLS + TURN + sslip.io + GCP NAT — replaced LiveKit Cloud entirely.

**🎨 Visual/artifact:** A **2×3 grid of icon cards** (mirror slide 8's card style for consistency). No chart.

---

## Slide 16 — Results & Roadmap
**Title:** Real-time, self-hosted, and extensible
**Stats (big numbers):** ~1.5 s end → first audio · 2 languages (hi · mr) · 1 GPU, 0 cloud AI calls
**Works today:**
- Real-time Hindi & Marathi speech-to-speech, fully self-hosted.
- Grammar-aware turn-taking + barge-in that feels conversational.
- Browser lip-synced avatar prototype at zero GPU cost.

**Roadmap:**
- Lip-sync accuracy pass — adaptive floor + phoneme-timeline blend.
- Host avatar on the VM (HTTPS) — any device, no local server.
- More languages — Gujarati / Punjabi / Bengali (per-script + reference clips).
- Lower TTS latency — single-pass / distilled Indic TTS engine.
- 3D-mesh avatar — driven by the existing viseme channel.

**🎨 Visual/artifact:** Top = a **row of 3 big stat callouts**. Below = two columns: "Works today" (left) and "Roadmap" (right). End with a closing/thank-you line (optionally repeat "Kabir Bhatia · GlobalLogic"). No chart.

---

### Appendix — hard numbers (for reference / backup slides)
- Presenter: **Kabir Bhatia** — Final Internship Presentation, **GlobalLogic**.
- VM: GCP `instance-20260713-005743`, asia-south1-b, static IP `34.100.253.119`, domain `34-100-253-119.sslip.io`.
- LLM: `gaunernst/gemma-3-12b-it-int4-awq`, awq_marlin, max-model-len 4096, gpu-util 0.55, served `gemma2-9b`, ~25 tok/s.
- STT: IndicConformer 600M multilingual, CTC (rnnt optional), `:8001`, onnxruntime-gpu 1.20.1 (CUDA 12), TF32.
- TTS: IndicF5, `:8002`, nfe 10 flat, cfg 2.0, fp32, speed 0.67, sentence-merge 24, transformers 4.49.0.
- Agent: livekit-agents 1.6.2, silero VAD min_silence 0.40, dynamic endpointing min 0.1 / max 2.0.
- Ports: LLM 8000, STT 8001, TTS 8002 (loopback); LiveKit 7880/7881, UDP 50000–60000, TURN 3478/5349, Caddy 443 (public).
- Latency (2-decimal): endpointing 0.54 s, STT 0.12 s, LLM 0.15 s (TTFT ~0.07 s), TTS first-audio 0.83 s, **end→first-audio ≈ 1.55 s** (state as ~1.5–1.6 s).
