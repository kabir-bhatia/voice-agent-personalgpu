"""OpenAI-compatible TTS server wrapping AI4Bharat IndicF5 (natural Hindi, voice-cloning).

Exposes POST /v1/audio/speech (same shape as OpenAI / Kokoro TTS) so the LiveKit `openai.TTS`
plugin can drive it exactly like it drove Kokoro. IndicF5 is flow-matching (non-AR) and clones a
fixed REFERENCE voice; synthesis is whole-utterance (non-streaming), so first-byte == synth time.

The reply text may be code-mixed (Devanagari + English). IndicF5 has no official English, so an
optional acronym-normalisation pass (ACRONYMS env / expand_acronyms) can spell out things like
"IIT" -> "आई आई टी" if pronunciation fumbles.

Run:
    HF_TOKEN=... .venv-f5/bin/uvicorn tts_server:app --host 0.0.0.0 --port 8002
"""
import importlib
import io
import os
import re
import time
from collections import OrderedDict
from itertools import count

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoModel

from visemes import text_to_visemes

MODEL_ID = os.environ.get("F5_MODEL", "ai4bharat/IndicF5")
REF_AUDIO = os.environ.get("F5_REF_AUDIO", os.path.expanduser("~/v6/hindi_tts_out.wav"))
REF_TEXT = os.environ.get(
    "F5_REF_TEXT", "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?"
)
SAMPLE_RATE = 24000  # IndicF5 native output rate; LiveKit openai.TTS reads 24k s16le mono PCM.
# Speed < 1.0 = slower speech (F5 scales duration). 0.67 = natural pace; benchmarked to have
# ~no effect on synth latency (synth time is dominated by NFE, not audio length).
SPEED = float(os.environ.get("F5_SPEED", "0.67"))
# Flow-matching steps. Stock F5 = 32; benchmark showed nfe is the dominant TTS latency lever:
# 16->8 nearly halves first-byte (~1.12s -> ~0.57s). 8 chosen for low latency w/ acceptable quality.
# Adaptive flow-matching steps (user-tuned): short sentences synthesize at fewer steps (faster, quality
# still good on short utterances), longer/more-complex sentences keep more steps for fidelity. Keyed off
# per-request text length — the pipeline streams one sentence per request, so short replies (a single
# short sentence) get NFE_SHORT and long replies (longer content sentences) get NFE_LONG.
NFE_LONG = int(os.environ.get("F5_NFE", "10"))
NFE_SHORT = int(os.environ.get("F5_NFE_SHORT", "10"))
NFE_CHAR_THRESHOLD = int(os.environ.get("F5_NFE_THRESHOLD", "50"))
# Merge consecutive sentence fragments shorter than this into the previous chunk before synth, so a
# short reply (e.g. "नमस्ते, कबीर! मेरा नाम प्रिया है।") is spoken as ONE flowing utterance instead of
# separate synths with a prosody reset between them (which sounded choppy / word-salad). Long replies
# still stream sentence-by-sentence because their sentences exceed this length.
SENT_MERGE_MIN = int(os.environ.get("F5_SENT_MERGE_MIN", "24"))

# --- Latency levers (deep-dive 2026-07-13) ---------------------------------------------------
# Compute dtype: L4 has ~2x bf16 throughput vs fp32. "bf16" wraps the synth forward in autocast;
# weights stay fp32. Watch for NaN/robotic artifacts in the flow-matching ODE (A/B quality-gated).
DTYPE = os.environ.get("F5_DTYPE", "fp32").lower()
_AUTOCAST_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(DTYPE)
# Classifier-free guidance strength. F5 default 2.0 runs the transformer TWICE per ODE step
# (conditional + unconditional). The CFM sampler only skips the second pass when cfg < 1e-5, so
# ONLY cfg=0.0 gives the ~2x transformer speedup (values like 1.0 change quality but not speed).
CFG_STRENGTH = float(os.environ.get("F5_CFG", "2.0"))


def _nfe_for(text: str) -> int:
    return NFE_SHORT if len(text.strip()) < NFE_CHAR_THRESHOLD else NFE_LONG

# Latin-letter -> Devanagari letter-name, so acronyms get spelled out (A I I T -> आई आई आई टी).
_LATIN_LETTER_HI = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ", "G": "जी",
    "H": "एच", "I": "आई", "J": "जे", "K": "के", "L": "एल", "M": "एम", "N": "एन",
    "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर", "S": "एस", "T": "टी", "U": "यू",
    "V": "वी", "W": "डब्ल्यू", "X": "एक्स", "Y": "वाई", "Z": "ज़ेड",
}
# Stand-alone all-caps tokens of length 2-6 (IIT, BTECH, NCR...) -> spelled out in Devanagari.
_ACRONYM = re.compile(r"\b([A-Z]{2,6})\b")
EXPAND_ACRONYMS = os.environ.get("F5_EXPAND_ACRONYMS", "1") == "1"


def expand_acronyms(text: str) -> str:
    if not EXPAND_ACRONYMS:
        return text
    return _ACRONYM.sub(
        lambda m: " ".join(_LATIN_LETTER_HI.get(ch, ch) for ch in m.group(1)), text
    )


# --- Number -> Hindi words (backstop) -------------------------------------------------------
# The LLM is instructed to spell numbers as words, but if an Arabic numeral slips through, IndicF5
# can't pronounce it. Convert integer runs (0-99999, covers years/quantities in conversation) to
# Devanagari number words so they're always spoken.
_HI_ONES = ["शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
    "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
    "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस",
    "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस",
    "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास",
    "इक्यावन", "बावन", "तिरेपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ",
    "इकसठ", "बासठ", "तिरेसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर",
    "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी", "अस्सी",
    "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी", "नब्बे",
    "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे"]


def _int_to_hindi(n: int) -> str:
    if n < 100:
        return _HI_ONES[n]
    if n < 1000:
        h = _HI_ONES[n // 100] + " सौ"
        return h if n % 100 == 0 else h + " " + _int_to_hindi(n % 100)
    if n < 100000:
        t = _int_to_hindi(n // 1000) + " हज़ार"
        return t if n % 1000 == 0 else t + " " + _int_to_hindi(n % 1000)
    return " ".join(_HI_ONES[int(d)] for d in str(n))  # too big: read digit by digit


_DIGITS = re.compile(r"[0-9०-९]+")  # Arabic + Devanagari digits


def _norm_digit(m: re.Match) -> str:
    s = m.group(0).translate({0x0966 + i: ord("0") + i for i in range(10)})  # Devanagari->ASCII
    try:
        return _int_to_hindi(int(s))
    except (ValueError, IndexError):
        return s


# The digit->word backstop emits HINDI number words, so only apply it for Hindi. For other languages
# (e.g. Marathi) it would speak Hindi number words; instead rely on the LLM to spell numbers in-language
# (it is prompted to) and leave any stray digits alone. Set F5_NUMBER_LANG=hi to enable the backstop.
NUMBER_LANG = os.environ.get("F5_NUMBER_LANG", "hi").lower()


def normalize_numbers(text: str) -> str:
    if NUMBER_LANG != "hi":
        return text
    return _DIGITS.sub(_norm_digit, text)


# --- Viseme timeline cache (for the browser lip-sync avatar) --------------------------------
# /v1/audio/speech synthesizes each sentence and, while it has the real waveform, computes an
# energy-accurate viseme timeline and stashes it here keyed on the NORMALIZED sentence text. The
# agent then calls /v1/visemes with the same sentence text to fetch it (a cache hit uses the
# audio-accurate timing; a miss falls back to a text-only estimate). Small LRU — a handful of
# in-flight sentences is all we ever need.
_VISEME_CACHE: "OrderedDict[str, list]" = OrderedDict()
_VISEME_CACHE_MAX = 256


def _norm_for_visemes(text: str) -> str:
    return normalize_numbers(expand_acronyms(text)).strip()


def _viseme_cache_put(norm_text: str, timeline: list) -> None:
    if not norm_text:
        return
    _VISEME_CACHE[norm_text] = timeline
    _VISEME_CACHE.move_to_end(norm_text)
    while len(_VISEME_CACHE) > _VISEME_CACHE_MAX:
        _VISEME_CACHE.popitem(last=False)


print(f"[f5] loading {MODEL_ID} ...")
_t0 = time.time()
# low_cpu_mem_usage=False: IndicF5's __init__ builds a vocos vocoder (real tensor math), which
# breaks under transformers' default meta-device lazy init ("Tensor on device cpu ... meta").
_model = AutoModel.from_pretrained(
    MODEL_ID, trust_remote_code=True, low_cpu_mem_usage=False
)
try:
    _model = _model.to("cuda")
    _DEVICE = "cuda"
except Exception as e:  # pragma: no cover
    print(f"[f5] cuda unavailable ({e}); using CPU")
    _DEVICE = "cpu"
# Slow the speech down a touch (config-driven; forward() reads self.config.speed).
_model.config.speed = SPEED

# IndicF5's forward() calls infer_process with the default nfe_step=32 and no way to override
# per-call. Patch the symbol in the model's own module namespace so we can drop it to NFE_STEP.
_mod = importlib.import_module(type(_model).__module__)
_orig_infer = _mod.infer_process


# Set per-synth by _synth() based on sentence length (see _nfe_for); defaults to the high-quality value.
_current_nfe = NFE_LONG


def _infer_with_nfe(*args, **kwargs):
    kwargs["nfe_step"] = _current_nfe
    kwargs["cfg_strength"] = CFG_STRENGTH  # 0.0 -> skip the unconditional pass (~2x transformer)
    return _orig_infer(*args, **kwargs)


_mod.infer_process = _infer_with_nfe
print(f"[f5] loaded in {time.time() - _t0:.1f}s on {_DEVICE}; ref={REF_AUDIO}; "
      f"speed={SPEED} nfe_short={NFE_SHORT} nfe_long={NFE_LONG} thr={NFE_CHAR_THRESHOLD} "
      f"dtype={DTYPE} cfg={CFG_STRENGTH}")

app = FastAPI()
_REQ_COUNTER = count(1)

# Split a reply into sentences so we can stream audio sentence-by-sentence (first sentence plays
# while the rest synthesises) — turns the non-streaming model into a low-first-byte one.
_SENT_SPLIT = re.compile(r"[^।?!\.]+[।?!\.]?")


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.findall(text) if p.strip()]
    # Merge very short fragments into the previous chunk to avoid choppy tiny synths.
    out: list[str] = []
    for p in parts:
        if out and len(p) < SENT_MERGE_MIN:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out or [text]


def _synth(text: str) -> np.ndarray:
    """text -> float32 mono waveform in [-1, 1] at SAMPLE_RATE."""
    global _current_nfe
    _current_nfe = _nfe_for(text)  # short sentence -> NFE_SHORT, long -> NFE_LONG (read by _infer_with_nfe)
    if _AUTOCAST_DTYPE is not None and _DEVICE == "cuda":
        with torch.autocast("cuda", dtype=_AUTOCAST_DTYPE):
            wav = _model(text, ref_audio_path=REF_AUDIO, ref_text=REF_TEXT)
    else:
        wav = _model(text, ref_audio_path=REF_AUDIO, ref_text=REF_TEXT)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.reshape(-1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:  # some builds return int16-range floats
        wav = wav / 32768.0
    return wav


def _to_pcm16(wav: np.ndarray) -> bytes:
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype("<i2").tobytes()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": _DEVICE, "sample_rate": SAMPLE_RATE}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "tts-1", "object": "model", "owned_by": "indicf5"}]}


@app.post("/v1/audio/speech")
async def speech(request: Request):
    req_id = next(_REQ_COUNTER)
    body = await request.json()
    text = (body.get("input") or "").strip()
    fmt = body.get("response_format", "pcm")
    if not text:
        return Response(content=b"", media_type="application/octet-stream")
    text = normalize_numbers(expand_acronyms(text))

    if fmt == "wav":
        # Non-streamed single WAV (used by test scripts).
        t0 = time.time()
        wav = _synth(text)
        dur = len(wav) / SAMPLE_RATE if len(wav) else 0.0
        print(f"[f5] {time.time()-t0:.2f}s synth -> {dur:.2f}s audio :: {text[:50]!r}")
        print(
            f"BENCH_TTS seq={req_id} kind=full synth_seconds={time.time()-t0:.2f} "
            f"audio_seconds={dur:.2f} rtf={(time.time()-t0)/dur:.2f} text={text[:60]!r}"
        )
        try:
            _viseme_cache_put(text.strip(), text_to_visemes(text, wav, SAMPLE_RATE))
        except Exception as e:
            print(f"[viseme] cache_put failed: {e}")
        buf = io.BytesIO()
        sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    # Default (LiveKit path): raw s16le PCM @ 24k mono, STREAMED per sentence so the first
    # sentence's audio is returned as soon as it's synthesised (low first-byte latency).
    sentences = _split_sentences(text)
    t_req = time.time()

    def _pcm_stream():
        for i, s in enumerate(sentences):
            t0 = time.time()
            wav = _synth(s)
            dur = len(wav) / SAMPLE_RATE if len(wav) else 0.0
            synth = time.time() - t0
            tag = "TTFB" if i == 0 else f"sent{i}"
            print(f"[f5] {tag} {synth:.2f}s synth -> {dur:.2f}s audio "
                  f"(rtf={synth/dur:.2f} since_req={time.time()-t_req:.2f}) :: {s[:40]!r}")
            kind = "ttfb" if i == 0 else "full"
            rtf = synth / dur if dur else 0.0
            print(
                f"BENCH_TTS seq={req_id} kind={kind} nfe={_nfe_for(s)} synth_seconds={synth:.2f} "
                f"audio_seconds={dur:.2f} rtf={rtf:.2f} text={s[:60]!r}"
            )
            # Audio-accurate viseme timeline for the avatar (from the real waveform), cached for
            # the agent's /v1/visemes fetch. Cheap (~ms) and off the audio bytes we yield.
            try:
                _viseme_cache_put(s.strip(), text_to_visemes(s, wav, SAMPLE_RATE))
            except Exception as e:  # never let lip-sync break audio
                print(f"[viseme] cache_put failed: {e}")
            yield _to_pcm16(wav)

    return StreamingResponse(_pcm_stream(), media_type="audio/pcm")


@app.post("/v1/visemes")
async def visemes(request: Request):
    """Return the viseme timeline for a reply's text, for the browser lip-sync avatar.

    The agent calls this per sentence with the SAME text it sent to /v1/audio/speech. We normalize
    identically, split into the same sentences, and concatenate each sentence's timeline (offset by
    its predecessors). Cache hits reuse the audio-accurate timing computed during synthesis; misses
    fall back to a text-only estimate so the avatar still moves.
    """
    body = await request.json()
    text = (body.get("input") or body.get("text") or "").strip()
    if not text:
        return JSONResponse({"visemes": [], "hit": False})

    text = _norm_for_visemes(text)
    timeline: list = []
    offset = 0
    all_hit = True
    for s in _split_sentences(text):
        key = s.strip()
        cached = _VISEME_CACHE.get(key)
        if cached is None:
            all_hit = False
            cached = text_to_visemes(s)  # text-only fallback (no waveform)
        else:
            _VISEME_CACHE.move_to_end(key)
        for f in cached:
            timeline.append({"t": f["t"] + offset, "v": f["v"], "w": f["w"]})
        offset = timeline[-1]["t"] if timeline else offset
    return JSONResponse({"visemes": timeline, "hit": all_hit})


@app.on_event("startup")
def _warmup():
    # First synth pays JIT/cudnn autotune cost (we saw rtf 1.6 on turn 1). Warm it up so the
    # first real reply isn't slow.
    try:
        t0 = time.time()
        _synth("नमस्ते।")
        print(f"[f5] warmup synth in {time.time()-t0:.2f}s")
    except Exception as e:  # pragma: no cover
        print(f"[f5] warmup failed: {e}")
