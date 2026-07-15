"""OpenAI-compatible STT server wrapping AI4Bharat IndicConformer (Hindi).

Exposes POST /v1/audio/transcriptions (same shape as OpenAI Whisper / faster-whisper-server)
so the LiveKit `openai.STT` plugin + `StreamAdapter` can drive it exactly like it drove
Whisper — the reliable VAD-chunked architecture (no streaming-socket silent-turn bug).

IndicConformer is offline (transcribes a whole utterance). VAD upstream cuts speech into
utterances, each POSTed here. CTC decoding is the default (fast); set ?decoder=rnnt or the
DECODER env for max accuracy.

Run:
    uv run uvicorn indic_server:app --host 0.0.0.0 --port 8001
"""
import io
import os
import tempfile
import time
from itertools import count

import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile
from transformers import AutoModel

MODEL_ID = os.environ.get("INDIC_MODEL", "ai4bharat/indic-conformer-600m-multilingual")
DEFAULT_LANG = os.environ.get("INDIC_LANG", "hi")
DEFAULT_DECODER = os.environ.get("DECODER", "ctc")  # "ctc" (fast) | "rnnt" (accurate)
TARGET_SR = 16000

# Free inference speedup on Ampere+ (L4): allow TF32 matmuls and let cuDNN autotune kernels.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

print(f"[indic] loading {MODEL_ID} ...")
_t0 = time.time()
_model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
try:
    _model = _model.to("cuda")
    _DEVICE = "cuda"
except Exception as e:  # pragma: no cover
    print(f"[indic] cuda unavailable ({e}); using CPU")
    _DEVICE = "cpu"
print(f"[indic] loaded in {time.time() - _t0:.1f}s on {_DEVICE}")

app = FastAPI()
_REQ_COUNTER = count(1)


@app.on_event("startup")
def _warmup():
    # First real transcription pays a cold-start cost (we saw transcription_delay ~3.5s on turn 1).
    # Run a dummy decode at startup so the first user turn is fast.
    try:
        t0 = time.time()
        dummy = torch.zeros(1, TARGET_SR, dtype=torch.float32)  # 1s of silence @ 16k
        with torch.no_grad():
            _model(dummy, DEFAULT_LANG, DEFAULT_DECODER)
        print(f"[indic] warmup decode in {time.time() - t0:.2f}s")
    except Exception as e:  # pragma: no cover
        print(f"[indic] warmup failed: {e}")


def _load_wav(raw: bytes) -> torch.Tensor:
    """Decode arbitrary audio bytes -> mono [1, samples] float tensor at 16 kHz."""
    try:
        wav, sr = torchaudio.load(io.BytesIO(raw))
    except Exception:
        # Some encoders need a real file on disk; fall back to a temp file.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            f.write(raw)
            f.flush()
            wav, sr = torchaudio.load(f.name)
    if wav.shape[0] > 1:  # stereo -> mono
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav


@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL_ID, "object": "model"}]}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": _DEVICE, "decoder": DEFAULT_DECODER}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_ID),
    language: str = Form(default=DEFAULT_LANG),
    response_format: str = Form(default="json"),
    decoder: str = Form(default=DEFAULT_DECODER),
):
    req_id = next(_REQ_COUNTER)
    t0 = time.time()
    raw = await file.read()
    wav = _load_wav(raw)
    lang = (language or DEFAULT_LANG)[:2]  # "hi-IN" -> "hi"
    audio_seconds = wav.shape[-1] / TARGET_SR if wav.shape[-1] else 0.0
    with torch.no_grad():
        text = _model(wav, lang, decoder)
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    text = (text or "").strip()
    latency = time.time() - t0
    print(
        f"BENCH_STT seq={req_id} lang={lang} audio_seconds={audio_seconds:.2f} "
        f"latency_seconds={latency:.2f} text={text[:80]!r}"
    )
    if response_format == "text":
        return text
    return {"text": text}
