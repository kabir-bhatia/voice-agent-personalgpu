import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from itertools import count

if sys.platform == "win32":
    import platform as _platform
    import socket as _socket

    if getattr(_platform, "_uname_cache", None) is None:
        try:
            _platform._uname_cache = _platform.uname_result(
                "Windows", _socket.gethostname(), "", "", ""
            )
        except Exception:
            pass

from dotenv import load_dotenv

load_dotenv(".env.local")

import functools

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    room_io,
    stt as stt_module,
    tts as tts_module,
)
from livekit.agents.tokenize import _basic_sent, token_stream, tokenizer as _tk
from livekit.plugins import noise_cancellation, openai, silero

from visemes import text_to_visemes


def _split_sentences_hi(text, *, min_sentence_len=20, retain_format=False):
    # The framework's English splitter treats only [.!?] as sentence ends, NOT the Hindi danda '।',
    # so a long Hindi reply is seen as ONE giant sentence and is only sent to TTS after the WHOLE
    # LLM generation finishes (~12s of silence). Map danda -> '.' (same length, so char offsets are
    # preserved) so each Hindi sentence is emitted to TTS as soon as the LLM produces it.
    return _basic_sent.split_sentences(
        text.replace("।", "."), min_sentence_len=min_sentence_len, retain_format=retain_format
    )


class HindiSentenceTokenizer(_tk.SentenceTokenizer):
    def __init__(self, *, min_sentence_len: int = 20, stream_context_len: int = 10) -> None:
        self._min = min_sentence_len
        self._ctx = stream_context_len

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [t[0] for t in _split_sentences_hi(text, min_sentence_len=self._min)]

    def stream(self, *, language: str | None = None) -> _tk.SentenceStream:
        return token_stream.BufferedSentenceStream(
            tokenizer=functools.partial(_split_sentences_hi, min_sentence_len=self._min),
            min_token_len=self._min,
            min_ctx_len=self._ctx,
        )

# --- Lip-sync avatar: publish a viseme timeline per spoken sentence -------------------------
# The browser avatar (avatar-client/) lip-syncs a cartoon face. The StreamAdapter tokenizes the
# LLM reply into sentences and calls the inner TTS's synthesize(sentence) once per sentence — the
# exact point where we know each sentence's text just as its audio is about to be produced. We
# derive a viseme timeline from the text (visemes.text_to_visemes) and publish it into the room on
# a data channel; the client anchors each timeline to the real audio onset and animates the mouth.
# This rides a parallel data channel — it never touches the tuned audio bytes or latency.
VISEME_TOPIC = "lk.viseme"
_viseme_seq = count(1)
_room = None  # set to ctx.room in entrypoint() so the TTS wrapper can publish


def _publish_visemes(text: str) -> None:
    room = _room
    if room is None or not text.strip():
        return
    try:
        timeline = text_to_visemes(text)
        payload = json.dumps(
            {"id": next(_viseme_seq), "visemes": timeline}, ensure_ascii=False
        )
        asyncio.get_running_loop().create_task(
            room.local_participant.publish_data(payload, topic=VISEME_TOPIC, reliable=True)
        )
    except Exception as e:  # lip-sync must never break speech
        print(f"[viseme] publish failed: {e}")


class VisemeTTS(openai.TTS):
    """openai.TTS that also publishes a viseme timeline for each sentence it synthesizes."""

    def synthesize(self, text, **kwargs):
        _publish_visemes(text)
        return super().synthesize(text, **kwargs)


STT_BASE_URL = os.environ.get("STT_BASE_URL", "http://localhost:8001/v1")
STT_MODEL = os.environ.get("STT_MODEL", "indic-conformer")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "hi")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma2-9b")

TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://localhost:8002/v1")
TTS_MODEL = os.environ.get("TTS_MODEL", "tts-1")

LANGUAGE_GREETINGS = {
    "hi": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
    "mr": "नमस्कार! मी तुमची कशी मदत करू शकतो?",
    "gu": "નમસ્તે! હું તમારી કેવી રીતે મદદ કરી શકું?",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
    "bn": "নমস্কার! আমি কীভাবে আপনাকে সাহায্য করতে পারি?",
}


def _iso_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def prewarm(proc: JobProcess) -> None:
    # Phase 2: shorten the end-of-speech silence wait (Silero default 0.55s) — the dominant chunk of
    # endpointing. 0.40s trims ~0.16s off every turn. User-validated as feeling good.
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.40)


def _instructions_for(lang: str) -> str:
    # Language name + example number/word transliterations per language. The rest of the prompt
    # structure (concise-by-default, plain speech, Devanagari-only) is shared. All current languages
    # use Devanagari script; add script-specific variants when extending beyond Devanagari.
    cfg = {
        "hi": ("Hindi", "Devanagari (Hindi)",
               "'2024' -> 'दो हज़ार चौबीस', 'IIT Delhi' -> 'आई आई टी दिल्ली'"),
        "mr": ("Marathi", "Devanagari (Marathi)",
               "'2024' -> 'दोन हजार चोवीस', 'IIT Delhi' -> 'आय आय टी दिल्ली'"),
    }
    name, script, examples = cfg.get(lang, cfg["hi"])
    return (
        "You are a friendly voice assistant on a PHONE CALL with an Indian user. "
        "CRITICAL rules for every reply:\n"
        "1. Be concise BY DEFAULT: for normal answers, reply in 1-2 short spoken sentences (max ~30 words). "
        "Keep the first sentence extremely short so you can reply instantly. BUT if the user explicitly asks for a "
        "story, explanation, or a list of things, give the FULL answer in one flowing, continuous reply — do not chop "
        f"it into tiny pieces or stop after one line. Still keep it natural spoken {name}.\n"
        "2. Plain speech ONLY - no markdown, no bullet points, no numbered lists, no asterisks or headings. It will be read aloud by a TTS engine.\n"
        f"3. LANGUAGE: Always reply in {name} only, even if the user speaks English or another "
        f"language. Keep it natural and conversational spoken {name}.\n"
        "4. Sound human and quick. If more detail is needed, give the short version first and offer to elaborate.\n"
        f"5. SCRIPT: Write your ENTIRE reply in {script} script only. The TTS engine "
        f"cannot pronounce Latin letters or digits. Spell ALL numbers as {name} words and transliterate every "
        f"English or proper noun into Devanagari (e.g. {examples}). "
        "Never output Latin characters or Arabic numerals.\n"
        "6. GREETING ONCE: Only greet at the very first message of the call. In EVERY later reply do NOT say "
        "नमस्ते/नमस्कार or repeat the user's name — that sounds robotic. Just answer directly and naturally."
    )


class VoiceAgent(Agent):
    def __init__(self, lang: str = "hi") -> None:
        super().__init__(instructions=_instructions_for(lang))


async def entrypoint(ctx: JobContext):
    global _room
    _room = ctx.room  # let VisemeTTS publish the lip-sync timeline into this room
    vad = ctx.proc.userdata["vad"]
    session = AgentSession(
        vad=vad,
        stt=stt_module.StreamAdapter(
            stt=openai.STT(
                model=STT_MODEL,
                base_url=STT_BASE_URL,
                api_key="local",
                language=STT_LANGUAGE,
            ),
            vad=vad,
        ),
        llm=openai.LLM(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key="local",
            temperature=0.7,
        ),
        tts=tts_module.StreamAdapter(
            tts=VisemeTTS(
                model=TTS_MODEL,
                voice="indicf5",
                api_key="not-needed",
                base_url=TTS_BASE_URL,
                response_format="pcm",
            ),
            sentence_tokenizer=HindiSentenceTokenizer(),
        ),
        # Dynamic endpointing + grammar-aware waiting (Part A): the Hindi-capable turn detector
        # (turn-detector-v1-mini, supports "hi") scores whether the utterance is semantically
        # complete. In dynamic mode the wait slides between min_delay (confident you're done -> fast)
        # and max_delay (looks incomplete -> hold for you to continue). max_delay raised 0.4 -> 2.0 so
        # incomplete sentences like "मुझे ये जानना है कि..." actually wait instead of being cut off,
        # while complete sentences still commit at ~min_delay.
        turn_handling={"endpointing": {"mode": "dynamic", "min_delay": 0.1, "max_delay": 2.0}},
    )

    from livekit.agents import (
        ConversationItemAddedEvent,
        MetricsCollectedEvent,
        UserInputTranscribedEvent,
        metrics as lk_metrics,
    )

    @session.on("user_input_transcribed")
    def _on_user(ev: UserInputTranscribedEvent):
        print(f"\n[USER SAID] {ev.transcript!r} (final={ev.is_final})")
        if ev.is_final:
            print(f"BENCH_USER ts={_iso_utc()} transcript={ev.transcript!r}")

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        lk_metrics.log_metrics(ev.metrics)
        print(f"BENCH_METRIC ts={_iso_utc()} metric={ev.metrics}")

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent):
        role = getattr(ev.item, "role", None)
        if role is None:
            return
        print(f"\n[{role.upper()}] {ev.item.text_content!r}")
        print(f"BENCH_ASSISTANT ts={_iso_utc()} role={role} text={ev.item.text_content!r}")

    await session.start(
        agent=VoiceAgent(lang=STT_LANGUAGE[:2]),
        room=ctx.room,
        room_input_options=room_io.RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    greeting = LANGUAGE_GREETINGS.get(STT_LANGUAGE[:2], LANGUAGE_GREETINGS["hi"])
    await session.generate_reply(
        instructions=(
            "Greet the user warmly in one short sentence in their language "
            f"(e.g. '{greeting}') and ask how you can help."
        )
    )


server = AgentServer(setup_fnc=prewarm)


@server.rtc_session()
async def _entrypoint(ctx):
    await entrypoint(ctx)


if __name__ == "__main__":
    cli.run_app(server)
