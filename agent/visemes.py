"""Devanagari text -> viseme timeline for the browser lip-sync avatar.

The TTS server (IndicF5) knows both the spoken text and the exact synthesized audio, so it is the
right place to derive mouth shapes. This module does a lightweight, rule-based grapheme->phoneme
pass over Devanagari (an abugida: every consonant carries an inherent 'a' vowel unless a virama or
a dependent vowel sign overrides it), maps each unit to a compact 12-viseme set (Oculus /
Preston-Blair), then lays the units out in time. When the synthesized waveform is available we
distribute timing across its real duration and modulate the jaw-open amount by the local audio
energy so the mouth tracks the actual speech; without audio we fall back to a text-only estimate.

Output: a list of {"t": <ms int>, "v": <viseme>, "w": <0..1 jaw-open>} keyframes. Pure Python +
numpy, a few milliseconds per sentence, no GPU. All current languages (Hindi, Marathi, ...) are
Devanagari, so one map covers them.
"""
from __future__ import annotations

import numpy as np

# 12-viseme set (plus sil). Keep names stable — the browser avatar keys mouth shapes off them.
VISEMES = ("sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou")

# Base jaw-open per viseme (how far the mouth opens for that shape), 0 = shut, 1 = wide.
_OPEN = {
    "sil": 0.0, "PP": 0.06, "FF": 0.12, "TH": 0.25, "DD": 0.30, "kk": 0.30,
    "CH": 0.35, "SS": 0.22, "nn": 0.28, "RR": 0.30,
    "aa": 1.0, "E": 0.6, "ih": 0.45, "oh": 0.8, "ou": 0.55,
}
_VOWELS = frozenset({"aa", "E", "ih", "oh", "ou"})
# Relative duration weight per unit type (vowels hold longer, consonants are brief, pauses longest).
_DUR = {"sil": 1.6}  # default vowel 1.2 / consonant 0.7 applied below

# Consonant letter -> viseme (independent of the inherent/attached vowel).
_CONS = {}
for _chs, _v in [
    ("कखगघङक़ग़ख़ह", "kk"),
    ("चछजझशषज़", "CH"),
    ("टठडढणड़ढ़तथदधन", "DD"),
    ("पफबभम", "PP"),
    ("फ़व", "FF"),
    ("स", "SS"),
    ("नणङञल", "nn"),
    ("रऱ", "RR"),
    ("य", "ih"),
]:
    for _c in _chs:
        _CONS[_c] = _v
# ण/ङ/ञ nasal preference over the DD group above:
for _c in "णङञ":
    _CONS[_c] = "nn"

# Independent vowels -> viseme.
_VOWEL_IND = {}
for _chs, _v in [
    ("अआ", "aa"), ("इई", "ih"), ("उऊ", "ou"), ("एऐ", "E"), ("ओऔ", "oh"), ("ऋ", "ih"),
]:
    for _c in _chs:
        _VOWEL_IND[_c] = _v

# Dependent vowel signs (matras) -> viseme. '' key would be the inherent 'a'.
_MATRA = {}
for _chs, _v in [
    ("ा", "aa"), ("िी", "ih"), ("ुू", "ou"), ("ेै", "E"), ("ोौ", "oh"), ("ृ", "ih"),
]:
    for _c in _chs:
        _MATRA[_c] = _v

_VIRAMA = "्"       # ् suppresses the inherent vowel (consonant cluster)
_ANUSVARA = "ंँ"  # ं ँ nasalisation
_VISARGA = "ः"      # ः


def _grapheme_units(text: str) -> list[str]:
    """Devanagari (+ any) text -> ordered sequence of viseme units."""
    units: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _CONS:
            units.append(_CONS[ch])
            # Look at what follows to decide the attached vowel.
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == _VIRAMA:
                i += 2  # cluster: no vowel, next consonant follows
                continue
            if nxt in _MATRA:
                units.append(_MATRA[nxt])
                i += 2
                continue
            if nxt in _ANUSVARA:
                units.append("aa"); units.append("nn"); i += 2; continue
            units.append("aa")  # inherent schwa
            i += 1
            continue
        if ch in _VOWEL_IND:
            units.append(_VOWEL_IND[ch]); i += 1; continue
        if ch in _MATRA:  # stray matra
            units.append(_MATRA[ch]); i += 1; continue
        if ch in _ANUSVARA:
            units.append("nn"); i += 1; continue
        if ch in _VISARGA:
            units.append("kk"); i += 1; continue
        if ch.isspace() or ch in ".।,?!;:—-\"'()":
            if units and units[-1] != "sil":
                units.append("sil")
            i += 1
            continue
        # Latin / digits / unknown: treat as a light open beat so the mouth still moves.
        if ch.isalnum():
            units.append("aa")
        i += 1
    return units


def _energy_envelope(wav: np.ndarray, sample_rate: int, n: int) -> np.ndarray:
    """RMS energy over `n` equal slices of the waveform, normalised to [0, 1]."""
    if wav is None or wav.size == 0 or n <= 0:
        return np.ones(max(n, 0))
    edges = np.linspace(0, wav.size, n + 1).astype(int)
    env = np.empty(n, dtype=np.float32)
    for k in range(n):
        seg = wav[edges[k]:edges[k + 1]]
        env[k] = float(np.sqrt(np.mean(seg.astype(np.float32) ** 2))) if seg.size else 0.0
    peak = float(env.max())
    return env / peak if peak > 1e-6 else np.zeros(n, dtype=np.float32)


def text_to_visemes(text: str, wav: np.ndarray | None = None, sample_rate: int = 24000,
                    *, est_unit_ms: float = 85.0) -> list[dict]:
    """Return a viseme keyframe timeline for `text`.

    If `wav` is given, timing spans its real duration and jaw-open tracks local audio energy;
    otherwise a text-only estimate (~est_unit_ms per unit) is used.
    """
    units = _grapheme_units(text)
    if not units:
        return [{"t": 0, "v": "sil", "w": 0.0}]

    dur_w = np.array([_DUR.get(u, 1.2 if u in _VOWELS else 0.7) for u in units], dtype=np.float64)
    if wav is not None and getattr(wav, "size", 0) > 0:
        total_ms = 1000.0 * wav.size / sample_rate
    else:
        total_ms = float(dur_w.sum()) * est_unit_ms
    starts_ms = np.concatenate([[0.0], np.cumsum(dur_w)[:-1]]) / dur_w.sum() * total_ms

    env = _energy_envelope(wav, sample_rate, len(units)) if wav is not None else None
    frames: list[dict] = []
    for k, u in enumerate(units):
        base = _OPEN.get(u, 0.3)
        if env is not None:
            w = base * (0.4 + 0.6 * float(env[k]))  # energy-modulated openness
        else:
            w = base
        frames.append({"t": int(round(starts_ms[k])), "v": u, "w": round(min(1.0, w), 2)})
    # Always end shut so the mouth closes after the utterance.
    frames.append({"t": int(round(total_ms)), "v": "sil", "w": 0.0})
    return frames
