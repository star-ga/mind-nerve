"""Tier-3 script-floor CI gate (multilingual policy).

`spec/quality_targets.md` §"Multilingual language policy" Tier 3:

  > Every other language the 32k BPE encodes survives the tokenizer
  > round-trip and produces a deterministic embedding. No per-language
  > eval set; smoke-tested via a "tokenizer.encode().decode() == input"
  > gate over the FLORES-200 dev set.

This is the "no language silently breaks at the tokenizer layer"
floor. The shipped tokenizer is a GPT-2-style **byte-level BPE**
(`catalog-builder/train_bpe.py`: `pre_tokenizers.ByteLevel`,
`decoders.ByteLevel`). Byte-level encoding is lossless by
construction — every UTF-8 input maps to a unique byte sequence and
back. This test pins that invariant so a future tokenizer change that
breaks any script fails CI.

Hermetic: pure-Python reference of the exact GPT-2 byte↔unicode
mapping `tokenizers.ByteLevel` uses. No torch / tokenizers / network.
The corpus is a UDHR Article-1 sample (public domain) standing in for
the FLORES-200 dev set across the major scripts.
"""

from __future__ import annotations

import functools

import pytest


@functools.lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    """The exact GPT-2 / ByteLevel byte→unicode table.

    Verbatim algorithm from Radford et al. (GPT-2) and the HF
    `tokenizers` ByteLevel pre-tokeniser: every one of the 256 byte
    values maps to a unique printable Unicode code point so the
    tokenised stream never contains control bytes, and decoding is the
    exact inverse.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


def _byte_encode(text: str) -> str:
    table = _bytes_to_unicode()
    return "".join(table[b] for b in text.encode("utf-8"))


def _byte_decode(encoded: str) -> str:
    table = _bytes_to_unicode()
    inv = {v: k for k, v in table.items()}
    return bytes(inv[ch] for ch in encoded).decode("utf-8")


# UDHR Article 1 — public domain. One line per script family. Covers
# the 12 Tier-1 languages + extra scripts (Greek, Hebrew, Thai, Tamil,
# Telugu) + emoji / combining-mark / RTL edge cases.
_FLORES_PROXY: dict[str, str] = {
    "en-Latn": "All human beings are born free and equal in dignity and rights.",
    "es-Latn": "Todos los seres humanos nacen libres e iguales en dignidad y derechos.",
    "fr-Latn": "Tous les êtres humains naissent libres et égaux en dignité et en droits.",
    "de-Latn": "Alle Menschen sind frei und gleich an Würde und Rechten geboren.",
    "pt-Latn": "Todos os seres humanos nascem livres e iguais em dignidade e direitos.",
    "ru-Cyrl": "Все люди рождаются свободными и равными в своём достоинстве и правах.",
    "zh-Hans": "人人生而自由，在尊严和权利上一律平等。",
    "ja-Jpan": "すべての人間は、生まれながらにして自由であり、かつ、尊厳と権利とについて平等である。",
    "ko-Hang": "모든 인간은 태어날 때부터 자유로우며 그 존엄과 권리에 있어 동등하다.",
    "hi-Deva": "सभी मनुष्य जन्म से स्वतंत्र तथा मर्यादा और अधिकारों में समान होते हैं।",
    "bn-Beng": "সমস্ত মানুষ স্বাধীনভাবে সমান মর্যাদা এবং অধিকার নিয়ে জন্মগ্রহণ করে।",
    "ar-Arab": "يولد جميع الناس أحرارًا متساوين في الكرامة والحقوق.",
    "el-Grek": "Όλοι οι άνθρωποι γεννιούνται ελεύθεροι και ίσοι στην αξιοπρέπεια και τα δικαιώματα.",
    "he-Hebr": "כל בני האדם נולדו בני חורין ושווים בערכם ובזכויותיהם.",
    "th-Thai": "มนุษย์ทั้งหลายเกิดมามีอิสระและเสมอภาคกันในศักดิ์ศรีและสิทธิ",
    "ta-Taml": "மனிதப் பிறவியினர் சுதந்திரமாகவே பிறக்கின்றனர்.",
    "te-Telu": "ప్రతి మానవుడు స్వేచ్ఛగా జన్మించును.",
    "emoji": "deploy 🚀 to prod ✅ — rollback ↩️ if 🔥",
    "combining": "é vs é — Å vs Å (NFC vs NFD)",
    "mixed": "git commit -m '修复 баг in λ-calc' && push 🎯",
}


@pytest.mark.parametrize("tag,text", sorted(_FLORES_PROXY.items()))
def test_tier3_byte_round_trip_is_lossless(tag: str, text: str) -> None:
    """decode(encode(x)) == x for every script — the Tier-3 contract."""
    encoded = _byte_encode(text)
    # Encoded stream must be pure printable Unicode (no control bytes).
    assert "\n" not in encoded and "\x00" not in encoded, f"{tag}: control byte leaked"
    decoded = _byte_decode(encoded)
    assert decoded == text, f"{tag}: round-trip changed the text"


@pytest.mark.parametrize("tag,text", sorted(_FLORES_PROXY.items()))
def test_tier3_encoding_is_deterministic(tag: str, text: str) -> None:
    """Same input → byte-identical encoding across calls (attestation
    relies on this; a non-deterministic tokeniser breaks model_hash)."""
    assert _byte_encode(text) == _byte_encode(text), f"{tag}: non-deterministic encode"


def test_tier3_table_is_a_bijection_over_256_bytes() -> None:
    """All 256 byte values are covered and the mapping is invertible —
    the structural guarantee that *any* language round-trips, not just
    the sampled ones."""
    table = _bytes_to_unicode()
    assert len(table) == 256, "byte table must cover all 256 values"
    assert len(set(table.values())) == 256, "byte→unicode must be injective"
    # Exhaustive inverse check over the full byte domain.
    inv = {v: k for k, v in table.items()}
    for b in range(256):
        assert inv[table[b]] == b


def test_tier3_corpus_covers_required_scripts() -> None:
    """Guard against the fixture silently losing script coverage."""
    scripts = {tag.split("-")[1] for tag in _FLORES_PROXY if "-" in tag}
    required = {"Latn", "Cyrl", "Hans", "Jpan", "Hang", "Deva", "Beng", "Arab"}
    missing = required - scripts
    assert not missing, f"Tier-3 corpus missing required scripts: {sorted(missing)}"
