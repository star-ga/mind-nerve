#!/usr/bin/env python3
"""Generate mind/luts/vocab_wp.mind — the pinned WordPiece vocabulary +
BertNormalizer exception tables for the native MIND tokenizer.

Source of truth: the runtime checkpoint's tokenizer.json (BERT WordPiece,
30,522 entries) and the `tokenizers` library's BertNormalizer evaluated
per codepoint (so the tables are exact by construction, not by
re-derivation).

Table layout (all heap buffers, i64 lanes unless noted):

  wp_tables_init() returns a 17-lane header block:
    slot 0   vocab_arena   packed token bytes (8 per i64 lane, LE)
    slot 1   vocab_ent_off  per-entry arena byte offset (sorted by bytes)
    slot 2   vocab_ent_len  per-entry byte length
    slot 3   vocab_ent_id   per-entry token id
    slot 4   vocab_n        entry count
    slot 5   drop_ranges    (lo, hi) codepoint pairs, inclusive
    slot 6   n_drop_ranges
    slot 7   decomp_cp      sorted cps that normalize to ' '+CJK+' '
    slot 8   decomp_tgt     the decomposed CJK target cp
    slot 9   n_decomp
    slot 10  map_cp         sorted cps with arbitrary replacements
    slot 11  map_off        arena offset of the replacement bytes
    slot 12  map_len        replacement byte length (0 = drop, but those
                           live in drop_ranges instead)
    slot 13  map_arena      packed replacement bytes
    slot 14  n_map
    slot 15  punct_arr      sorted punctuation codepoints (Unicode P*)
    slot 16  n_punct

Normalization rule for an input codepoint (first match wins):
  1. cp inside a drop range            -> emit nothing
  2. cp in decomp table                -> emit ' ' + utf8(target) + ' '
  3. cp in full map                    -> emit replacement bytes
  4. cp in a CJK range                 -> emit ' ' + utf8(cp) + ' '
  5. otherwise                         -> emit utf8(cp)

Usage:
    python3 tools/gen_vocab_wp.py [--checkpoint ~/.local/share/mind-nerve-runtime/checkpoint]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import unicodedata
from pathlib import Path

# HF `tokenizers` crate _is_chinese_char ranges (BertNormalizer).
CJK_RANGES = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
]


def _is_cjk(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in CJK_RANGES)


def _pack_bytes(data: bytes) -> list[int]:
    """Pack bytes little-endian into i64 lanes (8 per lane, zero-padded).

    Lanes are emitted as SIGNED i64 decimal literals (bit pattern two's
    complement): an unsigned lane >= 2^63 would overflow the parser's
    integer literal range, so bit 63 set means the literal is negative.
    """
    lanes = []
    for i in range(0, len(data), 8):
        chunk = data[i : i + 8]
        lane = 0
        for j, b in enumerate(chunk):
            lane |= b << (8 * j)
        if lane >= 1 << 63:
            lane -= 1 << 64
        lanes.append(lane)
    return lanes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=str(
            Path.home() / ".local/share/mind-nerve-runtime/checkpoint"
        ),
    )
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "mind/luts/vocab_wp.mind"),
    )
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tj = json.loads(Path(args.checkpoint, "tokenizer.json").read_text())
    model = tj["model"]
    assert model["type"] == "WordPiece", model["type"]
    assert model["unk_token"] == "[UNK]"
    assert model["continuing_subword_prefix"] == "##"
    assert int(model["max_input_chars_per_word"]) == 100
    vocab: dict[str, int] = model["vocab"]
    assert len(vocab) == 30522

    norm = Tokenizer.from_file(
        str(Path(args.checkpoint, "tokenizer.json"))
    ).normalizer

    # ── normalizer exception tables (per-codepoint, exact by construction) ──
    exceptions: dict[int, str] = {}
    for cp in range(0x110000):
        ch = chr(cp)
        try:
            out = norm.normalize_str(ch)
        except Exception:
            out = ch
        if _is_cjk(cp) and out == " " + ch + " ":
            continue  # simple pad rule — handled procedurally
        if out != ch:
            exceptions[cp] = out

    drop_cps = sorted(cp for cp, out in exceptions.items() if out == "")
    drop_ranges: list[tuple[int, int]] = []
    for _, g in itertools.groupby(enumerate(drop_cps), lambda t: t[1] - t[0]):
        g = list(g)
        drop_ranges.append((g[0][1], g[-1][1]))

    decomp: dict[int, int] = {}
    full: dict[int, str] = {}
    for cp, out in exceptions.items():
        if out == "":
            continue
        cps = [ord(c) for c in out]
        if len(cps) == 3 and cps[0] == 32 and cps[2] == 32 and _is_cjk(cps[1]):
            decomp[cp] = cps[1]
        else:
            full[cp] = out

    punct = sorted(
        cp for cp in range(0x110000) if unicodedata.category(chr(cp)).startswith("P")
    )

    # ── vocab sorted by raw UTF-8 bytes ──
    entries = sorted(vocab.items(), key=lambda kv: kv[0].encode("utf-8"))
    vocab_arena = bytearray()
    ent_off: list[int] = []
    ent_len: list[int] = []
    ent_id: list[int] = []
    for token, tid in entries:
        b = token.encode("utf-8")
        ent_off.append(len(vocab_arena))
        ent_len.append(len(b))
        ent_id.append(tid)
        vocab_arena += b

    map_arena = bytearray()
    map_cp: list[int] = []
    map_off: list[int] = []
    map_len: list[int] = []
    for cp in sorted(full):
        b = full[cp].encode("utf-8")
        map_cp.append(cp)
        map_off.append(len(map_arena))
        map_len.append(len(b))
        map_arena += b

    vocab_lanes = _pack_bytes(bytes(vocab_arena))
    map_lanes = _pack_bytes(bytes(map_arena))

    out_path = Path(args.out)
    with out_path.open("w") as f:
        f.write(
            "// vocab_wp.mind — pinned WordPiece vocab + BertNormalizer tables.\n"
            "// Auto-generated by tools/gen_vocab_wp.py. DO NOT edit by hand.\n"
            "// Regenerate: python3 tools/gen_vocab_wp.py\n"
            "//\n"
            "// Source: the runtime checkpoint's tokenizer.json (BERT WordPiece,\n"
            "// 30522 entries) + the `tokenizers` BertNormalizer evaluated per\n"
            "// codepoint (exact by construction). Layout is documented in the\n"
            "// generator docstring and mind/kernels/wordpiece.mind.\n"
            "// Bit-identity: byte-identical tables on every substrate; the\n"
            "// corpus gate (tests/python/test_tokenize_corpus.py) proves the\n"
            "// pipeline against the HF fast tokenizer byte-for-byte.\n\n"
        )

        def emit_i64_array(name: str, vals: list[int]) -> None:
            f.write(f"pub fn {name}() -> i64 {{\n")
            f.write(f"    let buf: i64 = __mind_alloc({len(vals) * 8});\n")
            for i, v in enumerate(vals):
                f.write(f"    __mind_store_i64(buf + {i * 8}, {v});\n")
            f.write("    buf\n}\n\n")

        emit_i64_array("wp_vocab_arena_init", vocab_lanes)
        emit_i64_array("wp_vocab_off_init", ent_off)
        emit_i64_array("wp_vocab_len_init", ent_len)
        emit_i64_array("wp_vocab_id_init", ent_id)
        emit_i64_array("wp_drop_ranges_init", [v for r in drop_ranges for v in r])
        emit_i64_array("wp_decomp_cp_init", sorted(decomp))
        emit_i64_array("wp_decomp_tgt_init", [decomp[cp] for cp in sorted(decomp)])
        emit_i64_array("wp_map_cp_init", map_cp)
        emit_i64_array("wp_map_off_init", map_off)
        emit_i64_array("wp_map_len_init", map_len)
        emit_i64_array("wp_map_arena_init", map_lanes)
        emit_i64_array("wp_punct_init", punct)

    total = (
        len(vocab_lanes) * 3 + len(ent_off) * 3 + len(drop_ranges) * 2
        + len(decomp) * 2 + len(map_cp) * 3 + len(map_lanes) + len(punct)
    )
    print(
        f"wrote {out_path}: vocab={len(entries)} entries "
        f"({len(vocab_arena)} arena bytes), drop_ranges={len(drop_ranges)}, "
        f"decomp={len(decomp)}, map={len(map_cp)} ({len(map_arena)} arena bytes), "
        f"punct={len(punct)}; ~{total} table lanes"
    )


if __name__ == "__main__":
    sys.exit(main())
