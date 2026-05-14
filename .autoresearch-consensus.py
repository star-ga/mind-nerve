#!/usr/bin/env python3
"""Multi-LLM consensus gate for autoresearch IMPLEMENT-phase commits.

Runs after autorun.py commits a new iteration. Reads the latest diff,
identifies the RFC the agent claims to have IMPLEMENTED, and broadcasts
a review prompt to the fleet (grok, deepseek, mistral) for a structured
approve/reject verdict. Prints `consensus_passed: 0|1` and
`consensus_score: <0-100>` for `.autoresearch-metric.sh` to consume.

If there's no new diff at HEAD (e.g. the agent only marked SKIPPED, or
this is the baseline iteration) the script prints `consensus_skipped: 1`
and exits 0 without calling any LLMs — autorun.py reads no penalty.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

# Use skill-improver's Fleet — same providers, same API keys, no duplication.
sys.path.insert(0, "/home/n/skill-improver/src")
from skill_improver.config import load_config as load_skill_config  # type: ignore
from skill_improver.fleet import Fleet  # type: ignore


REVIEW_PROMPT = """You are reviewing an autonomous code change to mind-nerve, a CPU-first
intent-classification preselector. The change must respect these
non-negotiables — any violation is grounds for rejection:

1. Pure MIND inference path. No PyTorch / ONNX / TF on the hot path.
2. Q16.16 activations + INT8 weights. No FP16 / BF16 / FP32.
3. Cross-arch bit-identity. Same bytes on x86, ARM, CUDA, WebGPU, NPU.
4. ≤30 ms p95 on 4-core x86 at 1024-token cap.
5. Single static binary. No external ML framework dependency.
6. Tamper-evident 212-byte envelope chain. Every inference still emits.

You also reject changes that:
- Claim "IMPLEMENTED" but only add a Status marker without real src/ code change
- Add code that obviously won't compile or has dead branches
- Break a load-bearing test
- Introduce randomness, clock reads, or non-deterministic reduction order
  in the hot path
- Add backwards-INCOMPATIBLE default behavior (every RFC default must
  keep the binary byte-identical to today when disabled)

## The RFC being implemented

{rfc_section}

## The commit message

{commit_msg}

## The diff

{diff}

## Your task

Respond with a single JSON object on one line, nothing else:

{{"verdict":"approve|reject|concerns","score":0-100,"reason":"one sentence"}}

- "approve" + score ≥ 75: change is sound and respects all non-negotiables
- "concerns" + score 40-74: minor issues but not blocking
- "reject" + score < 40: clear violation or broken change
"""


def run(cmd: list[str]) -> str:
    """Run a command, return stdout (empty on error)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout
    except Exception:
        return ""


def latest_diff(cwd: Path) -> str:
    """Return diff HEAD~1..HEAD, capped per-file with src/ prioritised.

    Naive truncation at 20 KB starved the LLM reviewers when the diff
    appended a giant RFC entry to `RFCs/INDEX.md` — the src/*.mind
    changes (the part that actually matters for non-negotiable review)
    fell off the end. We instead diff each file group separately and
    cap each contribution, putting src/ first so it survives.
    """
    # Split the diff by file, prioritise src/, allocate budgets.
    files_out = run(["git", "-C", str(cwd), "diff", "HEAD~1..HEAD", "--name-only"])
    files = [f for f in files_out.splitlines() if f.strip()]

    def diff_for(path: str) -> str:
        return run(["git", "-C", str(cwd), "diff", "HEAD~1..HEAD", "--", path])

    src_files = [f for f in files if f.startswith("src/") or f.endswith(".mind")]
    rfc_files = [f for f in files if f.endswith("RFCs/INDEX.md")]
    other_files = [f for f in files if f not in src_files and f not in rfc_files]

    pieces: list[str] = []
    used = 0
    # src/ gets up to 18 KB total (most important).
    per_src = 18000 // max(1, len(src_files)) if src_files else 0
    for f in src_files:
        d = diff_for(f)
        if len(d) > per_src:
            d = d[:per_src] + f"\n... [truncated for {f}]\n"
        pieces.append(d)
        used += len(d)
    # other files get up to 6 KB total.
    per_oth = 6000 // max(1, len(other_files)) if other_files else 0
    for f in other_files:
        d = diff_for(f)
        if len(d) > per_oth:
            d = d[:per_oth] + f"\n... [truncated for {f}]\n"
        pieces.append(d)
        used += len(d)
    # RFCs/INDEX.md gets the rest, capped at 6 KB. We don't need the
    # whole new RFC body — reviewers only need to see whether a real
    # Status marker was added and roughly which RFC.
    if rfc_files:
        d = diff_for(rfc_files[0])
        cap = 6000
        if len(d) > cap:
            d = d[:cap] + "\n... [truncated: RFC body omitted]\n"
        pieces.append(d)

    return "\n".join(pieces)


def latest_commit_msg(cwd: Path) -> str:
    return run(["git", "-C", str(cwd), "log", "-1", "--pretty=%B"]).strip()


def diff_added_implemented_marker(diff: str) -> bool:
    """Did the latest diff add a real `**Status:** IMPLEMENTED` line?

    Match an added line whose content (after the diff's `+` prefix)
    starts at column 0 with `**Status:**`. This avoids false positives
    when the substring appears mid-sentence in prose (e.g. program.md
    discussing how the marker works).
    """
    for line in diff.splitlines():
        if (
            line.startswith("+")
            and not line.startswith("+++")
            and line[1:].lstrip(" ").startswith("**Status:**")
            and "IMPLEMENTED" in line
        ):
            return True
    return False


def latest_implemented_rfc_section(index_md: Path, diff: str) -> str | None:
    """Find the RFC whose status was JUST set to IMPLEMENTED, return its section."""
    # Scan the diff for an added IMPLEMENTED status, walk back to find the
    # RFC header it belongs to (look in the file, not the diff, since the
    # diff may have only the status line in context).
    text = index_md.read_text(encoding="utf-8")
    headers = [(m.start(), m.group(0)) for m in re.finditer(r"^# RFC-\d+ .*$", text, re.M)]
    if not headers:
        return None
    # Walk through RFCs from last to first; the first one with IMPLEMENTED
    # status is the one most recently implemented (autoresearch generally
    # appends, not edits earlier RFCs).
    for i in range(len(headers) - 1, -1, -1):
        start = headers[i][0]
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        section = text[start:end]
        if re.search(r"^\*\*Status:\*\*\s+IMPLEMENTED", section, re.M):
            # Cap to 4000 chars so prompts stay reasonable.
            return section[:4000]
    return None


def parse_verdict(content: str) -> tuple[str, int, str] | None:
    """Pull a JSON object out of free-form LLM response."""
    m = re.search(r"\{.*?\}", content, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        v = str(d.get("verdict", "")).lower()
        s = int(d.get("score", 0))
        r = str(d.get("reason", ""))[:200]
        return (v, s, r)
    except Exception:
        return None


async def main_async() -> int:
    cwd = Path(".").resolve()
    diff = latest_diff(cwd)
    if not diff.strip():
        print("consensus_skipped: 1  # no diff at HEAD")
        return 0

    if not diff_added_implemented_marker(diff):
        print("consensus_skipped: 1  # no new IMPLEMENTED marker in diff")
        return 0

    rfc_section = latest_implemented_rfc_section(cwd / "RFCs" / "INDEX.md", diff) or "(unavailable)"
    commit_msg = latest_commit_msg(cwd)

    prompt = REVIEW_PROMPT.format(
        rfc_section=rfc_section, commit_msg=commit_msg, diff=diff,
    )

    cfg = load_skill_config()
    fleet = Fleet(cfg)

    # Same three the skill-improver pipeline uses for cross-model critique.
    providers = [p.id for p in cfg.fleet][:3]
    responses = await fleet.broadcast(prompt, providers)

    approves = 0
    rejects = 0
    concerns = 0
    scores: list[int] = []
    notes: list[str] = []

    for r in responses:
        parsed = parse_verdict(r.content)
        if parsed is None:
            notes.append(f"{r.provider_id}: unparseable")
            continue
        verdict, score, reason = parsed
        scores.append(score)
        notes.append(f"{r.provider_id}: {verdict}({score}) {reason}")
        if verdict == "approve":
            approves += 1
        elif verdict == "reject":
            rejects += 1
        else:
            concerns += 1

    # Pass requires at least 2 of 3 approves AND no rejects.
    total = approves + concerns + rejects
    passed = approves >= 2 and rejects == 0 and total >= 2
    avg = sum(scores) // len(scores) if scores else 0

    print(f"consensus_providers: {total}")
    print(f"consensus_approves: {approves}")
    print(f"consensus_concerns: {concerns}")
    print(f"consensus_rejects: {rejects}")
    print(f"consensus_score: {avg}")
    print(f"consensus_passed: {1 if passed else 0}")
    for n in notes:
        print(f"#  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
