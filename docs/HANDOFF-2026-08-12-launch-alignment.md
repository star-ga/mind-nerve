# Handoff — mind-nerve launch alignment + native-MIND port

**Created:** 2026-08-12 · **Origin:** Naestro session `agent:main:main`
**For:** the session completing the full native-MIND port + surface alignment
**Status of this doc:** every claim below was verified by direct inspection on
2026-08-12, not recalled. Where something is unverified it says so explicitly.

---

## 1. Why this handoff exists

A Phase-1 explainer video for mind-nerve is **rendered and gate-passed**, and
launch copy for X + LinkedIn is **drafted**. Publishing was held because
pre-flight verification turned up three real inconsistencies between the video,
the repo, the website, and PyPI.

Then the scope changed: the operator confirmed the Python package is being
**ported to native MIND, and there will probably be no PyPI**. That makes some
of the blockers moot and changes the launch narrative.

This doc hands off both halves: the launch artifacts, and the alignment work.

---

## 2. Verified state (2026-08-12)

### 2.1 Version — four surfaces, four answers

| Surface | Version | Notes |
|---|---|---|
| **Repo (canonical)** | **`0.3.0b9`** | `python/mind_nerve/__init__.py:18`; `pyproject.toml` reads it via `version = { attr = "mind_nerve.__version__" }` |
| GitHub releases | `v0.3.0-beta.2` | published 2026-05-18 — 7 betas behind |
| PyPI (live) | `0.2.0` | a full minor behind; what `pip install mind-nerve` serves today |
| mindlang.dev | `0.1.0a13` | badly stale; also still advertises `pip install mind-nerve` |

`0.3.0b9` is a **beta**. Publishing it as plain `0.3.0` would be a version bump
nobody has actually decided on. That decision is the operator's and was **not
made** — do not assume it.

### 2.2 Latency claim — the video is the only wrong surface

- **README** (`/home/n/mind-nerve/README.md`): warm-daemon p95 **~23 ms on GPU**,
  **~90 ms on a 4-core CPU**. The **≤30 ms-on-CPU** figure is the **Phase 2
  target**, explicitly *not* the Phase 1 result.
- **mindlang.dev**: correctly qualified — "23 ms p95 (warm daemon, GPU)".
- **Video narration (scene 4)**: says *"...with a warm-daemon p95 around twenty
  three milliseconds. All local. No network."* — **unqualified**. Most viewers
  read that as "on my laptop."

**Fix:** re-synthesize scene 4's single narration line with the GPU qualifier,
copying the site's wording. See §4.

Also worth aligning: the site says "CPU fallback when GPU is busy" where the
README gives a measured ~90 ms p95. "Fallback" reads as degradation; the number
reads as engineering. Prefer the number.

### 2.3 Working tree — the native port is already in flight

`git status --short` at HEAD `0de96cc` shows **~60 modified/deleted/untracked
paths**. This is not stray cruft; it is the mindc-0.10.2 port mid-flight:

- **8 `.mind` modules moved out of `src/`** → `src/dormant/`
  (`chain_log`, `clock`, `encoder_kernels`, `evidence`, `inference`, `loader`,
  `model`, `runtime_ffi`, `tokenizer`, plus copies of `lib`/`q16_16`/`sha256`/`top_k`)
- **7 unit tests moved** → `tests/unit/dormant/`
- **New `mind/Mind.toml`** — project manifest; mindc 0.10.2 resolves
  cross-module imports **only in project mode**, with every module listed
  explicitly in `[targets.cpu].sources`
- **New `tests/mindc_gate.sh`** — fail-closed gate over the 17 CI-gated files
  in `mind/`
- New Python surfaces: `acquire.py`, `security_scan.py`, `shared_env.py`
- New installer surface: `verify.ts` + tests
- New `docs/acquisition.md`

**Read `tests/mindc_gate.sh`'s header comment before touching any of this.** It
documents two anti-silent-green findings (audit 2026-08-10, D1/D2) that matter:

> `mindc build` and `mindc test` both exit 0 in degraded states — a module that
> fails to compile is embedded as a runtime-JIT fallback with only a `[WARN]`
> line; a failed native LINK emits a launcher-script fallback; `mindc test`
> prints "running 0 tests / test result: ok" when discovery finds nothing.

The gate therefore never trusts an exit code alone — it pattern-checks output
and asserts exact test counts. **Preserve that discipline.** This is the same
class of trap as the standing rule that a claimed compiler fix is not a
verified fix.

⚠️ **A release cut from this tree would ship whatever is half-finished in those
edits.** Review and commit deliberately before any publish.

### 2.4 Repo visibility

`star-ga/mind-nerve` is **PUBLIC** (verified via `gh repo view`). The link is
safe to post. Description: "Intent-classification preselector for agent runtimes
— decouples library size from token cost".

---

## 3. Launch copy (drafted, not posted)

Numbers used are README-verified: **4,400 skills for the price of 44**,
**11,922 routing candidates**, **96.06% top-5**.

Deliberately excluded: any "infinite capability" claim (the catalog is a real
finite number), and anything about **marketplace or federation** — both are
still `raise NotImplementedError` in the repo.

**Per the operator's "no PyPI probably": `pip install mind-nerve` has been
removed from all copy below.** Promoting an install path you're about to retire
trains people on the wrong entry point.

### X — thread opener

```
A skill your agent can't see is a skill it doesn't have.

Modern libraries hold thousands of skills, tools and MCP servers. Announcing all of them burns the prompt budget every session — so most agents surface ~150 by default. The rest are installed, paid for, and invisible.

mind-nerve fixes that by inverting the order: route first, load one.

Hosting 4,400 skills costs the same prompt budget as hosting 44.
```

### X — reply 1

```
How it works: an intent arrives in plain language, no skill name required. It's matched against 11,922 routing candidates, each embedded once, learned offline, pinned to the local runtime.

Ranked matches above the score floor survive. The winner alone is read from disk, on demand.

96.06% of the time the right route lands in the top five — across all 11,922.
```

### X — reply 2 (install line removed)

```
Phase 1 is shipped and open — Apache-2.0 end to end, weights on Hugging Face, everything runs locally with no network call.

Phase 2 is the real target: a native MIND Q16.16 encoder replacing the PyTorch path. Same top-K, bit-identical across x86, ARM, CUDA and WebGPU. CPU byte-identity is already verified on real hardware; the rest is in progress.
```

### First comment (both platforms)

```
Code, docs and weights: https://github.com/star-ga/mind-nerve

Apache-2.0 end to end. The README tracks Phase 2 honestly — what's landed, what's blocked, and on what.
```

### LinkedIn

Still needs the same surgery as X reply 2: drop the `pip install` clause, swap
in the native-port framing. Otherwise as drafted in the session transcript.

---

## 4. The video

**Rendered artifact:**
`/home/n/hf-projects/mind-nerve-explainer/renders/mind-nerve-explainer_2026-08-12_10-07-39.mp4`

- 1920×1080 h264, 118.7 s, 7.0 MB, audio mean −17.3 dB / peak −1.0 dB
- Voice: **Arnold** (ElevenLabs `VR6AewLTigWG4xSOukaG`, model `eleven_v3`)
- `npm run check`: 0 errors, 0 runtime, 0 layout issues across 9 samples,
  33/33 WCAG AA
- Rachel VO preserved at `vo-rachel-backup/` for A/B

**Outstanding fix — scene 4 narration.** Project at
`/home/n/hf-projects/mind-nerve-explainer/`. Procedure:

1. Edit paragraph 4 of `narration.txt` to qualify the latency ("…on GPU").
2. Re-run `ELEVENLABS_API_KEY=… MIND_NERVE_VOICE_ID=VR6AewLTigWG4xSOukaG python3 synth_vo.py`
   (key is in `/home/n/.claude-ultimate/.env`). It prints measured per-segment
   durations — **use those, never estimates**.
3. Re-time `index.html`: root `data-duration`, the 5 `<audio>` cues, the 5
   `.clip` sections, and the `S1..S5` constants in the script block. Inner
   animation beats are **scene-relative offsets** from `S1..S5`, so updating
   those constants re-syncs everything; only rescale inner offsets if a scene's
   VO length changes materially (scenes 2/3 were rescaled by ratio last time).
4. `npm run check` must pass — watch for clip-overlap errors from rounding
   (a 10 ms overlap at 77.6 s had to be fixed by trimming s3 by 0.01 s).
5. `nice -n 15 npm run render` (U1 CPU budget), then verify the artifact with
   `ffprobe` + an `ffmpeg volumedetect` pass — do not trust the renderer's
   success message alone.

---

## 5. Open questions for the operator

Both were asked and **not yet answered**:

1. **The demo.** Operator says one exists and is being ported. Where does it
   live, and is it public? If public it is the natural CTA replacing
   `pip install`. If not, the CTA is GitHub only.
2. **How public is "no PyPI"?** There's a real difference between "we're not
   promoting pip" (quiet, reversible) and "PyPI is deprecated" (a statement
   people hold you to). **Recommendation: say nothing about deprecation in the
   launch.** Announcing a retirement for a package that still has users, before
   the replacement ships, buys nothing and costs goodwill.

---

## 6. Suggested order of work

1. **Finish the native-MIND port** — the `src/dormant/` split, `mind/Mind.toml`
   project-mode wiring, and `tests/mindc_gate.sh` green (fail-closed, exact
   test counts asserted).
2. **Review and commit the dirty tree** deliberately. Do not bulk-commit.
3. **Decide the version story** — final `0.3.0`, keep betas, or drop PyPI.
4. **Align every surface** to that decision: README, CHANGELOG, ROADMAP,
   mindlang.dev (`0.1.0a13` + the `pip install` line + the PyPI link in the
   three-link row), GitHub release, HF card.
5. **Re-cut scene 4 + re-render** (§4). Independent of 1–4; can run in parallel.
6. **Post** video + first comment.

Steps 5 and 6 are the only ones with drafted artifacts ready to go.

---

## 7. Rails carried over

- Do not put unqualified latency numbers in public copy. GPU vs CPU is the
  distinction that gets checked first.
- Do not claim marketplace or federation — still `NotImplementedError`.
- "Tamper-evident", never "signed" (RFC 0016 Phase C pending).
- mindlang.dev is a public surface: **commit locally, do not push without an
  explicit go-ahead.**
- Verify artifacts, never self-reports. `mindc` exit codes lie in degraded
  states (§2.3).
