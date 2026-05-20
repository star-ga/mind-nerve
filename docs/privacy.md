# Privacy

mind-nerve runs **locally on the user's machine**. It is designed to keep
prompts and skill catalogues inside the user's own environment.

## Summary

| Property | Default |
| --- | --- |
| Network calls during routing | none |
| Telemetry | none |
| Prompt persistence | none |
| Log files | opt-in, off by default |
| Hugging Face downloads | once per runtime-dir, pinned revision |
| Catalog data sent off-host | none |

## What `mind-nerve route` does

`route(prompt, top_k=k)` is a pure local function. It:

1. Loads the runtime model and the precomputed `route_table.npy` from the
   runtime directory (the first call may download them from Hugging Face;
   see below).
2. Encodes the prompt to a pooled query vector locally.
3. Computes a dense dot product against the in-memory route table.
4. Returns the top-K route descriptors.

The prompt text is held in memory for the duration of the call and is not
written to disk, not logged, and not sent off-host by default.

## What `mind-nerve-routed` and the preselect hook do

`mind-nerve-routed` is a long-lived UNIX-socket daemon listening on
`$XDG_RUNTIME_DIR/mind-nerve.sock` (or the path in `MIND_NERVE_SOCKET`).
It accepts one request line per JSON object, returns one response line,
and never writes to disk during routing.

The `mind-nerve-preselect` `UserPromptSubmit` hook reads the user's
incoming prompt from the host CLI, asks the daemon for the top-K matching
skills, and atomically rewrites `~/.claude/skills/` (or
`$MIND_NERVE_PROJECTED_DIR`) as a directory of symlinks into the user's
real catalog. The prompt itself is forwarded only to the local daemon over
the UNIX socket; it is not persisted.

## What `mind-nerve scan` does

`mind-nerve learn` and the underlying `mind_nerve.discovery.scan(...)`
read local files on the host:

- They walk a user-specified directory (e.g. `~/.agents/skills/`).
- They parse YAML/Markdown frontmatter and short body excerpts.
- They write to the local runtime directory (`route_table.jsonl`,
  `route_table.npy`) using atomic temp-file + rename.
- They do **not** upload, mirror, or otherwise transmit any catalog
  content off-host.

By default `scan` ignores artefacts with no declared license (the
`include_unknown=False` default). License rules are documented in
[`docs/data_governance.md`](data_governance.md).

## Hugging Face model downloads

On first use, mind-nerve auto-downloads the Phase-1 weights from
[`star-ga/mind-nerve`](https://huggingface.co/star-ga/mind-nerve)
into the runtime directory. This is a one-time network call per machine
per runtime-dir.

You can:

- **Pre-seed** the runtime directory and disable the download by setting
  `MIND_NERVE_RUNTIME_DIR` to a populated directory.
- **Pin the revision** with `MIND_NERVE_HF_REVISION=<sha-or-tag>` for
  reproducible artefact pinning.
- **Avoid Hugging Face entirely** by setting
  `HF_HUB_OFFLINE=1` once the runtime directory is populated.

## Logging

The preselect hook supports an opt-in JSONL log at
`~/.mind-nerve/hook.log` (override with `MIND_NERVE_LOG`). It is **off by
default** in a fresh install — the installer must be invoked with
logging-enabled flags, or the user must set `MIND_NERVE_LOG` themselves.
Log lines record the timestamp, the projected skill IDs, and the round-trip
latency. The full prompt text is **not** recorded.

If you choose to enable logging on a shared or regulated host, please
ensure the log file's path is on a per-user volume and that retention is
set according to your organisation's policy. See
[`docs/data_governance.md`](data_governance.md#retention).

## Attestation envelopes

Each inference can emit an attestation envelope tying the request hash,
the model hash, and the result hash into the evidence chain (see
`python/mind_nerve/attestation.py`). Envelopes are produced and held in
memory; mind-nerve does not persist or transmit them. Downstream
consumers (e.g. governance tooling) may choose to do so under their own
policies.

## Telemetry

mind-nerve has **no telemetry**. There is no metrics endpoint, no usage
beacon, no opt-out flag — because there is nothing to opt out of.

If you find any code path in `python/mind_nerve/` that violates this
guarantee, please report it through the channel in
[`SECURITY.md`](../SECURITY.md).

## Data subject rights

Because mind-nerve is local-only, there is no STARGA-side data store to
exercise rights against. All prompts, logs, and catalog content sit on the
user's machine under the user's control; deletion is `rm` on the
appropriate path.

For the Phase-1 training corpus (`star-ga/mind-nerve` on Hugging
Face) and any reports of inappropriate content discovered in the public
catalog, contact [`info@star.ga`](mailto:info@star.ga).
