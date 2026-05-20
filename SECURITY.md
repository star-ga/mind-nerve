# Security Policy

mind-nerve is a routing component that sits between a user prompt and the
host LLM runtime. It also patches user-level configuration files for the
hosts it integrates with. We take vulnerability reports seriously.

## Reporting a vulnerability

**Please do not file a public GitHub issue** for security vulnerabilities.

Instead, email [`info@star.ga`](mailto:info@star.ga) with:

- A description of the vulnerability and the components affected.
- The mind-nerve version (`mind-nerve --version`) and platform.
- Reproduction steps, including any required configuration.
- The observed impact (e.g. local privilege escalation, data exposure,
  denial of service, integrity loss).
- Optionally, a suggested fix or mitigation.

If you would like an encrypted channel, request a PGP key in your first
email and we will respond with one.

## Response SLA

STARGA Inc. commits to the following response times for vulnerability
reports against the public mind-nerve repository and the
[`mind-nerve`](https://pypi.org/project/mind-nerve/) PyPI package:

| Stage                          | Target SLA      |
| ------------------------------ | --------------- |
| Acknowledge receipt            | within 3 business days |
| Triage decision (in/out of scope, severity) | within 7 business days |
| Fix or mitigation for critical | within 30 days  |
| Fix or mitigation for high     | within 60 days  |
| Fix or mitigation for medium/low | next regular release |

We will keep you updated as the fix progresses and will credit reporters
in the release notes once a patch is published, unless you request to
remain anonymous.

## Supported versions

We support the most recent minor release line on PyPI. Older versions may
receive fixes for critical issues at the maintainers' discretion.

| Version line | Status |
| ------------ | ------ |
| `0.3.x`      | actively supported |
| `0.2.x`      | end-of-life; please upgrade |
| `0.1.x`      | end-of-life; please upgrade |

## Scope

In scope:

- The mind-nerve Python wheel published on PyPI.
- The repository source at <https://github.com/star-ga/mind-nerve>.
- The Phase-1 weights published at
  <https://huggingface.co/star-ga/mind-nerve>.
- The MCP server (`mind-nerve-mcp`), the UNIX-socket daemon
  (`mind-nerve-routed`), the installer (`mind-nerve-install`), and the
  `mind-nerve-preselect` hook.

Out of scope:

- Third-party integrations, downstream forks, or modified builds.
- Issues that require physical access to the host machine or already-root
  privileges to exploit.
- Denial-of-service via resource exhaustion outside default bounds
  (e.g. spamming the daemon socket from the same user account on the
  same host).
- Vulnerabilities in third-party dependencies (`numpy`,
  `sentence-transformers`, `torch`, `huggingface_hub`, `cryptography`,
  `mcp`); please report those upstream. We will follow up on advisories
  that materially affect mind-nerve users.

## What to expect after disclosure

1. We acknowledge the report and begin triage.
2. If the report is in scope, we agree on a coordinated disclosure date
   with the reporter, typically aligned with the next patch release.
3. We prepare a fix on a private branch, test it, and ship a release.
4. We publish a security advisory referencing the CVE (where applicable)
   and crediting the reporter.

## Hardening guidance

Operators integrating mind-nerve into their environments should also:

- Pin the wheel version and the `MIND_NERVE_HF_REVISION` environment
  variable to a known-good revision.
- Run `mind-nerve-routed` as the same UID that owns
  `$XDG_RUNTIME_DIR` (the daemon enforces this; see `_runtime_dir.py`).
- Treat `~/.claude/skills/` and other projected directories as
  derived state — back up the canonical `skills.full/` catalog before
  enabling preselect.
- Keep logging opt-in (see [`docs/privacy.md`](docs/privacy.md)) on
  shared or regulated hosts.
