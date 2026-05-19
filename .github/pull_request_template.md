<!--
Thanks for opening a PR against mind-nerve. Please fill in this template
to help maintainers review your change quickly.

If this PR addresses one of the items from a published audit, link it
explicitly under "Audit finding" below.
-->

## Summary

<!-- One paragraph: what changes, and why. -->

## Motivation

<!--
What user-visible problem does this solve, or which internal invariant
does it tighten? Link to issues, RFCs, or audit findings as needed.
-->

## Audit finding

<!--
If this PR closes one or more items from a published technical audit,
list the audit ID / report and the specific finding it addresses.
Otherwise write "N/A".
-->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (requires version bump and migration notes)
- [ ] Documentation update
- [ ] Test or CI improvement
- [ ] Performance change
- [ ] Refactor (no behaviour change)

## Surface affected

- [ ] Python API (`mind_nerve.route`, `RouteResult`, ...)
- [ ] CLI (`mind-nerve`, `mind-nerve-install`, ...)
- [ ] Daemon (`mind-nerve-routed`) / preselect hook
- [ ] MCP server
- [ ] Training pipeline / dataset tooling
- [ ] Spec / architecture (requires RFC link)
- [ ] Docs / governance only

## Test plan

<!--
Required. Describe how this change was tested. Paste relevant pytest
output, manual repro commands, or CI run links. List any tests added.
-->

- [ ] `ruff check python/mind_nerve` clean
- [ ] `ruff format --check python/mind_nerve` clean
- [ ] `pytest tests/python -q` passing locally
- [ ] New regression test added (if a bug fix)
- [ ] Docs updated (if user-visible behaviour changed)

## Backwards compatibility

<!--
Does this change break the public Python API, CLI flags, daemon socket
protocol, MCP tool schema, or runtime-dir layout? If yes, describe the
migration path.
-->

## Checklist

- [ ] Commits authored as `STARGA Inc <noreply@star.ga>` with no
      co-author lines and no AI-tool attribution.
- [ ] No internal product names leaked into public-surface files.
- [ ] No secrets, tokens, or credentials in the diff.
- [ ] `CHANGELOG.md` updated under `[Unreleased]` if user-visible.
- [ ] I did not bump `python/mind_nerve/__version__` (release commit
      handles that).
