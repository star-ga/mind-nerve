# MHS Capability Routing in mind-nerve

> Status: roadmap/specification work. Anthropic announced the Model Hardware Standard (MHS) research preview on 2026-08-27. The final open-source MHS specification is not public yet. This document therefore defines MIND's stable internal boundaries and an MHS compatibility seam without claiming conformance to an unpublished specification. When the normative MHS specification is released, the adapter and conformance layer MUST be reconciled before any interoperability claim is made.

`mind-nerve` remains a **router/preselector, not a tool executor**. MHS/device integration must not
change that product boundary.

## What mind-nerve may consume

A normalized device-capability catalog derived from public DeviceManifest IR:

```text
capability_id
effect class
semantic tags
input/output types
units
bounds summary
availability/health summary
```

The router may rank/select which semantic capability or downstream specialist is relevant to a user
request.

## What it must never do

- hold device credentials;
- issue device writes;
- call vendor SDKs;
- mint authorization/grants;
- treat `device advertises capability` as `user/agent is permitted to use capability`.

The selected capability is passed to the host/orchestrator, which owns execution and approval.

## Public interoperability benefit

This allows a public mind-nerve installation to route toward MHS/device capabilities without requiring
private runtime code or exposing proprietary device policy. Capability metadata stays semantic and
portable across manufacturers.

## Roadmap

1. Add `device_capability` catalog kind.
2. Train/evaluate routing on observe vs effectful capability distinctions.
3. Include effect class as a routing feature but never as an authorization decision.
4. Add negative tests proving a routed capability cannot execute inside mind-nerve.
