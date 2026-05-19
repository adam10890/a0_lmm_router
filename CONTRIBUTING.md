# Contributing — dual-repo workflow

`lmm-router` (this repo) and `a0_lmm_router` (the Agent Zero plugin) share
a common origin and evolve in parallel. This document explains how to keep
improvements flowing between them without coupling them too tightly.

## Mental model

```
        ┌────────────────────────────────┐
        │  shared lineage (pre-split)    │
        └──────────────┬─────────────────┘
                       │
       ┌───────────────┴────────────────┐
       │                                │
┌──────▼────────┐               ┌───────▼────────┐
│ a0_lmm_router │               │   lmm-router   │
│ (A0 plugin)   │               │  (standalone)  │
│               │  ◄───────►    │                │
│ remote:       │  cherry-pick  │ remote:        │
│ adam10890/    │               │ origin (TBD)   │
│ a0_lmm_router │               │ a0-plugin →    │
└───────────────┘               │ adam10890/...  │
                                └────────────────┘
```

Both checkouts have the SAME `a0-plugin` remote (pointing at the original
GitHub repo) so cherry-picks between them flow through that shared remote.

## Branch naming

| Branch            | Lives in           | Purpose                                |
|-------------------|--------------------|----------------------------------------|
| `main`            | both               | A0-plugin compatible (frozen contract) |
| `standalone`      | lmm-router only    | Decoupling work (env vars, packaging)  |
| `feat/<name>`     | wherever it starts | Shared features — should land on both  |

## Workflow — propagating a feature both ways

### Scenario A: improvement made in standalone, want it in A0 plugin

```bash
# From inside ~/lmm-router/
git checkout standalone
# ... commit your change as commit SHA: abc123

# Push to the shared remote on a feature branch:
git push a0-plugin standalone:feat/my-feature

# Then in ~/agent-zero/agent-zero-2/usr/plugins/a0_lmm_router/:
git fetch origin
git checkout main
git cherry-pick abc123        # or merge feat/my-feature
```

### Scenario B: improvement made in A0 plugin, want it in standalone

```bash
# From inside the A0 plugin:
git commit -m "fix: thing"     # → commit SHA: def456
git push origin main

# Then in ~/lmm-router/:
git fetch a0-plugin
git checkout standalone
git cherry-pick def456
```

### Scenario C: change is standalone-only (e.g. pyproject.toml tweak)

Stay on the `standalone` branch. Never propagate to `main`.

### Scenario D: A0 plugin only (e.g. hooks.py changes)

Push only to the A0 plugin repo. Never touch the `standalone` branch with it.

## What belongs where

| Concern                  | Lives on `standalone`? | Lives on A0 `main`? |
|--------------------------|------------------------|---------------------|
| `pyproject.toml`         | ✅ yes                 | ❌ no               |
| `paths.py`               | ✅ yes                 | ✅ yes (compatible) |
| `hooks.py` (A0 lifecycle)| ❌ no                  | ✅ yes              |
| `launcher.py`            | ✅ yes (refactored)    | ✅ yes (A0 paths)   |
| `api/`, `helpers/`, `mcp_server/` | ✅ yes        | ✅ yes              |
| `webui/`                 | ✅ yes                 | ✅ yes              |
| Docker compose files     | ✅ yes                 | ✅ yes              |

## A divergence policy

When the standalone branch diverges enough that cherry-picks stop working,
that's the signal to either:

1. Promote `standalone` to a real separate repo with its own `origin`, OR
2. Refactor the A0 plugin to depend on `lmm-router` as a pip package
   (so the plugin becomes a thin shim).

We're not there yet. Stay on the cherry-pick rhythm until the friction
demands the next step.
