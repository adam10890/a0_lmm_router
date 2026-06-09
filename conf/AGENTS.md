# DOX contract - a0_lmm_router/conf

## Purpose

Router, fleet, provider, model preset, and standalone service configuration
files.

## Ownership

- Config files define default/operator-controlled behavior.
- Secrets and machine-local overrides should stay in env/config locations, not
  committed presets.

## Local Contracts

- `helpers/conf_resolver.py` owns safe path resolution for these files.
- Preset files should remain curated by default; do not expose every local GGUF
  unless explicitly requested.

## Work Guidance

- Update docs/tests when config schema or file names change.

## Verification

- Run config resolver tests for config path/schema changes.

## Child DOX Index

No child AGENTS.md files yet.
