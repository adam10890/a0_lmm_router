# DOX contract - a0_lmm_router/scripts

## Purpose

Operator scripts for standalone provider, preset rendering, smoke checks, and
host status workflows.

## Ownership

- Scripts may orchestrate local processes but should not hide destructive
  operations.

## Local Contracts

- Keep PowerShell and shell variants aligned when both exist.
- `render_router_preset.py` owns generated Router Mode preset rendering.

## Work Guidance

- Prefer explicit operator output and exit codes.

## Verification

- Run or dry-run focused scripts when script behavior changes.
- Run `python -m py_compile` on touched Python scripts.

## Child DOX Index

No child AGENTS.md files yet.
