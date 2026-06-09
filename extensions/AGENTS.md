# DOX contract - a0_lmm_router/extensions

## Purpose

Agent Zero hooks for smart routing, context guardrails, tool-call adapters, and
tool/MCP prompt exposure filtering.

## Ownership

- Extensions integrate with Agent Zero lifecycle.
- Helpers own routing, budget, and exposure policy.

## Local Contracts

- Hooks must fail safely and preserve non-local/cloud model behavior unless the
  config explicitly targets local fleet models.

## Work Guidance

- Keep hook logic narrow and delegate policy to helpers.

## Verification

- Run `python -m py_compile` on touched Python extension files.
- Run exposure/context tests when hook policy changes.

## Child DOX Index

No child AGENTS.md files yet.
