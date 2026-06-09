# DOX contract - a0_lmm_router/api

## Purpose

Agent Zero Web/API wrappers for router, fleet, model install, hardware, MCP,
and dashboard actions.

## Ownership

- API files should delegate core behavior to helpers/service code.
- They own request validation and UI-oriented response envelopes.

## Local Contracts

- Keep endpoint names aligned with dashboard callers.
- Do not place portable routing policy only in API wrappers.

## Work Guidance

- Preserve explicit error responses for missing fleet/router components.

## Verification

- Run `python -m py_compile` on touched API files.
- Run focused tests for endpoint helper behavior where present.

## Child DOX Index

No child AGENTS.md files yet.
