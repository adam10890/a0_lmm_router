# DOX contract - a0_lmm_router/webui

## Purpose

Local fleet dashboard, model management UI, and router control panels.

## Ownership

- UI owns presentation and interaction state.
- Service/helpers/API own routing and fleet behavior.

## Local Contracts

- Keep dashboard calls aligned with `api/`.
- Do not make UI-only state the source of truth for router config.

## Work Guidance

- Maintain responsive dashboard behavior and avoid hard-coded local host
  assumptions unless surfaced as config.

## Verification

- Inspect changed HTML/JS wiring and run UI-focused tests if present.

## Child DOX Index

No child AGENTS.md files yet.
