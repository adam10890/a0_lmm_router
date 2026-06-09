# DOX contract - a0_lmm_router/service

## Purpose

Standalone router/provider service, routing intent, observer, and Fleet Manager
control-plane code.

## Ownership

- Service code owns portable HTTP behavior.
- Docker/container permissions must stay outside request-path routing unless a
  hardened worker boundary is explicitly added.

## Local Contracts

- `POST /routing/request` is intent routing unless forwarding is implemented.
- OpenAI-compatible endpoints must use existing routing decisions and selected
  slots rather than duplicating router policy.

## Work Guidance

- Keep non-streaming forwarding, streaming forwarding, auth, and packaging as
  separate gates.

## Verification

- Run routing intent/provider tests for service changes.
- Run `python -m py_compile` on touched service files.

## Child DOX Index

No child AGENTS.md files yet.
