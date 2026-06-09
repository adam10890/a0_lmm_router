# DOX contract - a0_lmm_router/tests

## Purpose

Tests for config resolution, routing/failover, MCP, OpenAI provider behavior,
dashboard wiring, and helper logic.

## Ownership

- Tests should protect portable service/MCP contracts and local fleet safety
  behavior.

## Local Contracts

- Add focused tests for changed routing, context, exposure, or config policy.

## Work Guidance

- Prefer deterministic fixtures over live GPU/router requirements.

## Verification

- Run focused `pytest tests/<file>.py -v` for touched behavior.

## Child DOX Index

No child AGENTS.md files yet.
