# DOX contract - a0_lmm_router/tests

## Purpose

Tests for config resolution, routing/failover, MCP, OpenAI provider behavior,
dashboard wiring, and helper logic.

## Ownership

- Tests should protect portable service/MCP contracts and local fleet safety
  behavior.

## Local Contracts

- Add focused tests for changed routing, context, exposure, or config policy.
- Tests touching `helpers/router_context.py` context resolution must stub
  `fetch_router_model_ctx` and `read_slot_context_size` (see the
  `offline_ctx_sources` fixture in `test_router_context_guard.py`); otherwise
  a live Router Mode fleet or a leaked BackendManager singleton overrides the
  test's `ctx_length` and the assertions become environment-dependent.

## Work Guidance

- Prefer deterministic fixtures over live GPU/router requirements.

## Verification

- Run focused `pytest tests/<file>.py -v` for touched behavior.

## Child DOX Index

No child AGENTS.md files yet.
