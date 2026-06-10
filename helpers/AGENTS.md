# DOX contract - a0_lmm_router/helpers

## Purpose

Router helper layer for backends, config resolution, routing/failover,
context/budget planning, model fit, exposure filtering, fleet metadata, and
compute monitoring.

## Ownership

- Config path trust belongs in `conf_resolver.py`.
- Backend lifecycle belongs in backend/manager helpers.
- Prompt/tool/MCP exposure policy belongs in dedicated exposure helpers.

## Local Contracts

- Keep portable helper behavior independent from A0 WebUI assumptions.
- Do not bypass safe config roots or expose all local models by default.

## Work Guidance

- Keep standalone provider behavior and A0 plugin dashboard behavior separated.

## Verification

- Run focused tests under `tests/` for touched helper behavior.
- Run `python -m py_compile` on touched helper files.

## Child DOX Index

No child AGENTS.md files yet.
