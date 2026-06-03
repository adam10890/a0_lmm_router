# DOX contract — a0_lmm_router

## Purpose

Local llama.cpp fleet management and routing plugin for Agent Zero. It owns
slot lifecycle, health/failover, router dashboards, MCP tools, and the emerging
standalone OpenAI-compatible provider track.

## Ownership

- This plugin is a primary product track.
- Portable surfaces are the priority: MCP server and OpenAI-compatible HTTP.
- A0 WebUI and extension hooks are useful wrappers, not the core product.
- Keep branch-router work separate from future meta-router/RBAC work unless a
  task explicitly crosses that boundary.

## Local Contracts

- `helpers/llama_cpp_manager.py` owns BackendManager and slot orchestration.
- `service/` owns the standalone observer/router HTTP service.
- `mcp_server/` owns Streamable HTTP MCP tools/resources.
- `conf/llama_cpp_servers.yaml` describes desired local fleet configuration.
- `docker/` owns compose files and Windows/dev launch helpers.
- `scripts/` owns operator-facing smoke/run helpers for standalone and local
  workflows.
- `docs/` owns plugin-local runbooks such as the standalone provider package
  guide.
- Do not add memory ownership, agent identity ownership, or SharedBrain policy
  ownership to this plugin. It may enforce routing policy, but it must not
  become the whole operating system.
- Keep non-streaming forwarding, streaming forwarding, auth, and packaging as
  separate implementation gates.

## Work Guidance

- Before provider work, inspect `service/app.py`, `service/routing_intent.py`,
  `service/observer.py`, and `mcp_server/router_bridge.py`.
- The current `POST /routing/request` path is dry-run intent routing. Do not
  treat it as completed OpenAI-compatible forwarding.
- When adding `POST /v1/chat/completions`, use the existing routing decision
  path and forward to the selected llama.cpp slot without Docker control.
- Do not modify Agent Zero plugin behavior while building standalone service
  phases unless the task explicitly says to.
- Prefer server-agnostic behavior over Windows/RTX-4090 assumptions.

## Verification

- Run focused tests under `tests/` for touched behavior.
- For service/router edits, run tests covering routing intent and slot
  failover where practical.
- At minimum, compile touched Python files with `python -m py_compile`.

## Child DOX Index

No child AGENTS.md files yet. Add child docs for `service/` or `mcp_server/`
once those areas gain independent release gates.
