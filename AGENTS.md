# DOX contract — a0_lmm_router

## Purpose

Local llama.cpp fleet management and routing plugin for Agent Zero. It owns
slot lifecycle, health/failover, router dashboards, MCP tools, the standalone
OpenAI-compatible provider, and the emerging autonomous Fleet Manager control
plane.

## Ownership

- This plugin is a primary product track.
- Portable surfaces are the priority: MCP server and OpenAI-compatible HTTP.
- A0 WebUI and extension hooks are useful wrappers, not the core product.
- Keep branch-router work separate from future meta-router/RBAC work unless a
  task explicitly crosses that boundary.

## Local Contracts

- `helpers/llama_cpp_manager.py` owns BackendManager and slot orchestration.
- `helpers/conf_resolver.py` is the single source of truth for
  `llama_cpp_servers.yaml` resolution. API/service code must not consume
  `A0_LMM_ROUTER_CONFIG` directly; env overrides must stay inside safe config
  roots.
- `helpers/context_planner.py` owns max-feasible context planning. Role context
  values are minimums, not caps; generated Router Mode presets carry the
  planned hard context and compression uses the effective context ratio.
- `helpers/tool_exposure.py` owns Local Fleet tool-prompt exposure policy.
  Keep the main Agent Zero profile lean and route heavy tool surfaces to
  specialist profiles instead of toggling plugins globally during live chats.
- `helpers/mcp_exposure.py` owns Local Fleet MCP-prompt exposure policy.
  Filter rendered MCP prompts by profile instead of editing global MCP server
  settings or disabling MCP servers that specialist profiles still need.
- `helpers/output_budget.py` owns Local Fleet completion-token caps for Agent
  Zero model calls. Keep defaults conservative and scoped to local fleet
  models so cloud/external presets keep their configured behavior.
- `helpers/model_params_cache.py` persists per-GGUF llama.cpp plans under
  `data/model_params_cache.json`. First fleet warm computes ctx/KV/batch;
  later ignites reuse cache unless the file or VRAM fingerprint changes.
  `last_success_by_role` seeds global options when switching models on a slot.
- `service/` owns the standalone observer/router HTTP service and Fleet
  Manager V1 control plane: agent identity headers, SQLite telemetry, and
  bounded admission control. It must remain Docker-socket-free.
- `mcp_server/` owns Streamable HTTP MCP tools/resources.
- `conf/llama_cpp_servers.yaml` describes desired local fleet configuration.
- `docker/` owns compose files and Windows/dev launch helpers. Router Mode is
  preset-only by default; do not expose all GGUFs with `--models-dir` unless a
  task explicitly asks for directory discovery.
- `scripts/` owns operator-facing smoke/run helpers for standalone and local
  workflows, including `scripts/render_router_preset.py` for regenerating
  `conf/models_preset.ini` before Router Mode starts.
- `docs/` owns plugin-local runbooks such as the standalone provider and Fleet
  Manager guides.
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
- Fleet Manager work should keep container lifecycle operations out of
  `service/`; add a future host-side worker/sidecar for Docker permissions.
- MCP bridge calls may target Fleet Manager via `A0_FLEET_MANAGER_BASE_URL`;
  without that env they preserve the legacy BackendManager path.
- Do not modify Agent Zero plugin behavior while building standalone service
  phases unless the task explicitly says to.
- Prefer server-agnostic behavior over Windows/RTX-4090 assumptions.

## Verification

- Run focused tests under `tests/` for touched behavior.
- For service/router edits, run tests covering routing intent and slot
  failover where practical.
- At minimum, compile touched Python files with `python -m py_compile`.

## Child DOX Index

- `helpers/AGENTS.md` — backend orchestration, config resolution, routing,
  model fit, budget, exposure, and monitoring helpers.
- `service/AGENTS.md` — standalone OpenAI-compatible provider and Fleet Manager
  service boundary.
- `mcp_server/AGENTS.md` — MCP tools/resources and router bridge.
- `api/AGENTS.md` — Agent Zero Web/API wrappers.
- `webui/AGENTS.md` — dashboard and local fleet UI.
- `extensions/AGENTS.md` — Agent Zero hook integration.
- `conf/AGENTS.md` — router, fleet, model provider, and preset configuration.
- `docker/AGENTS.md` — compose files and container runtime helpers.
- `scripts/AGENTS.md` — operator scripts and provider smoke helpers.
- `tests/AGENTS.md` — routing, MCP, UI, config, and helper tests.
- `docs/AGENTS.md` — runbooks and durable architecture notes.
- `tools/AGENTS.md` — Agent Zero tool wrappers.
- `skills/AGENTS.md` — local fleet task-routing skill guidance.
