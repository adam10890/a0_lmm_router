# a0_lmm_router — Codebase Guide for AI Assistants

## What This Project Is

`a0_lmm_router` (v1.3.0) is an Agent Zero plugin that manages a fleet of local [llama.cpp](https://github.com/ggml-org/llama.cpp) inference servers (GGUF models) and routes requests across them. It is a **fleet orchestrator**, not a multi-provider cloud gateway. Every model runs locally via `llama-server`.

Core capabilities:
- Slot lifecycle control (start/stop/health-check llama-server instances)
- Role-based request routing with failover chains
- FastMCP Streamable HTTP server (port 8095) exposing 9 tools + 4 resources
- 27 REST API endpoints consumed by the Agent Zero web UI
- Alpine.js dashboard for fleet monitoring, model install, config, and testing

---

## Module Path Constraint — Critical

All imports of plugin internals **within Agent Zero** must use the full plugin path:

```python
from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager
```

This path is not configurable. Changing it breaks Agent Zero's plugin loader. Every test file bootstraps this with:

```python
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
```

---

## Architecture Map

```
a0_lmm_router/
├── api/                        # REST handlers — POST /plugins/a0_lmm_router/<name>
│   │                           # incl. fleet_reconnect.py (HTTP fleet detect/reconnect)
├── helpers/
│   ├── llama_cpp_manager.py    # BackendManager singleton (core orchestrator, line 840)
│   │                           # ServerConfig dataclass (line 58)
│   ├── backends/               # Pluggable execution layer
│   │   ├── base.py             # InferenceBackend ABC (line 45) + SlotStatus (line 24)
│   │   ├── factory.py          # Auto-detect: remote → docker → subprocess
│   │   ├── remote_backend.py   # HTTP health-only; no container lifecycle
│   │   ├── docker_backend.py   # docker SDK — create/destroy containers
│   │   └── subprocess_backend.py  # local llama-server processes
│   ├── smart_router/
│   │   ├── failover.py         # FailoverReason (line 66), CooldownTracker (line 305)
│   │   ├── workflow_registry.py  # WorkflowRegistry — loaded but NOT wired in
│   │   └── session_models.py
│   ├── fleet_mode.py           # detect_fleet_mode() — docker-ps based (host only)
│   ├── router_probe.py         # detect_fleet_http() — HTTP /props probe (works inside A0)
│   ├── fleet_models.py         # host-helper adapter for model install/delete/assign
│   ├── compute_monitor.py      # GPU/CPU/RAM via nvidia-smi + psutil; router-aware slots
│   ├── model_recommender.py    # hardware-aware model selection
│   ├── slot_recommender.py     # VRAM-aware slot count recommendations
│   ├── rate_limit_retry.py     # circuit breaker + exponential backoff decorator
│   └── stats_tracker.py        # failover event recording
├── mcp_server/
│   ├── server.py               # FastMCP app factory + main() entry point
│   ├── tools.py                # 9 MCP tools (register_tools())
│   ├── resources.py            # 4 MCP resources (register_resources())
│   └── router_bridge.py        # bridge: MCP tools → BackendManager
├── extensions/python/
│   ├── agent_init/
│   │   ├── _10_init_servers.py  # LlamaCppInitExtension — auto-start on agent init
│   │   └── _15_rate_limit_retry.py
│   └── message_loop_start/
│       └── _20_smart_router.py  # SmartRouterExtension — DISABLED (no-op)
├── tools/
│   ├── lmm_host_helper.py      # Windows host HTTP bridge for docker compose control
│   ├── fleet_ignite.py
│   └── llama_cpp_control.py
├── webui/                      # Alpine.js pages: dashboard, config, model-test, dev-tracker
├── conf/
│   ├── llama_cpp_servers.yaml  # Main config — slots, global, hardware, mcp_server
│   ├── model_providers.yaml
│   └── models_preset.ini       # Router Mode model aliases (.ini format)
├── docker/                     # docker-compose.lmm.yml / .router.yml / .mtp.yml
├── tests/                      # 10 pytest modules
├── launcher.py                 # CLI: start / stop / status / restart / mcp
├── hooks.py                    # Plugin install/uninstall lifecycle (pip install)
├── plugin.yaml                 # Agent Zero plugin manifest
├── default_config.yaml         # Agent Zero plugin settings schema
├── _smoke_test.py              # Standalone smoke test (python _smoke_test.py)
└── _e2e_test.py                # Standalone e2e test (python _e2e_test.py)
```

---

## Key Classes

| Class | File:Line | Purpose |
|---|---|---|
| `BackendManager` | `helpers/llama_cpp_manager.py:840` | Singleton orchestrator. Owns slot configs, backend instance, failover chains. Entry point for all slot operations. |
| `LlamaCppManager` | `helpers/llama_cpp_manager.py:155` | Legacy per-slot manager; still used by some direct subprocess paths. |
| `ServerConfig` | `helpers/llama_cpp_manager.py:58` | Dataclass for a single slot's full configuration (50+ fields). |
| `InferenceBackend` | `helpers/backends/base.py:45` | ABC with `start_slot`, `stop_slot`, `health_check`, `list_slots`, `cleanup`. |
| `SlotStatus` | `helpers/backends/base.py:24` | Runtime status dataclass returned by all backend operations. |
| `RemoteBackend` | `helpers/backends/remote_backend.py` | HTTP health-only; no container lifecycle. Used when `backend: remote`. |
| `DockerBackend` | `helpers/backends/docker_backend.py` | Creates/destroys Docker containers via docker SDK. |
| `SubprocessBackend` | `helpers/backends/subprocess_backend.py` | Spawns llama-server as a local process. |
| `FailoverReason` | `helpers/smart_router/failover.py:66` | Enum: TIMEOUT, RATE_LIMIT, QUOTA_EXHAUSTED, PROVIDER_ERROR, HTTP_ERROR, SLOT_UNHEALTHY, UNKNOWN_ERROR. |
| `CooldownTracker` | `helpers/smart_router/failover.py:305` | Tracks slots in ERROR state; manages recovery probes. |
| `WorkflowRegistry` | `helpers/smart_router/workflow_registry.py` | Regex-based workflow classification. Loaded but not wired into any request path. |
| `RouterBridge` (module) | `mcp_server/router_bridge.py` | Module-level functions bridging MCP tools to `BackendManager`. Not a class. |
| `LlamaCppInitExtension` | `extensions/python/agent_init/_10_init_servers.py` | Auto-starts fleet + MCP server on agent container init. |

---

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt
# Packages: aiohttp>=3.9.0, mcp>=1.22.0, pyyaml>=6.0

# Run the full pytest suite
pytest tests/

# Standalone smoke test — verifies critical module imports
python _smoke_test.py

# Standalone e2e test — verifies all API handlers and launcher imports
python _e2e_test.py

# CLI fleet control (launcher.py)
python launcher.py status
python launcher.py start
python launcher.py start slot_chat   # start a single named slot
python launcher.py stop
python launcher.py restart
python launcher.py mcp               # start MCP server on port 8095

# Start MCP server directly
python -m mcp_server.server
MCP_PORT=9000 python -m mcp_server.server

# Inspect MCP server
npx @modelcontextprotocol/inspector http://127.0.0.1:8095/mcp
```

There is no configured linter, formatter, or pre-commit hook. No CI/CD pipelines exist.

---

## Configuration

### Primary config: `conf/llama_cpp_servers.yaml`

Discovery order (first match wins):
1. `$A0_LMM_ROUTER_CONFIG` environment variable
2. `/a0/conf/llama_cpp_servers.yaml` (production container path)
3. `<plugin_root>/conf/llama_cpp_servers.yaml` (dev fallback)

**Top-level sections:**

```yaml
active_slots:    # List of slot dicts — each becomes one inference server instance
slot_defaults:   # Default values applied to every slot that doesn't override them
global:          # backend type, lmm_hosts URLs, docker config, health intervals
hardware:        # GPU/CPU/RAM inventory used by model recommender
ephemeral:       # Per-conversation container pool (disabled by default)
mcp_server:      # Port, host, auth token, allow_mutating_tools flag
```

Slot `id` field identifies each slot (e.g., `slot_chat`, `slot_utility`). Defaults to `slot_<port>` if absent.

**Runtime state overlay:** `conf/router_state.json` (created at runtime) stores per-slot overrides from the dashboard (e.g., `router_default_model`). Applied by `BackendManager._apply_router_state()`. In production this lives at `/a0/conf/router_state.json`.

### Environment Variables

| Variable | Purpose |
|---|---|
| `A0_LMM_ROUTER_CONFIG` | Override config file path |
| `A0_LMM_HOST_TOKEN` | Auth token for host helper |
| `A0_LMM_HOST_URL` | Full URL for host helper (overrides host+port) |
| `A0_LMM_HOST_HOST` | Host helper hostname |
| `A0_LMM_HOST_PORT` | Host helper port (default 55501) |
| `MCP_PORT` | MCP server port (default 8095) |
| `MCP_HOST` | MCP server bind host |
| `MCP_BIND_PUBLIC` | Set to `1` to allow 0.0.0.0 bind |
| `MCP_DISABLE_AUTH` | Set to `1` to skip bearer token auth |
| `MCP_ALLOW_MUTATING_TOOLS` | Set to `1` to expose start/stop/fleet MCP tools |

---

## Operational Modes

**Router Mode** (`router_mode: true` on a slot): One llama.cpp container registers all GGUFs in `router_models_dir` and hot-swaps on demand. Single container serves all roles. Uses `docker/docker-compose.lmm.router.yml`.

**Multi-slot (three-slot) Mode**: Separate containers per role — chat:8080, utility:8088, embedding:8082. Default mode. Uses `docker/docker-compose.lmm.yml`.

**Remote Mode** (`backend: remote` in global): Plugin does not manage container lifecycle. It only monitors health and routes to pre-running containers declared in `global.lmm_hosts`. Production default.

Fleet mode detection: `helpers/fleet_mode.py:detect_fleet_mode()` inspects running Docker containers to identify active mode and detect conflicts.

---

## API Patterns

All API handlers live in `api/` and follow this pattern:

```python
from __future__ import annotations
import os
from pathlib import Path
from flask import Request
from helpers.api import ApiHandler


def _resolve_conf_path() -> str:
    env_conf = os.environ.get("A0_LMM_ROUTER_CONFIG", "")
    here = Path(__file__).resolve()
    plugin_conf = str(here.parents[1] / "conf" / "llama_cpp_servers.yaml")
    root_conf = str(here.parents[4] / "conf" / "llama_cpp_servers.yaml")
    if env_conf and os.path.exists(env_conf):
        return env_conf
    return root_conf if os.path.exists(root_conf) else plugin_conf


class MyHandler(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager
            conf_path = _resolve_conf_path()
            manager = BackendManager.get_instance(conf_path)
            # ... do work ...
            return {"ok": True, "result": ...}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
```

**URL pattern**: `POST /plugins/a0_lmm_router/<snake_case_filename>`

Agent Zero auto-discovers handlers by filename — no registration step needed.

**Always return** `{"ok": True/False, ...}` — never raise from `process()`.

**Reference handlers:**
- `api/llamacpp_status.py` — read-only health probe, safe import pattern
- `api/llamacpp_control.py` — uses `BackendManager`, start/stop operations

---

## How to Add a New API Endpoint

1. Create `api/my_new_handler.py` with a class subclassing `ApiHandler`.
2. Implement `async def process(self, input: dict, request: Request) -> dict`.
3. Copy `_resolve_conf_path()` from an existing handler — use path-relative resolution (not `helpers.files`) for safety outside the A0 container.
4. Import plugin helpers **inside** the method body using `usr.plugins.a0_lmm_router.*`.
5. Always wrap the body in `try/except Exception` and return `{"ok": False, "error": ...}` on failure.
6. No registration required — Agent Zero discovers the handler automatically via the filename.

---

## How Backends Work

Backend is selected once at `BackendManager` load time via `helpers/backends/factory.py`:

| Config `backend:` | Or auto-detect condition | Backend chosen |
|---|---|---|
| `remote` | `lmm_hosts` populated | `RemoteBackend` |
| `docker` | Docker socket available | `DockerBackend` |
| `subprocess` | fallback | `SubprocessBackend` |
| `auto` | checks in order above | first match |

All backends implement `InferenceBackend` (`helpers/backends/base.py:45`):
- `start_slot(name, config) → SlotStatus`
- `stop_slot(name) → bool`
- `health_check(name) → SlotStatus`
- `list_slots() → dict[name, SlotStatus]`
- `cleanup() → None`

**Production default is `backend: remote`** — the plugin monitors health and proxies requests; Docker containers are managed externally via `docker-compose` (started by host scripts or `lmm_host_helper.py`). Calling `stop_slot()` on `RemoteBackend` only deregisters the slot from internal tracking; it does not stop any container.

**To add a new backend:** Subclass `InferenceBackend`, implement all 5 abstract methods, add a `BackendType` enum value to `base.py`, and update `factory.py` to instantiate it.

---

## Failover

Entry point: `BackendManager.select_slot_with_failover(role)`.

Flow:
1. Resolve a failover chain for the role from `global.failover_chains` (or `DEFAULT_CHAINS` in `failover.py`).
2. Synchronously health-check the primary slot (2 s timeout via `urllib.request`).
3. Walk the chain if primary is unhealthy.
4. Return a dict: `{slot_id, url, is_failover, failover_reason}`.

Default chains:
```python
DEFAULT_CHAINS = {
    "chat":    ["chat", "utility", "openrouter_fallback"],
    "utility": ["utility", "chat", "openrouter_fallback"],
    "embed":   ["embed"],
}
```

`BackendManager.mark_slot_error(slot_id, msg)` puts a slot into `CooldownTracker`, causing it to return `unhealthy` until recovery probes succeed.

---

## MCP Server

Entry point: `mcp_server/server.py`. `create_app()` returns a configured `FastMCP` instance.

**9 tools** in `mcp_server/tools.py`:
- Always registered: `chat_completion`, `utility_completion`, `route_completion`, `get_embeddings`, `fleet_status`, `list_slots`
- Mutating (only when `allow_mutating_tools=True` or `MCP_ALLOW_MUTATING_TOOLS=1`): `start_fleet`, `start_slot`, `stop_slot`

**4 resources** in `mcp_server/resources.py`:
- `models://fleet/status`
- `models://{slot_id}/info`
- `models://hardware/profile`
- `models://slots/list`

**Auth:** Bearer token read from `/host/a0_lmm_host.key` or `/a0/tmp/lmm_host_token`. Disable with `MCP_DISABLE_AUTH=1`.

**Bind:** Defaults to `127.0.0.1`. Set `MCP_BIND_PUBLIC=1` for `0.0.0.0`.

MCP tools call `mcp_server/router_bridge.py` which calls `BackendManager.get_instance()` and dispatches `aiohttp` requests to slot endpoints.

---

## Testing Patterns

Tests live in `tests/` and use pytest. Each test file sets up `sys.path` to resolve the plugin as if it were in an `/a0` container:

```python
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

Common patterns:
- `monkeypatch.setenv` / `delenv` for environment variable isolation
- `monkeypatch.setattr` to replace module-level bridge functions with stubs
- `tmp_path` for token files and compose path tests
- `asyncio.run(...)` for testing async methods directly
- Pass a `status_provider` callable to `detect_fleet_mode()` to inject fake Docker state (see `tests/test_fleet_mode.py`)

`_smoke_test.py` and `_e2e_test.py` at repo root are standalone scripts (not pytest) — run them with `python _smoke_test.py`.

---

## Disabled / Incomplete Features

**`SmartRouterExtension`** (`extensions/python/message_loop_start/_20_smart_router.py`): Registered as an Agent Zero extension but `execute()` is a no-op. Do not add logic here without re-wiring it end-to-end.

**`WorkflowRegistry`** (`helpers/smart_router/workflow_registry.py`): Loads workflow definitions from `conf/routing_config.yaml` if that file exists, but no code path currently calls `get_workflow_registry()` or `route_request()` during inference.

**Ephemeral pool** (`helpers/ephemeral_pool.py`): Per-conversation container pooling. Disabled by default (`ephemeral.enabled: false`) because it requires Docker socket access from inside the A0 container.

---

## Gotchas

**`simpleeval` missing in A0 container:** `helpers.files` transitively imports `simpleeval`, which may be absent from the Agent Zero image's venv. In `api/` handlers, prefer path-relative `_resolve_conf_path()` over `helpers.files.get_abs_path()`. Inside `extensions/` it is safe because Agent Zero guarantees the framework is loaded. Some older `api/` handlers use `from helpers import files` — this works in production but can break in dev environments without the full A0 venv.

**`BackendManager` is a singleton:** `BackendManager.get_instance()` caches across calls within a process. `launcher.py` explicitly resets it (`BackendManager._instance = None`) before constructing a fresh one for CLI use.

**Cooldown probes need a running event loop:** `BackendManager._start_cooldown_probes()` is deferred to `start_all()`. It silently skips if no event loop is running during `__init__`.

**`router_state.json` path:** Written at runtime to the same directory as `llama_cpp_servers.yaml`. In production: `/a0/conf/router_state.json`.

**Mutating MCP tools are off by default:** `start_fleet`, `start_slot`, `stop_slot` are not registered unless `allow_mutating_tools=True` in config or `MCP_ALLOW_MUTATING_TOOLS=1` is set. This is intentional for security.

**`backend: remote` cannot manage containers:** Calling `start_slot()` / `stop_slot()` on `RemoteBackend` only registers/deregisters health tracking. Actual container control goes through `tools/lmm_host_helper.py` (Windows host bridge) or direct `docker compose` commands on the host machine.

**Config can lie about what's running:** `conf/llama_cpp_servers.yaml` describes the *intended* fleet, not the *actual* one. If an operator starts Router Mode (`docker-compose.lmm.router.yml`) while the YAML still lists `slot_chat/utility/embedding`, the old config-driven code path showed stale/empty slots. The fix is HTTP reality-detection: `helpers/router_probe.py:detect_fleet_http()` probes `GET /props` (a native router returns `{"role":"router"}`) and works **inside the A0 container** (no Docker socket, unlike `fleet_mode.detect_fleet_mode()` which shells out to `docker ps`). `compute_monitor._query_slots()` and `api/router_aliases.py` both consult it and synthesize a `slot_router` when a live router is found. The dashboard's **🔌 Reconnect** button hits `api/fleet_reconnect.py`, which re-probes and resets the `BackendManager` singleton.

**Router preset INI is picky (image b8840-9e5647aff):** `--models-preset` accepts only `alias =`, `model =`, and `embedding = true` per section. `ctx_size`/`n_gpu_layers`/`flash_attn`/`cache_type_*` are rejected ("option '<key>' not recognized in preset"). `model =` must be an ABSOLUTE container path (`/models/...`); it is NOT joined with `--models-dir`. Embeddings need `embedding = true` in that model's section or the router returns 501. The preset bind cannot live under the read-only `/models` mount — it's mounted at `/etc/llama/preset.ini`.
