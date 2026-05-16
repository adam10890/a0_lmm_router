# CLAUDE.md — a0_lmm_router

## Project Overview

`a0_lmm_router` is an Agent Zero **plugin** (v1.2.2) that unifies two previously separate concerns:

1. **LMM Fleet Management** — Start, stop, monitor, and configure `llama.cpp` server slots (via remote HTTP, Docker SDK, or subprocess)
2. **Smart Routing** — Classify incoming messages and route them to the most appropriate local model slot based on task type, complexity, and hardware capacity

This plugin replaces the deprecated `a0_lmm` and `a0_smart_router` plugins. It targets Agent Zero v0.9.7+.

---

## Repository Structure

```
a0_lmm_router/
├── plugin.yaml                  # Plugin manifest (name, version, settings_sections)
├── launcher.py                  # CLI entry point for fleet management (start/stop/status/restart)
├── api/                         # HTTP API handlers — one file per endpoint
│   ├── assign_model.py          # Assign a model to a slot
│   ├── cancel_job.py            # Cancel a download/install job
│   ├── clear_hf_token.py        # Clear HuggingFace token
│   ├── delete_model.py          # Delete an installed model
│   ├── fleet_status.py          # Get fleet-wide status
│   ├── fleet_upgrade.py         # Upgrade llama.cpp image
│   ├── fleet_upgrade_rollback.py# Rollback llama.cpp upgrade
│   ├── job_status.py            # Check download job status
│   ├── llamacpp_control.py      # Start/stop individual slots
│   ├── llamacpp_list_models.py  # List available models
│   ├── llamacpp_status.py       # Per-slot health and status
│   ├── lmm_compute_stats.py     # Compute resource stats
│   ├── lmm_fleet_ignite.py      # Bulk fleet startup
│   ├── lmm_hardware_recommend.py# Hardware-based model recommendations
│   ├── lmm_host_ignite.py       # Host-side fleet startup
│   ├── lmm_model_install.py     # Install a model from HuggingFace
│   ├── lmm_model_recommend.py   # Recommend models for a slot
│   ├── lmm_slot_recommendations.py # Slot configuration recommendations
│   ├── lmm_stats_summary.py     # Aggregated stats summary
│   ├── lmm_test_prompt.py       # Test a prompt against a slot
│   └── set_hf_token.py          # Set HuggingFace token
├── conf/
│   ├── llama_cpp_servers.yaml   # Slot definitions, global backend, hardware inventory
│   └── model_providers.yaml     # Model provider configurations
├── extensions/
│   ├── python/                  # Agent Zero Python extensions (lifecycle hooks)
│   └── webui/                   # Agent Zero WebUI extensions (injected JS/CSS)
├── helpers/
│   ├── backends/                # Backend implementations (remote/docker/subprocess)
│   ├── smart_router/            # Message classification and routing logic
│   ├── llama_cpp_manager.py     # Core singleton manager — BackendManager class
│   ├── compute_monitor.py       # GPU/CPU utilization monitoring
│   ├── context_calculator.py    # Token context window calculations
│   ├── fleet_models.py          # Fleet-level model data models
│   ├── hardware_inspector.py    # Hardware detection and profiling
│   ├── model_recommender.py     # Model recommendation engine
│   ├── rate_limit_retry.py      # Rate limiting and retry logic
│   ├── slot_recommender.py      # Slot configuration recommender
│   ├── stats_tracker.py         # Usage statistics tracking
│   └── tier_catalog.py          # Model tier catalog
├── tools/
│   ├── fleet_ignite.py          # Agent Zero tool: bulk fleet start
│   ├── llama_cpp_control.py     # Agent Zero tool: per-slot control
│   └── lmm_host_helper.py       # Agent Zero tool: host-side ops (primary tool, ~46KB)
├── webui/
│   ├── dashboard.html           # Fleet dashboard (real-time status)
│   ├── config.html              # Configuration UI
│   ├── model-test.html          # Model testing/prompting UI
│   ├── dev-tracker.html         # Development tracker UI
│   └── js/                      # JavaScript modules for WebUI pages
├── tests/
│   ├── test_host_helper_models.py
│   └── test_patch_order.py
├── scripts/                     # Utility scripts
├── docker/                      # Docker configuration files
├── _e2e_test.py                 # End-to-end tests
├── _smoke_test.py               # Smoke tests
├── _check_async.py              # Async diagnostic utility
└── _fix_structure.py            # File structure repair utility
```

---

## Installation Location

When deployed inside Agent Zero:
```
/a0/usr/plugins/a0_lmm_router/   # plugin root
/a0/conf/llama_cpp_servers.yaml  # optional host-level config override
```

---

## Configuration

### Config Discovery Order

`launcher.py` and the init extension discover config in this priority order:
1. `$A0_LMM_ROUTER_CONFIG` environment variable (if set and readable)
2. `/a0/conf/llama_cpp_servers.yaml` (host-mounted preferred location)
3. `<plugin_dir>/conf/llama_cpp_servers.yaml` (bundled fallback)

### Key Config Files

**`conf/llama_cpp_servers.yaml`** — Primary runtime config:
- `active_slots` — List of llama.cpp server slots to manage (port, model_id, role, context_size, etc.)
- `slot_defaults` — Default parameters applied to every slot
- `global` — Backend selection (`remote`/`docker`/`subprocess`/`auto`), Docker settings, paths, timeouts
- `hardware` — GPU/CPU inventory and memory limits for the model recommender

**`conf/model_providers.yaml`** — Defines model providers and their endpoints.

### Slot Roles
- `chat` — Primary chat/reasoning model
- `utility` — Fast utility/routing model
- `embedding` — Text embedding model
- `code` — Code-specialized model
- `router` — Ultra-fast classification model
- `internal_api` — For external tools (Aider, Continue, etc.)

---

## Core Architecture

### BackendManager (`helpers/llama_cpp_manager.py`)
- **Singleton** — reset via `BackendManager._instance = None` to force re-initialization
- Manages slot lifecycle across all backend types
- Key methods: `start_all()`, `stop_all()`, `start_slot(id)`, `stop_slot(id)`
- All lifecycle methods are `async`

### Backend Types (`helpers/backends/`)
- `remote` — Plugin acts as HTTP client to pre-running llama-server containers
- `docker` — Plugin manages containers via Docker SDK
- `subprocess` — Plugin spawns `llama-server` locally
- `auto` — Tries each backend in order, uses first that works

### Smart Router (`helpers/smart_router/`)
Classifies messages by task type and selects the best available slot. Used when Agent Zero receives a message and needs to decide which local model to invoke.

---

## API Handler Pattern

Each file in `api/` is a standalone module loaded by Agent Zero's HTTP router. Pattern:

```python
async def handler(request: dict, context) -> dict:
    # ...
    return {"status": "ok", "data": ...}
```

Handlers delegate to `BackendManager` or the relevant helper module. Return `{"error": "..."}` on failure rather than raising exceptions.

---

## Agent Zero Tools (`tools/`)

Tools in `tools/` are Agent Zero tool classes. The most important is `lmm_host_helper.py` (~46KB), which provides host-side operations:
- List/install/delete/assign models
- Track download jobs with cancellation support
- Manage HuggingFace tokens
- Fleet upgrade and rollback

---

## CLI Usage

```bash
# Inside the A0 container
python /a0/usr/plugins/a0_lmm_router/launcher.py status
python /a0/usr/plugins/a0_lmm_router/launcher.py start
python /a0/usr/plugins/a0_lmm_router/launcher.py start slot_chat
python /a0/usr/plugins/a0_lmm_router/launcher.py stop
python /a0/usr/plugins/a0_lmm_router/launcher.py restart
```

Exit codes: `0` success, `2` config missing, `3` partial failure, `4` unhandled error.

The launcher respects `global.auto_start: false` — it exits cleanly without touching slots when disabled.

---

## WebUI Pages

Served by Agent Zero's web server under the plugin's URL namespace:

| Page | Description |
|------|-------------|
| `dashboard.html` | Real-time fleet status, VRAM, slot health |
| `config.html` | Edit slot configuration and global settings |
| `model-test.html` | Send test prompts to any slot |
| `dev-tracker.html` | Development task tracker |

---

## Development Conventions

### Python Style
- All backend/manager methods are `async` / `await`
- Use `asyncio.run()` only at top-level CLI entry points (e.g., `launcher.py`)
- `BackendManager` is a singleton; reset `_instance = None` to force re-initialization
- No hard-coded paths — always resolve via config discovery order
- Return dicts from API handlers (not exceptions); include `"error"` key on failure
- Type hints on all public functions; use `from __future__ import annotations` for forward refs

### Testing
```bash
# Unit tests
pytest tests/

# Smoke test (lightweight, no llama.cpp required)
python _smoke_test.py

# E2E test (requires running llama.cpp slots)
python _e2e_test.py
```

### Adding a New API Endpoint
1. Create `api/my_endpoint.py` with an async `handler(request, context)` function
2. Register it in Agent Zero's HTTP router (in the plugin's extension under `extensions/python/`)

### Adding a New Tool
1. Create `tools/my_tool.py` extending Agent Zero's `Tool` base class
2. Register in the plugin's tool list in the extensions init file

### Config Changes
- Edit `conf/llama_cpp_servers.yaml` for slot/hardware changes
- The plugin re-reads config on each `BackendManager` instantiation (no hot-reload — restart required)
- Never commit HF tokens or API keys; use environment variables or Agent Zero's secrets management

---

## Key Files for Common Tasks

| Task | File(s) |
|------|---------|
| Add/modify slots | `conf/llama_cpp_servers.yaml` |
| Slot lifecycle logic | `helpers/llama_cpp_manager.py` |
| Hardware detection | `helpers/hardware_inspector.py` |
| Model recommendations | `helpers/model_recommender.py`, `helpers/slot_recommender.py` |
| Routing logic | `helpers/smart_router/` |
| Host-side model ops | `tools/lmm_host_helper.py` |
| CLI entry point | `launcher.py` |
| Fleet dashboard UI | `webui/dashboard.html` |
| API endpoint handlers | `api/` |

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `A0_LMM_ROUTER_CONFIG` | Override config file path |
| `LLAMA_CPP_MODELS_DIR` | Override models directory |
| `HF_TOKEN` | HuggingFace token for private model downloads |

---

## Relationship to agent-zero

This plugin is installed into Agent Zero under `usr/plugins/a0_lmm_router/` at runtime. It depends on Agent Zero's plugin infrastructure (extension loading, WebUI injection, HTTP routing). When developing, changes here need to be tested inside a running Agent Zero instance.
