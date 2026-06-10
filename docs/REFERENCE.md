# a0_lmm_router — Reference

**Version:** 1.3.0 | Agent Zero v0.9.7–v1.17

## Overview

Unified LMM server management and smart routing plugin. Manages `llama.cpp` server slots, classifies tasks to route to the best model, and exposes an MCP server for external clients.

## Installation

```bash
# ZIP upload: Settings → Plugins → Upload ZIP
# Git: Settings → Plugins → Install from Git
docker cp ./a0_lmm_router <container>:/a0/usr/plugins/a0_lmm_router
```

The `hooks.py` install hook auto-installs Python deps (`mcp`, `aiohttp`, `pyyaml`) on first activation.

## Configuration

Config section: `router` (Settings → Router)

| Key | Description |
|---|---|
| `slots` | Array of llama.cpp slot definitions |
| `routing.enabled` | Enable smart task routing |
| `routing.strategy` | Routing strategy (`round_robin`, `task_type`, …) |
| `mcp_server.port` | MCP HTTP server port (default: 8095) |

`per_project_config: true` — project-level overrides supported.

## Backend types

| Type | Description |
|---|---|
| `remote` | llama.cpp in a separate Docker container (HTTP) |
| `docker` | Local Docker SDK — plugin manages container lifecycle |
| `subprocess` | Child process on the host |

## Dashboard

Access at `/plugins/a0_lmm_router/webui/` in the Agent Zero browser frame.

Features:
- Live slot status + VRAM monitor
- Model finder + HuggingFace install form
- Slot assignment (drag-and-drop model → slot)
- Download job progress

## MCP Server

Port: `8095` | Protocol: Streamable HTTP

Connect any MCP client to `http://localhost:8095/mcp`.

**9 tools:** slot status, model list, model install, model delete, slot assign, slot unassign, fleet upgrade, benchmark, HF token management

**4 resources:** fleet status, model catalog, active slots, hardware info

## Fleet operations

| Operation | Description |
|---|---|
| List models | Running slots + installed models |
| Install model | Download GGUF from HuggingFace into a slot |
| Assign | Map a model to a specific slot |
| Unassign | Remove model from slot |
| Upgrade | Pull latest llama.cpp image with rollback |

## Integration with llmfit_advisor

When `llmfit_advisor` is also active on the same instance:
- `llmfit list` → queries `/plugins/a0_lmm_router/llamacpp_status`
- `llmfit download` → delegates to `/plugins/a0_lmm_router/lmm_model_install`
- `llmfit recommend` enriches results with `fleet_status: running|available`
