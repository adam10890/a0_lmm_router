# AGENTS.md — a0_lmm_router

**Version:** 1.3.0 | **Target:** Agent Zero v0.9.7–v1.17

## What this plugin does

Full-stack LMM (Local Multimodal Model) management layer for Agent Zero. Combines:
- **Backend** — start/stop/monitor `llama.cpp` server slots across remote Docker containers, local Docker SDK, or local subprocesses
- **Smart routing** — classify messages and route to the best slot by task type, complexity, and VRAM
- **MCP server** — Streamable HTTP server on port 8095 exposing 9 tools + 4 resources to any MCP client

## Key files

| Path | Role |
|---|---|
| `plugin.yaml` | Manifest — settings section: `router` |
| `hooks.py` | `install(**kwargs)` — pip-installs `requirements.txt` deps |
| `launcher.py` | MCP + router server entry point |
| `helpers/` | Slot managers, health monitor, model recommendation, HF token |
| `tools/` | Agent-facing tools |
| `api/` | REST endpoints registered with A0 |
| `extensions/python/` | Python lifecycle hooks |
| `extensions/webui/` | Frontend injection points |
| `webui/` | Real-time dashboard (VRAM monitor, model finder, slot assignment) |
| `mcp_server/` | Streamable HTTP MCP server (port 8095) |
| `conf/` | Default configuration files |
| `requirements.txt` | `mcp`, `aiohttp`, `pyyaml` |

## Backend types

| Type | When to use |
|---|---|
| `remote` | llama.cpp running in a separate Docker container on the network |
| `docker` | Local Docker SDK — plugin manages container lifecycle |
| `subprocess` | Child process on the same host |

## MCP server

Port 8095. 9 tools + 4 resources. Exposed to any MCP client (Cursor, Claude Desktop, etc.). Started by `launcher.py`. Config in `conf/`.

## How to add a new agent tool

1. Create `tools/<tool_name>.py` with a `Tool` subclass.
2. A0 auto-discovers all files in `tools/`.

## How to add a REST endpoint

Add `api/<endpoint>.py` — A0 auto-discovers.

## How to add an MCP tool

Add to `mcp_server/` following the existing tool pattern and register in the tool list.

## Dependency install

`hooks.py::install(**kwargs)` parses `requirements.txt` and calls `pip install` only for missing packages — idempotent and offline-safe when deps are already present.

## Constraints

- MCP server port 8095 — must not be used by another plugin.
- `per_project_config: true` — each project can override router config.
- Config section name is `router` (not `agent`) — custom settings section.
- `hooks.py` does not uninstall shared deps on removal (other plugins may depend on them).
