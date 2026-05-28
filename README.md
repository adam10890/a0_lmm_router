# a0_lmm_router

**Unified LMM (Local Multimodal Model) Server Management + Smart Routing for Agent Zero**

Version: `1.3.0` · Merged from `a0_lmm` + `a0_smart_router` + MCP server · Target: Agent Zero v0.9.7 – v1.15

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [API Reference](#api-reference)
8. [WebUI Pages](#webui-pages)
9. [File Structure](#file-structure)
10. [Development Status](#development-status)
11. [Troubleshooting](#troubleshooting)
12. [Requirements](#requirements)

---

## Overview

`a0_lmm_router` is a full-stack Agent Zero plugin that unifies two previously separate concerns:

- **LMM server management** — start, stop, monitor, and interact with `llama.cpp` server slots (remote Docker containers, local Docker SDK, or subprocess on host)
- **Smart routing** — classify incoming user messages and route them to the most appropriate local model slot based on task type, complexity, and hardware capacity

This plugin replaces the deprecated `a0_lmm` and `a0_smart_router` plugins with a single, cohesive namespace.

## Features

### Backend

- **Fleet model management** — centralized model operations via host helper (list, install, delete, assign)
- **Transactional assignment** — model assignments with automatic rollback on failure
- **Download job tracking** — progress monitoring for model installations with cancellation support
- **HF token management** — secure HuggingFace token handling for private models
- **Fleet upgrade** — llama.cpp image upgrade with rollback support
- **llama.cpp slot control** — start/stop individual slots or all at once via API
- **Health monitoring** — polling `/health`, `/slots`, `/metrics`, `/props` of each llama-server
- **Real-time compute stats** — GPU (VRAM, utilization, temp) via `nvidia-smi`, CPU/RAM via `psutil`
- **Model recommender** — delegates to llmfit_advisor plugin with curated fallback
- **Smart router** — classifies incoming messages and selects the best slot (planned: LLM-based classifier via utility slot)

### Frontend

- **Config panel** (`config.html`) — slot list, start/stop controls, model picker, status badges
- **Dashboard** (`dashboard.html`) — real-time mission-control view of GPU, CPU, RAM, and slot health with 5s polling
- **Dev Tracker** (`dev-tracker.html`) — visual development status radiator (phases, bugs, APIs, self-review results)

### HTTP-based fleet detection & Reconnect (v1.3+)

The plugin trusts `conf/llama_cpp_servers.yaml`, but the **running** fleet can
differ from what the config describes — most commonly when an operator starts
Router Mode out-of-band while the YAML still lists the 3 fixed slots. In that
case the dashboard used to show stale/empty data because it only knew about
the configured slots.

The plugin now reconciles config against reality **over HTTP**, which works
even when it runs inside the Agent Zero container (no Docker socket required):

- `helpers/router_probe.py` probes `GET /props` (a native router answers
  `{"role":"router",...}`), `GET /health`, and `GET /v1/models`.
- `helpers/compute_monitor._query_slots()` returns a synthetic `slot_router`
  (with `router_mode: true` and the live registered model list) whenever a
  router is detected — lighting up the dashboard's ROUTER badge/panel.
- `api/router_aliases.py` falls back to an HTTP probe when the config has no
  `slot_router`, so role bindings populate from the live router.
- `api/fleet_reconnect.py` backs the dashboard **🔌 Reconnect** button: it
  re-probes the fleet, resets the `BackendManager` singleton, and reports the
  detected mode. Use it whenever "the container is running but the dashboard
  is blank."

Detection signal of record: **`GET /props` → `role == "router"`**.

### MCP Server (v1.3+)

The plugin includes a **Model Context Protocol (MCP) server** that exposes router capabilities to any MCP-aware client (Agent Zero, Claude Desktop, Cursor, mcp-inspector, etc.). Runs as Streamable HTTP on port 8095, spec version `2025-06-18`.

**Tools exposed:** `chat_completion`, `utility_completion`, `route_completion`, `get_embeddings`, `fleet_status`, `start_fleet`, `start_slot`, `stop_slot`, `list_slots`.

**Resources exposed:** `models://fleet/status`, `models://{slot_id}/info`, `models://hardware/profile`, `models://slots/list`.

#### Quick wiring (Agent Zero v1.15)

1. Start the MCP server inside the A0 container:
   ```cmd
   docker exec -d agent-zero-2 /opt/venv-a0/bin/python /a0/usr/plugins/a0_lmm_router/launcher.py mcp
   ```
   `start_agent_zero.bat` does this automatically since v1.3.
2. In the A0 WebUI → **Settings → MCP Servers Configuration JSON**, add:
   ```json
   "lmm-router": {
     "transport": "streamable-http",
     "url": "http://host.docker.internal:8095/mcp"
   }
   ```
   > **Note (Docker):** A0 runs inside a container, so use `host.docker.internal` to reach the MCP server on the host. On native Linux use `172.17.0.1` (Docker bridge) or the host's LAN IP instead.
   >
   > **Note (A0 version):** Some older A0 builds require `"type"` instead of `"transport"`. If the server shows as SSE instead of Streamable HTTP, switch the key to `"type": "streamable-http"`.
3. Click **Apply now**. The server appears in the status list with 9 tools.

#### Smoke test from inside the container
```cmd
docker exec agent-zero-2 /opt/venv-a0/bin/python -c "
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
async def main():
    async with streamablehttp_client('http://localhost:8095/mcp') as (r,w,sid):
        async with ClientSession(r,w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print([t.name for t in tools.tools])
asyncio.run(main())
"
```

#### Logs and lifecycle
- Live log: `docker exec agent-zero-2 cat /tmp/mcp_server.log`
- Restart: `docker exec agent-zero-2 pkill -f "launcher.py mcp"` then re-run the launcher command
- `stop_agent_zero.bat` shuts down the MCP server before stopping the container

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Zero WebUI                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ config.html  │  │dashboard.html│  │  dev-tracker.html    │  │
│  │ (Alpine.js)  │  │ (Alpine.js)  │  │  (Alpine.js)         │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘  │
│         │                 │                                     │
│         │ POST /plugins/a0_lmm_router/...                       │
└─────────┼─────────────────┼─────────────────────────────────────┘
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  API Handlers (api/*.py)                        │
│  llamacpp_control · llamacpp_status · llamacpp_list_models      │
│  lmm_model_install · lmm_model_recommend                        │
│  fleet_status · set_hf_token · clear_hf_token                   │
│  delete_model · assign_model · fleet_upgrade                     │
│  job_status · cancel_job                                         │
└────────┬──────────────────┬─────────────────────┬───────────────┘
         ▼                  ▼                     ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Fleet Models    │  │ ComputeMonitor   │  │ ModelRecommender │
│ (fleet_models)  │  │ (nvidia-smi,     │  │ (llmfit_advisor,  │
│                 │  │  psutil)         │  │  huggingface-cli)│
└────────┬────────┘  └──────────────────┘  └──────────────────┘
         │
         │ HTTP via X-Token auth
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Host Helper (lmm_host_helper.py)                    │
│  - Model manifest management                                      │
│  - Download jobs with progress tracking                           │
│  - Model assignment (transactional with rollback)                │
│  - HF token management                                           │
│  - Fleet upgrade (llama.cpp image pull + restart)               │
└────────┬────────────────────────────────────────────────────────┘
         │
         │ Docker Compose control
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Docker Compose Fleet                            │
│   usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml + .env      │
│   a0-llama-chat:8080  ·  a0-llama-utility:8088  ·  embed:8082  │
└─────────────────────────────────────────────────────────────────┘
```

### Extension Points

- `extensions/python/agent_init/_10_init_servers.py` — Initializes LMM slots on agent start
- `extensions/python/message_loop_start/_20_smart_router.py` — Intercepts user messages for routing

---

## Installation

### One-click install (recommended for new users)

1. **Download** the release ZIP — `dist/a0_lmm_router-1.3.0.zip` from this repo or from your GitHub release page.
2. **Extract** into your Agent Zero workspace under `usr/plugins/`:
   ```powershell
   Expand-Archive -Path "a0_lmm_router-1.3.0.zip" `
                  -DestinationPath "C:\path\to\agent-zero-2\usr\plugins\a0_lmm_router" -Force
   ```
3. **Restart** Agent Zero. The plugin's `hooks.install()` runs automatically and installs Python deps (`mcp`, `aiohttp`, `pyyaml`) into the A0 venv if missing.
4. **Activate** via the WebUI: Plugin List → a0_lmm_router → toggle ON. (Or create file `.toggle-1` at the plugin root.)
5. **Wire the MCP server** (one-time): WebUI → Settings → MCP Servers Configuration JSON → add the snippet shown in [MCP Server (v1.3+)](#mcp-server-v13) above. Click **Apply now**.

That's it. The plugin self-registers, its dashboard appears under `/a0/usr/plugins/a0_lmm_router/webui/dashboard.html`, and 9 MCP tools become available to any A0 agent.

### Option B: Already present in this repo (dev workflow)

The plugin is installed at `usr/plugins/a0_lmm_router/`. It is activated by the presence of `.toggle-1`. Use the included `start_agent_zero.bat` which boots the llama.cpp fleet, the host helper, the A0 container, and the MCP server in the right order.

### Option C: Manual install from ZIP into another A0 instance

```powershell
# Extract the ZIP into the target A0 instance's plugins directory
Expand-Archive -Path "dist\a0_lmm_router-1.3.0.zip" `
               -DestinationPath "C:\path\to\agent-zero\usr\plugins\a0_lmm_router" `
               -Force

# If old plugins exist, delete them to avoid conflicts
Remove-Item -Recurse -Force `
  "C:\path\to\agent-zero\usr\plugins\a0_lmm", `
  "C:\path\to\agent-zero\usr\plugins\a0_smart_router"
```

Then restart the A0 container. The plugin auto-registers on startup via the `agent_init` extension and `hooks.install()` ensures Python deps are present.

### Activation

Toggle files control activation:

- `.toggle-1` — plugin is **enabled** globally
- `.toggle-0` — plugin is **disabled** globally
- Per-agent / per-project rules are managed via the plugin Switch modal in the A0 WebUI

---

## Configuration

### Primary config files

| File | Purpose | Location |
|---|---|---|
| `conf/llama_cpp_servers.yaml` | Active slots, backend selection, LMM container hosts | A0 root `conf/` |
| `conf/installed_models.yaml` | Installed GGUF models, roles, VRAM, capabilities | A0 root `conf/` |
| `conf/compute_resources.yaml` | Hardware inventory and VRAM allocation profiles | A0 root `conf/` |
| `conf/model_providers.yaml` | Agent Zero provider metadata for the LMM Router OpenAI-compatible endpoint | Plugin `conf/` |

### Agent Zero Provider (`lmm_router`)

When the plugin is enabled, `conf/model_providers.yaml` registers an
`lmm_router` provider for chat and embedding models. The provider delegates to
LiteLLM's `openai` compatibility mode with `custom_llm_provider: openai`, so
Agent Zero can call any model alias exposed by the local llama.cpp Router Mode
endpoint.

Recommended Model Configuration / preset values:

```yaml
chat:
  provider: lmm_router
  name: chat
  api_base: http://host.docker.internal:8080/v1
  kwargs:
    api_key: not-needed
utility:
  provider: lmm_router
  name: utility
  api_base: http://host.docker.internal:8080/v1
  kwargs:
    api_key: not-needed
embedding:
  provider: lmm_router
  name: embedding
  api_base: http://host.docker.internal:8080/v1
  kwargs:
    api_key: not-needed
```

In Router Mode, the model names must match aliases from
`conf/models_preset.ini` and `/v1/models`: `chat`, `utility`, and `embedding`.
Do not use `llama` unless you explicitly add a `llama` alias to
`models_preset.ini`; otherwise llama.cpp returns `model 'llama' not found`.

Keep `ROUTER_PARALLEL=1` for long Agent Zero conversations. llama.cpp divides
`--ctx-size` across parallel slots, so `ROUTER_CTX_SIZE=65536` with
`ROUTER_PARALLEL=4` gives each request only `n_ctx_slot=16384` and can fail with
`request (...) exceeds the available context size (16384 tokens)`.

### Backend selection

Edit `conf/llama_cpp_servers.yaml`:

```yaml
global:
  backend: "remote"        # remote | docker | subprocess | auto
  lmm_hosts:
    chat: "a0-llama-chat:8080"
    utility: "a0-llama-utility:8088"
    embedding: "a0-llama-embed:8082"
```

- **`remote`** (recommended for Docker) — plugin is an HTTP client to pre-running LMM containers
- **`docker`** — plugin manages Docker containers via the Docker SDK
- **`subprocess`** — plugin spawns `llama-server.exe` processes on the host
- **`auto`** — auto-detects based on environment

### Connectivity model (how Agent Zero reaches the LMM fleet)

The plugin is an **outbound HTTP client** to the llama-server fleet. That
means the Agent Zero container does **not** need any additional host
ports published for LMM traffic — the only port A0 exposes is its own
WebUI (e.g. `5080:80`). What does matter is how the `lmm_hosts` URLs in
`conf/llama_cpp_servers.yaml` resolve from *inside* the A0 container.

Two supported topologies:

#### 1. Host-loopback via `host.docker.internal` (default, simplest)

The LMM containers publish their ports on the host (8080 / 8088 / 8082),
and A0 reaches them through the Docker Desktop loopback alias:

```yaml
# conf/llama_cpp_servers.yaml
global:
  backend: "remote"
  lmm_hosts:
    chat:      "host.docker.internal:8080"
    utility:   "host.docker.internal:8088"
    embedding: "host.docker.internal:8082"
```

Requirements in the A0 `docker-compose.yml`:

```yaml
services:
  agent-zero-2:
    # …
    extra_hosts:
      - "host.docker.internal:host-gateway"   # explicit on Linux, no-op on Docker Desktop
    networks:
      - a0-lmm-net

networks:
  a0-lmm-net:
    external: true          # created by usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml
```

Pros: works out of the box with Docker Desktop on Windows/macOS; the LMM
endpoints are reachable from the host too (useful for `curl` / Postman
debugging). Cons: the llama-servers are exposed on the host's network
interface, which may not be desirable in multi-tenant or untrusted
environments.

#### 2. Shared Docker network (more isolated, Docker-native)

Attach the A0 container to the same user-defined bridge as the LMM
fleet (`a0-lmm-net`) and address the llama-servers by their container
DNS names. The LMM containers then no longer need to publish ports on
the host at all.

```yaml
# docker-compose.yml  (A0 side)
services:
  agent-zero-2:
    # … same as before …
    networks:
      - a0-lmm-net

networks:
  a0-lmm-net:
    external: true          # created by usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml
```

```yaml
# conf/llama_cpp_servers.yaml
global:
  backend: "remote"
  lmm_hosts:
    chat:      "a0-llama-chat:8080"
    utility:   "a0-llama-utility:8080"   # note: container-internal port, not the host port
    embedding: "a0-llama-embed:8080"
```

Optional hardening: remove the `ports:` stanzas from
`usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml` so the fleet is reachable only from inside
`a0-lmm-net`.

Pros: llama endpoints are invisible from the host; cleaner DNS; easier
to scale to multiple A0 instances sharing one fleet. Cons: slightly
harder to poke at the endpoints from the host for ad-hoc debugging
(you'll need `docker exec` or a temporary published port).

**Rule of thumb:** stay on model #1 while iterating locally; switch to
model #2 when you want the fleet "locked down" behind Docker's internal
network or when multiple agents / projects will share the same fleet.

---

## Fleet Model Management

The plugin now uses a centralized fleet model management system via the host helper. This replaces the old local model manager with a more robust architecture.

### Architecture

- **Host Helper** (`lmm_host_helper.py`) — Runs on the Windows host, exposes HTTP API on port 55501
- **Fleet Models Abstraction** (`helpers/fleet_models.py`) — Python client library for host helper communication
- **Model Manifest** — JSON file tracking installed models, assignments, and download jobs
- **Docker Compose** — `usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml` + `.env` for fleet configuration

### Host Helper

The host helper provides:

- Model manifest management (add, remove, update models)
- Download jobs with progress tracking (via `huggingface-cli download`)
- Model assignment to slots (chat, utility, embed)
- Transactional assignment with automatic rollback on failure
- HuggingFace token management for private models
- Fleet upgrade (pull new llama.cpp image, restart containers)
- Job status polling and cancellation

Authentication is via `X-Token` header using a token from `A0_LMM_HOST_TOKEN_PATH` environment variable.

### Model Manifest

The manifest file tracks:

```json
{
  "models": [
    {
      "model_id": "qwen3.5_9b",
      "repo_id": "bartowski/Qwen3.5-9B-GGUF",
      "filename": "Qwen3.5-9B-Q4_K_M.gguf",
      "role": "chat",
      "assigned_slot": "chat",
      "size_gb": 5.7,
      "pending_assignment": false
    }
  ],
  "jobs": [
    {
      "job_id": "uuid",
      "repo_id": "...",
      "filename": "...",
      "status": "downloading|completed|failed|cancelled",
      "progress": 75
    }
  ]
}
```

### Transactional Assignment

When assigning a model to a slot:

1. Save previous assignment state (model's current slot, slot's current model)
2. Mark model as `pending_assignment: true`
3. Write environment variable to `usr/plugins/a0_lmm_router/docker/docker-compose.lmm.env`
4. Restart the Docker service
5. Wait for health check
6. On success: clear `pending_assignment` flag
7. On failure: rollback to previous state, restore previous model to slot

The GUI shows a PENDING badge for models with `pending_assignment: true` and disables assign/delete buttons during assignment.

### Docker Compose Configuration

Model paths and parameters are configured via environment variables in `usr/plugins/a0_lmm_router/docker/docker-compose.lmm.env`:

```env
LMM_CHAT_MODEL=/models/chat/qwen3.5_9b/Qwen3.5-9B-Q4_K_M.gguf
LMM_CHAT_CTX_SIZE=65536
LMM_CHAT_N_PARALLEL=1
LMM_CHAT_N_BATCH=512
LMM_CHAT_FLASH_ATTN=1

LMM_UTILITY_MODEL=/models/utility/qwen3.5_9b/Qwen3.5-9B-Q4_K_M.gguf
LMM_UTILITY_CTX_SIZE=16384
LMM_UTILITY_N_PARALLEL=1
LMM_UTILITY_N_BATCH=512
LMM_UTILITY_FLASH_ATTN=1

LMM_EMBED_MODEL=/models/embed/nomic-embed-text-v1.5.Q4_K_M.gguf
LMM_EMBED_CTX_SIZE=8192
LMM_EMBED_N_PARALLEL=1
LMM_EMBED_N_BATCH=512
LMM_EMBED_FLASH_ATTN=0
```

The `usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml` file references these variables for each service.

### Optimization Flags Reference

All flags are **opt-in** — they are not passed to llama.cpp unless the
corresponding env var is set in `.env`. This keeps existing setups
unchanged.

| Flag | Env Var | What It Does | When To Use |
|------|---------|-------------|-------------|
| `--no-mmap` | `<SLOT>_NO_MMAP=1` | Pin full model in RAM at startup | Faster init, needs enough RAM for full model |
| `--mlock` | `<SLOT>_MLOCK=1` | Lock model memory, prevent OS swap | Reduces latency spikes; needs `memlock:-1` in compose (already set) |
| `--n-cpu-moe N` | `<SLOT>_CPU_MOE=N` | Offload N MoE experts to CPU | For MoE models that exceed VRAM (e.g. Qwen3.6-35B-A3B) |
| `--cache-type-k X` | `<SLOT>_CACHE_TYPE_K=X` | KV cache K quantization | Halves KV cache VRAM; values: `f16`, `q8_0`, `q5_0`, `q4_0` |
| `--cache-type-v X` | `<SLOT>_CACHE_TYPE_V=X` | KV cache V quantization | Same as above for V cache |

**Example:** To enable quantized KV cache for the chat slot, uncomment in `.env`:
```env
CHAT_CACHE_TYPE_K=q4_0
CHAT_CACHE_TYPE_V=q4_0
```
Then restart: `docker compose -f usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml up -d`

**TurboQuant** (`--ctk-v` / `--ctv-v`) is from a fork and not supported
by the official `ghcr.io/ggml-org/llama.cpp:server-cuda` image. Use
`--cache-type-k` / `--cache-type-v` instead.

### Removal of Local Model Manager

The old local model manager has been retired. All model management operations now go through the host helper and Docker Compose fleet.

Changes made:
- Removed local model manager startup/shutdown from `start_agent_zero.bat` and `stop_agent_zero.bat`
- Removed local model manager health checks from `lmm_manager.bat` and `status_agent_zero.bat`
- Replaced local model manager APIs with fleet model management APIs
- Updated GUI config panel to use fleet status and model management

----

## GUI-Driven Model Installation (v1.2.2)

The dashboard now supports installing arbitrary GGUF models from HuggingFace and assigning them to slots directly from the WebUI.

### Installing a Model from HuggingFace

1. Open the **Dashboard** (Settings → Plugins → LMM Router → Dashboard button)
2. Scroll to the **Install Model** section
3. Enter:
   - **Repo ID**: e.g. `Jiunsong/supergemma4-26b-uncensored-gguf-v2` or `OBLITERATUS/gemma-4-E4B-it-OBLITERATED`
   - **Filename**: the `.gguf` file name from that repo (e.g. `supergemma4-26b-uncensored-q4_k_m.gguf`)
   - **Role** (optional): chat, utility, embedding, or vision — helps with organization
4. Click **Download** — a progress bar shows download status
5. The model appears in **Installed Models** when complete

### Assigning a Model to a Slot

1. In the **llama.cpp Slots** section, each slot shows:
   - Current status (healthy/unhealthy/stopped)
   - A dropdown with all installed models
   - **Assign** button (disabled if no model selected)
   - **Load** / **Unload** buttons for VRAM management

2. Select a model from the dropdown and click **Assign**
3. The slot's container restarts automatically with the new model
4. A spinner shows "Restarting container…" during the swap (~10-90s depending on model size)

### VRAM-Aware Load/Unload

- **Load**: Starts the slot's container with its currently assigned model
- **Unload**: Stops the container, freeing VRAM
- The UI shows a warning if loading a model would exceed available VRAM
- For your 24GB RTX 4090: keep only one large chat model loaded at a time (e.g. supergemma4-26b @ ~16-18GB OR gemma-4-E4B @ ~3-5GB)

### How It Works (Architecture)

- **Files on disk**: Multiple GGUFs live in the shared host models folder (`C:/Users/frant/A0-Data-Permanent/A0_v.adam/models`), mounted read-only at `/models` in every container
- **One process = one model**: Each `llama-server` container loads exactly one GGUF via `--model`. To change models, the container is recreated with a new `--model` arg.
- **Restart-to-swap**: When you assign a model, the host helper:
  1. Rewrites the `*_MODEL_PATH` in `docker-compose.lmm.env`
  2. Automatically calculates optimal context window size based on:
     - GGUF metadata (max training context `n_ctx_train`)
     - Available GPU VRAM
     - Model size and other running slots
  3. Rewrites the `*_CTX_SIZE` in `docker-compose.lmm.env` with the calculated value
  4. Runs `docker compose up -d --force-recreate <service>` for that slot only
  5. Other slots are unaffected; A0 keeps talking to the same stable host:port

### Automatic Context Window Sizing

The system now automatically calculates the optimal context window size for each model assignment. This ensures you get the maximum context your hardware can support without exceeding VRAM limits.

**Critical constraint: Agent Zero system prompts**

The context window must be large enough to accommodate Agent Zero's system prompts for each role:

- **Chat agent**: Minimum 16K tokens (real minimum) - full system prompt (~3.3K tokens) + conversation history + tool results. With 8K the agent will choke after a few steps.
- **Utility/subagent**: Minimum 8K tokens (16K recommended) - shorter prompt + specific task + tool result
- **Embedding**: Minimal context needed (for vectors only)
- **Vision/Reasoning**: Minimum 16K tokens

This is why qwen3.5-9B runs as both chat and utility - smaller models cannot handle the system prompt overhead.

**How it works:**

1. When you assign a model to a slot, the host helper reads the GGUF metadata to extract:
   - `n_ctx_train`: The model's maximum training context length
   - `n_layer`: Number of layers
   - `n_embd`: Embedding dimension

2. It calculates the optimal context based on:
   - **Role minimum**: Never falls below the minimum required for that role (chat: 16K, utility: 8K, etc.)
   - **Model's max context**: Never exceeds the model's training limit
   - **Available VRAM**: Accounts for model weights + KV cache + other running slots
   - **KV cache formula**: `ctx_size * 2 * n_layer * n_embd * bytes_per_token`

3. The calculated context is rounded down to a power of 2 for efficiency

4. The `*_CTX_SIZE` environment variable is updated in `docker-compose.lmm.env`

**Example:**

For a 24GB RTX 4090 with Qwen3.5-9B (5.7GB file) assigned to chat:
- Role minimum: 16K tokens (required for system prompt)
- Model supports 262K tokens (n_ctx_train)
- Available VRAM after weights: ~18GB
- Calculated optimal context: ~65K tokens
- Result: `CHAT_CTX_SIZE=65536`

**Fallback behavior:**

If GGUF metadata cannot be read or calculation fails, the system uses the role minimum:
- Chat: 32K tokens (minimum 16K enforced)
- Utility: 16K tokens (minimum 8K enforced)
- Embedding: 8K tokens
- Vision/Reasoning: 32K tokens (minimum 16K enforced)

This design means:
- No proxy/router between A0 and slots needed
- 10-90s swap time (model load from disk into VRAM)
- Only one big model resident at a time fits in 24GB VRAM
- Multiple models can be installed and ready for quick swap

----

## Usage

### From the Agent Zero WebUI

1. Open **Settings → Plugins → LMM Router** (click the settings cog)
2. Use the config panel to start/stop individual slots or all slots
3. Click **DASHBOARD** for the real-time monitoring view
4. Click **DEV STATUS** for the development tracker

### From .bat scripts (Windows)

```powershell
# Full startup (A0 container + LMM Docker containers + Local Model Manager + Host Helper)
.\start_agent_zero.bat

# Stop everything
.\stop_agent_zero.bat

# LMM-only manager (run from repo root: agent-zero-2/)
.\lmm_manager.bat start     # start chat slot only
.\lmm_manager.bat full      # start chat + utility + embed
.\lmm_manager.bat stop      # stop all LMM containers
.\lmm_manager.bat restart   # restart running containers (auto-detect profile)
.\lmm_manager.bat status    # container status + health checks + GPU stats
.\lmm_manager.bat refresh   # stop → pull latest images → start → verify
.\lmm_manager.bat logs      # tail last 50 lines per container
.\lmm_manager.bat help      # show full command reference
```

### From the agent (via tool)

```python
# Agent invokes the llama_cpp_control tool
{
  "thoughts": ["I need to stop the chat slot"],
  "tool_name": "llama_cpp_control",
  "tool_args": { "operation": "stop", "server": "slot_chat" }
}
```

---

## API Reference

All endpoints are POST and accept JSON bodies. Path prefix: `/plugins/a0_lmm_router/`

### Fleet Model Management (via Host Helper)

| Endpoint | Body | Returns | Status |
|---|---|---|---|
| `fleet_status` | `{}` | `{ ok, fleet_status, hf_token_configured }` | **live** |
| `set_hf_token` | `{ token }` | `{ ok, message }` | **live** |
| `clear_hf_token` | `{}` | `{ ok, message }` | **live** |
| `delete_model` | `{ model_id }` | `{ ok, message }` | **live** |
| `assign_model` | `{ slot, model_id, apply_now }` | `{ ok, message }` | **live** |
| `fleet_upgrade` | `{}` | `{ ok, message }` | **live** |
| `fleet_upgrade_rollback` | `{}` | `{ ok, message }` | **live** |
| `job_status` | `{ job_id }` | `{ ok, job }` | **live** |
| `cancel_job` | `{ job_id }` | `{ ok, message }` | **live** |

### Legacy Slot Control

| Endpoint | Body | Returns | Status |
|---|---|---|---|
| `llamacpp_status` | `{}` | `{ ok, slots: [...] }` | **live** |
| `llamacpp_control` | `{ data: { operation, server } }` | `{ ok, message }` | **live** |
| `llamacpp_list_models` | `{}` | `{ ok, models }` | **live** |
| `lmm_compute_stats` | `{}` | `{ ok, gpu, cpu, ram, slots }` | **live** |
| `lmm_model_recommend` | `{ role?, max_vram_gb? }` | `{ ok, recommendations }` | **live** |
| `lmm_model_install` | `{ repo_id, filename, role? }` | `{ ok, job_id, message }` | **live** |
| `lmm_test_prompt` | `{ slot, prompt, max_tokens?, temperature?, system? }` | `{ ok, content, reasoning_content, usage, timings, ... }` | **live** |
| `lmm_host_ignite` | `{ action: ignite\|extinguish\|status\|run-bat\|health }` | `{ ok, http_status, stdout, stderr }` | **live** |

### Example: get compute snapshot

```bash
curl -X POST http://localhost:5080/plugins/a0_lmm_router/lmm_compute_stats \
     -H "Content-Type: application/json" \
     --cookie-jar cookies.txt -d '{}'
```

Response:

```json
{
  "ok": true,
  "gpu": [{ "name": "NVIDIA RTX 4090", "vram_used_mb": 10240, "vram_total_mb": 24576, "util_pct": 42 }],
  "cpu": { "percent": 28.4, "cores": 16 },
  "ram": { "used_gb": 24.1, "total_gb": 63 },
  "slots": [{ "id": "slot_chat", "running": true, "healthy": true }]
}
```

---

## WebUI Pages

### `config.html` — Settings Panel

Opens when the user clicks **Settings → LMM Router**. Provides:

- Fleet status (host helper version, llama.cpp image version, HF token status)
- Fleet models list with assign/delete buttons and pending badges
- Install model modal (repo, filename, role)
- Active jobs section with progress tracking and cancel button
- HF token management (set/clear token modal)
- Fleet upgrade controls (upgrade/rollback)
- Status badge (RUNNING / IDLE)
- **DASHBOARD** button → opens `dashboard.html` as a modal
- **DEV STATUS** button → opens `dev-tracker.html` as a modal

### `dashboard.html` — Real-time Monitor

Mission-control aesthetic. Polls `/lmm_compute_stats` every 5 seconds and shows:

- GPU bars (VRAM, utilization, temperature)
- CPU and RAM gauges
- Per-slot health and request metrics
- Model recommendations panel with one-click install

### `dev-tracker.html` — Development Status Radiator

Visual snapshot of plugin development state:

- Progress strip (done / wip / todo / blocked)
- 16 development phases with status and priority
- API endpoint registry (live / planned)
- Self-review checklist results
- Bug tracker (open / fixed / wontfix)
- File inventory (27 files) with status
- Notes section for blockers and next priorities

Data is defined in a plain JavaScript object inside the file — update it after each development phase.

---

## File Structure

```text
usr/plugins/a0_lmm_router/
├── plugin.yaml                       # Metadata, version, settings_sections
├── README.md                         # This file
├── .toggle-1                         # Global activation flag
│
├── api/                              # API handlers (ApiHandler subclasses)
│   ├── llamacpp_control.py           # start/stop slot operations
│   ├── llamacpp_status.py            # slot health snapshot
│   ├── llamacpp_list_models.py       # list GGUF models on disk
│   ├── lmm_compute_stats.py          # GPU/CPU/RAM + slot stats
│   ├── lmm_model_recommend.py        # hardware-aware model picks
│   └── lmm_model_install.py          # HuggingFace download
│
├── conf/
│   └── model_providers.yaml          # LiteLLM provider entries
│
├── extensions/python/
│   ├── agent_init/
│   │   └── _10_init_servers.py       # Initialize LMM slots at agent start
│   └── message_loop_start/
│       └── _20_smart_router.py       # Route messages to best slot
│
├── helpers/
│   ├── llama_cpp_manager.py          # Main LMM orchestrator
│   ├── compute_monitor.py            # GPU/CPU/RAM telemetry
│   ├── model_recommender.py          # Catalog + hardware fit logic
│   ├── backends/                     # Pluggable LMM backends
│   │   ├── base.py
│   │   ├── remote_backend.py         # HTTP to llama-server
│   │   ├── docker_backend.py         # Docker SDK
│   │   ├── subprocess_backend.py     # Host-native llama-server.exe
│   │   └── factory.py                # Backend selection
│   └── smart_router/
│       ├── session_models.py         # Pydantic models for sessions
│       └── workflow_registry.py      # Workflow definitions
│
├── tools/
│   ├── llama_cpp_control.py          # Agent tool for LMM control
│   └── lmm_host_helper.py            # Host-side HTTP bridge for Docker control + GPU stats
│
└── webui/
    ├── config.html                   # Settings panel (Alpine.js)
    ├── dashboard.html                # Real-time monitoring dashboard
    ├── dev-tracker.html              # Development status page
    └── js/
        └── dashboard-store.js        # Alpine store for dashboard
```

Total: **27 files** + `README.md`

---

## Development Status

Current: **v1.0.0** — merge complete, ready for inference-client upgrade (see `work/plans/PLAN_LMM_ROUTER_UPGRADE.md`)

| Phase | Status |
|---|---|
| Plugin merge (a0_lmm + a0_smart_router → a0_lmm_router) | done |
| Import path migration | done |
| Config & activation | done |
| Compute monitor | done |
| Model recommender + installer | done |
| Dashboard UI | done |
| Dev Tracker UI | done |
| Self-review workflow | done |
| LMM inference client (chat_completion, streaming, fallback) | **todo** |
| `local_inference` tool + system prompt templates | **todo** |
| Task classifier + executor | **todo** |
| Terminal delegation | **todo** |

See `webui/dev-tracker.html` for the full live breakdown.

---

## Current Fleet (RTX 4090, 24GB VRAM)

> Last updated: 2026-04-26 — Both slots use Qwen3.5-9B; Phi-4-14B removed (16K native limit, YaRN scaling unsupported by this GGUF).
> Models use **dynamic KV cache allocation** (allocated only for tokens actually in the prompt), so large context windows are safe even with limited VRAM.

| Slot | Model | Params | Weights (Q4_K_M) | Context | KV Cache (full) | Role |
|------|-------|--------|------------------|---------|-----------------|------|
| **chat** | Qwen3.5-9B | 9B dense | ~5.7 GB | **64K** | ~9 GB | Main agent reasoning |
| **utility** | Qwen3.5-9B | 9B dense | ~5.7 GB | **16K** | ~2.3 GB | Sub-agents, wiki ops, tool calls |
| **embed** | nomic-embed-text-v1.5 | — | ~0.5 GB | **8K** | ~0.4 GB | Embeddings |

**VRAM usage at typical load:**
- qwen3.5-9B @ 64K context with ~32K prompt: ~5.7 GB weights + ~4.5 GB KV = ~10.2 GB
- qwen3.5-9B @ 16K context with ~8K prompt: ~5.7 GB weights + ~1.1 GB KV = ~6.8 GB
- nomic-embed @ 8K: ~0.5 GB weights + negligible KV
- **Total active:** ~10.2 GB (only chat loaded) to ~17 GB (chat + utility loaded)

**VRAM margin:** ~7 GB buffer on 24 GB card when all slots active.

**Rationale for this configuration:**
- **Phi-4-14B was removed** — its GGUF has a hard `n_ctx_train=16384` limit. Attempts to extend via `--rope-scaling yarn` + `--rope-scale 4.0` and `--override-kv` failed to change the server's reported `n_ctx`. The model simply rejected prompts >16K.
- **Qwen3.5-9B** natively supports 262K context (`n_ctx_train=262144`), so 64K works without any tricks. Both slots share the same GGUF file on disk.
- **Chat @ 64K** handles long Agent Zero conversations without truncation (was failing at 16K with 28K+ token prompts).
- **Utility @ 16K** provides fast sub-agent/wiki operations while conserving VRAM when both slots are loaded simultaneously.
- **Dynamic KV allocation** means VRAM is consumed proportionally to actual prompt length, not the full context window.
- **Router aliases** (`chat`, `utility`, `embedding`) are the stable model IDs Agent Zero sends to the local Router Mode endpoint; swap the backing GGUF files in `models_preset.ini`.

### GGUF Sources

| Model | HuggingFace Repo | File |
|-------|-----------------|------|
| Qwen3.5-9B | `bartowski/Qwen3.5-9B-GGUF` | `Qwen3.5-9B-Q4_K_M.gguf` |
| nomic-embed | (existing) | `nomic-embed-text-v1.5.Q4_K_M.gguf` |

> **Note:** Both chat and utility slots use the **same** Qwen3.5-9B GGUF file (`models/utility/qwen3.5_9b/Qwen3.5-9B-Q4_K_M.gguf`). The file is loaded independently by each llama.cpp server process (no shared weights across containers on Windows Docker Desktop).

---

## Wiki Routing via `wiki_librarian` Sub-Agent

All wiki operations (`wiki_query`, `wiki_ingest`, `wiki_lint`, etc.) are offloaded from the main chat model to the **utility slot** via a dedicated sub-agent profile: **`wiki_librarian`**.

This preserves the chat model's context window for reasoning and tool calls, while the utility model handles knowledge retrieval and cross-wiki synthesis.

### How it works

1. Main chat agent (Qwen3.5-9B @ 64K) detects a wiki-relevant query.
2. It invokes `call_subordinate` with `profile: "wiki_librarian"`.
3. The sub-agent runs on the **utility slot** (Qwen3.5-9B, 16K context).
4. It queries the SharedBrain wikis, synthesizes a cited answer, and returns it.

### Sub-agent profile location

```
usr/agents/wiki_librarian/
├── agent.yaml                           # metadata
├── prompts/
│   ├── agent.system.main.role.md       # identity & scope
│   ├── agent.system.main.specifics.md  # grants, privacy, citation rules
│   ├── agent.system.main.communication.md  # Answer / Citations / Coverage format
│   ├── agent.system.main.solving.md    # standard operating flow
│   ├── agent.system.main.tips.md       # edge cases (contradictions, stale data)
│   └── agent.system.tools.md           # whitelist: wiki_* + response only
└── README.md                            # activation & context budget
```

### Context budget (by slot)

**Chat slot (64K context):**

| Component | Tokens |
|-----------|--------|
| System prompts + framework | ~3.3K |
| Wiki payload (index + 3–5 pages) | ~4.0K |
| User conversation history | ~20.0K |
| Tool results buffer | ~4.0K |
| Final answer + reasoning | ~12.0K |
| **Total** | **~43K / 64K** |

**Utility slot (128K context):**

| Component | Tokens |
|-----------|--------|
| System prompts (wiki_librarian) | ~2.0K |
| Framework overhead | ~1.3K |
| Wiki payload (index + 2–3 pages) | ~3.0K |
| User question + short history | ~1.5K |
| Tool results buffer | ~2.0K |
| Final answer + reasoning | ~6.2K |
| **Total** | **~16K / 16K** |

### Registry grants

`wiki_librarian` is registered in `SharedBrain/registry.yaml` with read access to all wikis and write access to `commons`, `general`, `slr_project`, `llm_wiki_project`. Write access to `about_user` is **excluded** — privacy-sensitive queries are flagged and require main-agent approval.

---

## Event Log

Chronological record of configuration changes and troubleshooting sessions.

### 2026-04-26 — Context Size Fix + Model Swap (Phi-4 → Qwen3.5-9B)

**Problem:** `litellm.exceptions.BadRequestError: prompt (28902 tokens) exceeds available context size (16384 tokens)`

**Root cause:** Phi-4-14B GGUF has `n_ctx_train=16384` hard limit. Agent Zero sent a 28K-token prompt to the chat slot.

**Failed attempts:**

| Attempt | Method | Result |
|---------|--------|--------|
| 1 | `--rope-scaling yarn --rope-scale 4.0` | ❌ Server still reports `n_ctx: 16384` |
| 2 | `--override-kv llama.context_length=u32:65536` | ❌ Type `u32` rejected by llama.cpp |
| 3 | `--override-kv llama.context_length=int:65536` + `--rope-freq-scale 0.25` | ❌ Phi-4 GGUF ignores all overrides |

**Resolution:** Swapped chat slot model from Phi-4-14B → **Qwen3.5-9B** (native `n_ctx_train=262144`). No tricks needed — 64K context works immediately.

**Files changed:**

| File | Change |
|------|--------|
| `usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml` | Chat slot: Phi-4 path → Qwen3.5-9B path, `ctx-size 16384` → `65536` |
| `usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml` | Utility slot: `ctx-size 65536` → `16384` (VRAM conservation) |
| `llama_cpp_servers.yaml` | Chat model_id → `qwen3.5_9b`, context_size → `65536` |
| `llama_cpp_servers.yaml` | Utility context_size → `16384` |
| `model_providers.yaml` | Registers the `lmm_router` OpenAI-compatible provider for chat/embedding |
| `presets.yaml` | `ctx_length`: chat `65536`, utility `16384`, embed `8192` |
| `presets.yaml` | `api_key: "not-needed"` added to all slots (avoids AuthenticationError) |
| `README.md` | Updated Current Fleet table, rationale, context budget |

**Final fleet state:**

| Slot | Model | Context | Status |
|------|-------|---------|--------|
| Chat | Qwen3.5-9B | **64K** | ✅ First successful local response achieved |
| Utility | Qwen3.5-9B | **16K** | ✅ Healthy |
| Embed | nomic-embed | **8K** | ✅ Healthy |

**VRAM impact:** Chat @ 64K with ~32K prompt ≈ 10.2 GB. Total active (chat + utility + embed) ≈ 17 GB / 24 GB.

**Note:** A0 container (`agent-zero-2`) required restart after preset changes to clear cached `ctx_length` values.

---

## Model Testing & Comparison Framework

This section helps you evaluate multiple models and make data-driven decisions about which LMM to use for different tasks.

### Why Compare Models?

Different models excel at different tasks:
- **Chat/General** → Larger context, good reasoning (Hermes 3, Qwen 2.5)
- **Code** → Code-trained models (DeepSeek-Coder, CodeLlama)
- **Fast/Utility** → Small, efficient models (Phi-3.5, Qwen 2.5 1.5B)
- **Reasoning** → Specialized for step-by-step thinking (DeepSeek-R1-Distill)

### Testing Methodology

#### 1. Establish Baseline Metrics

For each model you test, record these metrics:

| Metric | How to Measure | Target |
|--------|---------------|--------|
| **TTFT** (Time to First Token) | Dashboard or API call timing | < 500ms |
| **TPS** (Tokens Per Second) | `tokens_generated / generation_time` | > 20 TPS |
| **VRAM Usage** | `nvidia-smi` or Dashboard | < 80% of available |
| **Context Length** | Model card / `llama-server` startup logs | Match your needs |
| **Quality Score** | Manual evaluation on test prompts | Subjective 1-10 |

#### 2. Test Prompt Suite

Create a file `test_prompts.json` in your project:

```json
{
  "tests": [
    {
      "id": "chat_simple",
      "category": "chat",
      "prompt": "Explain quantum computing in one paragraph",
      "expected_length": 100,
      "criteria": ["accurate", "concise", "accessible"]
    },
    {
      "id": "code_python",
      "category": "code",
      "prompt": "Write a Python function to find prime numbers up to N using the Sieve of Eratosthenes",
      "expected_length": 50,
      "criteria": ["correct", "efficient", "documented"]
    },
    {
      "id": "reasoning_logic",
      "category": "reasoning",
      "prompt": "If a train travels 60 km in 30 minutes, and another travels 80 km in 40 minutes, which is faster? Show your reasoning.",
      "expected_length": 150,
      "criteria": ["step_by_step", "correct_math", "clear_conclusion"]
    },
    {
      "id": "utility_extraction",
      "category": "utility",
      "prompt": "Extract all email addresses from: Contact us at support@example.com or sales@company.co.il",
      "expected_length": 30,
      "criteria": ["complete", "no_false_positives"]
    }
  ]
}
```

#### 3. Run Comparison Tests

Use the built-in inference API to test each model:

```bash
# Test a specific slot
$body = @{
  operation = "inference"
  server = "slot_chat"
  data = @{
    prompt = "Explain quantum computing in one paragraph"
    max_tokens = 200
  }
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5080/plugins/a0_lmm_router/llamacpp_control" `
  -Method POST -ContentType "application/json" -Body $body
```

Or use the comparison script (create `scripts/compare_models.py`):

```python
#!/usr/bin/env python3
"""Compare multiple LMM models on the same test suite."""
import json
import time
import statistics
from pathlib import Path

# Configuration
SLOTS_TO_TEST = [
    {"name": "slot_chat", "url": "http://host.docker.internal:8080/v1"},
    {"name": "slot_utility", "url": "http://host.docker.internal:8088/v1"},
]

TEST_PROMPTS = Path("test_prompts.json").read_text()
prompts = json.loads(TEST_PROMPTS)["tests"]

results = []

for slot in SLOTS_TO_TEST:
    slot_results = {"slot": slot["name"], "tests": []}
    
    for test in prompts:
        # Run inference (pseudo-code, adapt to actual API)
        start = time.time()
        # response = call_llm(slot["url"], test["prompt"])
        elapsed = time.time() - start
        
        slot_results["tests"].append({
            "test_id": test["id"],
            "category": test["category"],
            "ttft_ms": elapsed * 1000,
            "tokens": len(response.split()),
            "tps": len(response.split()) / elapsed
        })
    
    results.append(slot_results)

# Save results
Path("model_comparison_results.json").write_text(
    json.dumps(results, indent=2)
)

# Print summary
for r in results:
    ttfts = [t["ttft_ms"] for t in r["tests"]]
    tps_vals = [t["tps"] for t in r["tests"]]
    print(f"\n{r['slot']}:")
    print(f"  Avg TTFT: {statistics.mean(ttfts):.1f}ms")
    print(f"  Avg TPS:  {statistics.mean(tps_vals):.1f}")
```

### Decision Matrix

Use this framework to decide which model to use:

```
Decision Tree:
│
├─ Is it a simple acknowledgment/confirmation?
│  └─ YES → Use Tiny Router CANNED response (skip LLM entirely)
│
├─ Is speed critical (< 100ms total)?
│  └─ YES → slot_utility (Phi-3.5 3.8B) or Tiny Router
│
├─ Is it complex reasoning/math/coding?
│  ├─ YES → Do you have VRAM for 70B+?
│  │  ├─ YES → slot_reasoning (DeepSeek-R1-Distill 70B)
│  │  └─ NO  → slot_chat (Hermes 3 8B) with reasoning prompt
│  └─ NO → Continue...
│
├─ Is context length > 8K tokens needed?
│  ├─ YES → slot_chat (Qwen 2.5 14B, 32K context)
│  └─ NO  → Continue...
│
└─ Default: slot_chat (best general-purpose balance)
```

### Configuration for Model Testing

Add to your `conf/llama_cpp_servers.yaml`:

```yaml
# Test slots - uncomment to enable multiple model comparison
test_slots:
  hermes3_8b:
    enabled: false  # Set true to test
    model: "Hermes-3-Llama-3.1-8B-Q4_K_M.gguf"
    port: 8090
    ctx_size: 8192
    
  qwen_14b:
    enabled: false
    model: "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    port: 8091
    ctx_size: 32768
    
  phi35_mini:
    enabled: true  # Utility slot comparison
    model: "Phi-3.5-mini-instruct-Q4_K_M.gguf"
    port: 8088
    ctx_size: 4096
```

### Tiny Router Integration (Future)

The plugin now includes analysis of `tiny_router` (see `_SYSTEM_MAP.html` Appendix A). Key patterns to adopt:

1. **Phase-based rollout**:
   - Phase 1: Log routing decisions only
   - Phase 2: Apply preset override (switch slots dynamically)
   - Phase 3: Skip LLM for canned responses

2. **Stats tracking**:
   ```python
   # After each routing decision
   record_routing(
       route="chat",           # Decision made
       slot_url="...:8080",    # Slot used
       inference_ms=45.2,      # Time spent
       model_used="Hermes3-8B" # Model name
   )
   ```

3. **Failover chain**:
   ```yaml
   slot_routing:
     primary: "http://host.docker.internal:8080/v1"   # slot_chat
     fallback_chain:
       - "http://host.docker.internal:8088/v1"        # slot_utility
       - "https://openrouter.ai/api/v1"               # API backup
   ```

### Recommended Test Workflow

1. **Week 1: Baseline**
   - Run test suite on current `slot_chat` model
   - Record TTFT, TPS, quality scores
   - Document VRAM usage under load

2. **Week 2: Compare Alternatives**
   - Enable one alternative slot at a time
   - Run same test suite
   - Compare metrics side-by-side

3. **Week 3: Decision**
   - Choose primary model based on your use case mix
   - Set up fallback chain
   - Document reasoning in project notes

4. **Ongoing: Monitor**
   - Dashboard shows real-time stats
   - Review `model_comparison_results.json` monthly
   - Re-test when new GGUF releases available

---

## Troubleshooting

### `exceed_context_size_error` / request exceeds available context size (65536)

Agent Zero compresses **history only** against `ctx_length × ctx_history` *before* the system prompt is built. With `lmm_router`, llama.cpp enforces a **hard** `n_ctx` on the full request (system + history + extras + completion).

**Fix (v1.3+):** extension `message_loop_prompts_after/_20_router_context_guard.py` runs **only** when the active chat model is the **Local Fleet (llama.cpp RTX 4090)** preset (global Settings or per-chat model switcher). All other presets use Agent Zero’s built-in history compression only.

1. Reads live `n_ctx` from router `GET /v1/models` (fallback: `ROUTER_CTX_SIZE`, slot config, preset `ctx_length`).
2. After the system prompt is assembled, computes a history budget:  
   `n_ctx × 0.9 − system_tokens − extras_estimate − 8192` (completion reserve).
3. Temporarily overrides `History._get_ctx_size_for_history()` to that budget and runs A0’s built-in `history.compress()` until the prompt fits.

You should see **“LMM Router context guard”** in the agent log when compression runs. If history still exceeds the budget after compression, start a new chat or reduce memories / system prompt size.

### Plugin does not appear in the Settings sidebar

- Ensure `.toggle-1` exists in the plugin directory
- Ensure old `a0_lmm/` and `a0_smart_router/` directories are removed (or at least have `.toggle-0`)
- Restart the A0 container: `docker restart agent-zero-2`

### `ImportError: No module named usr.plugins.a0_lmm_router`

- Confirm the plugin lives at `usr/plugins/a0_lmm_router/` (not nested deeper)
- A0 framework auto-adds `usr/plugins/` to `sys.path` — no `__init__.py` needed at plugin root
- Restart the A0 container after copying plugin files

### Dashboard shows "0 slots" or "manager offline"

- Confirm the LMM containers are running: `lmm_manager.bat status`
- Confirm the shared Docker network exists: `docker network ls | findstr run_default`
- Confirm `conf/llama_cpp_servers.yaml` has `backend: "remote"` and correct `lmm_hosts`
- **simpleeval bug workaround (2026-04 onwards):** some `agent0ai/agent-zero:latest`
  image builds ship without the `simpleeval` package even though it is listed in
  `requirements.txt`. That broke `_query_slots()` silently because the plugin
  imported `helpers.files` which transitively needs simpleeval. Fixed in-plugin
  as of this version — the plugin resolves its config path without touching
  A0 core helpers. If you see `ModuleNotFoundError: simpleeval` in agent logs,
  you can also install it directly:
  `docker exec agent-zero-2 /opt/venv-a0/bin/pip install simpleeval==1.0.3`

### Dashboard shows "No GPU detected (nvidia-smi unavailable)"

The A0 container does **not** have GPU passthrough by design — `nvidia-smi`
is only available on the host. The plugin bridges this via the host helper:

1. Make sure `tools/lmm_host_helper.py` is running on the host
   (`start_agent_zero.bat` launches it automatically).
2. Make sure `docker-compose.yml` for A0 mounts the host's `$TEMP` directory
   into `/host:ro` — that's where `a0_lmm_host.key` lives and the plugin
   reads it to authenticate against the helper's `/gpu-stats` endpoint.
3. Verify from inside the A0 container:
   `docker exec agent-zero-2 ls /host/a0_lmm_host.key` → should exist.
4. If the token is missing or the helper is down, the dashboard gracefully
   falls back to "GPU unavailable" without breaking.

### Models install fails

- Confirm `huggingface-cli` is installed inside the A0 container: `docker exec agent-zero-2 /opt/venv-a0/bin/pip show huggingface-hub`
- Check free disk space in the target models volume
- Large models (>5GB) may need `HF_HUB_ENABLE_HF_TRANSFER=1` for speed

### Lint warnings for `flask`, `aiohttp`, `psutil`, `yaml`

- These are harmless — the packages exist inside the A0 Docker container, not in your local IDE's Python env
- To silence, install them into your local dev venv: `pip install flask aiohttp psutil pyyaml`

---

## Cross-Plugin Integration

This plugin operates at a specific layer in the A0 model-selection stack. Understanding the 6 layers prevents confusion about "which router does what":

| Layer | Plugin | What it decides |
|-------|--------|---------------|
| 0 | `_model_config` | Static preset catalog (chat/utility/embed models) |
| 1 | `tiny_router` | Per-message preset override via DeBERTa classification |
| 2 | `a0_lmm_router` (smart_router) | **DISABLED** — workflow pattern matching currently no-op |
| 3 | `a0_lmm_router` (init_servers + rate_limit_retry) | Boots llama.cpp fleet; patches LLM calls for 429 backoff |
| 4 | `parallel_swarm` | Sub-task complexity classifier (per spawned sub-agent only) |
| 5 | `_tool_args_guard` | Normalizes malformed tool-call JSON before validation |

### Key Interactions

- **Fleet Bootstrap → Preset**: After clicking **Ignite Fleet** in the dashboard, select the "Local Fleet (llama.cpp RTX 4090)" preset in A0 Settings to route chat, utility, and embedding calls to the Router Mode endpoint (`:8080`) instead of OpenRouter.
- **Rate Limit Retry + Tiny Router**: Both patch the LLM path. Order is `_15_` (init) before `_20_` (message_loop), so rate-limit patches apply first — safe.
- **Tool Args Guard**: Orthogonal safety net; runs before strict validation to fix malformed `tool_request` dicts from weaker models (e.g., Nemotron-free tiers).

See `docs/lmm_plugins_survey.md` for the full cross-plugin analysis.

---

## Requirements

### Runtime (inside A0 container)

- Python 3.12+
- `aiohttp`, `flask`, `psutil`, `pyyaml`, `pydantic` (already present in A0 base image)
- Optional: `huggingface-hub` for model downloads
- Optional: `docker` Python SDK (for `docker` backend)

### Host

- **Windows:** Docker Desktop with NVIDIA Container Toolkit for GPU passthrough
- **NVIDIA drivers** (for `nvidia-smi` compute monitoring)
- `curl` (used by `.bat` scripts)

### LMM containers

- `ghcr.io/ggml-org/llama.cpp:server-cuda` (GPU slots)
- `ghcr.io/ggml-org/llama.cpp:server` (CPU slots)
- Managed via `usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml` (inside the plugin)

---

## ROADMAP

Priority-ordered list of next features. Each item is a self-contained piece
of work — pick the next one when current work is stable.

### P1 — Auto-discovery of LMM slots (zero-config install)

**Goal:** A fresh A0 container should find its llama.cpp slots automatically,
with no manual `conf/llama_cpp_servers.yaml` editing. This makes the plugin
drop-in installable on any A0 instance that sits on the same Docker network
(or has the host helper running).

**Why this matters:** the current setup forces every new A0 container to
hand-wire slot IDs, roles, and ports. That's a footgun when the llama fleet
port layout drifts (e.g. we already renamed `a0-v098` → `agent-zero-2` and
changed the UI port from `50001` to `5080`).

**Proposed design — three layered strategies, try in order:**

1. **Docker-socket introspection (strongest signal).**
   If `/var/run/docker.sock` is bind-mounted into A0, enumerate running
   containers whose image matches `ghcr.io/ggml-org/llama.cpp*` and read
   their published ports + `--alias` arg. Populate slot config from that.
2. **Host-helper introspection (preferred in Docker Desktop).**
   New endpoint on `tools/lmm_host_helper.py`: `/compose-ls` which runs
   `docker compose -f usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml ps --format json` on the host
   and returns the container/port list. The plugin calls this on startup
   and caches the result. Requires only the existing host-helper token —
   no extra privileges inside A0.
3. **DNS probing (last-resort fallback).**
   For each known llama container DNS name (`a0-llama-chat`,
   `a0-llama-utility`, `a0-llama-embed`) on standard ports (8080/8088/8082),
   probe `/health`. Any 2xx response registers the slot. Works even
   without docker access, as long as A0 and the fleet share a Docker
   network.

**UI:** "🔍 Auto-discover slots" button on the dashboard (next to "Refresh")
that runs the three strategies and shows a diff ("Found new slot X on
:8091 — add to config? [Save] [Ignore]"). Saving writes back to
`conf/llama_cpp_servers.yaml`.

**Value:** installing this plugin into a new A0 container becomes a single
action (`cp -r` the plugin, restart A0). No YAML editing, no drift bugs.

### P2 — Per-slot VRAM attribution (partially blocked on Windows)

Show **how much VRAM each slot is consuming**, not just the GPU total.
Currently the dashboard shows "23.7 / 24 GB" without saying which slot
holds what.

**Why this is hard on our setup:** true per-process VRAM accounting needs
nvidia-smi's `--query-compute-apps` to report `used_memory`, but Windows
WDDM mode (which RTX 4090 consumer cards use) returns `[N/A]` for that
field. TCC mode would work but is locked to datacenter GPUs.

**Pragmatic path forward:**

- Add a `vram_est_mb` field on each `SlotInfo`, computed as the GGUF
  model file size × 1.15 (rough weights + KV cache + context overhead).
  Mark it "~est" in the UI so users don't mistake it for a real
  measurement. Requires a new host-helper endpoint `/file-size?path=...`
  (the A0 container doesn't see the model files directly).
- For accurate numbers on Linux hosts / TCC setups, add a real
  `/gpu-processes` endpoint that calls `nvidia-smi --query-compute-apps`
  and cross-references PIDs with `docker inspect -f '{{.State.Pid}}'`.
  Skip silently on WDDM.

### P3 — Reasoning-content display in the main A0 chat

The Model Test panel already renders `reasoning_content` as a collapsible
section. The main A0 chat (outside the plugin) currently drops it on the
floor — you see an empty response when a reasoning model burns all its
tokens on thought. Fixing this requires edits to A0 core (`/a0/webui/...`)
to surface `reasoning_content` when present, which is out of scope for a
plugin but should be pushed upstream.

### P4 — Dashboard polish

- "Pull latest llama.cpp image" button (uses `/ignite` with a `--pull`
  action — one-click upgrade to new llama.cpp builds).
- Host-helper status indicator (green/red dot next to the "LIVE" badge).
- Per-slot context-length + quant info in the slot cards (already in the
  API response via `llamacpp_status`, just not rendered).

### P5 — Smart router re-enable

Layer 2 in the cross-plugin stack (see "Cross-Plugin Integration" above)
is currently a no-op. The extension lives at
`extensions/python/message_loop_start/_20_smart_router.py` but its
classifier is not wired. Re-enable it with the utility slot as the
classifier backend, guarded by a `plugin.yaml` setting so users can
opt in.

---

## Recent Changes (2026-04-20 session)

Captured here so you can reconstruct "what shipped" without diffing the
whole plugin history.

- **GPU visibility fix.** Added `/gpu-stats` endpoint to
  `tools/lmm_host_helper.py`. `helpers/compute_monitor.py::_query_gpus`
  now falls back to the host helper when `nvidia-smi` isn't available
  in-process — which is the default for the A0 container (no GPU
  passthrough). Dashboard now shows real RTX 4090 + VRAM stats from
  inside A0.
- **Token mount.** `docker-compose.yml` bind-mounts the host's `$TEMP`
  directory into `/host:ro` so the plugin can read `a0_lmm_host.key`
  and authenticate to the helper.
- **`helpers.files` dependency removed.** `compute_monitor.py`,
  `llama_cpp_manager.py`, and `api/llamacpp_status.py` previously
  imported `helpers.files` from A0 core, which transitively imports
  `simpleeval`. That package is declared in `/a0/requirements.txt` but
  ships missing from the current `agent0ai/agent-zero:latest` image, so
  every slot query was silently failing with `slots=[]`. The plugin now
  resolves its config path via `Path(__file__)` and keeps working even
  with a broken A0 core import chain.
- **Model Test panel.** New `api/lmm_test_prompt.py` + `webui/model-test.html`
  let you fire a one-off chat completion at any slot and inspect
  `content`, `reasoning_content`, `usage`, and `timings` side-by-side.
  Useful for comparing models and for seeing reasoning models (Gemma 4,
  DeepSeek-R1) without needing the full A0 agent loop. "🧪 Model Test"
  button added to the dashboard header.
- **Stale references cleaned.** Replaced `a0-v098` → `agent-zero-2` and
  `localhost:50001` → `localhost:5080` across README and
  `_SYSTEM_MAP.html`.

---

## License

Part of the Agent Zero Enhanced fork — same license as the parent project.

## Credits

- Merged from `a0_lmm` (llama.cpp management) and `a0_smart_router` (smart routing)
- Built on top of the Agent Zero plugin framework
- Uses llama.cpp server by Georgi Gerganov
