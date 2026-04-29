# a0_lmm_router

**Unified LMM (Local Multimodal Model) Server Management + Smart Routing for Agent Zero**

Version: `1.0.0` · Merged from `a0_lmm` + `a0_smart_router` · Target: Agent Zero v0.9.7 / v1.9+

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

- **Multi-backend LMM support** — `remote` (HTTP to pre-running containers), `docker` (Docker SDK), `subprocess` (host-native `llama-server`)
- **llama.cpp slot control** — start/stop individual slots or all at once via API
- **Health monitoring** — polling `/health`, `/slots`, `/metrics`, `/props` of each llama-server
- **Real-time compute stats** — GPU (VRAM, utilization, temp) via `nvidia-smi`, CPU/RAM via `psutil`
- **Model recommender** — hardware-aware recommendations from a curated GGUF catalog
- **Model installer** — one-click download of GGUF models from HuggingFace via `huggingface-cli`
- **Smart router** — classifies incoming messages and selects the best slot (planned: LLM-based classifier via utility slot)

### Frontend

- **Config panel** (`config.html`) — slot list, start/stop controls, model picker, status badges
- **Dashboard** (`dashboard.html`) — real-time mission-control view of GPU, CPU, RAM, and slot health with 5s polling
- **Dev Tracker** (`dev-tracker.html`) — visual development status radiator (phases, bugs, APIs, self-review results)

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
│  lmm_compute_stats · lmm_model_recommend · lmm_model_install    │
└────────┬──────────────────┬─────────────────────┬───────────────┘
         ▼                  ▼                     ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ LlamaCppManager │  │ ComputeMonitor   │  │ ModelRecommender │
│ (backends/…)    │  │ (nvidia-smi,     │  │ (catalog,        │
│                 │  │  psutil)         │  │  huggingface-cli)│
└────────┬────────┘  └──────────────────┘  └──────────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────┐
│         Backend adapters (helpers/backends/)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐    │
│  │ remote      │  │ docker      │  │ subprocess           │    │
│  │ (HTTP)      │  │ (SDK)       │  │ (llama-server.exe)   │    │
│  └─────────────┘  └─────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  llama.cpp server instances                     │
│   a0-llama-chat:8080  ·  a0-llama-utility:8088  ·  :8082        │
└─────────────────────────────────────────────────────────────────┘
```

### Extension Points

- `extensions/python/agent_init/_10_init_servers.py` — Initializes LMM slots on agent start
- `extensions/python/message_loop_start/_20_smart_router.py` — Intercepts user messages for routing

---

## Installation

### Option A: Already present in this repo

The plugin is installed at `usr/plugins/a0_lmm_router/`. It is activated by the presence of `.toggle-1`.

### Option B: Install from ZIP into another A0 instance

```powershell
# Extract the ZIP into the target A0 instance's plugins directory
Expand-Archive -Path "dist\a0_lmm_router.zip" `
               -DestinationPath "C:\path\to\agent-zero\usr\plugins\a0_lmm_router" `
               -Force

# If old plugins exist, delete them to avoid conflicts
Remove-Item -Recurse -Force `
  "C:\path\to\agent-zero\usr\plugins\a0_lmm", `
  "C:\path\to\agent-zero\usr\plugins\a0_smart_router"
```

Then restart the A0 container. The plugin auto-registers on startup via the `agent_init` extension.

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
| `conf/model_providers.yaml` | LiteLLM provider entries + generic model ID mappings | Plugin `conf/` |

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
    external: true          # created by docker-compose.lmm.yml
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
    external: true          # created by docker-compose.lmm.yml
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
`docker-compose.lmm.yml` so the fleet is reachable only from inside
`a0-lmm-net`.

Pros: llama endpoints are invisible from the host; cleaner DNS; easier
to scale to multiple A0 instances sharing one fleet. Cons: slightly
harder to poke at the endpoints from the host for ad-hoc debugging
(you'll need `docker exec` or a temporary published port).

**Rule of thumb:** stay on model #1 while iterating locally; switch to
model #2 when you want the fleet "locked down" behind Docker's internal
network or when multiple agents / projects will share the same fleet.

---

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

| Endpoint | Body | Returns | Status |
|---|---|---|---|
| `llamacpp_status` | `{}` | `{ ok, slots: [...] }` | **live** |
| `llamacpp_control` | `{ data: { operation, server } }` | `{ ok, message }` | **live** |
| `llamacpp_list_models` | `{}` | `{ ok, models, models_dir }` | **live** |
| `lmm_compute_stats` | `{}` | `{ ok, gpu, cpu, ram, slots }` | **live** |
| `lmm_model_recommend` | `{ role?, max_vram_gb? }` | `{ ok, recommendations }` | **live** |
| `lmm_model_install` | `{ model_id }` | `{ ok, message, path }` | **live** |
| `lmm_test_prompt` | `{ slot, prompt, max_tokens?, temperature?, system? }` | `{ ok, content, reasoning_content, usage, timings, ... }` | **live** |
| `lmm_host_ignite` | `{ action: ignite\|extinguish\|status\|run-bat\|health }` | `{ ok, http_status, stdout, stderr }` | **live** |
| `lmm_host_helper` | Host-side HTTP bridge for container control + GPU stats from A0 container | Flask server on port 55501 | **live** |
| `lmm_inference` | `{ prompt, slot?, params? }` | `{ ok, response }` | *planned* |
| `lmm_task_classify` | `{ message }` | `{ ok, task_type, complexity }` | *planned* |

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

- Status badge (RUNNING / IDLE)
- **DASHBOARD** button → opens `dashboard.html` as a modal
- **DEV STATUS** button → opens `dev-tracker.html` as a modal
- Slot list with per-slot start/stop buttons
- Model dropdown with GGUF files from `models_dir`
- Plugin settings form (bound to `plugin.yaml` `settings_sections`)

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
- **Generic model IDs** (`local-chat`, `local-utility`, `local-embedding`) allow swapping models in `model_providers.yaml` without touching presets.

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
| `docker-compose.lmm.yml` | Chat slot: Phi-4 path → Qwen3.5-9B path, `ctx-size 16384` → `65536` |
| `docker-compose.lmm.yml` | Utility slot: `ctx-size 65536` → `16384` (VRAM conservation) |
| `llama_cpp_servers.yaml` | Chat model_id → `qwen3.5_9b`, context_size → `65536` |
| `llama_cpp_servers.yaml` | Utility context_size → `16384` |
| `model_providers.yaml` | Generic IDs `local-chat`/`local-utility` → map to `qwen3.5_9b` |
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

- **Fleet Bootstrap → Preset**: After clicking **Ignite Fleet** in the dashboard, select the "Local Fleet (llama.cpp RTX 4090)" preset in A0 Settings to route chat/utility calls to your local slots (`:8080`, `:8088`) instead of OpenRouter.
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
- Managed via `docker-compose.lmm.yml` in the A0 repo root

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
   `docker compose -f docker-compose.lmm.yml ps --format json` on the host
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
