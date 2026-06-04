# Fleet Manager Runbook

The Fleet Manager is the standalone control plane for local OpenAI-compatible
agents. It keeps the existing router/provider endpoints and adds agent-aware
state, bounded queueing, and fleet status APIs.

## What V1 Provides

- `GET /health`
- `GET /slots`
- `GET /health/slots`
- `POST /routing/request`
- `POST /v1/chat/completions`
- `GET /fleet/status`
- `GET /fleet/agents`
- `POST /fleet/agents/register`

`POST /v1/chat/completions` accepts these optional headers:

```http
X-Agent-ID: hermes-1
X-Agent-Type: hermes
X-Priority: normal
```

Priority must be `low`, `normal`, or `high`. High-priority requests move ahead
of lower-priority queued requests, but running requests are not preempted.

Every successful inference response includes:

- `x-a0-agent-id`
- `x-a0-agent-type`
- `x-a0-fleet-request-id`
- `x-a0-fleet-queue-depth`
- `x-a0-fleet-vram-available-gb`
- `x-selected-model`
- `x-hard-ctx`
- `x-effective-ctx`
- `x-context-occupancy`
- existing router headers such as `x-a0-router-slot-id`

`hard_ctx` is the llama.cpp context loaded for the selected alias. `effective_ctx`
is the quality-safe prompt budget window, defaulting to 70% of `hard_ctx` before
system/extras/output reserve are subtracted by the Agent Zero compression guard.
`/fleet/status` exposes the same context window data under `context_windows`.

## Agent Zero Prompt Budgeting

When Agent Zero uses the Local Fleet preset, plugin extensions reduce prompt
pressure before the request reaches llama.cpp:

- `tool_exposure` keeps the main profile's native A0 tool prompt small and
  routes heavyweight tools to specialist profiles.
- `mcp_exposure` filters rendered MCP tool prompts by profile without changing
  global MCP server settings.
- `output_budget` injects conservative completion caps for local chat and
  utility calls so local models do not decode thousands of unwanted tokens.

The latest prompt-surface manifests are written under `data/`:

- `tool_exposure_manifest.json`
- `mcp_exposure_manifest.json`

The active agent also receives telemetry keys:

- `a0_lmm_router_tool_exposure`
- `a0_lmm_router_mcp_exposure`
- `a0_lmm_router_output_budget`
- `a0_lmm_router_context_guard`

## Safety Boundary

The Fleet Manager does not mount Docker socket and does not start or stop
containers. A future `fleet-node` worker may hold Docker permissions and
register with the manager. This keeps multi-agent clients from gaining direct
host/container control through the inference API.

## Local Run

From the plugin root:

```powershell
.\scripts\run_provider.ps1 -Port 9000 -ApiKey "change-me" -InstallDeps
.\scripts\smoke_provider.ps1 -BaseUrl "http://127.0.0.1:9000" -ApiKey "change-me"
```

Linux/WSL:

```bash
./scripts/run_provider.sh --port 9000 --api-key "change-me" --install-deps
./scripts/smoke_provider.sh --base-url http://127.0.0.1:9000 --api-key "change-me"
```

## Docker Compose

```bash
docker network create a0-fleet-net 2>/dev/null || true
cp .env.fleet.example .env.fleet
docker compose --env-file .env.fleet -f docker/compose.fleet.yml up -d
```

The compose service runs only the control plane. Start llama.cpp Router Mode
separately with `docker/docker-compose.lmm.router.yml` or another managed
upstream declared in `conf/llama_cpp_servers.yaml`.

Before starting Router Mode, regenerate the curated preset:

```bash
python scripts/render_router_preset.py
```

The renderer treats `CHAT_CTX_SIZE`, `UTILITY_CTX_SIZE`, and `EMBED_CTX_SIZE`
as minimums, reads GGUF metadata and local VRAM, then writes per-alias
`ctx-size`, KV cache, GPU-layer, and flash-attention settings to
`conf/models_preset.ini`.

An optional CPU utility worker is available for subagents/background jobs:

```bash
docker compose -f docker/docker-compose.lmm.cpu-worker.yml \
  --env-file docker/docker-compose.lmm.env --profile cpu-worker up -d
```

After starting it, enable the disabled `utility_cpu` slot in
`conf/llama_cpp_servers.yaml`.

## MCP Bridge

MCP tools keep their legacy direct `BackendManager` behavior unless this env is
set:

```text
A0_FLEET_MANAGER_BASE_URL=http://127.0.0.1:9000
```

With that value set, MCP chat/status calls use the Fleet Manager API. Mutating
MCP tools such as `start_slot` return a clear unsupported response because V1
does not hold Docker permissions.

## State

SQLite is used for V1 telemetry:

- agents
- requests
- queue events
- model residency snapshots

Set `A0_FLEET_STATE_DB` to choose the database path. If unset, local runs use a
temp directory path outside the repository.

## Config Path Safety

`A0_LMM_ROUTER_CONFIG` must point to a file named `llama_cpp_servers.yaml`
inside a safe config root such as `/a0/conf`,
`/a0/usr/plugins/a0_lmm_router/conf`, `/app/conf`, or the plugin `conf/`
directory. Local dev/test runs also allow temp/workspace roots.

To allow another deployment-specific root, set:

```text
A0_LMM_ROUTER_CONF_ALLOW_ROOTS=/path/to/config/root
```

## Not In V1

- WebSocket/SSE fleet events
- Docker/container lifecycle control
- RBAC beyond bearer-token auth
- Hermes-specific config commands beyond using an OpenAI-compatible base URL
- VRAM attribution beyond the placeholder response shape
