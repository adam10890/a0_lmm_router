# LMM Router — Future development: multi-container fleet orchestration

> Captured 2026-06-07 while wiring the `scribe` role for the a0_scribe plugin.

## Vision

Today the router operates in one of a few static modes (Router Mode hot-swap,
fixed multi-slot, or remote). The next major capability is **dynamic
multi-container fleet orchestration**:

> The router gains the ability to **launch and control multiple llama.cpp
> containers**, each hosting one or more models in parallel, and to **scale them
> on demand**. Roles (chat / utility / scribe / embedding and future roles) can
> each get dedicated *or* pooled containers, sized to live load — instead of the
> operator hand-running one `docker-compose` file per slot.

## What it enables

- **Per-role dedicated containers** managed automatically (no manual compose).
- **Parallel model hosting** — multiple models live across multiple containers,
  beyond a single container's `--models-max` hot-swap budget.
- **Elastic scaling** — spin containers up under load, down when idle, within the
  hardware envelope (VRAM/RAM-aware, reusing `compute_monitor` + `slot_recommender`).
- **Pooled roles** — e.g. several `scribe` containers for many concurrent chats
  (see the a0_scribe plugin), or several ephemeral chat containers.

## Relationship to existing pieces

- `helpers/backends/docker_backend.py` already creates/destroys containers via the
  Docker SDK — the orchestration layer would drive it across *many* containers.
- `ephemeral` (per-conversation containers) and `helpers/ephemeral_pool.py` are an
  early, single-purpose version of this idea; generalize the warm-pool model.
- `helpers/fleet_mode.py` / `router_probe.py` already reality-detect running
  containers; extend them to manage a dynamic set.

## Consumer note

The a0_scribe super-ego is the first concrete consumer: it currently runs one
`scribe_gpu` container (`docker/docker-compose.lmm.scribe.yml`). Under multi-
container orchestration it would get pooled/parallel scribe containers managed by
the router. See `usr/plugins/a0_scribe/docs/FUTURE.md`.
