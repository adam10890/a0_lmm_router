# DOX contract - a0_lmm_router/docker

## Purpose

Docker Compose files and container startup helpers for router/fleet workers.

## Ownership

- Compose files describe runtime topology.
- Do not store generated container state, logs, or secrets here.

## Local Contracts

- Keep role/slot names aligned with config and docs.
- Scribe/router compose changes must stay compatible with `a0_scribe` role
  expectations.

## Work Guidance

- Treat Docker lifecycle permissions as operator/worker concerns, not request
  path code.

## Verification

- Validate YAML by inspection or parser when available.
- Run relevant smoke scripts when compose behavior changes.

## Child DOX Index

No child AGENTS.md files yet.
