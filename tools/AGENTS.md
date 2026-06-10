# DOX contract - a0_lmm_router/tools

## Purpose

Agent Zero tool wrappers for router and fleet capabilities.

## Ownership

- Tools own agent-facing argument parsing and response shape.
- Helpers/service/MCP code own core behavior.

## Local Contracts

- Keep tool names and prompt exposure policy aligned.

## Work Guidance

- Prefer compact responses that do not flood the main agent context.

## Verification

- Run `python -m py_compile` on touched tool files.
- Inspect prompt/exposure behavior when tool availability changes.

## Child DOX Index

No child AGENTS.md files yet.
