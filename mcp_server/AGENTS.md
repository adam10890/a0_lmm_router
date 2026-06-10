# DOX contract - a0_lmm_router/mcp_server

## Purpose

MCP server tools, resources, and router bridge for local fleet control.

## Ownership

- MCP surfaces are portable integration contracts.
- Router bridge owns translation between MCP calls and backend/service actions.

## Local Contracts

- Preserve MCP exposure filtering and security tests.
- Without Fleet Manager env config, keep legacy BackendManager fallback behavior.

## Work Guidance

- Do not add A0 WebUI-only assumptions to MCP tool behavior.

## Verification

- Run MCP-focused tests under `tests/`.
- Run `python -m py_compile` on touched MCP files.

## Child DOX Index

No child AGENTS.md files yet.
