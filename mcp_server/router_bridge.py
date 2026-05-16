"""
router_bridge.py — Bridge between MCP tool calls and the BackendManager.

Uses the existing BackendManager singleton (llama_cpp_manager.py) for
slot lifecycle and failover, and aiohttp for proxying HTTP requests to
the appropriate llama.cpp container.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import aiohttp

logger = logging.getLogger("lmm_router.mcp.bridge")

# Allow running outside the /a0 container for testing.
_A0_ROOT = "/a0"
if _A0_ROOT not in sys.path and os.path.isdir(_A0_ROOT):
    sys.path.insert(0, _A0_ROOT)


def _get_manager():
    """Return the BackendManager singleton."""
    from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager  # noqa: PLC0415
    return BackendManager.get_instance()


def _slot_url(role: str, fallback_port_map: dict[str, int] | None = None) -> str | None:
    """Return the base v1 URL for a slot by role, using failover if needed."""
    mgr = _get_manager()

    # select_slot_with_failover returns a decision dict with 'url'
    decision = mgr.select_slot_with_failover(role)
    if decision:
        url = decision.get("url", "")
        if url:
            return url

    # Fallback: try lmm_hosts from global config
    hosts: dict[str, str] = mgr.global_config.get("lmm_hosts", {})
    if role in hosts:
        return f"http://{hosts[role]}/v1"

    # Last resort: static port map
    defaults = fallback_port_map or {"chat": 8080, "utility": 8088, "embedding": 8082}
    port = defaults.get(role)
    return f"http://localhost:{port}/v1" if port else None


async def chat_complete(
    messages: list[dict[str, str]],
    role: str = "chat",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    stream: bool = False,
) -> dict[str, Any]:
    """Forward a chat completion request to the appropriate slot."""
    url = _slot_url(role)
    if not url:
        return {"error": f"No healthy slot found for role '{role}'"}

    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    _get_manager().mark_slot_error(
                        _decision_slot_id(role), f"HTTP {resp.status}"
                    )
                return data
    except aiohttp.ClientError as exc:
        _get_manager().mark_slot_error(_decision_slot_id(role), str(exc))
        return {"error": str(exc)}


async def get_embeddings(texts: list[str]) -> dict[str, Any]:
    """Forward an embedding request to slot_embedding."""
    url = _slot_url("embedding")
    if not url:
        return {"error": "No healthy embedding slot found"}

    payload = {"input": texts, "model": "local-embed"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{url}/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                return await resp.json()
    except aiohttp.ClientError as exc:
        return {"error": str(exc)}


async def fleet_status() -> dict[str, Any]:
    """Return current status of all slots."""
    mgr = _get_manager()
    try:
        slots = await mgr.status()
    except Exception as exc:
        slots = {}
        logger.warning("fleet_status async failed: %s", exc)

    failover_info = {}
    try:
        failover_info = mgr.get_failover_status()
    except Exception:
        pass

    return {
        "slots": slots,
        "failover": failover_info,
        "backend": mgr.backend_type,
    }


async def start_slot(slot_id: str) -> dict[str, Any]:
    return await _get_manager().start_slot(slot_id)


async def stop_slot(slot_id: str) -> bool:
    return await _get_manager().stop_slot(slot_id)


async def start_fleet() -> dict[str, Any]:
    return await _get_manager().start_all()


def slot_configs() -> dict[str, Any]:
    """Return raw slot configs (for resource introspection)."""
    mgr = _get_manager()
    return getattr(mgr, "_slot_configs", {})


def hardware_profile() -> dict[str, Any]:
    """Return hardware info from config."""
    mgr = _get_manager()
    try:
        import yaml  # noqa: PLC0415
        with open(mgr.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("hardware", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decision_slot_id(role: str) -> str:
    """Get the slot id used for the given role (for error marking)."""
    mgr = _get_manager()
    decision = mgr.select_slot_with_failover(role)
    return decision.get("slot_id", f"slot_{role}") if decision else f"slot_{role}"
