"""Return status of all configured llama.cpp slots, with live HTTP health probe."""
from __future__ import annotations

import asyncio
import os
import socket
import time

from pathlib import Path

import aiohttp
from flask import Request

from helpers.api import ApiHandler


HEALTH_TIMEOUT_SEC = 2.0


async def _probe_http(host: str, port: int) -> dict:
    """Quick /health ping; on failure also do a raw TCP connect so the UI can
    distinguish 'port closed' from 'model still loading'."""
    url = f"http://{host}:{port}/health"
    started = time.monotonic()
    try:
        timeout = aiohttp.ClientTimeout(total=HEALTH_TIMEOUT_SEC)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                text = await resp.text()
                return {
                    "reachable": True,
                    "http_status": resp.status,
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "body": text[:200],
                }
    except aiohttp.ClientResponseError as exc:
        return {"reachable": True, "http_status": exc.status, "error": str(exc)}
    except (aiohttp.ClientConnectorError, asyncio.TimeoutError, OSError):
        # Fall through to raw TCP probe below.
        pass
    except Exception as exc:  # pragma: no cover — unexpected client bug
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}

    # TCP-level probe — lets the UI show "port open, HTTP not ready" during
    # model load, vs "service down".
    try:
        with socket.create_connection((host, port), timeout=HEALTH_TIMEOUT_SEC):
            return {"reachable": True, "http_status": None, "note": "tcp_only"}
    except OSError as exc:
        return {"reachable": False, "error": str(exc)}


def _resolve_conf_path() -> str:
    """Resolve llama_cpp_servers.yaml path without depending on `helpers.files`
    (which transitively imports `simpleeval`, a package that ships broken in
    the current agent0ai/agent-zero:latest image). Kept self-contained so the
    plugin keeps working even when A0 core has import-chain issues."""
    here = Path(__file__).resolve()
    # .../usr/plugins/a0_lmm_router/api/llamacpp_status.py
    #    parents[1] = a0_lmm_router, parents[4] = /a0
    plugin_conf = str(here.parents[1] / "conf" / "llama_cpp_servers.yaml")
    root_conf = str(here.parents[4] / "conf" / "llama_cpp_servers.yaml")
    return plugin_conf if os.path.exists(plugin_conf) else root_conf


class LlamacppStatus(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import LlamaCppManager

        try:
            conf_path = _resolve_conf_path()
            # Reset singleton so config edits show up without a container restart.
            LlamaCppManager._instance = None  # noqa: SLF001
            manager = LlamaCppManager.get_instance(conf_path)

            lmm_hosts = (manager.global_config or {}).get("lmm_hosts", {}) or {}
            backend = (manager.global_config or {}).get("backend", "auto")

            slots = []
            for sid, srv in manager.servers.items():
                cfg = srv.config
                host = lmm_hosts.get(cfg.role.value, "host.docker.internal")
                host_only = host.split(":")[0] if ":" in host else host
                probe = await _probe_http(host_only, cfg.port)

                slots.append({
                    "id": sid,
                    "port": cfg.port,
                    "role": cfg.role.value,
                    "model_id": cfg.model_id or cfg.specialty or "unknown",
                    "model_path": cfg.model_path,
                    "enabled": cfg.enabled,
                    "status": srv.status.value,
                    "host": f"{host_only}:{cfg.port}",
                    "health": probe,
                })
            return {
                "ok": True,
                "backend": backend,
                "config_path": conf_path,
                "slots": slots,
            }
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "slots": []}
