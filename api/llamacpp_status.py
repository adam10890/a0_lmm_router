"""Return status of all configured llama.cpp slots, with live HTTP health probe."""
from __future__ import annotations

import asyncio
import socket
import time

import aiohttp
from flask import Request

from helpers.api import ApiHandler
try:
    from usr.plugins.a0_lmm_router.helpers.conf_resolver import resolve_conf_path
except ImportError:
    from helpers.conf_resolver import resolve_conf_path


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


class LlamacppStatus(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            import yaml

            conf_path = resolve_conf_path(__file__)
            with open(conf_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}

            global_cfg = cfg.get("global", {}) or {}
            lmm_hosts = global_cfg.get("lmm_hosts", {}) or {}
            backend = global_cfg.get("backend", "auto")

            slots = []
            for slot in cfg.get("active_slots", []) or []:
                if not slot or not slot.get("enabled", True):
                    continue

                sid = slot.get("id", f"slot_{slot.get('port', 'unknown')}")
                role = slot.get("role", "chat")
                port = int(slot.get("port", 0) or 0)
                host = lmm_hosts.get(role, "host.docker.internal")
                if ":" in host:
                    host_only, host_port = host.rsplit(":", 1)
                    probe_port = int(host_port)
                else:
                    host_only = host
                    probe_port = port
                probe = await _probe_http(host_only, probe_port)
                running = bool(probe.get("reachable"))

                slots.append({
                    "id": sid,
                    "port": probe_port,
                    "role": role,
                    "model_id": slot.get("model_id", "unknown"),
                    "model_path": "",
                    "enabled": slot.get("enabled", True),
                    "running": running,
                    "status": "running" if running else "stopped",
                    "host": f"{host_only}:{probe_port}",
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
