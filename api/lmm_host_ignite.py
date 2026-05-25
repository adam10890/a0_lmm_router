"""API endpoint: /api/plugins/a0_lmm_router/lmm_host_ignite

Proxies to the host-side helper (tools/lmm_host_helper.py) so the A0
container can actually run `docker compose` even without a docker socket.

Discovery order for the auth token:
    1. $A0_LMM_HOST_TOKEN environment variable
    2. /host/a0_lmm_host.key     (live host $TEMP bind mount)
    3. /a0/tmp/lmm_host_token    (copied fallback)
    4. empty (endpoint returns 503 with a clear setup message)

Host is resolved in this order:
    1. $A0_LMM_HOST_URL               full URL including scheme+port
    2. $A0_LMM_HOST_HOST:PORT         split host & port
    3. http://host.docker.internal:55501  (Docker Desktop default)
"""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import aiohttp
from flask import Request

from helpers.api import ApiHandler


DEFAULT_PORT = 55501
TOKEN_CANDIDATES = ("/host/a0_lmm_host.key", "/a0/tmp/lmm_host_token")
ACTIONS = {"ignite", "extinguish", "status", "run-bat", "health", "start_slot", "stop_slot"}

# Actions that map to /models/* endpoints on the host helper
_MODEL_ACTIONS = {"start_slot": "models/start", "stop_slot": "models/stop"}


def _resolve_token() -> str:
    tok = os.environ.get("A0_LMM_HOST_TOKEN", "").strip()
    if tok:
        return tok
    for path in TOKEN_CANDIDATES:
        try:
            p = Path(path)
            if p.is_file():
                return p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
    return ""


def _resolve_host_url() -> str:
    url = os.environ.get("A0_LMM_HOST_URL", "").strip()
    if url:
        return url.rstrip("/")
    host = os.environ.get("A0_LMM_HOST_HOST", "host.docker.internal").strip()
    port = os.environ.get("A0_LMM_HOST_PORT", str(DEFAULT_PORT)).strip()
    return f"http://{host}:{port}"


def _tcp_alive(host: str, port: int, timeout: float = 0.8) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class LmmHostIgnite(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        action = str(input.get("action", "ignite")).lower().strip()
        if action not in ACTIONS:
            return {
                "ok": False,
                "error": f"unknown action {action!r}",
                "supported": sorted(ACTIONS),
            }

        token = _resolve_token()
        if not token:
            return {
                "ok": False,
                "error": "host helper token missing",
                "hint": (
                    "Start tools/lmm_host_helper.py on the host and "
                    "either set A0_LMM_HOST_TOKEN or mount the token file "
                    "into /a0/tmp/lmm_host_token."
                ),
            }

        base = _resolve_host_url()
        # Route model actions to /models/* endpoints
        if action in _MODEL_ACTIONS:
            url = f"{base}/{_MODEL_ACTIONS[action]}"
        else:
            url = f"{base}/{action}"

        # Cheap preflight: fail fast with a friendly message when the helper
        # is not running at all.
        try:
            host, port = base.replace("http://", "").replace("https://", "").split(":", 1)
            if not _tcp_alive(host, int(port)):
                return {
                    "ok": False,
                    "error": "host helper not reachable",
                    "host_url": base,
                    "hint": "Run start_agent_zero.bat OR python tools/lmm_host_helper.py on the host.",
                }
        except Exception:
            pass

        try:
            timeout = aiohttp.ClientTimeout(total=180)
            headers = {
                "Content-Type": "application/json",
                "X-Token": token,
            }
            # Build request body — pass slot for start_slot/stop_slot
            req_body = {}
            if action in ("start_slot", "stop_slot"):
                req_body["slot"] = str(input.get("slot", "")).strip()
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.post(url, data=json.dumps(req_body)) as resp:
                    text = await resp.text()
                    try:
                        payload = json.loads(text)
                    except Exception:
                        payload = {"raw": text[:4000]}
                    return {
                        "ok": bool(payload.get("ok", 200 <= resp.status < 300)),
                        "http_status": resp.status,
                        "host_url": base,
                        "action": action,
                        **payload,
                    }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "host_url": base,
                "action": action,
            }
