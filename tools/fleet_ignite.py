"""Agent tool: ignite / extinguish / status the llama.cpp fleet.

Usage from the agent:
    {
      "thoughts": ["I need to bring the local models up"],
      "tool_name": "fleet_ignite",
      "tool_args": { "action": "ignite" }      # or "extinguish" / "status" / "run-bat"
    }

Under the hood this calls the plugin's `lmm_host_ignite` API which in
turn talks to the host helper (tools/lmm_host_helper.py). The helper
runs `docker compose -f docker-compose.lmm.yml up -d` on the Windows
host — something the container itself cannot do.

When the host helper isn't running the tool returns a clear message
telling the operator to start `start_agent_zero.bat`.
"""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import aiohttp

from helpers.tool import Tool, Response


DEFAULT_PORT = 55501
TOKEN_CANDIDATES = ("/a0/tmp/lmm_host_token", "/host/a0_lmm_host.key")
SUPPORTED_ACTIONS = ("ignite", "extinguish", "status", "run-bat", "health")


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


class FleetIgnite(Tool):
    """Ignite / extinguish / status-check the llama.cpp fleet via the host helper."""

    async def execute(self, **kwargs) -> Response:
        action = str(kwargs.get("action", "ignite")).strip().lower()
        if action not in SUPPORTED_ACTIONS:
            return Response(
                message=(
                    f"Unknown action {action!r}. "
                    f"Supported: {', '.join(SUPPORTED_ACTIONS)}."
                ),
                break_loop=False,
            )

        token = _resolve_token()
        if not token:
            return Response(
                message=(
                    "Host helper token missing. Start "
                    "`python tools/lmm_host_helper.py` on the Windows host (or run "
                    "start_agent_zero.bat which starts it) and mount the token file "
                    "into /a0/tmp/lmm_host_token, or set A0_LMM_HOST_TOKEN."
                ),
                break_loop=False,
            )

        base = _resolve_host_url()
        try:
            host_part, port_part = base.replace("http://", "").replace("https://", "").split(":", 1)
            if not _tcp_alive(host_part, int(port_part)):
                return Response(
                    message=(
                        f"Host helper not reachable at {base}. "
                        "Run start_agent_zero.bat or start tools/lmm_host_helper.py manually."
                    ),
                    break_loop=False,
                )
        except Exception:
            pass

        url = f"{base}/{action}"
        try:
            timeout = aiohttp.ClientTimeout(total=180)
            headers = {"Content-Type": "application/json", "X-Token": token}
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.post(url, data=json.dumps({})) as resp:
                    text = await resp.text()
                    try:
                        payload = json.loads(text)
                    except Exception:
                        payload = {"raw": text[:3000]}
        except Exception as exc:
            return Response(
                message=f"Host helper call failed: {type(exc).__name__}: {exc}",
                break_loop=False,
            )

        ok = bool(payload.get("ok"))
        lines = [f"## Fleet `{action}` — {'OK' if ok else 'FAILED'}"]
        if "message" in payload:
            lines.append(str(payload["message"]))
        for key in ("stdout", "stderr"):
            content = str(payload.get(key, "") or "").strip()
            if content:
                lines.append(f"\n### {key}\n```\n{content[:2000]}\n```")
        if not ok and "error" in payload:
            lines.append(f"\n**error:** {payload['error']}")
        return Response(message="\n".join(lines), break_loop=False)
