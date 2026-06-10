"""API endpoint: /api/plugins/a0_lmm_router/mcp_control

Status, start, stop, and restart the LMM Router MCP server (Streamable HTTP)
that runs inside the Agent Zero container on mcp_server.port (default 8095).

Actions (input.action):
    status   — probe port + read config (default)
    start    — launch launcher.py mcp in background
    stop     — terminate MCP listener process(es)
    restart  — stop then start
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from flask import Request

from helpers.api import ApiHandler

_ACTIONS = frozenset({"status", "start", "stop", "restart"})
_LAUNCHER = "/a0/usr/plugins/a0_lmm_router/launcher.py"
_CONFIG_CANDIDATES = (
    os.environ.get("A0_LMM_ROUTER_CONFIG", ""),
    "/a0/conf/llama_cpp_servers.yaml",
    "/a0/usr/plugins/a0_lmm_router/conf/llama_cpp_servers.yaml",
)
_DEBUG_LOG = "/a0/usr/plugins/a0_lmm_router/data/debug-e401df.log"


def _dbg(hypothesis_id: str, location: str, message: str, data: dict | None = None) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "e401df",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion


def _find_config() -> str | None:
    for candidate in _CONFIG_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _load_mcp_cfg(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {"enabled": True, "host": "127.0.0.1", "port": 8095}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        mcp = data.get("mcp_server", {}) if isinstance(data, dict) else {}
        return {
            "enabled": bool(mcp.get("enabled", True)),
            "host": str(mcp.get("host", "127.0.0.1") or "127.0.0.1"),
            "port": int(mcp.get("port", 8095)),
        }
    except Exception:
        return {"enabled": True, "host": "127.0.0.1", "port": 8095}


def _port_open(host: str, port: int, timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _launcher_path() -> str:
    if Path(_LAUNCHER).is_file():
        return _LAUNCHER
    alt = Path(__file__).resolve().parent.parent / "launcher.py"
    return str(alt) if alt.is_file() else _LAUNCHER


def _start_mcp(config_path: str | None) -> dict[str, Any]:
    cfg = _load_mcp_cfg(config_path)
    if not cfg.get("enabled", True):
        return {"ok": False, "error": "MCP server disabled in llama_cpp_servers.yaml (mcp_server.enabled: false)"}

    host = str(cfg.get("host", "127.0.0.1"))
    port = int(cfg.get("port", 8095))
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host

    if _port_open(probe_host, port):
        return {"ok": True, "message": f"MCP already listening on {probe_host}:{port}", "already_running": True}

    launcher = _launcher_path()
    if not Path(launcher).is_file():
        return {"ok": False, "error": f"launcher not found at {launcher}"}

    log_path = "/tmp/mcp_server.log"
    try:
        log_fh = open(log_path, "a", encoding="utf-8")
    except OSError:
        log_fh = subprocess.DEVNULL  # type: ignore[assignment]

    subprocess.Popen(
        [sys.executable, launcher, "mcp"],
        stdout=log_fh,
        stderr=subprocess.STDOUT if log_fh != subprocess.DEVNULL else subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(Path(launcher).parent),
    )

    for _ in range(12):
        time.sleep(0.25)
        if _port_open(probe_host, port):
            return {
                "ok": True,
                "message": f"MCP started on {probe_host}:{port}",
                "log_path": log_path,
            }

    return {
        "ok": False,
        "error": f"MCP process launched but port {port} not listening after 3s",
        "log_path": log_path,
        "hint": f"Check {log_path} inside the agent-zero-2 container",
    }


def _stop_mcp() -> dict[str, Any]:
    patterns = ("launcher.py mcp", "mcp_server.server", "mcp_server/server.py")
    stopped = False
    for pat in patterns:
        rc = subprocess.run(
            ["pkill", "-f", pat],
            capture_output=True,
            text=True,
        )
        if rc.returncode == 0:
            stopped = True
    return {"ok": True, "message": "MCP stopped" if stopped else "MCP was not running", "stopped": stopped}


def _status(config_path: str | None) -> dict[str, Any]:
    cfg = _load_mcp_cfg(config_path)
    host = str(cfg.get("host", "127.0.0.1"))
    port = int(cfg.get("port", 8095))
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    running = _port_open(probe_host, port)
    endpoint = f"http://{host}:{port}/mcp" if host not in ("0.0.0.0", "::") else f"http://127.0.0.1:{port}/mcp"
    a0_client_url = f"http://host.docker.internal:{port}/mcp"
    return {
        "ok": True,
        "running": running,
        "enabled": bool(cfg.get("enabled", True)),
        "host": host,
        "port": port,
        "endpoint": endpoint,
        "a0_settings_url": a0_client_url,
        "log_path": "/tmp/mcp_server.log",
    }


class McpControl(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        action = str(input.get("action", "status")).lower().strip()
        if action not in _ACTIONS:
            return {"ok": False, "error": f"unknown action {action!r}", "supported": sorted(_ACTIONS)}

        config_path = _find_config()
        _dbg("MCP", "mcp_control.py:process", "mcp_control action", {"action": action, "config": config_path})

        try:
            if action == "status":
                return _status(config_path)
            if action == "start":
                result = _start_mcp(config_path)
                result.update(_status(config_path))
                return result
            if action == "stop":
                stop_result = _stop_mcp()
                status = _status(config_path)
                return {**status, **stop_result, "running": status.get("running", False)}
            if action == "restart":
                _stop_mcp()
                time.sleep(0.4)
                start_result = _start_mcp(config_path)
                status = _status(config_path)
                return {**status, **start_result}
        except Exception as exc:
            _dbg("MCP", "mcp_control.py:process", "mcp_control error", {"action": action, "error": str(exc)})
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        return {"ok": False, "error": "unhandled action"}
