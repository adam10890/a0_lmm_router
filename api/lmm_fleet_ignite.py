"""API endpoint: /plugins/a0_lmm_router/lmm_fleet_ignite

Entry point invoked by the Dashboard "Ignite Fleet" button.

What it does:
  * From INSIDE the A0 container we cannot talk to the Docker daemon (no
    socket mounted and no docker CLI installed). So this handler does NOT
    spawn containers itself.
  * It runs the plugin's own `launcher.py start` in-process. For the
    configured `remote` backend that revalidates connectivity to each
    llama-server and reports which slots are reachable.
  * It returns a payload the UI uses to tell the user what to do next:
    either "all green" or "run start_agent_zero.bat to spin containers".

The actual container spin-up is the job of `docker-compose.lmm.yml`,
invoked by `start_agent_zero.bat` on the host.
"""
from __future__ import annotations

import asyncio
import os
import socket
import sys
import time
from pathlib import Path

from flask import Request

from helpers.api import ApiHandler
from helpers import files


PLUGIN_DIR = "/a0/usr/plugins/a0_lmm_router"
LAUNCHER_PATH = f"{PLUGIN_DIR}/launcher.py"
HOST_BAT_HINT = "start_agent_zero.bat (on the Windows host)"


def _resolve_conf_path() -> str:
    plugin_conf = files.get_abs_path("usr/plugins/a0_lmm_router/conf/llama_cpp_servers.yaml")
    root_conf = files.get_abs_path("conf/llama_cpp_servers.yaml")
    return plugin_conf if os.path.exists(plugin_conf) else root_conf


def _probe(host: str, port: int, timeout: float = 1.5) -> dict:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        return {"reachable": False, "error": str(exc)}

    try:
        import urllib.request
        t0 = time.monotonic()
        with urllib.request.urlopen(  # noqa: S310
            f"http://{host}:{port}/health", timeout=timeout
        ) as resp:
            return {
                "reachable": True,
                "healthy": 200 <= resp.status < 300,
                "http_status": resp.status,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }
    except Exception as exc:
        return {"reachable": True, "healthy": False, "error": f"{type(exc).__name__}: {exc}"}


async def _run_launcher(cmd: str) -> tuple[int, str, str]:
    """Run `launcher.py <cmd>` as a child process; capture output."""
    py = "/opt/venv-a0/bin/python"
    if not Path(py).exists():
        py = sys.executable
    proc = await asyncio.create_subprocess_exec(
        py,
        LAUNCHER_PATH,
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    return (
        proc.returncode if proc.returncode is not None else -1,
        stdout_b.decode(errors="replace"),
        stderr_b.decode(errors="replace"),
    )


class LmmFleetIgnite(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        import yaml

        try:
            conf_path = _resolve_conf_path()
            if not os.path.exists(conf_path):
                return {
                    "ok": False,
                    "error": "config missing",
                    "hint": f"expected {conf_path}",
                }

            with open(conf_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}

            global_cfg = cfg.get("global", {}) or {}
            backend = global_cfg.get("backend", "auto")
            lmm_hosts = global_cfg.get("lmm_hosts", {}) or {}
            active_slots = cfg.get("active_slots", []) or []

            # Probe every configured slot.
            probes = []
            for slot in active_slots:
                role = slot.get("role", "")
                port = int(slot.get("port", 0) or 0)
                host_cfg = lmm_hosts.get(role, "host.docker.internal")
                host_only = host_cfg.split(":")[0] if ":" in host_cfg else host_cfg
                probe = _probe(host_only, port) if port else {"reachable": False, "error": "no port"}
                probes.append({
                    "id": slot.get("id", f"{role}_{port}"),
                    "role": role,
                    "host": f"{host_only}:{port}",
                    **probe,
                })

            reachable = [p for p in probes if p.get("reachable")]
            healthy = [p for p in probes if p.get("healthy")]

            # Ask the launcher to emit its own status so operators can diff
            # against what the manager thinks.
            rc, stdout, stderr = await _run_launcher("status")

            result = {
                "ok": True,
                "backend": backend,
                "config_path": conf_path,
                "slot_count": len(probes),
                "reachable_count": len(reachable),
                "healthy_count": len(healthy),
                "slots": probes,
                "launcher": {
                    "returncode": rc,
                    "stdout": stdout[-4000:],
                    "stderr": stderr[-2000:],
                },
            }

            if len(healthy) == len(probes) and probes:
                result["state"] = "fleet_healthy"
                result["message"] = f"All {len(probes)} slots healthy."
            elif not probes:
                result["state"] = "no_slots"
                result["message"] = "No active_slots configured. Edit llama_cpp_servers.yaml."
            elif backend == "remote":
                # Remote backend can't spawn; operator needs to run compose.
                result["state"] = "needs_host_ignition"
                result["message"] = (
                    f"{len(healthy)}/{len(probes)} slots healthy. "
                    "Run the host-side ignition to bring the rest online."
                )
                result["host_command"] = HOST_BAT_HINT
                result["docker_compose_hint"] = (
                    "docker compose -f usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml up -d"
                )
            else:
                result["state"] = "partial"
                result["message"] = f"{len(healthy)}/{len(probes)} slots healthy."

            return result
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
