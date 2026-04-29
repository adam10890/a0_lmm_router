"""
LMM Host Helper — lightweight HTTP bridge that runs on the Windows host
so the A0 container (which has no Docker CLI) can start/stop the llama.cpp
fleet and query GPU stats.

Endpoints (all POST except /health):
    POST /ignite       — docker compose -f docker-compose.lmm.yml up -d
    POST /extinguish   — docker compose -f docker-compose.lmm.yml down
    POST /status       — list running LMM containers + health
    POST /run-bat      — execute a whitelisted .bat file by name
    POST /gpu-stats    — nvidia-smi output as JSON
    GET  /health       — alive check

The helper writes a random token to $TEMP/a0_lmm_host.key on first run.
A0 reads it from /host/a0_lmm_host.key (bind-mounted in docker-compose.yml).

Usage:
    python lmm_host_helper.py --port 55501 --compose docker-compose.lmm.yml
"""

import argparse
import json
import os
import secrets
import subprocess
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_PORT = 55501
TOKEN_FILENAME = "a0_lmm_host.key"
COMPOSE_FILE = "docker-compose.lmm.yml"

# Whitelist of .bat files that /run-bat is allowed to execute (basename only)
BAT_WHITELIST = {
    "lmm_manager.bat",
    "start_agent_zero.bat",
    "stop_agent_zero.bat",
    "status_agent_zero.bat",
}

# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

def _get_token_path() -> Path:
    temp = os.environ.get("TEMP", os.environ.get("TMP", "/tmp"))
    return Path(temp) / TOKEN_FILENAME


def _ensure_token() -> str:
    """Return existing token or generate a new one and write it to disk."""
    p = _get_token_path()
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    tok = secrets.token_urlsafe(32)
    p.write_text(tok, encoding="utf-8")
    print(f"[INIT] Wrote host-helper token to {p}")
    return tok


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _query_gpu_stats() -> dict:
    """Run nvidia-smi and return parsed GPU stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                "utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip() or "nvidia-smi failed", "gpus": []}

        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append(
                    {
                        "id": int(parts[0]),
                        "name": parts[1],
                        "total_vram_mb": int(parts[2]),
                        "used_vram_mb": int(parts[3]),
                        "free_vram_mb": int(parts[4]),
                        "utilization_pct": int(parts[5]),
                        "temperature_c": int(parts[6]),
                    }
                )
        return {"ok": True, "gpus": gpus, "count": len(gpus)}
    except FileNotFoundError:
        return {"ok": False, "error": "nvidia-smi not found", "gpus": []}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "nvidia-smi timed out", "gpus": []}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "gpus": []}


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def _run_docker_compose(compose_path: str, *args: str) -> dict:
    """Run docker compose with given args and return result dict."""
    cmd = ["docker", "compose", "-f", compose_path, *args]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError:
        return {"ok": False, "error": "docker not found in PATH"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "docker compose timed out"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _container_status() -> dict:
    """Return status of the three LMM containers."""
    names = ["a0-llama-chat", "a0-llama-utility", "a0-llama-embed"]
    containers = {}
    for name in names:
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}\t{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            if lines:
                parts = lines[0].split("\t")
                containers[name] = {"running": True, "status": parts[1] if len(parts) > 1 else "unknown"}
            else:
                containers[name] = {"running": False, "status": "not found"}
        except Exception as exc:
            containers[name] = {"running": False, "status": str(exc)}
    return containers


# ---------------------------------------------------------------------------
# BAT runner
# ---------------------------------------------------------------------------

def _run_bat(project_dir: str, bat_name: str, *args: str) -> dict:
    """Execute a whitelisted .bat file by name."""
    if bat_name not in BAT_WHITELIST:
        return {"ok": False, "error": f"'{bat_name}' is not in the whitelist"}

    bat_path = Path(project_dir) / bat_name
    if not bat_path.is_file():
        return {"ok": False, "error": f"{bat_path} not found"}

    try:
        # Use cmd /c to run the bat file with args
        cmd = ["cmd", "/c", str(bat_path), *args]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        # Suppress default logging; we print our own
        pass

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # CORS — A0 container needs to call this
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Token")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _check_token(self) -> bool:
        expected = _ensure_token()
        header_tok = self.headers.get("X-Token", "").strip()
        return header_tok == expected

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Token")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(200, {"ok": True, "service": "lmm_host_helper"})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        body = self._read_body()

        # /health is allowed without token (public health check)
        if parsed.path == "/health":
            self._send_json(200, {"ok": True, "service": "lmm_host_helper"})
            return

        # All other endpoints require token
        if not self._check_token():
            self._send_json(403, {"ok": False, "error": "invalid or missing X-Token"})
            return

        compose = body.get("compose", self.server.compose_path)
        project_dir = body.get("project_dir", self.server.project_dir)

        if parsed.path == "/ignite":
            result = _run_docker_compose(compose, "up", "-d")
            result["action"] = "ignite"
            self._send_json(200 if result["ok"] else 500, result)

        elif parsed.path == "/extinguish":
            result = _run_docker_compose(compose, "--profile", "full", "down")
            result["action"] = "extinguish"
            self._send_json(200 if result["ok"] else 500, result)

        elif parsed.path == "/status":
            containers = _container_status()
            self._send_json(200, {"ok": True, "containers": containers})

        elif parsed.path == "/run-bat":
            bat_name = body.get("bat", "")
            bat_args = body.get("args", [])
            if isinstance(bat_args, str):
                bat_args = bat_args.split()
            result = _run_bat(project_dir, bat_name, *bat_args)
            self._send_json(200 if result["ok"] else 500, result)

        elif parsed.path == "/gpu-stats":
            stats = _query_gpu_stats()
            self._send_json(200 if stats["ok"] else 500, stats)

        else:
            self._send_json(404, {"ok": False, "error": f"unknown endpoint: {parsed.path}"})


class Server(HTTPServer):
    def __init__(self, address, handler, compose_path: str, project_dir: str):
        super().__init__(address, handler)
        self.compose_path = compose_path
        self.project_dir = project_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="LMM Host Helper")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Listen port")
    parser.add_argument("--compose", default=COMPOSE_FILE, help="Path to docker-compose.lmm.yml")
    parser.add_argument("--project-dir", default=os.getcwd(), help="Project directory for .bat resolution")
    args = parser.parse_args()

    # Resolve compose path relative to project-dir if needed
    compose_path = args.compose
    if not Path(compose_path).is_absolute():
        compose_path = str(Path(args.project_dir) / compose_path)

    # Ensure token exists before starting
    _ensure_token()

    server = Server(("", args.port), Handler, compose_path, args.project_dir)
    print(f"[READY] LMM Host Helper listening on port {args.port}")
    print(f"[READY] Compose file: {compose_path}")
    print(f"[READY] Project dir:  {args.project_dir}")
    print(f"[READY] Token file:   {_get_token_path()}")
    print("[READY] Endpoints: /ignite /extinguish /status /run-bat /gpu-stats /health")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[EXIT] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
