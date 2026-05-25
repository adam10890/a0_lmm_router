"""
Compute Monitor — real-time GPU/CPU stats + LMM slot aggregation.

Wraps nvidia-smi (or fallback) and merges with BackendManager slot data
so the dashboard can display a single unified view.
"""

import json
import os
import subprocess
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import yaml

logger = logging.getLogger(__name__)

# Host helper fallback config (used when nvidia-smi is not available inside
# this process, e.g. when running inside the A0 Docker container which has
# no GPU passthrough). Mirrors the discovery logic in
# api/lmm_host_ignite.py so operators configure it in one place.
_HOST_TOKEN_ENV = "A0_LMM_HOST_TOKEN"
_HOST_URL_ENV = "A0_LMM_HOST_URL"
_HOST_HOST_ENV = "A0_LMM_HOST_HOST"
_HOST_PORT_ENV = "A0_LMM_HOST_PORT"
_HOST_DEFAULT_PORT = 55501
_HOST_TOKEN_CANDIDATES = ("/host/a0_lmm_host.key", "/a0/tmp/lmm_host_token")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUStats:
    id: int
    name: str
    total_vram_mb: int
    used_vram_mb: int
    free_vram_mb: int
    utilization_pct: int
    temperature_c: int

@dataclass
class CPUStats:
    load_pct: float          # 0-100
    ram_total_mb: int
    ram_used_mb: int
    ram_free_mb: int

@dataclass
class SlotInfo:
    id: str
    role: str
    model_id: str
    port: Optional[int]
    running: bool
    healthy: bool
    router_mode: bool = False
    router_models_dir: str = ""
    router_models_preset: str = ""
    router_models_max: int = 1
    router_models_autoload: bool = True

@dataclass
class ComputeSnapshot:
    ts: float                 # epoch seconds
    gpus: List[GPUStats]
    cpu: CPUStats
    slots: List[SlotInfo]


# ---------------------------------------------------------------------------
# GPU helpers (nvidia-smi)
# ---------------------------------------------------------------------------

def _resolve_host_token() -> str:
    tok = os.environ.get(_HOST_TOKEN_ENV, "").strip()
    if tok:
        return tok
    for path in _HOST_TOKEN_CANDIDATES:
        try:
            p = Path(path)
            if p.is_file():
                return p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
    return ""


def _resolve_host_url() -> str:
    url = os.environ.get(_HOST_URL_ENV, "").strip()
    if url:
        return url.rstrip("/")
    host = os.environ.get(_HOST_HOST_ENV, "host.docker.internal").strip()
    port = os.environ.get(_HOST_PORT_ENV, str(_HOST_DEFAULT_PORT)).strip()
    return f"http://{host}:{port}"


def _query_gpus_local() -> List[GPUStats]:
    """Try `nvidia-smi` in the current process. Empty list if unavailable."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            logger.debug("nvidia-smi returned non-zero: %s", result.stderr.strip())
            return []
        gpus: List[GPUStats] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append(GPUStats(
                    id=int(parts[0]),
                    name=parts[1],
                    total_vram_mb=int(parts[2]),
                    used_vram_mb=int(parts[3]),
                    free_vram_mb=int(parts[4]),
                    utilization_pct=int(parts[5]),
                    temperature_c=int(parts[6]),
                ))
        return gpus
    except FileNotFoundError:
        logger.debug("nvidia-smi not found locally")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
        return []
    except Exception as exc:
        logger.warning("Local GPU query failed: %s", exc)
        return []


def _query_gpus_via_host(timeout: float = 3.0) -> List[GPUStats]:
    """Fallback: ask the host helper for GPU stats over HTTP.

    Used when the current process (typically the A0 Docker container) has
    no GPU passthrough so `nvidia-smi` is absent. Silently returns [] if
    the helper is unreachable or the token is missing — the caller treats
    an empty list the same as "GPU unavailable".
    """
    token = _resolve_host_token()
    if not token:
        logger.debug("Host helper token not found; skipping GPU fallback")
        return []

    url = f"{_resolve_host_url()}/gpu-stats"
    try:
        req = urllib.request.Request(
            url,
            method="POST",
            data=b"{}",
            headers={"Content-Type": "application/json", "X-Token": token},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        logger.debug("Host helper GPU fallback unreachable: %s", exc)
        return []
    except Exception as exc:
        logger.debug("Host helper GPU fallback error: %s", exc)
        return []

    try:
        payload = json.loads(body)
    except Exception:
        logger.debug("Host helper returned non-JSON: %r", body[:200])
        return []

    if not payload.get("ok"):
        logger.debug("Host helper GPU error: %s", payload.get("error"))
        return []

    gpus: List[GPUStats] = []
    for g in payload.get("gpus", []) or []:
        try:
            gpus.append(GPUStats(
                id=int(g["id"]),
                name=str(g["name"]),
                total_vram_mb=int(g["total_vram_mb"]),
                used_vram_mb=int(g["used_vram_mb"]),
                free_vram_mb=int(g["free_vram_mb"]),
                utilization_pct=int(g["utilization_pct"]),
                temperature_c=int(g["temperature_c"]),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return gpus


def _query_gpus() -> List[GPUStats]:
    """Query GPU stats: try local nvidia-smi, then fall back to host helper."""
    gpus = _query_gpus_local()
    if gpus:
        return gpus
    return _query_gpus_via_host()


# ---------------------------------------------------------------------------
# CPU / RAM helpers
# ---------------------------------------------------------------------------

def _query_cpu() -> CPUStats:
    """Return basic CPU / RAM stats using cross-platform approach."""
    load_pct = 0.0
    ram_total = 0
    ram_used = 0
    ram_free = 0
    try:
        import psutil  # available in A0 Docker runtime
        load_pct = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        ram_total = int(mem.total / (1024 * 1024))
        ram_used = int(mem.used / (1024 * 1024))
        ram_free = int(mem.available / (1024 * 1024))
    except ImportError:
        # Fallback: try /proc/meminfo (Linux)
        try:
            with open("/proc/meminfo") as f:
                info: Dict[str, int] = {}
                for line in f:
                    k, v = line.split(":")[:2]
                    info[k.strip()] = int(v.strip().split()[0])  # kB
                ram_total = info.get("MemTotal", 0) // 1024
                ram_free = info.get("MemAvailable", info.get("MemFree", 0)) // 1024
                ram_used = ram_total - ram_free
        except Exception:
            pass
    except Exception:
        pass
    return CPUStats(load_pct=load_pct, ram_total_mb=ram_total, ram_used_mb=ram_used, ram_free_mb=ram_free)


# ---------------------------------------------------------------------------
# LMM slot aggregation
# ---------------------------------------------------------------------------

def _probe_slot(host: str, port: int, timeout: float = 1.5) -> tuple[bool, bool]:
    """Return (running, healthy). `running` = port open; `healthy` = /health 2xx."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError:
        return (False, False)

    # Port is open; probe /health for a proper healthy signal.
    try:
        import urllib.request
        req = urllib.request.Request(f"http://{host}:{port}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return (True, 200 <= resp.status < 300)
    except Exception:
        # Port open but HTTP not ready (model still loading) -> running=True, healthy=False
        return (True, False)


def _query_slots() -> List[SlotInfo]:
    slots: List[SlotInfo] = []
    try:
        conf = os.environ.get("A0_LMM_ROUTER_CONFIG", "").strip()
        if not conf:
            here = Path(__file__).resolve()
            plugin_conf = str(here.parent.parent / "conf" / "llama_cpp_servers.yaml")
            root_conf = str(here.parents[4] / "conf" / "llama_cpp_servers.yaml")
            conf = plugin_conf if os.path.exists(plugin_conf) else root_conf

        with open(conf, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        global_config = data.get("global", {}) or {}
        lmm_hosts = global_config.get("lmm_hosts", {}) or {}

        for slot in data.get("active_slots", []) or []:
            if not slot or not slot.get("enabled", True):
                continue

            sid = slot.get("id", f"slot_{slot.get('port', 'unknown')}")
            role = str(slot.get("role", ""))
            port = int(slot.get("port", 0) or 0)
            host_cfg = str(lmm_hosts.get(role, f"host.docker.internal:{port}"))
            host_cfg = host_cfg.replace("http://", "").replace("https://", "").split("/", 1)[0]
            host_only = host_cfg
            probe_port = port
            if ":" in host_cfg:
                host_only, port_text = host_cfg.rsplit(":", 1)
                if port_text.isdigit():
                    probe_port = int(port_text)
            running, healthy = _probe_slot(host_only, probe_port)

            # Try to get the REAL model_id from the container's /v1/models endpoint.
            # Falls back to config value if the container is unreachable.
            config_model_id = str(slot.get("model_id") or slot.get("specialty") or "")
            model_id = _query_container_model_id(host_only, probe_port, config_model_id)

            slots.append(SlotInfo(
                id=sid,
                role=role,
                model_id=model_id,
                port=probe_port,
                running=running,
                healthy=healthy,
                router_mode=bool(slot.get("router_mode", False)),
                router_models_dir=str(slot.get("router_models_dir", "") or ""),
                router_models_preset=str(slot.get("router_models_preset", "") or ""),
                router_models_max=int(slot.get("router_models_max", 1) or 1),
                router_models_autoload=bool(slot.get("router_models_autoload", True)),
            ))
    except Exception as exc:
        logger.debug("Slot query skipped: %s", exc)
    return slots


def _query_container_model_id(host: str, port: int, fallback: str) -> str:
    """Query the llama.cpp container's /v1/models to get the real model ID.

    Returns the model filename stem from the API, or fallback if unreachable.
    """
    if not port:
        return fallback
    try:
        import urllib.request
        url = f"http://{host}:{port}/v1/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2.0) as resp:  # noqa: S310
            if 200 <= resp.status < 300:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("data", []) if isinstance(data, dict) else []
                if models:
                    first = models[0]
                    model_id = first.get("id", "")
                    if model_id:
                        # Strip path prefix and .gguf suffix for clean display
                        model_id = model_id.replace(".gguf", "")
                        if "/" in model_id:
                            model_id = model_id.rsplit("/", 1)[-1]
                        return model_id
    except Exception:
        pass
    return fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_compute_snapshot() -> Dict[str, Any]:
    """Return a serialisable dict with current compute + LMM stats."""
    snap = ComputeSnapshot(
        ts=time.time(),
        gpus=_query_gpus(),
        cpu=_query_cpu(),
        slots=_query_slots(),
    )
    return {
        "ts": snap.ts,
        "gpus": [asdict(g) for g in snap.gpus],
        "cpu": asdict(snap.cpu),
        "slots": [asdict(s) for s in snap.slots],
    }
