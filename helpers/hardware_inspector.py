"""
Hardware Inspector — cross-OS detection + EIM + Tier mapping.

Implements the local-llm-recommender skill (snapshot 11-05-2026) inside the
A0 plugin. The A0 container has no GPU passthrough and sees only docker-
allocated RAM, so this module first asks the host helper at
host.docker.internal:55501/hardware-scan for the real machine specs.
Falls back to local detection only when the host helper is unreachable
(e.g. when running on the host directly).

Pipeline:
    scan_hardware() -> compute_eim_gb() -> map_tier() -> HardwareReport

The HardwareReport is consumed by helpers/tier_catalog.py to choose three
recommendations (COMFORTABLE / BALANCED / STRETCH).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Host helper connection (mirrors compute_monitor.py — single source of truth
# would be nice eventually, but duplicating four constants is cheaper than a
# circular import).
_HOST_TOKEN_ENV = "A0_LMM_HOST_TOKEN"
_HOST_URL_ENV = "A0_LMM_HOST_URL"
_HOST_HOST_ENV = "A0_LMM_HOST_HOST"
_HOST_PORT_ENV = "A0_LMM_HOST_PORT"
_HOST_DEFAULT_PORT = 55501
_HOST_TOKEN_CANDIDATES = ("/a0/tmp/lmm_host_token", "/host/a0_lmm_host.key")


# ---------------------------------------------------------------------------
# Tier boundaries (per skill, EIM in GB)
# ---------------------------------------------------------------------------
# (lower_bound_inclusive_gb, tier_name) — sorted ascending
TIER_BOUNDARIES = [
    (0.0,   "T0"),   # < 6 GB
    (6.0,   "T1"),   # 6-9
    (10.0,  "T2"),   # 10-14
    (16.0,  "T3"),   # 16-20
    (22.0,  "T4"),   # 22-28
    (30.0,  "T5"),   # 30-40
    (45.0,  "T6"),   # 45-65
    (70.0,  "T7"),   # 70-95
    (100.0, "T8"),   # 100-150
    (200.0, "T9"),   # 200+
]


@dataclass
class HardwareReport:
    ok: bool
    source: str                       # "host_helper" | "local" | "error"
    snapshot: str                     # AA Index snapshot date
    os_name: str                      # Windows / Linux / Darwin
    os_version: str
    cpu_name: str
    gpus: List[Dict[str, Any]] = field(default_factory=list)
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_path: str = ""
    eim_gb: float = 0.0
    eim_basis: str = ""               # "vram" | "unified_memory" | "system_ram"
    tier: str = "T0"
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Host helper bridge
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


def _fetch_hardware_via_host(timeout: float = 6.0) -> Optional[Dict[str, Any]]:
    """Call the host helper /hardware-scan endpoint. Returns None on failure."""
    token = _resolve_host_token()
    if not token:
        logger.debug("Host helper token not found; skipping host-side scan")
        return None
    url = f"{_resolve_host_url()}/hardware-scan"
    try:
        req = urllib.request.Request(
            url, method="POST", data=b"{}",
            headers={"Content-Type": "application/json", "X-Token": token},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        logger.debug("Host helper /hardware-scan unreachable: %s", exc)
        return None
    except Exception as exc:
        logger.debug("Host helper /hardware-scan error: %s", exc)
        return None
    try:
        payload = json.loads(body)
    except Exception:
        logger.debug("Host helper /hardware-scan returned non-JSON: %r", body[:200])
        return None
    if not payload.get("ok"):
        logger.debug("Host helper /hardware-scan error: %s", payload.get("error"))
        return None
    return payload


# ---------------------------------------------------------------------------
# Local-fallback detection (used only when host helper unreachable, e.g. when
# the inspector runs on the host directly)
# ---------------------------------------------------------------------------

def _local_scan() -> Dict[str, Any]:
    cpu_name = platform.processor() or "Unknown CPU"
    ram_gb = 0.0
    gpus: List[Dict[str, Any]] = []
    sys_name = platform.system()

    # CPU
    try:
        if sys_name == "Linux":
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        elif sys_name == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0:
                cpu_name = result.stdout.strip() or cpu_name
    except Exception:
        pass

    # RAM
    try:
        import psutil  # type: ignore
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        try:
            if sys_name == "Linux":
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            ram_gb = round(int(line.split()[1]) / (1024 * 1024), 1)
                            break
        except Exception:
            pass

    # GPU — nvidia-smi only on the fallback path; richer detection lives on
    # the host helper side.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[1].isdigit():
                    gpus.append({
                        "name": parts[0],
                        "total_vram_mb": int(parts[1]),
                        "vendor": "NVIDIA",
                        "discrete": True,
                    })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass

    # Disk
    disk_info = {"path": "", "free_gb": 0.0, "total_gb": 0.0}
    try:
        home = Path.home()
        usage = shutil.disk_usage(home)
        disk_info = {
            "path": str(home),
            "free_gb": round(usage.free / (1024 ** 3), 1),
            "total_gb": round(usage.total / (1024 ** 3), 1),
        }
    except Exception:
        pass

    return {
        "ok": True,
        "snapshot": "11-05-2026",
        "os": {
            "system": sys_name,
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {"name": cpu_name},
        "gpus": gpus,
        "ram_gb": ram_gb,
        "disk": disk_info,
    }


# ---------------------------------------------------------------------------
# EIM computation
# ---------------------------------------------------------------------------

def compute_eim_gb(hw: Dict[str, Any]) -> tuple[float, str]:
    """
    Effective Inference Memory — the pool model weights can live in.

    Rules per skill:
        * Discrete NVIDIA / AMD GPU  -> EIM = dedicated VRAM
        * Apple Silicon              -> EIM = unified RAM * 0.75
        * iGPU only / CPU only       -> EIM = system RAM * 0.5
    """
    gpus = hw.get("gpus", []) or []
    ram_gb = float(hw.get("ram_gb", 0) or 0)
    os_system = (hw.get("os", {}) or {}).get("system", "")

    # Apple Silicon — unified memory pool
    if os_system == "Darwin" or any(g.get("unified_memory") for g in gpus):
        # Default cap 0.75 — user can raise via iogpu.wired_limit_mb
        return (round(ram_gb * 0.75, 1), "unified_memory")

    # Discrete GPU — pick the best discrete VRAM
    best_vram_gb = 0.0
    for g in gpus:
        if not g.get("discrete"):
            continue
        vram_mb = int(g.get("total_vram_mb", 0) or 0)
        if vram_mb <= 0:
            continue
        # Windows CIM AdapterRAM is 32-bit-bugged for cards ≥4 GB. Trust the
        # value only if it's clearly ≥4 GB; below that we assume the overflow
        # clipped it and skip it.
        if g.get("vram_unreliable") and vram_mb < 4096:
            continue
        best_vram_gb = max(best_vram_gb, vram_mb / 1024.0)

    if best_vram_gb >= 1.0:
        return (round(best_vram_gb, 1), "vram")

    # iGPU / CPU-only
    return (round(ram_gb * 0.5, 1), "system_ram")


# ---------------------------------------------------------------------------
# Tier mapping
# ---------------------------------------------------------------------------

def map_tier(eim_gb: float) -> str:
    """Map EIM (GB) to tier name T0-T9 per skill boundaries."""
    last_tier = "T0"
    for threshold, tier in TIER_BOUNDARIES:
        if eim_gb >= threshold:
            last_tier = tier
        else:
            break
    return last_tier


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def scan_hardware(prefer_host: bool = True) -> HardwareReport:
    """
    Run a full hardware scan. Returns a HardwareReport ready for the
    tier_catalog. Never raises — returns ok=False on detection failure.
    """
    raw: Optional[Dict[str, Any]] = None
    source = "local"

    if prefer_host:
        raw = _fetch_hardware_via_host()
        if raw is not None:
            source = "host_helper"

    if raw is None:
        try:
            raw = _local_scan()
        except Exception as exc:
            logger.warning("Local hardware scan failed: %s", exc)
            return HardwareReport(
                ok=False, source="error",
                snapshot="11-05-2026",
                os_name=platform.system(),
                os_version=platform.release(),
                cpu_name="Unknown",
                error=str(exc),
            )

    os_info = raw.get("os", {}) or {}
    disk_info = raw.get("disk", {}) or {}
    eim_gb, eim_basis = compute_eim_gb(raw)
    tier = map_tier(eim_gb)

    notes: List[str] = []

    # Windows VRAM caveat
    if any(g.get("vram_unreliable") for g in raw.get("gpus", []) or []):
        notes.append(
            "Windows CIM AdapterRAM is 32-bit and clips VRAM ≥4 GB. Install "
            "nvidia-smi or rocm-smi on the host for accurate VRAM."
        )

    # Apple Silicon — mention the cap
    if eim_basis == "unified_memory":
        notes.append(
            "Apple Silicon unified memory: macOS caps GPU allocation at ~75% "
            "of RAM. Raise it with: sudo sysctl iogpu.wired_limit_mb=<MB>"
        )

    # CPU-class machines
    if eim_basis == "system_ram":
        notes.append(
            "No discrete GPU detected — running on CPU. Expect ~2-10 tok/s "
            "on a 7B-class model."
        )

    # Disk freshness
    disk_free_gb = float(disk_info.get("free_gb", 0) or 0)
    if disk_free_gb < 30:
        notes.append(
            f"Free disk is low ({disk_free_gb} GB). Most picks need ≥2× the "
            f"model file size during download — clear space first."
        )

    return HardwareReport(
        ok=True,
        source=source,
        snapshot=raw.get("snapshot", "11-05-2026"),
        os_name=os_info.get("system", platform.system()),
        os_version=os_info.get("release", platform.release()),
        cpu_name=(raw.get("cpu", {}) or {}).get("name", "Unknown CPU"),
        gpus=raw.get("gpus", []) or [],
        ram_gb=float(raw.get("ram_gb", 0) or 0),
        disk_free_gb=disk_free_gb,
        disk_path=disk_info.get("path", ""),
        eim_gb=eim_gb,
        eim_basis=eim_basis,
        tier=tier,
        notes=notes,
    )


def report_to_dict(report: HardwareReport) -> Dict[str, Any]:
    """asdict but tolerant of None error field."""
    return asdict(report)


# ---------------------------------------------------------------------------
# Live derivation — used by the dashboard header chip and slot recommender.
# Reads the existing compute_monitor snapshot instead of re-scanning. No
# host-helper round-trip needed; compute_monitor already handles that.
# ---------------------------------------------------------------------------

def derive_tier_from_stats(compute_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive {eim_gb, eim_basis, tier} from a live compute_monitor snapshot.

    compute_snapshot is the dict returned by
    `helpers.compute_monitor.get_compute_snapshot()`:
        {
            "ts": float,
            "gpus": [{id, name, total_vram_mb, used_vram_mb, free_vram_mb,
                      utilization_pct, temperature_c}, ...],
            "cpu":  {load_pct, ram_total_mb, ram_used_mb, ram_free_mb},
            "slots": [...],
        }

    Returns a dict with eim_gb / eim_basis / tier, plus a flattened
    "gpus" list and "ram_gb" for convenience. Never raises — returns
    sensible defaults if the snapshot is missing fields.
    """
    gpus_raw = compute_snapshot.get("gpus", []) or []
    cpu = compute_snapshot.get("cpu", {}) or {}
    ram_gb = round(int(cpu.get("ram_total_mb", 0) or 0) / 1024.0, 1)

    # Translate compute_monitor's GPU records into hardware_inspector's
    # canonical shape (vendor + discrete flag).
    gpus: List[Dict[str, Any]] = []
    for g in gpus_raw:
        name = str(g.get("name", "")) or "Unknown GPU"
        upper = name.upper()
        if "NVIDIA" in upper or "GEFORCE" in upper or "RTX" in upper or "TESLA" in upper:
            vendor = "NVIDIA"
            discrete = True
        elif "AMD" in upper or "RADEON" in upper:
            vendor = "AMD"
            discrete = True
        elif "APPLE" in upper or "M1" in upper or "M2" in upper or "M3" in upper or "M4" in upper:
            vendor = "Apple"
            discrete = False
        elif "INTEL" in upper:
            vendor = "Intel"
            discrete = False
        else:
            vendor = "unknown"
            discrete = False
        gpus.append({
            "name": name,
            "total_vram_mb": int(g.get("total_vram_mb", 0) or 0),
            "vendor": vendor,
            "discrete": discrete,
        })

    eim_gb, eim_basis = compute_eim_gb({
        "gpus": gpus, "ram_gb": ram_gb, "os": {"system": platform.system()},
    })
    tier = map_tier(eim_gb)

    return {
        "gpus": gpus,
        "ram_gb": ram_gb,
        "eim_gb": eim_gb,
        "eim_basis": eim_basis,
        "tier": tier,
    }
