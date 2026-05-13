"""
LMM Host Helper — lightweight HTTP bridge that runs on the Windows host
so the A0 container (which has no Docker CLI) can start/stop the llama.cpp
fleet and query GPU stats.

Endpoints (all POST except /health and /models/list):
    POST /ignite       — docker compose up -f usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml up -d
    POST /extinguish   — docker compose down
    POST /status       — list running LMM containers + health
    POST /run-bat      — execute a whitelisted .bat file by name
    POST /gpu-stats    — nvidia-smi output as JSON
    GET  /health       — alive check

    GET  /models/list              — list installed models from manifest
    POST /models/install           — download model from HuggingFace (returns job_id)
    GET  /models/jobs/{job_id}     — get download job status/progress
    POST /models/jobs/{job_id}/cancel — cancel download job
    POST /models/delete            — delete a model file
    POST /models/verify            — verify model sha256
    POST /models/assign            — assign model to slot (rewrite env + restart container)

    GET  /tokens/hf                — check HF token status
    POST /tokens/hf                — set HF token
    DELETE /tokens/hf              — clear HF token

The helper writes a random token to $TEMP/a0_lmm_host.key on first run.
A0 reads it from /host/a0_lmm_host.key (bind-mounted in docker-compose.yml).

Usage:
    python lmm_host_helper.py --port 55501 --compose usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml
"""

import argparse
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse, unquote

# Optional huggingface_hub for model downloads
try:
    from huggingface_hub import hf_hub_download, HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_PORT = 55501
TOKEN_FILENAME = "a0_lmm_host.key"
COMPOSE_FILE = "usr/plugins/a0_lmm_router/docker/docker-compose.lmm.yml"

# Whitelist of .bat files that /run-bat is allowed to execute (basename only)
BAT_WHITELIST = {
    "lmm_manager.bat",
    "start_agent_zero.bat",
    "stop_agent_zero.bat",
    "status_agent_zero.bat",
}

# ---------------------------------------------------------------------------
# Model management globals
# ---------------------------------------------------------------------------
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_job_counter = 0

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
# HF Token management
# ---------------------------------------------------------------------------

def _get_hf_token_path() -> Path:
    home = Path.home()
    config_dir = home / ".lmm_helper"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "hf_token"


def _read_hf_token() -> str:
    try:
        return _get_hf_token_path().read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _write_hf_token(token: str) -> None:
    p = _get_hf_token_path()
    p.write_text(token, encoding="utf-8")
    # Best effort: chmod 600 on Unix
    try:
        import stat
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass


def _delete_hf_token() -> None:
    try:
        _get_hf_token_path().unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model manifest (installed_models.yaml)
# ---------------------------------------------------------------------------

def _get_manifest_path(models_dir: str) -> Path:
    return Path(models_dir) / "installed_models.yaml"


def _load_manifest(models_dir: str) -> dict:
    """Load installed_models.yaml or return empty structure."""
    p = _get_manifest_path(models_dir)
    if not p.exists():
        return {"models": {}}
    try:
        import yaml
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {"models": {}}
    except Exception:
        return {"models": {}}


def _save_manifest(models_dir: str, manifest: dict) -> None:
    """Atomic write of manifest."""
    try:
        import yaml
        p = _get_manifest_path(models_dir)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(manifest, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        tmp.replace(p)
    except Exception as e:
        print(f"[WARN] Failed to save manifest: {e}")


def _compute_sha256(filepath: Path, max_mb: int = 500) -> str:
    """Compute SHA256 of file (first max_mb MB for large files)."""
    h = hashlib.sha256()
    try:
        max_bytes = max_mb * 1024 * 1024
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
                if f.tell() > max_bytes:
                    break
        return h.hexdigest()
    except Exception:
        return ""


def _scan_models_dir(models_dir: str) -> dict:
    """Scan models dir for GGUF files and build manifest entries."""
    models = {}
    base = Path(models_dir)
    if not base.exists():
        return models

    for path in base.rglob("*.gguf"):
        rel = path.relative_to(base).as_posix()
        parts = rel.split("/")
        # Infer role from folder name
        role_hint = "utility"
        if len(parts) >= 2:
            folder = parts[0].lower()
            if folder in ("chat", "utility", "embedding", "vision", "reasoning"):
                role_hint = folder

        size_gb = round(path.stat().st_size / (1024**3), 2)
        model_id = path.stem  # filename without .gguf

        models[model_id] = {
            "file": path.name,
            "path": str(Path(rel).parent) if len(parts) > 1 else "",
            "repo_id": "local",  # unknown origin
            "size_gb": size_gb,
            "role_hint": role_hint,
            "sha256": "",  # computed on verify
        }
    return models


def _ensure_manifest(models_dir: str) -> dict:
    """Load or bootstrap manifest from disk scan."""
    manifest = _load_manifest(models_dir)
    if not manifest.get("models"):
        manifest["models"] = _scan_models_dir(models_dir)
        _save_manifest(models_dir, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Env file rewriter for slot assignment
# ---------------------------------------------------------------------------

def _get_env_path(compose_path: str) -> Path:
    """Get the .env file path next to docker-compose.lmm.yml."""
    compose_p = Path(compose_path)
    return compose_p.with_suffix(".env") if compose_p.suffix == ".yml" else compose_p.parent / "docker-compose.lmm.env"


_SLOT_ENV_KEYS = {
    "chat": "CHAT_MODEL_PATH",
    "utility": "UTILITY_MODEL_PATH",
    "embedding": "EMBED_MODEL_PATH",
    "embed": "EMBED_MODEL_PATH",
    "vision": "VISION_MODEL_PATH",
    "reasoning": "REASONING_MODEL_PATH",
}


def _rewrite_env_slot_model(env_path: Path, slot: str, model_path: str) -> bool:
    """Rewrite the MODEL_PATH for a slot in the .env file."""
    key = _SLOT_ENV_KEYS.get(slot.lower())
    if not key:
        return False

    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    # Update or append the key
    new_lines = []
    found = False
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={model_path}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={model_path}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return True


def _get_current_model_path(env_path: Path, slot: str) -> str:
    """Get current MODEL_PATH for a slot from .env."""
    key = _SLOT_ENV_KEYS.get(slot.lower())
    if not key or not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line[len(key)+1:].strip()
    return ""


# ---------------------------------------------------------------------------
# Download job worker
# ---------------------------------------------------------------------------

def _download_worker(job_id: str, repo_id: str, filename: str, models_dir: str, token: str) -> None:
    """Background thread to download a model from HuggingFace."""
    try:
        if not HF_HUB_AVAILABLE:
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = "huggingface_hub not installed on host"
            return

        # Prepare local path
        # Use repo_id structure: repo_id/model_name.gguf
        safe_repo = repo_id.replace("/", "--")
        local_dir = Path(models_dir) / safe_repo
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / filename

        # Progress callback
        def on_progress(data: dict) -> None:
            downloaded = data.get("downloaded_bytes", 0)
            total = data.get("total_bytes", 0)
            pct = round((downloaded / total) * 100, 1) if total else 0
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["downloaded_bytes"] = downloaded
                    _jobs[job_id]["total_bytes"] = total
                    _jobs[job_id]["percent"] = pct

        # Download
        with _jobs_lock:
            _jobs[job_id]["status"] = "downloading"
            _jobs[job_id]["local_path"] = str(local_path)

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            token=token if token else None,
            resume_download=True,
            local_files_only=False,
        )

        # Update manifest
        manifest = _ensure_manifest(models_dir)
        model_id = Path(filename).stem
        size_gb = round(Path(downloaded_path).stat().st_size / (1024**3), 2)

        # Infer role from path or filename
        role_hint = "utility"
        fname_lower = filename.lower()
        if "chat" in fname_lower or "instruct" in fname_lower:
            role_hint = "chat"
        elif "embed" in fname_lower:
            role_hint = "embedding"
        elif "vision" in fname_lower:
            role_hint = "vision"

        rel_path = str(Path(downloaded_path).relative_to(Path(models_dir)).parent.as_posix()) if downloaded_path.startswith(str(models_dir)) else ""

        manifest["models"][model_id] = {
            "file": filename,
            "path": rel_path,
            "repo_id": repo_id,
            "size_gb": size_gb,
            "role_hint": role_hint,
            "sha256": "",
        }
        _save_manifest(models_dir, manifest)

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["model_id"] = model_id
            _jobs[job_id]["percent"] = 100.0

    except Exception as e:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(e)


def _start_download_job(repo_id: str, filename: str, models_dir: str) -> str:
    """Start a background download job and return job_id."""
    global _job_counter
    with _jobs_lock:
        _job_counter += 1
        job_id = f"dl_{_job_counter}_{int(time.time())}"
        _jobs[job_id] = {
            "status": "queued",
            "repo_id": repo_id,
            "filename": filename,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "percent": 0.0,
            "error": None,
            "model_id": None,
        }
        token = _read_hf_token()
        t = threading.Thread(target=_download_worker, args=(job_id, repo_id, filename, models_dir, token), daemon=True)
        t.start()
        return job_id


def _get_job_status(job_id: str) -> dict:
    with _jobs_lock:
        return dict(_jobs.get(job_id, {"status": "unknown"}))


def _cancel_job(job_id: str) -> bool:
    # Note: actual cancel would require cooperative threading; we just mark it
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "cancelled"
            return True
    return False


# ---------------------------------------------------------------------------
# Helpers for compose env operations
# ---------------------------------------------------------------------------

def _get_models_dir_from_env(env_path: Path) -> str:
    """Extract LLAMA_MODELS_DIR from env file, with fallback."""
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("LLAMA_MODELS_DIR="):
                return line.split("=", 1)[1].strip()
    # Fallback: derive from common patterns
    return "C:/Users/frant/A0-Data-Permanent/A0_v.adam/models"


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
# Hardware scan (for local-llm-recommender skill)
# ---------------------------------------------------------------------------
#
# Returns CPU / GPU / VRAM / RAM / free-disk so the in-container
# hardware_inspector can compute EIM (Effective Inference Memory) and map
# to a tier T0-T9. Runs cross-OS commands per the public skill spec
# (snapshot 11-05-2026). Containers cannot see real hardware; this endpoint
# must be called by the A0 plugin via the host helper.

def _detect_cpu_name() -> str:
    sys_name = platform.system()
    try:
        if sys_name == "Windows":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=8, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                # If multiple CPUs, take the first non-empty line
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line:
                        return line
        elif sys_name == "Linux":
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except OSError:
                pass
        elif sys_name == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def _detect_gpus_for_scan() -> list:
    """Detect GPUs cross-OS. Returns list of dicts with name/total_vram_mb/vendor."""
    # 1) NVIDIA path — works on all OSes when nvidia-smi is installed
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[1].isdigit():
                    gpus.append({
                        "name": parts[0],
                        "total_vram_mb": int(parts[1]),
                        "vendor": "NVIDIA",
                        "discrete": True,
                    })
            if gpus:
                return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass

    sys_name = platform.system()

    # 2) Linux AMD: rocm-smi
    if sys_name == "Linux":
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                gpus = []
                for k, v in (data.items() if isinstance(data, dict) else []):
                    if not isinstance(v, dict) or not k.lower().startswith("card"):
                        continue
                    name = v.get("Card series") or v.get("Card model") or "AMD GPU"
                    vram_raw = v.get("VRAM Total Memory (B)") or v.get("VRAM Total (B)") or "0"
                    try:
                        total_vram_mb = int(int(vram_raw) / (1024 * 1024))
                    except (TypeError, ValueError):
                        total_vram_mb = 0
                    gpus.append({
                        "name": name,
                        "total_vram_mb": total_vram_mb,
                        "vendor": "AMD",
                        "discrete": True,
                    })
                if gpus:
                    return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        except Exception:
            pass
        # Fallback: lspci (no VRAM info, just the device name)
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    lower = line.lower()
                    if "vga" in lower or "3d" in lower:
                        name = line.split(":", 2)[-1].strip()
                        return [{
                            "name": name, "total_vram_mb": 0,
                            "vendor": "unknown", "discrete": False,
                        }]
        except Exception:
            pass

    # 3) Windows non-NVIDIA: Win32_VideoController via PowerShell
    elif sys_name == "Windows":
        try:
            ps_cmd = (
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name, @{n='VRAM_MB';e={[math]::Round($_.AdapterRAM/1MB,0)}} | "
                "ConvertTo-Json -Compress"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=8, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                parsed = json.loads(result.stdout.strip())
                items = parsed if isinstance(parsed, list) else [parsed]
                gpus = []
                for item in items:
                    name = item.get("Name", "Unknown GPU")
                    try:
                        vram_mb = int(item.get("VRAM_MB", 0) or 0)
                    except (TypeError, ValueError):
                        vram_mb = 0
                    upper = name.upper()
                    if "NVIDIA" in upper:
                        vendor = "NVIDIA"
                    elif "AMD" in upper or "RADEON" in upper:
                        vendor = "AMD"
                    elif "INTEL" in upper:
                        vendor = "Intel"
                    else:
                        vendor = "unknown"
                    # AdapterRAM is a known 32-bit field that overflows on cards
                    # with >4 GB VRAM — flag the value as unreliable so the
                    # inspector can decide what to do with it.
                    gpus.append({
                        "name": name,
                        "total_vram_mb": vram_mb,
                        "vendor": vendor,
                        "discrete": vendor in ("NVIDIA", "AMD"),
                        "vram_unreliable": True,
                    })
                if gpus:
                    return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        except Exception:
            pass

    # 4) macOS: system_profiler (Apple Silicon = unified memory, no VRAM)
    elif sys_name == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                displays = data.get("SPDisplaysDataType", []) or []
                gpus = []
                for d in displays:
                    name = d.get("sppci_model") or d.get("_name") or "Apple GPU"
                    cores = d.get("sppci_cores") or d.get("spdisplays_ndrvs")
                    gpus.append({
                        "name": name,
                        "total_vram_mb": 0,
                        "vendor": "Apple",
                        "cores": cores,
                        "unified_memory": True,
                        "discrete": False,
                    })
                if gpus:
                    return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        except Exception:
            pass

    return []


def _detect_ram_gb() -> float:
    sys_name = platform.system()
    try:
        if sys_name == "Windows":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"],
                capture_output=True, text=True, timeout=8, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        elif sys_name == "Linux":
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            return round(kb / (1024 * 1024), 1)
            except OSError:
                pass
        elif sys_name == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return round(int(result.stdout.strip()) / (1024 ** 3), 1)
    except Exception:
        pass
    return 0.0


def _detect_home_disk() -> dict:
    """Return free/total GB on the user's home volume."""
    try:
        home = Path.home()
        usage = shutil.disk_usage(home)
        return {
            "path": str(home),
            "free_gb": round(usage.free / (1024 ** 3), 1),
            "total_gb": round(usage.total / (1024 ** 3), 1),
        }
    except Exception as exc:
        return {"path": "", "free_gb": 0.0, "total_gb": 0.0, "error": str(exc)}


def _query_hardware_scan() -> dict:
    """Aggregate detection for the /hardware-scan endpoint."""
    return {
        "ok": True,
        "snapshot": "11-05-2026",  # AA Intelligence Index freshness window
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {"name": _detect_cpu_name()},
        "gpus": _detect_gpus_for_scan(),
        "ram_gb": _detect_ram_gb(),
        "disk": _detect_home_disk(),
    }


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

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        body = self._read_body()

        if not self._check_token():
            self._send_json(403, {"ok": False, "error": "invalid or missing X-Token"})
            return

        # DELETE /tokens/hf
        if parsed.path == "/tokens/hf":
            _delete_hf_token()
            self._send_json(200, {"ok": True, "message": "HF token cleared"})
            return

        # CORS fallback for unsupported DELETE
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Token")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        # Public health check (no token)
        if parsed.path == "/health":
            self._send_json(200, {"ok": True, "service": "lmm_host_helper"})
            return

        # All other GET endpoints require token
        if not self._check_token():
            self._send_json(403, {"ok": False, "error": "invalid or missing X-Token"})
            return

        # GET /models/list
        if parsed.path == "/models/list":
            env_path = _get_env_path(self.server.compose_path)
            models_dir = _get_models_dir_from_env(env_path)
            manifest = _ensure_manifest(models_dir)
            self._send_json(200, {"ok": True, "models": manifest.get("models", {}), "models_dir": models_dir})
            return

        # GET /models/jobs/{job_id}
        match = re.match(r"^/models/jobs/(.+)$", parsed.path)
        if match:
            job_id = unquote(match.group(1))
            status = _get_job_status(job_id)
            self._send_json(200, {"ok": True, "job_id": job_id, **status})
            return

        # GET /tokens/hf
        if parsed.path == "/tokens/hf":
            token = _read_hf_token()
            self._send_json(200, {"ok": True, "has_token": bool(token), "token_prefix": token[:4] + "..." if token else None})
            return

        self._send_json(404, {"ok": False, "error": f"unknown endpoint: {parsed.path}"})

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

        elif parsed.path == "/hardware-scan":
            scan = _query_hardware_scan()
            self._send_json(200 if scan.get("ok") else 500, scan)

        # Model management endpoints
        elif parsed.path == "/models/install":
            repo_id = body.get("repo_id", "").strip()
            filename = body.get("filename", "").strip()
            role = body.get("role", "")
            if not repo_id or not filename:
                self._send_json(400, {"ok": False, "error": "repo_id and filename are required"})
                return
            env_path = _get_env_path(self.server.compose_path)
            models_dir = _get_models_dir_from_env(env_path)
            job_id = _start_download_job(repo_id, filename, models_dir)
            self._send_json(202, {"ok": True, "job_id": job_id, "status": "queued"})

        elif parsed.path == "/models/assign":
            slot = body.get("slot", "").strip()
            model_id = body.get("model_id", "").strip()
            apply_now = body.get("apply_now", True)
            if not slot or not model_id:
                self._send_json(400, {"ok": False, "error": "slot and model_id are required"})
                return

            env_path = _get_env_path(self.server.compose_path)
            models_dir = _get_models_dir_from_env(env_path)
            manifest = _ensure_manifest(models_dir)

            model = manifest.get("models", {}).get(model_id)
            if not model:
                self._send_json(404, {"ok": False, "error": f"model '{model_id}' not found in manifest"})
                return

            # Build model path: /models/{path}/{file}
            model_file = model.get("file", "")
            model_path = model.get("path", "")
            full_model_path = f"/models/{model_path}/{model_file}" if model_path else f"/models/{model_file}"

            # Rewrite env
            ok = _rewrite_env_slot_model(env_path, slot, full_model_path)
            if not ok:
                self._send_json(400, {"ok": False, "error": f"invalid slot '{slot}'"})
                return

            restarted = False
            if apply_now:
                # Restart only this slot's container
                service_map = {
                    "chat": "a0-llama-chat",
                    "utility": "a0-llama-utility",
                    "embedding": "a0-llama-embed",
                    "embed": "a0-llama-embed",
                    "vision": "a0-llama-vision",
                    "reasoning": "a0-llama-reasoning",
                }
                service = service_map.get(slot.lower())
                if service:
                    result = _run_docker_compose(self.server.compose_path, "up", "-d", "--force-recreate", service)
                    restarted = result.get("ok", False)

            self._send_json(200, {"ok": True, "slot": slot, "model_id": model_id, "restarted": restarted, "model_path": full_model_path})

        elif parsed.path == "/models/delete":
            model_id = body.get("model_id", "").strip()
            if not model_id:
                self._send_json(400, {"ok": False, "error": "model_id is required"})
                return

            env_path = _get_env_path(self.server.compose_path)
            models_dir = _get_models_dir_from_env(env_path)
            manifest = _ensure_manifest(models_dir)

            model = manifest.get("models", {}).get(model_id)
            if not model:
                self._send_json(404, {"ok": False, "error": f"model '{model_id}' not found"})
                return

            # Check if currently assigned to any slot
            model_path = model.get("path", "")
            model_file = model.get("file", "")
            full_path = f"/models/{model_path}/{model_file}" if model_path else f"/models/{model_file}"

            for slot_name in ["chat", "utility", "embedding", "vision", "reasoning"]:
                current = _get_current_model_path(env_path, slot_name)
                if current == full_path:
                    self._send_json(409, {"ok": False, "error": f"model is currently assigned to slot '{slot_name}'"})
                    return

            # Delete file
            try:
                file_path = Path(models_dir) / model_path / model_file if model_path else Path(models_dir) / model_file
                file_path.unlink(missing_ok=True)
                # Remove from manifest
                del manifest["models"][model_id]
                _save_manifest(models_dir, manifest)
                self._send_json(200, {"ok": True, "deleted": model_id})
            except Exception as e:
                self._send_json(500, {"ok": False, "error": str(e)})

        elif parsed.path == "/models/verify":
            model_id = body.get("model_id", "").strip()
            if not model_id:
                self._send_json(400, {"ok": False, "error": "model_id is required"})
                return

            env_path = _get_env_path(self.server.compose_path)
            models_dir = _get_models_dir_from_env(env_path)
            manifest = _ensure_manifest(models_dir)

            model = manifest.get("models", {}).get(model_id)
            if not model:
                self._send_json(404, {"ok": False, "error": f"model '{model_id}' not found"})
                return

            try:
                model_path = model.get("path", "")
                model_file = model.get("file", "")
                file_path = Path(models_dir) / model_path / model_file if model_path else Path(models_dir) / model_file
                sha256 = _compute_sha256(file_path)
                # Update manifest
                manifest["models"][model_id]["sha256"] = sha256
                _save_manifest(models_dir, manifest)
                self._send_json(200, {"ok": True, "model_id": model_id, "sha256": sha256})
            except Exception as e:
                self._send_json(500, {"ok": False, "error": str(e)})

        elif parsed.path == "/models/jobs/cancel":
            job_id = body.get("job_id", "").strip()
            if not job_id:
                self._send_json(400, {"ok": False, "error": "job_id is required"})
                return
            ok = _cancel_job(job_id)
            self._send_json(200 if ok else 404, {"ok": ok, "job_id": job_id})

        # Token management
        elif parsed.path == "/tokens/hf":
            token = body.get("token", "").strip()
            if token:
                _write_hf_token(token)
                self._send_json(200, {"ok": True, "message": "HF token set"})
            else:
                self._send_json(400, {"ok": False, "error": "token is required in body"})

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
    parser.add_argument("--compose", default=COMPOSE_FILE, help="Path to docker-compose.lmm.yml (inside plugin)")
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
    print("[READY] Endpoints: /ignite /extinguish /status /run-bat /gpu-stats /hardware-scan /health")
    print("[READY] Model endpoints: /models/list /models/install /models/assign /models/delete /models/verify /models/jobs/{id}")
    print("[READY] Token endpoints: /tokens/hf")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[EXIT] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
