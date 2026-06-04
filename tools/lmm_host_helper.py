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
import hmac
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

# Add helpers to path for context calculator
sys.path.insert(0, str(Path(__file__).parent.parent / "helpers"))
try:
    from context_calculator import calculate_optimal_context, read_gguf_metadata
    calculate_optimal_context = calculate_optimal_context  # type: ignore
    read_gguf_metadata = read_gguf_metadata  # type: ignore
except ImportError:
    calculate_optimal_context = None  # type: ignore
    read_gguf_metadata = None  # type: ignore

try:
    from model_params_cache import (
        plan_model_with_cache,
        record_role_success,
        render_fleet_preset_to_file,
        store_plan_in_cache,
        warm_fleet_params,
    )
    MODEL_PARAMS_CACHE_AVAILABLE = True
except ImportError:
    MODEL_PARAMS_CACHE_AVAILABLE = False
    plan_model_with_cache = None  # type: ignore
    record_role_success = None  # type: ignore
    render_fleet_preset_to_file = None  # type: ignore
    store_plan_in_cache = None  # type: ignore
    warm_fleet_params = None  # type: ignore

try:
    from fleet_mode import compose_target_mode, detect_fleet_mode, is_conflicting_mode
except ImportError:
    compose_target_mode = None  # type: ignore
    detect_fleet_mode = None  # type: ignore
    is_conflicting_mode = None  # type: ignore

# Optional huggingface_hub for model downloads
try:
    from huggingface_hub import hf_hub_download, HfApi
    from huggingface_hub.utils import tqdm as hf_hub_tqdm
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    hf_hub_tqdm = None  # type: ignore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_PORT = 55501
TOKEN_FILENAME = "a0_lmm_host.key"
COMPOSE_FILE = "usr/plugins/a0_lmm_router/docker/docker-compose.lmm.router.yml"
RATE_LIMIT_MAX = 30
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_hits: dict[str, list[float]] = {}
_rate_limit_lock = threading.Lock()

LOCAL_CORS_ORIGINS = {
    "http://127.0.0.1:5080",
    "http://localhost:5080",
}
MUTATING_ENDPOINTS = {
    "/ignite",
    "/extinguish",
    "/run-bat",
    "/models/install",
    "/models/assign",
    "/models/start",
    "/models/stop",
    "/models/delete",
    "/models/verify",
    "/models/jobs/cancel",
    "/tokens/hf",
    "/router/write_preset_ini",
    "/router/restart",
    "/router/warm_params",
    "/router/record_success",
}

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


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_bind_host(bind_arg: str) -> str:
    """Return the interface the helper should bind to."""
    bind = os.environ.get("A0_LMM_HOST_BIND", bind_arg or "").strip()
    if not bind:
        return "127.0.0.1"
    if bind in {"0.0.0.0", "::", ""} and not _truthy(os.environ.get("A0_LMM_HOST_BIND_PUBLIC")):
        return "127.0.0.1"
    return bind


def _token_matches(header_token: str, expected_token: str) -> bool:
    return hmac.compare_digest(header_token or "", expected_token or "")


def _allowed_cors_origin(origin: str | None) -> str:
    origin = (origin or "").strip()
    return origin if origin in LOCAL_CORS_ORIGINS else "null"


def _compose_allowlist(project_dir: str) -> set[str]:
    project = Path(project_dir).resolve()
    plugin_docker = project / "usr" / "plugins" / "a0_lmm_router" / "docker"
    allowed: set[str] = set()
    for pattern_root, pattern in [
        (plugin_docker, "docker-compose*.yml"),
        (plugin_docker, "docker-compose*.yaml"),
        (project, "docker-compose*.yml"),
        (project, "docker-compose*.yaml"),
    ]:
        if pattern_root.is_dir():
            allowed.update(str(p.resolve()) for p in pattern_root.glob(pattern) if p.is_file())
    return allowed


def _resolve_compose_path(requested: str | None, default_compose: str, project_dir: str) -> str:
    """Resolve compose path and reject files outside the local allowlist."""
    candidate = requested or default_compose
    path = Path(candidate)
    if not path.is_absolute():
        path = Path(project_dir) / path
    resolved = str(path.resolve())
    if resolved not in _compose_allowlist(project_dir):
        raise ValueError(f"compose path not allowed: {resolved}")
    return resolved


def _default_preset_host_path(project_dir: str) -> Path:
    env_path = os.environ.get("LLAMACPP_PRESET_HOST", "").strip()
    if env_path:
        return Path(env_path).resolve()
    return (Path(project_dir) / "usr" / "plugins" / "a0_lmm_router" / "conf" / "models_preset.ini").resolve()


def _resolve_preset_path(requested: str | None, project_dir: str) -> Path:
    allowed = _default_preset_host_path(project_dir)
    candidate = Path(requested).resolve() if requested else allowed
    if candidate != allowed:
        raise ValueError(f"preset path not allowed: {candidate}")
    return candidate


def _rewrite_preset_alias_model(content: str, alias: str, model_path: str) -> tuple[str, str]:
    """Replace only the model line in the requested alias section."""
    if not alias or not model_path:
        raise ValueError("alias and model_path are required")
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    in_section = False
    found_section = False
    replaced = False
    snippet: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_section:
                in_section = False
            section = stripped[1:-1].strip()
            if section == alias:
                in_section = True
                found_section = True
        if in_section and stripped.lower().startswith("model") and "=" in stripped:
            newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            line = f"model = {model_path}{newline}"
            replaced = True
        out.append(line)
        if in_section:
            snippet.append(line)

    if not found_section:
        raise ValueError(f"alias section not found: {alias}")
    if not replaced:
        raise ValueError(f"model line not found in alias section: {alias}")
    return "".join(out), "".join(snippet)


def _write_preset_ini_atomic(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    return backup


def _restart_router_container() -> dict:
    try:
        result = subprocess.run(
            ["docker", "restart", "a0-llama-router"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
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
        return {"ok": False, "error": "docker restart timed out"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _preflight_fleet_mode(compose_path: str) -> dict:
    if not (compose_target_mode and detect_fleet_mode and is_conflicting_mode):
        return {"ok": True, "mode": "unknown"}
    target_mode = compose_target_mode(compose_path)
    fleet_mode = detect_fleet_mode()
    active_mode = fleet_mode.get("mode", "idle")
    if is_conflicting_mode(active_mode, target_mode):
        return {
            "ok": False,
            "status": 409,
            "error": "conflicting llama.cpp fleet is already running",
            "target_mode": target_mode,
            "fleet_mode": fleet_mode,
        }
    return {"ok": True, "target_mode": target_mode, "fleet_mode": fleet_mode}


def _rate_limit_allow(client_ip: str, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    with _rate_limit_lock:
        hits = [t for t in _rate_limit_hits.get(client_ip, []) if t > window_start]
        if len(hits) >= RATE_LIMIT_MAX:
            _rate_limit_hits[client_ip] = hits
            return False
        hits.append(now)
        _rate_limit_hits[client_ip] = hits
        return True


def _audit_log_path() -> Path:
    base = Path(os.environ.get("A0_LMM_AUDIT_DIR", "tmp"))
    return base / "lmm_host_helper_audit.log"


def _audit_mutating_call(action: str, path: str, client_ip: str, token: str, status: str) -> None:
    try:
        p = _audit_log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        token_hash = hashlib.sha256((token or "").encode("utf-8")).hexdigest()[:12] if token else ""
        row = {
            "ts": time.time(),
            "action": action,
            "path": path,
            "ip": client_ip,
            "token_hash_prefix": token_hash,
            "status": status,
        }
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception:
        pass


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

    # Import benchmark fetcher
    try:
        import sys
        plugin_dir = Path(__file__).parent.parent
        if str(plugin_dir) not in sys.path:
            sys.path.insert(0, str(plugin_dir))
        from helpers.benchmark_fetcher import get_benchmark_data
        benchmark_available = True
    except Exception as e:
        print(f"[WARNING] Benchmark fetcher not available: {e}")
        benchmark_available = False

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

        # Read GGUF metadata if context_calculator is available
        gguf_metadata = {}
        if read_gguf_metadata:
            try:
                gguf_metadata = read_gguf_metadata(str(path))
            except Exception as e:
                print(f"[WARNING] Failed to read GGUF metadata for {path.name}: {e}")

        # Fetch benchmark data if available
        benchmark_data = {}
        if benchmark_available:
            try:
                # Try to match model name to benchmark data
                # Use filename as hint, also try repo_id if known
                benchmark_data = get_benchmark_data(model_id)
            except Exception as e:
                print(f"[WARNING] Failed to fetch benchmark data for {model_id}: {e}")

        models[model_id] = {
            "file": path.name,
            "path": str(Path(rel).parent) if len(parts) > 1 else "",
            "repo_id": "local",  # unknown origin
            "size_gb": size_gb,
            "role_hint": role_hint,
            "sha256": "",  # computed on verify
            # GGUF metadata for context calculation and UI display
            "n_ctx_train": gguf_metadata.get("n_ctx_train"),
            "n_layer": gguf_metadata.get("n_layer"),
            "n_embd": gguf_metadata.get("n_embd"),
            # Benchmark data
            "benchmark_score": benchmark_data.get("score", 0),
            "benchmark_sources": benchmark_data.get("sources", []),
            "benchmark_date": benchmark_data.get("date", ""),
            "benchmark_confidence": benchmark_data.get("confidence", "unknown"),
            "task_profiles": benchmark_data.get("task_profiles", []),
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


def _read_env_file_dict(env_path: Path) -> dict[str, str]:
    """Parse docker-compose.lmm.env into a flat dict (merged with os.environ)."""
    data = dict(os.environ)
    if not env_path.is_file():
        return data
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


_SLOT_ENV_KEYS = {
    "chat": "CHAT_MODEL_PATH",
    "utility": "UTILITY_MODEL_PATH",
    "embedding": "EMBED_MODEL_PATH",
    "embed": "EMBED_MODEL_PATH",
    "vision": "VISION_MODEL_PATH",
    "reasoning": "REASONING_MODEL_PATH",
}

_SLOT_CTX_KEYS = {
    "chat": "CHAT_CTX_SIZE",
    "utility": "UTILITY_CTX_SIZE",
    "embedding": "EMBED_CTX_SIZE",
    "embed": "EMBED_CTX_SIZE",
    "vision": "VISION_CTX_SIZE",
    "reasoning": "REASONING_CTX_SIZE",
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


def _rewrite_env_slot_ctx(env_path: Path, slot: str, ctx_size: int) -> bool:
    """Rewrite the CTX_SIZE for a slot in the .env file."""
    key = _SLOT_CTX_KEYS.get(slot.lower())
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
            new_lines.append(f"{key}={ctx_size}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={ctx_size}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Download job worker
# ---------------------------------------------------------------------------

def _make_download_tqdm_class(job_id: str):
    """Tqdm subclass that mirrors HF download bytes into the job registry."""
    base = hf_hub_tqdm

    class _DownloadJobTqdm(base):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._lmm_job_id = job_id

        def _sync_job_progress(self) -> None:
            total = int(self.total or 0)
            downloaded = int(self.n or 0)
            pct = round((downloaded / total) * 100, 1) if total else 0
            with _jobs_lock:
                if self._lmm_job_id in _jobs:
                    _jobs[self._lmm_job_id]["downloaded_bytes"] = downloaded
                    _jobs[self._lmm_job_id]["total_bytes"] = total
                    _jobs[self._lmm_job_id]["percent"] = pct

        def update(self, n=1):
            result = super().update(n)
            self._sync_job_progress()
            return result

        def close(self):
            try:
                self._sync_job_progress()
            finally:
                super().close()

    return _DownloadJobTqdm


def _fetch_remote_file_size(repo_id: str, filename: str, token: str) -> int:
    """Best-effort remote size for progress when tqdm total is missing."""
    if not HF_HUB_AVAILABLE:
        return 0
    try:
        api = HfApi(token=token if token else None)
        info = api.repo_file_info(repo_id=repo_id, filename=filename, repo_type="model")
        return int(getattr(info, "size", 0) or 0)
    except Exception:
        return 0


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

        remote_size = _fetch_remote_file_size(repo_id, filename, token)

        with _jobs_lock:
            _jobs[job_id]["status"] = "downloading"
            _jobs[job_id]["local_path"] = str(local_path)
            if remote_size > 0:
                _jobs[job_id]["total_bytes"] = remote_size

        tqdm_class = _make_download_tqdm_class(job_id) if hf_hub_tqdm is not None else None

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            token=token if token else None,
            resume_download=True,
            local_files_only=False,
            tqdm_class=tqdm_class,
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
    # Fallback: check env var
    fallback = os.environ.get("LLAMA_MODELS_DIR", "")
    if fallback:
        return fallback
    # Last resort: parse docker-compose.yml for default value
    compose_path = env_path.with_suffix(".yml") if env_path.suffix == ".env" else env_path.parent / "docker-compose.lmm.yml"
    if compose_path.exists():
        try:
            import yaml
            with open(compose_path, "r", encoding="utf-8") as f:
                compose_data = yaml.safe_load(f)
                # Extract from volume mount in x-llama-base
                llama_base = compose_data.get("x-llama-base", {})
                volumes = llama_base.get("volumes", [])
                for vol in volumes:
                    if isinstance(vol, dict) and vol.get("target") == "/models":
                        source = vol.get("source", "")
                        # Parse ${VAR:-default} syntax
                        if source.startswith("${") and ":-" in source:
                            default = source.split(":-", 1)[1].rstrip("}")
                            return default
                        return source
        except Exception:
            pass
    # Ultimate fallback: use the known default from docker-compose
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
        self.send_header("Access-Control-Allow-Origin", _allowed_cors_origin(self.headers.get("Origin")))
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
        return _token_matches(header_tok, expected)

    def _client_ip(self) -> str:
        return str(self.client_address[0]) if self.client_address else "unknown"

    def _is_mutating(self, path: str) -> bool:
        return path in MUTATING_ENDPOINTS

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
        self.send_header("Access-Control-Allow-Origin", _allowed_cors_origin(self.headers.get("Origin")))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Token")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        # Public health check (no token)
        if parsed.path == "/health":
            self._send_json(200, {
                "ok": True,
                "service": "lmm_host_helper",
                "version": 2,
                "capabilities": [
                    "router/write_preset_ini",
                    "router/restart",
                    "router/warm_params",
                    "router/record_success",
                ],
            })
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
            self._send_json(200, {
                "ok": True,
                "service": "lmm_host_helper",
                "version": 2,
                "capabilities": [
                    "router/write_preset_ini",
                    "router/restart",
                    "router/warm_params",
                    "router/record_success",
                ],
            })
            return

        # All other endpoints require token
        if not self._check_token():
            self._send_json(403, {"ok": False, "error": "invalid or missing X-Token"})
            return

        if self._is_mutating(parsed.path):
            if not _rate_limit_allow(self._client_ip()):
                self._send_json(429, {"ok": False, "error": "rate limit exceeded"})
                return
            _audit_mutating_call(
                action=parsed.path.strip("/") or "root",
                path=parsed.path,
                client_ip=self._client_ip(),
                token=self.headers.get("X-Token", "").strip(),
                status="accepted",
            )

        project_dir = self.server.project_dir
        try:
            compose = _resolve_compose_path(body.get("compose"), self.server.compose_path, project_dir)
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
            return

        if parsed.path == "/ignite":
            preflight = _preflight_fleet_mode(compose)
            if not preflight.get("ok"):
                self._send_json(int(preflight.get("status", 409)), preflight)
                return
            result = _run_docker_compose(compose, "up", "-d")
            result["action"] = "ignite"
            self._send_json(200 if result["ok"] else 500, result)

        elif parsed.path == "/extinguish":
            result = _run_docker_compose(compose, "--profile", "full", "down")
            result["action"] = "extinguish"
            self._send_json(200 if result["ok"] else 500, result)

        elif parsed.path == "/status":
            containers = _container_status()
            fleet_mode = detect_fleet_mode() if detect_fleet_mode else {"mode": "unknown"}
            self._send_json(200, {"ok": True, "containers": containers, "fleet_mode": fleet_mode})

        elif parsed.path == "/router/write_preset_ini":
            try:
                preset_path = _resolve_preset_path(body.get("preset_path"), project_dir)
                content = body.get("content")
                snippet = ""
                if content is None:
                    alias = body.get("alias", "").strip()
                    model_path = body.get("model_path", "").strip()
                    current = preset_path.read_text(encoding="utf-8")
                    content, snippet = _rewrite_preset_alias_model(current, alias, model_path)
                backup = _write_preset_ini_atomic(preset_path, str(content))
                self._send_json(200, {
                    "ok": True,
                    "preset_path": str(preset_path),
                    "backup_path": str(backup),
                    "snippet": snippet,
                })
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})

        elif parsed.path == "/router/restart":
            result = _restart_router_container()
            result["action"] = "restart_router"
            self._send_json(200 if result.get("ok") else 500, result)

        elif parsed.path == "/router/warm_params":
            if not MODEL_PARAMS_CACHE_AVAILABLE:
                self._send_json(503, {"ok": False, "error": "model_params_cache module unavailable"})
                return
            try:
                env_path = _get_env_path(self.server.compose_path)
                env = _read_env_file_dict(env_path)
                force_refresh = bool(body.get("force_refresh", False))
                restart = bool(body.get("restart", True))
                preset_path = _default_preset_host_path(project_dir)
                warm = render_fleet_preset_to_file(env, preset_path, force_refresh=force_refresh)
                restarted = False
                if restart and "router" in str(self.server.compose_path).lower():
                    rr = _restart_router_container()
                    restarted = bool(rr.get("ok"))
                self._send_json(200, {
                    "ok": True,
                    "action": "warm_params",
                    "preset_path": str(preset_path),
                    "stats": warm.get("stats"),
                    "cache_path": warm.get("cache_path"),
                    "restarted": restarted,
                    "plans": [p.as_dict() for p in warm.get("entries") or []],
                })
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})

        elif parsed.path == "/router/record_success":
            if not MODEL_PARAMS_CACHE_AVAILABLE:
                self._send_json(503, {"ok": False, "error": "model_params_cache module unavailable"})
                return
            try:
                env_path = _get_env_path(self.server.compose_path)
                env = _read_env_file_dict(env_path)
                if body.get("all"):
                    for role, _alias, model_var in (
                        ("chat", "chat", "CHAT_MODEL_PATH"),
                        ("utility", "utility", "UTILITY_MODEL_PATH"),
                        ("embed", "embedding", "EMBED_MODEL_PATH"),
                    ):
                        path = (env.get(model_var) or "").strip()
                        if path:
                            record_role_success(role, container_model_path=path, env=env)
                    self._send_json(200, {"ok": True, "action": "record_success", "scope": "all"})
                    return
                role = (body.get("role") or "").strip().lower()
                model_path = (body.get("model_path") or "").strip()
                if not role or not model_path:
                    self._send_json(400, {"ok": False, "error": "role and model_path required (or all=true)"})
                    return
                record_role_success(role, container_model_path=model_path, env=env)
                self._send_json(200, {"ok": True, "action": "record_success", "role": role})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})

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
            model_path = str(model.get("path", "")).replace("\\", "/")
            full_model_path = f"/models/{model_path}/{model_file}" if model_path else f"/models/{model_file}"

            env_map = _read_env_file_dict(env_path)
            role_key = slot.lower()
            if role_key == "embedding":
                role_key = "embed"
            alias_map = {"chat": "chat", "utility": "utility", "embed": "embedding", "embedding": "embedding"}
            alias = alias_map.get(role_key, role_key)

            recommended_ctx = None
            params_source = None
            host_model_path = str(Path(models_dir) / model_path / model_file) if model_path else str(Path(models_dir) / model_file)

            if MODEL_PARAMS_CACHE_AVAILABLE and plan_model_with_cache:
                try:
                    gpus = _detect_gpus_for_scan()
                    total_vram_gb = sum(g.get("total_vram_mb", 0) / 1024 for g in gpus) if gpus else 0.0
                    if total_vram_gb > 0:
                        env_map["A0_LMM_AVAILABLE_VRAM_GB"] = str(round(total_vram_gb, 2))
                    plan, _g_opts, _p_opts, params_source = plan_model_with_cache(
                        alias=alias,
                        role=role_key if role_key != "embedding" else "embed",
                        container_model_path=full_model_path,
                        env=env_map,
                        force_refresh=bool(body.get("force_refresh_params", False)),
                    )
                    recommended_ctx = plan.hard_ctx
                    if store_plan_in_cache:
                        store_plan_in_cache(
                            alias=alias,
                            role=plan.role,
                            container_model_path=full_model_path,
                            host_model_path=host_model_path,
                            plan=plan,
                            global_options=_g_opts,
                            per_model_options=_p_opts,
                            source=params_source or "computed",
                            env=env_map,
                            available_vram_gb=total_vram_gb or None,
                        )
                except Exception as e:
                    print(f"[WARNING] model_params_cache planning failed: {e}")

            if recommended_ctx is None and calculate_optimal_context:
                try:
                    gpus = _detect_gpus_for_scan()
                    total_vram_gb = sum(g.get("total_vram_mb", 0) / 1024 for g in gpus) if gpus else 0.0
                    other_slots_vram_gb = 0.0
                    for other_slot in ["chat", "utility", "embedding", "vision", "reasoning"]:
                        if other_slot != slot.lower():
                            other_model_id = _get_current_model_path(env_path, other_slot)
                            if other_model_id:
                                other_model = manifest.get("models", {}).get(
                                    other_model_id.split("/")[-1].replace(".gguf", "")
                                )
                                if other_model:
                                    other_slots_vram_gb += other_model.get("size_gb", 0) * 1.15
                    ctx_calc_result = calculate_optimal_context(
                        host_model_path,
                        slot.lower(),
                        total_vram_gb,
                        other_slots_vram_gb,
                    )
                    recommended_ctx = ctx_calc_result.get("recommended_ctx")
                    params_source = "legacy_calculator"
                except Exception as e:
                    print(f"[WARNING] Context calculation failed: {e}")

            ok = _rewrite_env_slot_model(env_path, slot, full_model_path)
            if not ok:
                self._send_json(400, {"ok": False, "error": f"invalid slot '{slot}'"})
                return

            if recommended_ctx:
                _rewrite_env_slot_ctx(env_path, slot, int(recommended_ctx))

            restarted = False
            preset_refreshed = False
            if apply_now:
                compose_name = str(self.server.compose_path).lower()
                if "router" in compose_name and MODEL_PARAMS_CACHE_AVAILABLE and render_fleet_preset_to_file:
                    try:
                        env_map = _read_env_file_dict(env_path)
                        preset_path = _default_preset_host_path(project_dir)
                        render_fleet_preset_to_file(env_map, preset_path, force_refresh=False)
                        preset_refreshed = True
                        rr = _restart_router_container()
                        restarted = bool(rr.get("ok"))
                    except Exception as e:
                        print(f"[WARNING] router preset refresh after assign failed: {e}")
                else:
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
                        result = _run_docker_compose(
                            self.server.compose_path, "up", "-d", "--force-recreate", service
                        )
                        restarted = result.get("ok", False)

            response_data = {
                "ok": True,
                "slot": slot,
                "model_id": model_id,
                "restarted": restarted,
                "preset_refreshed": preset_refreshed,
                "model_path": full_model_path,
                "recommended_ctx": recommended_ctx,
                "params_source": params_source,
            }

            self._send_json(200, response_data)

        elif parsed.path == "/models/start":
            slot = body.get("slot", "").strip()
            if not slot:
                self._send_json(400, {"ok": False, "error": "slot is required"})
                return
            service_map = {
                "chat": "a0-llama-chat",
                "utility": "a0-llama-utility",
                "embedding": "a0-llama-embed",
                "embed": "a0-llama-embed",
                "vision": "a0-llama-vision",
                "reasoning": "a0-llama-reasoning",
            }
            service = service_map.get(slot.lower())
            if not service:
                self._send_json(400, {"ok": False, "error": f"unknown slot '{slot}'"})
                return
            result = _run_docker_compose(self.server.compose_path, "up", "-d", service)
            self._send_json(200, {"ok": result.get("ok", False), "slot": slot, "service": service, "output": result.get("stdout", "")})

        elif parsed.path == "/models/stop":
            slot = body.get("slot", "").strip()
            if not slot:
                self._send_json(400, {"ok": False, "error": "slot is required"})
                return
            service_map = {
                "chat": "a0-llama-chat",
                "utility": "a0-llama-utility",
                "embedding": "a0-llama-embed",
                "embed": "a0-llama-embed",
                "vision": "a0-llama-vision",
                "reasoning": "a0-llama-reasoning",
            }
            service = service_map.get(slot.lower())
            if not service:
                self._send_json(400, {"ok": False, "error": f"unknown slot '{slot}'"})
                return
            result = _run_docker_compose(self.server.compose_path, "stop", service)
            self._send_json(200, {"ok": result.get("ok", False), "slot": slot, "service": service, "output": result.get("stdout", "")})

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
    parser.add_argument("--bind", default="", help="Listen address (default: 127.0.0.1)")
    parser.add_argument("--compose", default=COMPOSE_FILE, help="Path to docker-compose.lmm.yml (inside plugin)")
    parser.add_argument("--project-dir", default=os.getcwd(), help="Project directory for .bat resolution")
    args = parser.parse_args()

    # Resolve compose path relative to project-dir if needed
    compose_path = args.compose
    if not Path(compose_path).is_absolute():
        compose_path = str(Path(args.project_dir) / compose_path)

    # Ensure token exists before starting
    _ensure_token()

    bind_host = _resolve_bind_host(args.bind)
    if bind_host in {"0.0.0.0", "::"}:
        print(f"[WARN] LMM Host Helper is publicly bound on {bind_host}; keep the token private.")

    server = Server((bind_host, args.port), Handler, compose_path, args.project_dir)
    print(f"[READY] LMM Host Helper listening on {bind_host}:{args.port}")
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
