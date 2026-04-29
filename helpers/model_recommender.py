"""
Model Recommender — suggest and install GGUF models based on hardware + role.

Reads installed_models.yaml + compute_resources.yaml to understand current
capacity, then queries HuggingFace for compatible models.
"""

import os
import logging
import subprocess
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelCandidate:
    name: str
    repo_id: str           # HuggingFace repo  e.g.  "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    filename: str          # e.g.  "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    size_gb: float
    vram_required_gb: float
    quant: str             # e.g. Q4_K_M, Q5_K_M
    role: str              # chat, code, math, embedding, utility, router
    reason: str            # why recommended


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file, return empty dict on failure."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.debug("YAML load failed for %s: %s", path, exc)
        return {}


def _get_installed_roles(installed: Dict[str, Any]) -> Dict[str, str]:
    """Map role -> best installed model name."""
    roles: Dict[str, str] = {}
    for mid, minfo in installed.get("models", {}).items():
        role = minfo.get("role", "")
        if role and role not in roles:
            roles[role] = mid
    return roles


def _get_available_vram_gb(compute: Dict[str, Any]) -> float:
    """Get total usable VRAM from compute_resources.yaml."""
    gpus = compute.get("hardware", {}).get("gpus", [])
    total = 0.0
    for g in gpus:
        if g.get("status") == "active":
            total += g.get("vram_usable_mb", 0) / 1024.0
    return total


# ---------------------------------------------------------------------------
# Curated recommendation catalog (offline, no HF API needed)
# ---------------------------------------------------------------------------

_CURATED: List[Dict[str, Any]] = [
    {
        "name": "Mistral-Small-3.1-24B-Instruct (Q4_K_M)",
        "repo_id": "bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF",
        "filename": "Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf",
        "size_gb": 14.2, "vram_required_gb": 17.0, "quant": "Q4_K_M",
        "role": "chat", "reason": "Excellent multilingual chat, strong reasoning",
    },
    {
        "name": "DeepSeek-Coder-V2-Lite (Q4_K_M)",
        "repo_id": "TheBloke/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "filename": "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        "size_gb": 9.65, "vram_required_gb": 11.6, "quant": "Q4_K_M",
        "role": "code", "reason": "Top code completion/review model in its class",
    },
    {
        "name": "Qwen2.5-7B-Instruct (Q4_K_M)",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_gb": 4.7, "vram_required_gb": 5.6, "quant": "Q4_K_M",
        "role": "utility", "reason": "Versatile utility model, good instruction following",
    },
    {
        "name": "Phi-3.5-mini (Q4_K_M)",
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size_gb": 2.23, "vram_required_gb": 2.7, "quant": "Q4_K_M",
        "role": "router", "reason": "Lightweight router/classifier, 128K context",
    },
    {
        "name": "Nomic-Embed-Text-v1.5 (Q4_K_M)",
        "repo_id": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "filename": "nomic-embed-text-v1.5.Q4_K_M.gguf",
        "size_gb": 0.08, "vram_required_gb": 0.1, "quant": "Q4_K_M",
        "role": "embedding", "reason": "High quality embeddings, tiny footprint",
    },
    {
        "name": "Gemma-3-4B-Instruct (Q4_K_M)",
        "repo_id": "bartowski/gemma-3-4b-it-GGUF",
        "filename": "gemma-3-4b-it-Q4_K_M.gguf",
        "size_gb": 2.5, "vram_required_gb": 3.0, "quant": "Q4_K_M",
        "role": "utility", "reason": "Very capable for its size, good for lightweight tasks",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recommendations(
    installed_yaml: str,
    compute_yaml: str,
    role_filter: Optional[str] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Return a list of model recommendations based on hardware + gaps.

    Parameters
    ----------
    installed_yaml : path to installed_models.yaml
    compute_yaml   : path to compute_resources.yaml
    role_filter    : optional – only recommend for this role
    max_results    : cap on returned items
    """
    installed = _load_yaml(installed_yaml)
    compute = _load_yaml(compute_yaml)
    vram_gb = _get_available_vram_gb(compute)
    existing_roles = _get_installed_roles(installed)

    results: List[Dict[str, Any]] = []
    for c in _CURATED:
        if role_filter and c["role"] != role_filter:
            continue
        # Fits in VRAM?
        if c["vram_required_gb"] > vram_gb:
            continue
        # Already installed?
        already = c["role"] in existing_roles
        results.append({
            **c,
            "already_installed": already,
            "fits_vram": True,
        })
    # Sort: uninstalled first, then by size ascending
    results.sort(key=lambda x: (x["already_installed"], x["size_gb"]))
    return results[:max_results]


def install_model(
    repo_id: str,
    filename: str,
    target_dir: str,
) -> Dict[str, Any]:
    """
    Download a GGUF file from HuggingFace using huggingface-cli.

    Returns dict with ok, message, path.
    """
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)

    if os.path.exists(target_path):
        return {"ok": True, "message": "Already exists", "path": target_path}

    try:
        cmd = [
            "huggingface-cli", "download",
            repo_id, filename,
            "--local-dir", target_dir,
            "--local-dir-use-symlinks", "False",
        ]
        logger.info("Downloading model: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            return {"ok": True, "message": "Download complete", "path": target_path}
        else:
            return {"ok": False, "message": f"Download failed: {result.stderr.strip()}", "path": ""}
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "Download timed out (10 min)", "path": ""}
    except FileNotFoundError:
        return {"ok": False, "message": "huggingface-cli not found. Install with: pip install huggingface-hub", "path": ""}
    except Exception as exc:
        return {"ok": False, "message": str(exc), "path": ""}
