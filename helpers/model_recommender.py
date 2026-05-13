"""
Model Recommender — suggest and install GGUF models based on hardware + role.

Reads installed_models.yaml + compute_resources.yaml to understand current
capacity, then queries HuggingFace for compatible models.
"""

import logging
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
        "recommended_flags": {"cache_type_k": "q8_0", "cache_type_v": "q8_0"},
    },
    {
        "name": "DeepSeek-Coder-V2-Lite (Q4_K_M)",
        "repo_id": "TheBloke/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "filename": "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        "size_gb": 9.65, "vram_required_gb": 11.6, "quant": "Q4_K_M",
        "role": "code", "reason": "Top code completion/review model in its class",
        "recommended_flags": {"cache_type_k": "q8_0"},
    },
    {
        "name": "Qwen2.5-7B-Instruct (Q4_K_M)",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_gb": 4.7, "vram_required_gb": 5.6, "quant": "Q4_K_M",
        "role": "utility", "reason": "Versatile utility model, good instruction following",
        "recommended_flags": {},
    },
    {
        "name": "Phi-3.5-mini (Q4_K_M)",
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size_gb": 2.23, "vram_required_gb": 2.7, "quant": "Q4_K_M",
        "role": "router", "reason": "Lightweight router/classifier, 128K context",
        "recommended_flags": {},
    },
    {
        "name": "Nomic-Embed-Text-v1.5 (Q4_K_M)",
        "repo_id": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "filename": "nomic-embed-text-v1.5.Q4_K_M.gguf",
        "size_gb": 0.08, "vram_required_gb": 0.1, "quant": "Q4_K_M",
        "role": "embedding", "reason": "High quality embeddings, tiny footprint",
        "recommended_flags": {},
    },
    {
        "name": "Gemma-3-4B-Instruct (Q4_K_M)",
        "repo_id": "bartowski/gemma-3-4b-it-GGUF",
        "filename": "gemma-3-4b-it-Q4_K_M.gguf",
        "size_gb": 2.5, "vram_required_gb": 3.0, "quant": "Q4_K_M",
        "role": "utility", "reason": "Very capable for its size, good for lightweight tasks",
        "recommended_flags": {},
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recommendations(
    installed_yaml: str = "",
    compute_yaml: str = "",
    role_filter: Optional[str] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Return model recommendations, delegated to llmfit_advisor if available.

    Falls back to a minimal curated list when the advisor plugin is absent.
    """
    try:
        from usr.plugins.llmfit_advisor.helpers.llmfit import recommend as llmfit_recommend
        use_case = role_filter or "general"
        result = llmfit_recommend(use_case=use_case, limit=max_results, enrich_fleet=True)
        if "models" in result:
            return result["models"]
        if "error" not in result:
            return [result]
        return _fallback(role_filter, max_results)
    except ImportError:
        return _fallback(role_filter, max_results)


def _fallback(role_filter: Optional[str], max_results: int) -> List[Dict[str, Any]]:
    """Minimal fallback when llmfit_advisor is not available."""
    fallback_models = [
        {"name": "Qwen3.5-9B (Q4_K_M)", "repo_id": "Qwen/Qwen3.5-9B-Instruct-GGUF", "filename": "qwen3.5-9b-instruct-q4_k_m.gguf", "size_gb": 5.7, "vram_required_gb": 6.8, "quant": "Q4_K_M", "role": "chat", "reason": "Strong general-purpose model, good reasoning", "recommended_flags": {"cache_type_k": "q8_0", "cache_type_v": "q8_0"}},
        {"name": "Nomic-Embed-Text-v1.5 (Q4_K_M)", "repo_id": "nomic-ai/nomic-embed-text-v1.5-GGUF", "filename": "nomic-embed-text-v1.5.Q4_K_M.gguf", "size_gb": 0.08, "vram_required_gb": 0.1, "quant": "Q4_K_M", "role": "embedding", "reason": "High quality embeddings, tiny footprint", "recommended_flags": {}},
        {"name": "Phi-3.5-mini (Q4_K_M)", "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF", "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf", "size_gb": 2.23, "vram_required_gb": 2.7, "quant": "Q4_K_M", "role": "utility", "reason": "Lightweight, 128K context", "recommended_flags": {}},
    ]
    results = [m for m in fallback_models if not role_filter or m["role"] == role_filter]
    for m in results:
        m["source"] = "fallback"
        m["note"] = "llmfit_advisor plugin not available"
    return results[:max_results]


def install_model(
    repo_id: str,
    filename: str,
    target_dir: str = "",
) -> Dict[str, Any]:
    try:
        from usr.plugins.a0_lmm_router.helpers.fleet_models import install_model as fleet_install
        return fleet_install(repo_id=repo_id, filename=filename)
    except ImportError:
        return {
            "ok": False,
            "error": "fleet_models module not available",
        }
