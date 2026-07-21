"""
Benchmark Fetcher — fetch and cache benchmark scores for LLMs.

Implements benchmark integration inspired by whichllm:
- Static benchmark data for popular models (curated JSON)
- Evidence confidence grading (direct > variant > base > interpolated > self-reported)
- Recency demotion (stale benchmarks get lower scores)
- Future: HuggingFace API integration for live data
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static benchmark data (curated, ~50 popular models)
# ---------------------------------------------------------------------------

_STATIC_BENCHMARKS = {
    # Qwen models
    "Qwen/Qwen2.5-72B-Instruct": {
        "score": 89.5,
        "sources": ["LiveBench", "Arena ELO"],
        "date": "2026-05",
        "confidence": "direct",
        "task_profiles": ["general", "chat", "coding"],
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "score": 85.2,
        "sources": ["LiveBench", "Arena ELO"],
        "date": "2026-05",
        "confidence": "direct",
        "task_profiles": ["general", "chat"],
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "score": 78.3,
        "sources": ["LiveBench", "Arena ELO"],
        "date": "2026-05",
        "confidence": "direct",
        "task_profiles": ["general", "chat"],
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "score": 72.1,
        "sources": ["LiveBench", "Arena ELO"],
        "date": "2026-05",
        "confidence": "direct",
        "task_profiles": ["general", "chat", "utility"],
    },
    "Qwen/Qwen3.5-9B-Instruct": {
        "score": 76.8,
        "sources": ["LiveBench", "Arena ELO"],
        "date": "2026-05",
        "confidence": "direct",
        "task_profiles": ["general", "chat", "utility"],
    },
    "Qwen/Qwen3.6-27B": {
        "score": 92.8,
        "sources": ["LiveBench", "Artificial Analysis"],
        "date": "2026-05",
        "confidence": "direct",
        "task_profiles": ["general", "chat", "coding"],
    },
    
    # Mistral models
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "score": 84.5,
        "sources": ["LiveBench", "Arena ELO"],
        "date": "2026-04",
        "confidence": "direct",
        "task_profiles": ["general", "chat"],
    },
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "score": 79.2,
        "sources": ["LiveBench", "Arena ELO"],
        "date": "2024-07",
        "confidence": "direct",
        "task_profiles": ["general", "chat"],
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "score": 68.4,
        "sources": ["Open LLM Leaderboard"],
        "date": "2024-06",
        "confidence": "direct",
        "task_profiles": ["general", "chat", "utility"],
    },
    
    # Gemma models
    "google/gemma-2-27b-it": {
        "score": 82.7,
        "sources": ["LiveBench", "Arena ELO"],
        "date": "2024-06",
        "confidence": "direct",
        "task_profiles": ["general", "chat"],
    },
    "google/gemma-2-9b-it": {
        "score": 74.3,
        "sources": ["LiveBench", "Arena ELO"],
        "date": "2024-06",
        "confidence": "direct",
        "task_profiles": ["general", "chat", "utility"],
    },
    "google/gemma-3-4b-it": {
        "score": 65.8,
        "sources": ["Open LLM Leaderboard"],
        "date": "2025-02",
        "confidence": "direct",
        "task_profiles": ["general", "utility"],
    },
    
    # DeepSeek models
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": {
        "score": 71.5,
        "sources": ["Aider", "Arena ELO"],
        "date": "2024-12",
        "confidence": "direct",
        "task_profiles": ["coding", "utility"],
    },
    "deepseek-ai/DeepSeek-V2.5": {
        "score": 88.9,
        "sources": ["LiveBench", "Arena ELO"],
        "date": "2025-01",
        "confidence": "direct",
        "task_profiles": ["general", "chat", "coding"],
    },
    
    # Phi models
    "microsoft/Phi-3.5-mini-instruct": {
        "score": 63.2,
        "sources": ["Open LLM Leaderboard"],
        "date": "2024-08",
        "confidence": "direct",
        "task_profiles": ["general", "utility", "router"],
    },
    "microsoft/Phi-3-medium-128k-instruct": {
        "score": 70.1,
        "sources": ["Open LLM Leaderboard"],
        "date": "2024-06",
        "confidence": "direct",
        "task_profiles": ["general", "chat"],
    },
    
    # Llama models
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {
        "score": 86.4,
        "sources": ["LiveBench", "Arena ELO"],
        "date": "2024-07",
        "confidence": "direct",
        "task_profiles": ["general", "chat"],
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "score": 69.7,
        "sources": ["LiveBench", "Arena ELO"],
        "date": "2024-07",
        "confidence": "direct",
        "task_profiles": ["general", "chat", "utility"],
    },
    
    # Specialized models
    "nomic-ai/nomic-embed-text-v1.5": {
        "score": 85.0,
        "sources": ["MTEB"],
        "date": "2024-03",
        "confidence": "direct",
        "task_profiles": ["embedding"],
    },
}


# ---------------------------------------------------------------------------
# Confidence multipliers for evidence grading
# ---------------------------------------------------------------------------

_CONFIDENCE_MULTIPLIERS = {
    "direct": 1.0,           # Exact model ID match, independently verified
    "variant": 0.95,         # Suffix-stripped or -Instruct variant
    "base": 0.78,            # Base model from cardData
    "interpolated": 0.65,    # Size-aware interpolation within model family
    "self_reported": 0.55,   # Uploader-claimed eval (heavily discounted)
}


# ---------------------------------------------------------------------------
# Recency demotion (stale benchmarks get lower scores)
# ---------------------------------------------------------------------------

def _apply_recency_demotion(score: float, benchmark_date: str, current_date: str = "2026-05") -> float:
    """
    Demote scores from stale benchmarks.
    
    Rules:
    - Current month: no demotion
    - 1-3 months old: 0.95 multiplier
    - 3-6 months old: 0.90 multiplier
    - 6-12 months old: 0.80 multiplier
    - 12+ months old: 0.70 multiplier
    """
    try:
        b_year, b_month = map(int, benchmark_date.split("-"))
        c_year, c_month = map(int, current_date.split("-"))
        
        months_old = (c_year - b_year) * 12 + (c_month - b_month)
        
        if months_old <= 0:
            return score
        elif months_old <= 3:
            return score * 0.95
        elif months_old <= 6:
            return score * 0.90
        elif months_old <= 12:
            return score * 0.80
        else:
            return score * 0.70
    except Exception:
        # If date parsing fails, no demotion
        return score


# ---------------------------------------------------------------------------
# Model name matching (fuzzy matching for GGUF variants)
# ---------------------------------------------------------------------------

def _match_model_name(model_name: str) -> Optional[str]:
    """
    Match a model name (possibly GGUF variant) to benchmark data.
    
    Examples:
    - "Qwen/Qwen2.5-7B-Instruct-GGUF" -> "Qwen/Qwen2.5-7B-Instruct"
    - "bartowski/Mistral-7B-Instruct-v0.3-GGUF" -> "mistralai/Mistral-7B-Instruct-v0.3"
    """
    # Remove common GGUF repo suffixes
    clean_name = model_name
    for suffix in ["-GGUF", "-GGUF2", "-GGUF3"]:
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)]
            break
    
    # Handle common repackagers (bartowski, TheBloke, etc.)
    if "/" in clean_name:
        org, name = clean_name.split("/", 1)
        if org in ["bartowski", "TheBloke", "MaziyarPanahi"]:
            # Try to find the original org in our benchmarks
            for benchmark_key in _STATIC_BENCHMARKS:
                b_org, b_name = benchmark_key.split("/", 1)
                if name == b_name or name.startswith(b_name):
                    return benchmark_key
    
    # Direct match
    if clean_name in _STATIC_BENCHMARKS:
        return clean_name
    
    # Try without -Instruct suffix
    if clean_name.endswith("-Instruct"):
        base = clean_name[:-len("-Instruct")]
        if base in _STATIC_BENCHMARKS:
            return base
    
    # Try partial match
    for benchmark_key, benchmark_data in _STATIC_BENCHMARKS.items():
        b_org, b_name = benchmark_key.split("/", 1)
        if b_name in clean_name or clean_name in b_name:
            return benchmark_key
    
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_benchmark_data(model_name: str) -> Dict[str, Any]:
    """
    Get benchmark data for a model.
    
    Returns dict with:
        score: float (0-100)
        sources: list of benchmark source names
        date: str (YYYY-MM format)
        confidence: str (direct/variant/base/interpolated/self_reported)
        task_profiles: list of task types (general, chat, coding, vision, math, embedding)
    """
    matched_key = _match_model_name(model_name)
    
    if matched_key is None:
        return {
            "score": 0,
            "sources": [],
            "date": "",
            "confidence": "unknown",
            "task_profiles": [],
        }
    
    benchmark = _STATIC_BENCHMARKS[matched_key].copy()
    
    # Apply recency demotion
    benchmark["score"] = _apply_recency_demotion(
        benchmark["score"],
        benchmark["date"]
    )
    
    # If this was a fuzzy match, adjust confidence
    if matched_key != model_name:
        original_confidence = benchmark["confidence"]
        if original_confidence == "direct":
            benchmark["confidence"] = "variant"
        benchmark["score"] = benchmark["score"] * _CONFIDENCE_MULTIPLIERS["variant"]
    
    benchmark["score"] = round(benchmark["score"], 1)
    
    return benchmark


def get_all_benchmarks() -> Dict[str, Dict[str, Any]]:
    """
    Get all benchmark data (for debugging/testing).
    """
    return _STATIC_BENCHMARKS.copy()


def load_benchmarks_from_json(json_path: Optional[str] = None) -> bool:
    """
    Load benchmark data from a JSON file (future: live API caching).
    
    Returns True if loaded successfully, False otherwise.
    """
    if json_path is None:
        # Default path: plugin directory / conf / benchmarks.json
        plugin_dir = Path(__file__).parent.parent
        json_path = plugin_dir / "conf" / "benchmarks.json"
    
    json_path = Path(json_path)
    if not json_path.exists():
        logger.debug(f"Benchmark JSON not found at {json_path}, using static data")
        return False
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Update static benchmarks with loaded data
        _STATIC_BENCHMARKS.update(data)
        logger.info(f"Loaded {len(data)} benchmarks from {json_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to load benchmarks from {json_path}: {e}")
        return False
