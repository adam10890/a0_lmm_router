"""
Tier Catalog — T0-T9 mapping of open-weight LLMs per the local-llm-recommender
skill (snapshot 11-05-2026, Artificial Analysis Intelligence Index open-source
rankings).

Each tier holds an ordered list of picks; the first item in each tier is the
"headline" / recommended pick (per the source skill's curation order).

Each pick is tagged with `roles` — the slot roles it's well-suited for. The
canonical roles in this plugin are:
    chat        — multi-turn assistant
    utility     — fast small-context helper (classification, extraction, etc.)
    reasoning   — chain-of-thought / planner
    code        — code completion / review
    embedding   — embedding vectors (separate catalog tier "embeddings")
    vision      — multimodal (text + image)

Note: the embedding/vision rows live in the special "EMBEDDINGS" / "VISION"
slots of the catalog because they don't scale on the AA Intelligence Index
the way text generators do.

pick_three(tier) returns:
    COMFORTABLE = headline of (tier - 1)
    BALANCED    = headline of (tier)
    STRETCH     = headline of (tier + 1)

Edge cases per skill:
    * T0 user: no useful COMFORTABLE pick — suggest cloud / upgrade
    * T9 user: STRETCH falls back to BALANCED — at hardware ceiling
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Canonical slot roles known to the plugin (extend by adding entries here +
# corresponding role tags in the catalog below).
KNOWN_ROLES = ("chat", "utility", "reasoning", "code", "embedding", "vision")

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
# Fields per pick:
#   name              — model display name
#   params_b          — total parameters (B)
#   active_params_b   — active params for MoE (None for dense)
#   quant             — quantization label
#   size_gb           — on-disk file size (approximate, model_params * 0.55/0.6)
#   aa_score          — AA Intelligence Index score (None = off-chart)
#   speed_class       — "Fast" | "Usable" | "Slow" | "Painful"
#   engine            — "ollama" | "llamacpp"
#   install           — one-line install command
#   reason            — one-line why-it-fits text
#
# Snapshot date is 11-05-2026; rankings have a freshness window.

_CATALOG: Dict[str, List[Dict[str, Any]]] = {
    "T0": [
        {
            "name": "Llama 3.2 3B Instruct",
            "params_b": 3, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 1.9,
            "aa_score": None, "speed_class": "Usable",
            "engine": "ollama",
            "install": "ollama run llama3.2:3b",
            "reason": "Smallest viable chat model; CPU-class machines.",
            "roles": ["utility", "chat"],
        },
        {
            "name": "Qwen3 4B Instruct",
            "params_b": 4, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 2.4,
            "aa_score": None, "speed_class": "Usable",
            "engine": "ollama",
            "install": "ollama run qwen3:4b",
            "reason": "Lightweight multilingual; better quality than Llama 3.2 3B.",
            "roles": ["utility", "chat"],
        },
    ],
    "T1": [
        {
            "name": "Qwen3.5 9B Instruct",
            "params_b": 9, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 5.4,
            "aa_score": 27, "speed_class": "Fast",
            "engine": "ollama",
            "install": "ollama run qwen3.5:9b",
            "reason": "Best score that fits 6-9 GB EIM (tight at Q4).",
            "roles": ["chat", "utility"],
        },
        {
            "name": "Nemotron Nano 9B V2",
            "params_b": 9, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 5.4,
            "aa_score": 15, "speed_class": "Fast",
            "engine": "ollama",
            "install": "ollama run nemotron-nano:9b",
            "reason": "NVIDIA-curated reasoning tune, low AA but well-supported.",
            "roles": ["reasoning", "utility"],
        },
    ],
    "T2": [
        {
            "name": "gpt-oss-20B",
            "params_b": 20, "active_params_b": None,
            "quant": "MXFP4", "size_gb": 12.0,
            "aa_score": 24, "speed_class": "Fast",
            "engine": "ollama",
            "install": "ollama run gpt-oss:20b",
            "reason": "OpenAI's open-weight; MXFP4 fits 10-14 GB cleanly.",
            "roles": ["chat", "utility", "reasoning"],
        },
        {
            "name": "Qwen3.5 9B Instruct",
            "params_b": 9, "active_params_b": None,
            "quant": "Q8_0", "size_gb": 9.5,
            "aa_score": 27, "speed_class": "Fast",
            "engine": "ollama",
            "install": "ollama run qwen3.5:9b-q8_0",
            "reason": "Higher quant of Qwen3.5 9B — better quality, same family.",
            "roles": ["chat", "utility"],
        },
    ],
    "T3": [
        {
            "name": "Qwen3.6-27B",
            "params_b": 27, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 16.2,
            "aa_score": 37, "speed_class": "Fast",
            "engine": "llamacpp",
            "install": "llama-cli -hf unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL",
            "reason": "Top score that fits 16-20 GB; dense (strong on coding).",
            "note": "Ollama lacks Qwen3.6 mmproj support — use llama.cpp + Unsloth.",
            "roles": ["chat", "code", "reasoning"],
        },
        {
            "name": "gpt-oss-20B",
            "params_b": 20, "active_params_b": None,
            "quant": "MXFP4", "size_gb": 12.0,
            "aa_score": 24, "speed_class": "Fast",
            "engine": "ollama",
            "install": "ollama run gpt-oss:20b",
            "reason": "Lower score but faster + Ollama-native.",
            "roles": ["chat", "utility", "reasoning"],
        },
        {
            "name": "Apriel-v1.6-15B-Thinker",
            "params_b": 15, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 9.0,
            "aa_score": 28, "speed_class": "Fast",
            "engine": "llamacpp",
            "install": "llama-cli -hf unsloth/Apriel-v1.6-15B-Thinker-GGUF:UD-Q4_K_XL",
            "reason": "Reasoning-tuned 15B; tight middle option.",
            "roles": ["reasoning", "chat"],
        },
    ],
    "T4": [
        {
            "name": "Qwen3.6-35B-A3B",
            "params_b": 35, "active_params_b": 3,
            "quant": "Q4_K_M", "size_gb": 21.0,
            "aa_score": 43, "speed_class": "Fast",
            "engine": "llamacpp",
            "install": "llama-cli -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL",
            "reason": "Top open-weight AA score; MoE runs at 3B-dense speed.",
            "note": "Ollama lacks Qwen3.6 mmproj — use llama.cpp + Unsloth.",
            "roles": ["chat", "reasoning", "utility"],
        },
        {
            "name": "Gemma 4 31B Instruct",
            "params_b": 31, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 18.6,
            "aa_score": 39, "speed_class": "Fast",
            "engine": "ollama",
            "install": "ollama run gemma4:31b",
            "reason": "Dense Google model; strong all-rounder for 22-28 GB EIM.",
            "roles": ["chat", "code"],
        },
        {
            "name": "Qwen3.6-27B",
            "params_b": 27, "active_params_b": None,
            "quant": "Q5_K_M", "size_gb": 19.0,
            "aa_score": 37, "speed_class": "Fast",
            "engine": "llamacpp",
            "install": "llama-cli -hf unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL",
            "reason": "Higher quant of the T3 pick — fits T4 with room.",
            "roles": ["chat", "code", "reasoning"],
        },
    ],
    "T5": [
        {
            "name": "Qwen3.6-35B-A3B",
            "params_b": 35, "active_params_b": 3,
            "quant": "Q6_K", "size_gb": 28.0,
            "aa_score": 43, "speed_class": "Fast",
            "engine": "llamacpp",
            "install": "llama-cli -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q6_K_XL",
            "reason": "Headroom for long context at top AA score.",
            "roles": ["chat", "reasoning"],
        },
        {
            "name": "Qwen3.6-27B",
            "params_b": 27, "active_params_b": None,
            "quant": "Q8_0", "size_gb": 27.0,
            "aa_score": 37, "speed_class": "Fast",
            "engine": "llamacpp",
            "install": "llama-cli -hf unsloth/Qwen3.6-27B-GGUF:UD-Q8_0",
            "reason": "Near-FP16 quality of Qwen3.6 dense; fits T5 cleanly.",
            "roles": ["chat", "code", "reasoning"],
        },
        {
            "name": "Gemma 4 31B Instruct",
            "params_b": 31, "active_params_b": None,
            "quant": "Q5_K_M", "size_gb": 22.0,
            "aa_score": 39, "speed_class": "Fast",
            "engine": "ollama",
            "install": "ollama run gemma4:31b-q5_k_m",
            "reason": "Higher quant of the T4 Gemma 4 pick.",
            "roles": ["chat", "code"],
        },
    ],
    "T6": [
        {
            "name": "gpt-oss-120B",
            "params_b": 120, "active_params_b": None,
            "quant": "MXFP4", "size_gb": 63.0,
            "aa_score": 33, "speed_class": "Usable",
            "engine": "ollama",
            "install": "ollama run gpt-oss:120b",
            "reason": "o4-mini-class reasoning at MXFP4; fits 45-65 GB EIM.",
            "roles": ["reasoning", "chat"],
        },
        {
            "name": "Nemotron 3 Super",
            "params_b": 70, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 42.0,
            "aa_score": 36, "speed_class": "Fast",
            "engine": "ollama",
            "install": "ollama run nemotron:super",
            "reason": "Higher AA than gpt-oss-120B; smaller and faster.",
            "roles": ["chat", "reasoning"],
        },
        {
            "name": "Qwen3 Next 80B-A3B",
            "params_b": 80, "active_params_b": 3,
            "quant": "Q4_K_M", "size_gb": 48.0,
            "aa_score": 27, "speed_class": "Fast",
            "engine": "ollama",
            "install": "ollama run qwen3-next:80b-a3b",
            "reason": "MoE running at 3B-dense speed; lower AA than alternatives.",
            "roles": ["chat", "utility"],
        },
    ],
    "T7": [
        {
            "name": "Qwen3.5-122B-A10B",
            "params_b": 122, "active_params_b": 10,
            "quant": "Q4_K_M", "size_gb": 75.0,
            "aa_score": 42, "speed_class": "Usable",
            "engine": "llamacpp",
            "install": "llama-cli -hf unsloth/Qwen3.5-122B-A10B-GGUF:UD-Q4_K_XL",
            "reason": "Highest-quality MoE in 70-95 GB range.",
            "roles": ["chat", "reasoning", "code"],
        },
        {
            "name": "gpt-oss-120B",
            "params_b": 120, "active_params_b": None,
            "quant": "Q8_0", "size_gb": 90.0,
            "aa_score": 33, "speed_class": "Usable",
            "engine": "ollama",
            "install": "ollama run gpt-oss:120b-q8_0",
            "reason": "Near-FP16 quant of gpt-oss; very close to BF16 quality.",
            "roles": ["reasoning", "chat"],
        },
    ],
    "T8": [
        {
            "name": "Qwen3.5-122B-A10B",
            "params_b": 122, "active_params_b": 10,
            "quant": "Q6_K", "size_gb": 105.0,
            "aa_score": 42, "speed_class": "Usable",
            "engine": "llamacpp",
            "install": "llama-cli -hf unsloth/Qwen3.5-122B-A10B-GGUF:UD-Q6_K_XL",
            "reason": "Higher quant of T7 headline; gains a bit of quality.",
            "roles": ["chat", "reasoning", "code"],
        },
        {
            "name": "Mistral Medium 3.5",
            "params_b": 70, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 42.0,
            "aa_score": 39, "speed_class": "Fast",
            "engine": "llamacpp",
            "install": "llama-cli -hf unsloth/Mistral-Medium-3.5-GGUF:UD-Q4_K_XL",
            "reason": "Strong Mistral release — if you have the license.",
            "roles": ["chat", "code"],
        },
    ],
    "T9": [
        {
            "name": "Qwen3.5-397B-A17B",
            "params_b": 397, "active_params_b": 17,
            "quant": "Q4_K_M", "size_gb": 238.0,
            "aa_score": None, "speed_class": "Usable",
            "engine": "llamacpp",
            "install": "llama-cli -hf unsloth/Qwen3.5-397B-A17B-GGUF:UD-Q4_K_XL",
            "reason": "Top-tier open-weight MoE; needs 200+ GB EIM.",
            "roles": ["chat", "reasoning", "code"],
        },
    ],

    # ---------------------------------------------------------------------
    # Role-specific catalogs — sit outside the T0..T9 hierarchy because
    # they don't compete on the AA Intelligence Index. Sorted by quality.
    # ---------------------------------------------------------------------
    "EMBEDDINGS": [
        {
            "name": "Nomic Embed Text v1.5",
            "params_b": 0.14, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 0.08,
            "aa_score": None, "speed_class": "Fast",
            "engine": "llamacpp",
            "install": (
                "huggingface-cli download nomic-ai/nomic-embed-text-v1.5-GGUF "
                "nomic-embed-text-v1.5.Q4_K_M.gguf --local-dir models/embedding/nomic"
            ),
            "reason": "Default embedding for the plugin; 768d, 8K context, tiny.",
            "roles": ["embedding"],
            "min_tier": "T0",
        },
        {
            "name": "BGE-M3",
            "params_b": 0.56, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 0.34,
            "aa_score": None, "speed_class": "Fast",
            "engine": "llamacpp",
            "install": (
                "huggingface-cli download BAAI/bge-m3 model.safetensors "
                "--local-dir models/embedding/bge-m3"
            ),
            "reason": "Multilingual (100+ langs), 1024d, dense + sparse + colbert.",
            "roles": ["embedding"],
            "min_tier": "T0",
        },
        {
            "name": "BGE Large EN v1.5",
            "params_b": 0.34, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 0.21,
            "aa_score": None, "speed_class": "Fast",
            "engine": "llamacpp",
            "install": (
                "huggingface-cli download CompendiumLabs/bge-large-en-v1.5-gguf "
                "bge-large-en-v1.5-q4_k_m.gguf --local-dir models/embedding/bge-large"
            ),
            "reason": "English-only but stronger than Nomic on MTEB benchmarks.",
            "roles": ["embedding"],
            "min_tier": "T0",
        },
    ],
    "VISION": [
        {
            "name": "Qwen2.5-VL 7B Instruct",
            "params_b": 7, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 4.7,
            "aa_score": None, "speed_class": "Fast",
            "engine": "llamacpp",
            "install": "llama-cli -hf unsloth/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M",
            "reason": "Best small VLM; needs mmproj file alongside weights.",
            "roles": ["vision"],
            "min_tier": "T1",
        },
        {
            "name": "Gemma 3 27B Vision",
            "params_b": 27, "active_params_b": None,
            "quant": "Q4_K_M", "size_gb": 16.2,
            "aa_score": None, "speed_class": "Fast",
            "engine": "llamacpp",
            "install": "llama-cli -hf unsloth/gemma-3-27b-it-GGUF:Q4_K_M",
            "reason": "Strong multimodal at 27B; needs T3+ to fit comfortably.",
            "roles": ["vision"],
            "min_tier": "T3",
        },
    ],
}


# Tier order, used to navigate adjacent tiers
_TIERS_ORDERED = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _adjacent_tier(tier: str, offset: int) -> Optional[str]:
    try:
        idx = _TIERS_ORDERED.index(tier)
    except ValueError:
        return None
    new_idx = idx + offset
    if 0 <= new_idx < len(_TIERS_ORDERED):
        return _TIERS_ORDERED[new_idx]
    return None


def _headline_for(tier: str) -> Optional[Dict[str, Any]]:
    picks = _CATALOG.get(tier, [])
    return dict(picks[0]) if picks else None


def list_tier(tier: str) -> List[Dict[str, Any]]:
    """Return all picks for a tier (defensive copies)."""
    return [dict(p) for p in _CATALOG.get(tier, [])]


def pick_three(tier: str, disk_free_gb: float = 0.0) -> Dict[str, Any]:
    """
    Return the canonical COMFORTABLE / BALANCED / STRETCH triple for the
    given tier. Applies the skill's edge-case rules for T0 and T9.

    disk_free_gb: when > 0, picks whose size_gb * 2 exceeds the free disk
    get a `disk_warning` field set.
    """
    comfortable = _headline_for(_adjacent_tier(tier, -1) or "")
    balanced = _headline_for(tier)
    stretch = _headline_for(_adjacent_tier(tier, +1) or "")

    # Edge case: T0 has no usable COMFORTABLE
    if tier == "T0":
        comfortable = None

    # Edge case: T9 has no STRETCH — duplicate BALANCED with note
    if tier == "T9" and balanced is not None:
        stretch = dict(balanced)
        stretch["note_stretch"] = "No stretch — at hardware ceiling."

    # Annotate speed expectations: COMFORTABLE always Fast/Usable;
    # STRETCH is typically Slow/Painful due to RAM offload at next tier.
    if comfortable is not None and comfortable.get("speed_class") not in ("Fast", "Usable"):
        comfortable["speed_class"] = "Usable"
    if stretch is not None and stretch.get("speed_class") in ("Fast", "Usable"):
        # We're using stretch as next tier headline — physical fit assumed
        # marginal, so bias toward Usable when it was Fast at home tier.
        if stretch.get("speed_class") == "Fast":
            stretch["speed_class"] = "Usable"

    # Disk warnings (skill rule: need ≥ 2× the model file size)
    def _annotate_disk(pick: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not pick or disk_free_gb <= 0:
            return pick
        required = float(pick.get("size_gb", 0)) * 2.0
        if required > disk_free_gb * 0.9:
            pick["disk_warning"] = (
                f"Needs ~{required:.0f} GB free for safe download; "
                f"only {disk_free_gb:.0f} GB available."
            )
        return pick

    return {
        "tier": tier,
        "comfortable": _annotate_disk(comfortable),
        "balanced": _annotate_disk(balanced),
        "stretch": _annotate_disk(stretch),
    }


def tier_summary() -> List[Dict[str, Any]]:
    """All tiers + the headline picks for each — useful for debugging/UI."""
    out = []
    for t in _TIERS_ORDERED:
        head = _headline_for(t)
        out.append({
            "tier": t,
            "headline": head.get("name") if head else None,
            "aa_score": head.get("aa_score") if head else None,
        })
    return out


# ---------------------------------------------------------------------------
# Role-aware lookup (used by slot_recommender)
# ---------------------------------------------------------------------------

def picks_for_role(
    role: str,
    tier: str,
    max_results: int = 3,
    include_adjacent_tiers: bool = True,
) -> List[Dict[str, Any]]:
    """
    Return picks suitable for `role` at the user's `tier`.

    For text-generation roles (chat/utility/reasoning/code): pulls from the
    T0..T9 catalog, optionally including one tier below and the same tier
    (we don't suggest STRETCH picks per-slot to avoid recommending models
    the user can't actually load alongside other slots).

    For specialty roles (embedding/vision): pulls from the matching role-
    specific catalog, filtered by `min_tier` so a T0 user doesn't get the
    27B Gemma 3 Vision pick.
    """
    role = role.lower()

    # Specialty catalogs
    if role == "embedding":
        return [
            dict(p) for p in _CATALOG.get("EMBEDDINGS", [])
            if _tier_index(p.get("min_tier", "T0")) <= _tier_index(tier)
        ][:max_results]

    if role == "vision":
        return [
            dict(p) for p in _CATALOG.get("VISION", [])
            if _tier_index(p.get("min_tier", "T0")) <= _tier_index(tier)
        ][:max_results]

    # Text-generation roles — pull from current tier + (optionally) tier-1
    candidates: List[Dict[str, Any]] = []
    for t in _candidate_tiers(tier, include_adjacent_tiers):
        for pick in _CATALOG.get(t, []):
            pick_roles = [r.lower() for r in pick.get("roles", [])]
            if role in pick_roles:
                p = dict(pick)
                p["_tier"] = t
                candidates.append(p)

    # Sort: same-tier first, then by AA score desc, then by size asc
    def _sort_key(p: Dict[str, Any]) -> tuple:
        same_tier = 0 if p.get("_tier") == tier else 1
        aa = p.get("aa_score")
        # None scores rank below numeric ones
        aa_key = -1 if aa is None else -aa
        return (same_tier, aa_key, p.get("size_gb", 0))

    candidates.sort(key=_sort_key)
    # De-duplicate by model name (keeps the higher-tier version if same name
    # appears multiple times)
    seen = set()
    unique: List[Dict[str, Any]] = []
    for c in candidates:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        unique.append(c)
        if len(unique) >= max_results:
            break
    return unique


def _candidate_tiers(tier: str, include_adjacent: bool) -> List[str]:
    """Tiers to scan for role picks — current + tier-1 by default."""
    if not include_adjacent:
        return [tier]
    prev_tier = _adjacent_tier(tier, -1)
    out = [tier]
    if prev_tier:
        out.append(prev_tier)
    return out


def _tier_index(tier: str) -> int:
    try:
        return _TIERS_ORDERED.index(tier)
    except ValueError:
        return 0
