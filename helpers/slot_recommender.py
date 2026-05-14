"""
Slot Recommender — role-aware suggestions for llama.cpp slots.

For each configured slot (chat / utility / embedding / reasoning / vision /
code), decides:

    current_status   — how the currently-assigned model fits the slot
                       (OPTIMAL / FITS / OVERSIZED / UNDERSIZED / EMPTY)
    suggestions      — up to N picks suitable for the slot's role, ordered:
                         1. already-installed models that fit (instant Assign)
                         2. higher-quality candidates that need download

Reads:
    * Live tier (from hardware_inspector.derive_tier_from_stats)
    * Installed-model manifest (from fleet_models.list_models)
    * Slot definitions (from conf/llama_cpp_servers.yaml)
    * Role-tagged tier_catalog

Never raises — returns an "empty" slot suggestion on any failure so the UI
can still render.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Resilient import: prefer the A0 container path, fall back for standalone
# tests / out-of-container execution.
try:
    from usr.plugins.a0_lmm_router.helpers.tier_catalog import picks_for_role
except ImportError:  # pragma: no cover — standalone path
    try:
        from .tier_catalog import picks_for_role  # type: ignore
    except ImportError:
        from tier_catalog import picks_for_role  # type: ignore


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Suggestion:
    name: str                       # display name from catalog
    source: str                     # "installed" | "download"
    model_id: Optional[str] = None  # only when source == "installed"
    repo_id: Optional[str] = None   # for download — parsed from install cmd
    filename: Optional[str] = None
    install: str = ""
    engine: str = ""                # "ollama" | "llamacpp"
    quant: str = ""
    size_gb: float = 0.0
    aa_score: Optional[int] = None
    speed_class: str = ""
    reason: str = ""
    note: Optional[str] = None


@dataclass
class SlotSuggestion:
    slot_id: str
    role: str
    current_model_id: str = ""
    current_size_gb: float = 0.0
    current_status: str = "EMPTY"   # OPTIMAL | FITS | OVERSIZED | UNDERSIZED | EMPTY
    current_status_reason: str = ""
    suggestions: List[Suggestion] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Lowercase + strip punctuation, for fuzzy matching catalog ↔ filename."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _find_installed_match(catalog_pick: Dict[str, Any],
                          installed: Dict[str, Any]) -> Optional[str]:
    """
    Try to find an installed model_id that corresponds to this catalog pick.

    Matches on:
        1. Exact model_id == lowercased catalog name with punctuation stripped
        2. file or model_id contains a normalized substring of the catalog name
        3. repo_id match (when catalog install parses to a known HF repo)
    """
    target = _normalize(catalog_pick.get("name", ""))
    if not target:
        return None

    for mid, minfo in installed.items():
        mid_norm = _normalize(mid)
        file_norm = _normalize(minfo.get("file", ""))
        repo_norm = _normalize(minfo.get("repo_id", ""))

        # The catalog name might be e.g. "Qwen3.5 9B Instruct" → "qwen359binstruct"
        # whereas the installed id is "qwen3_5_9b" → "qwen359b". Compare core
        # tokens: model family + size are usually shared.
        if mid_norm in target or target in mid_norm:
            return mid
        if file_norm and (file_norm in target or target in file_norm):
            return mid
        if repo_norm and (repo_norm in target or target in repo_norm):
            return mid

    return None


def _parse_repo_filename_from_install(cmd: str) -> tuple[Optional[str], Optional[str]]:
    """Best-effort parse `llama-cli -hf repo:tag` or huggingface-cli download repo file."""
    if not cmd:
        return (None, None)
    cmd_l = cmd.strip()

    # llama-cli -hf user/repo:tag  → repo = "user/repo", filename = "tag"
    m = re.search(r"-hf\s+([^\s:]+):([^\s]+)", cmd_l)
    if m:
        return (m.group(1), m.group(2))

    # huggingface-cli download user/repo filename ...
    m = re.search(r"huggingface-cli\s+download\s+(\S+)\s+(\S+)", cmd_l)
    if m:
        return (m.group(1), m.group(2))

    # ollama run model:tag  → no repo/file, leave None
    return (None, None)


def _status_for_current(current_size_gb: float, tier_size_gb: float) -> tuple[str, str]:
    """
    Classify how the currently-assigned model fits the tier headline pick.

    Heuristic: if current is within 80%-120% of tier headline size → OPTIMAL.
    < 60% → UNDERSIZED. > 130% → OVERSIZED (might not fit headroom).
    """
    if current_size_gb <= 0 or tier_size_gb <= 0:
        return ("FITS", "")
    ratio = current_size_gb / tier_size_gb
    if 0.8 <= ratio <= 1.2:
        return ("OPTIMAL", "Sized well for your tier.")
    if ratio < 0.6:
        return ("UNDERSIZED",
                f"~{int((1 - ratio) * 100)}% of tier-headline size — you could "
                f"run a bigger / higher-quality model.")
    if ratio > 1.3:
        return ("OVERSIZED",
                f"~{int((ratio - 1) * 100)}% over tier-headline size — may "
                f"crowd VRAM if you load other slots.")
    return ("FITS", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def suggest_for_slot(
    role: str,
    tier: str,
    installed_models: Dict[str, Any],
    current_model_id: str = "",
    max_suggestions: int = 3,
) -> SlotSuggestion:
    """
    Build a SlotSuggestion for one slot.

    role / tier / installed_models all come from upstream (compute_monitor,
    derive_tier_from_stats, fleet_models.list_models). current_model_id is
    the slot's currently-assigned model.
    """
    role_norm = (role or "").lower()
    installed = installed_models or {}
    current_info = installed.get(current_model_id, {}) if current_model_id else {}
    current_size_gb = float(current_info.get("size_gb", 0) or 0)

    picks = picks_for_role(role_norm, tier, max_results=max_suggestions + 2)

    # Headline tier size for status calculation — first pick is the canonical
    # one for this role at this tier.
    tier_size_gb = float(picks[0].get("size_gb", 0)) if picks else 0.0
    status, status_reason = _status_for_current(current_size_gb, tier_size_gb)
    if not current_model_id:
        status = "EMPTY"
        status_reason = "No model assigned."

    # Build suggestions: installed first, then download
    suggestions: List[Suggestion] = []
    seen_names = set()
    for pick in picks:
        if pick["name"] in seen_names:
            continue
        seen_names.add(pick["name"])

        installed_mid = _find_installed_match(pick, installed)
        repo_id, filename = _parse_repo_filename_from_install(pick.get("install", ""))

        # Skip the suggestion entirely if it's the model already loaded here
        if installed_mid and installed_mid == current_model_id:
            continue

        suggestions.append(Suggestion(
            name=pick["name"],
            source="installed" if installed_mid else "download",
            model_id=installed_mid,
            repo_id=repo_id,
            filename=filename,
            install=pick.get("install", ""),
            engine=pick.get("engine", ""),
            quant=pick.get("quant", ""),
            size_gb=float(pick.get("size_gb", 0) or 0),
            aa_score=pick.get("aa_score"),
            speed_class=pick.get("speed_class", ""),
            reason=pick.get("reason", ""),
            note=pick.get("note"),
        ))
        if len(suggestions) >= max_suggestions:
            break

    return SlotSuggestion(
        slot_id="",                     # filled by caller
        role=role_norm,
        current_model_id=current_model_id or "",
        current_size_gb=current_size_gb,
        current_status=status,
        current_status_reason=status_reason,
        suggestions=suggestions,
    )


def suggest_for_all_slots(
    slots: List[Dict[str, Any]],
    tier: str,
    installed_models: Dict[str, Any],
    max_per_slot: int = 3,
) -> List[Dict[str, Any]]:
    """
    Compute suggestions for every slot. Each entry in `slots` is the
    SlotInfo dict from compute_monitor (id, role, model_id, ...).
    Returns a list of dicts (asdict of SlotSuggestion).
    """
    out: List[Dict[str, Any]] = []
    for s in slots or []:
        try:
            slot_id = str(s.get("id", ""))
            role = str(s.get("role", ""))
            current = str(s.get("model_id", "")) or ""
            ss = suggest_for_slot(
                role=role, tier=tier,
                installed_models=installed_models,
                current_model_id=current,
                max_suggestions=max_per_slot,
            )
            ss.slot_id = slot_id
            out.append(asdict(ss))
        except Exception as exc:
            logger.warning("slot_recommender failed for %r: %s", s, exc)
            out.append(asdict(SlotSuggestion(
                slot_id=str(s.get("id", "")),
                role=str(s.get("role", "")),
                current_status="EMPTY",
                current_status_reason=f"recommender error: {exc}",
            )))
    return out
