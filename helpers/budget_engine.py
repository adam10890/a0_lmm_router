"""
Budget Engine for LMM Router — provider registry, rolling-window budgets,
local capacity snapshots, and deterministic task routing ("token economy").

Everything is computed on demand: a budget is pure time-window arithmetic
over the usage ledger, so there is no background loop to maintain.

Registry sources (merged by load_registry):
  conf/provider_limits.yaml   — owned by this plugin, safe to write back
  conf/model_providers.yaml   — auto-merged by A0, READ-ONLY here; its local
                                llama.cpp entries fold into the `local` provider
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml

try:
    from . import compute_monitor, usage_ledger
except ImportError:  # direct import with plugin root on sys.path (MCP process, tests)
    from helpers import compute_monitor, usage_ledger

logger = logging.getLogger("lmm_router.budget")

_PLUGIN_DIR = Path(__file__).resolve().parent.parent

# Overridable in tests.
LIMITS_PATH: Path = _PLUGIN_DIR / "conf" / "provider_limits.yaml"
MANIFEST_PATH: Path = _PLUGIN_DIR / "conf" / "model_providers.yaml"

_save_lock = Lock()

_WINDOW_RE = re.compile(r"^(\d+)([hd])$")
_LOCAL_URL_MARKERS = ("host.docker.internal", "localhost", "127.0.0.1")

WARN_PCT = 80.0
DEFAULT_NEED_TOKENS = 2000
RESERVATION_TTL_S = 600
MIN_LOCAL_FREE_VRAM_MB = 1024

# ponytail: in-memory, MCP-process only — persist if multi-client demand matters
_reservations: Dict[str, List[tuple]] = {}  # provider_id -> [(expiry_ts, tokens)]
_res_counter = 0


def _cooldown_ids() -> set:
    """Provider ids currently cooling down (best-effort, per-process).

    The failover tracker's package pulls in A0-container-only modules
    (helpers.files) — standalone it is unimportable, so treat as none.
    """
    try:
        from .smart_router.failover import get_cooldown_tracker
    except Exception:
        try:
            from helpers.smart_router.failover import get_cooldown_tracker
        except Exception:
            return set()
    try:
        return set(get_cooldown_tracker().get_error_slots())
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def parse_window(spec: str) -> int:
    """'5h' / '7d' -> seconds. Raises ValueError on anything else."""
    m = _WINDOW_RE.match(str(spec).strip())
    if not m:
        raise ValueError(f"bad window spec: {spec!r} (expected e.g. '5h' or '7d')")
    n, unit = int(m.group(1)), m.group(2)
    return n * (3600 if unit == "h" else 86400)


def _load_limits_file() -> dict:
    if LIMITS_PATH.exists():
        return yaml.safe_load(LIMITS_PATH.read_text(encoding="utf-8")) or {}
    return {"version": 1, "providers": {}}


def load_registry() -> Dict[str, dict]:
    """provider_id -> config, merging provider_limits.yaml with model_providers.yaml."""
    providers: Dict[str, dict] = {}
    for pid, cfg in (_load_limits_file().get("providers") or {}).items():
        providers[str(pid)] = dict(cfg or {})

    # Fold A0's manifest in (read-only). Local llama.cpp entries are covered
    # by the `local` provider; any other entry becomes a tracked-but-unlimited
    # external provider unless provider_limits.yaml already defines it.
    try:
        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    except OSError:
        manifest = {}
    for entry in manifest.get("providers") or []:
        pid = (entry or {}).get("id", "")
        base_url = (entry or {}).get("base_url", "") or ""
        if not pid or pid in providers:
            continue
        if any(m in base_url for m in _LOCAL_URL_MARKERS):
            continue
        providers[pid] = {
            "name": entry.get("name", pid),
            "kind": "external",
            "invoke": entry.get("type", "openai_compatible"),
            "base_url": base_url,
            "priority": 50,
            "enabled": entry.get("enabled", True),
            "limits": [],
        }
    return providers


def save_provider(provider_id: str, cfg: dict) -> None:
    """Write one provider entry into provider_limits.yaml (never the manifest)."""
    with _save_lock:
        data = _load_limits_file()
        data.setdefault("providers", {})[provider_id] = cfg
        LIMITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LIMITS_PATH.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


def delete_provider(provider_id: str) -> bool:
    with _save_lock:
        data = _load_limits_file()
        if provider_id not in (data.get("providers") or {}):
            return False
        del data["providers"][provider_id]
        LIMITS_PATH.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return True


# ---------------------------------------------------------------------------
# Budget math
# ---------------------------------------------------------------------------

def provider_budget(provider_id: str, cfg: dict, now: Optional[float] = None) -> dict:
    """Per-window budget state for one subscription/external provider."""
    now = now if now is not None else time.time()
    burn_per_hour = usage_ledger.window_totals(provider_id, 3600, now=now)["tokens"]
    windows: List[dict] = []
    worst_pct = 0.0
    for lim in cfg.get("limits") or []:
        try:
            seconds = parse_window(lim.get("window", "1d"))
        except ValueError:
            continue
        totals = usage_ledger.window_totals(provider_id, seconds, now=now)
        max_tokens = lim.get("max_tokens")
        max_requests = lim.get("max_requests")
        pct = 0.0
        remaining = None
        if max_tokens:
            remaining = max(int(max_tokens) - totals["tokens"], 0)
            pct = max(pct, totals["tokens"] / int(max_tokens) * 100)
        if max_requests:
            pct = max(pct, totals["requests"] / int(max_requests) * 100)
        exhausts = (
            round(remaining / burn_per_hour, 1)
            if remaining is not None and burn_per_hour > 0
            else None
        )
        windows.append({
            "window": lim.get("window"),
            "used_tokens": totals["tokens"],
            "max_tokens": max_tokens,
            "used_requests": totals["requests"],
            "max_requests": max_requests,
            "remaining": remaining,
            "pct": round(pct, 1),
            "burn_per_hour": burn_per_hour,
            "exhausts_in_hours": exhausts,
        })
        worst_pct = max(worst_pct, pct)

    in_cooldown = provider_id in _cooldown_ids()
    status = (
        "exhausted" if worst_pct >= 100.0
        else "warn" if (worst_pct >= WARN_PCT or in_cooldown)
        else "ok"
    )
    return {"windows": windows, "status": status, "worst_pct": round(worst_pct, 1)}


def local_capacity() -> dict:
    """Gross/net local compute from the live snapshot (VRAM, RAM, slot health)."""
    snap = compute_monitor.get_compute_snapshot()
    gpus = snap.get("gpus") or []
    cpu = snap.get("cpu") or {}
    slots = [
        {
            "id": s.get("id"),
            "role": s.get("role"),
            "model_id": s.get("model_id"),
            "port": s.get("port"),
            "running": bool(s.get("running")),
            "healthy": bool(s.get("healthy")),
            # ponytail: tok/s needs the model file size (extra host-helper hop);
            # add via speed_estimator when the dashboard needs it
            "est_tok_per_sec": None,
        }
        for s in snap.get("slots") or []
    ]
    return {
        "gross_vram_mb": sum(g.get("total_vram_mb", 0) for g in gpus),
        "free_vram_mb": sum(g.get("free_vram_mb", 0) for g in gpus),
        "ram_total_mb": cpu.get("ram_total_mb", 0),
        "ram_free_mb": cpu.get("ram_free_mb", 0),
        "slots": slots,
    }


def compute_budget() -> dict:
    """Full budget state for all providers — the dashboard/MCP payload."""
    registry = load_registry()
    providers: List[dict] = []
    for pid, cfg in sorted(registry.items(), key=lambda kv: (kv[1].get("priority", 50), kv[0])):
        entry = {
            "id": pid,
            "name": cfg.get("name", pid),
            "kind": cfg.get("kind", "subscription"),
            "invoke": cfg.get("invoke", "openai_compatible"),
            "base_url": cfg.get("base_url", ""),
            "default_model": cfg.get("default_model", ""),
            "priority": cfg.get("priority", 50),
            "enabled": cfg.get("enabled", True),
        }
        if entry["kind"] == "local":
            try:
                cap = local_capacity()
                entry["capacity"] = cap
                entry["status"] = "ok" if any(s["healthy"] for s in cap["slots"]) else "warn"
            except Exception as e:
                logger.warning("local capacity unavailable: %s", e)
                entry["capacity"] = {}
                entry["status"] = "warn"
        else:
            entry.update(provider_budget(pid, cfg))
        providers.append(entry)
    return {"ts": time.time(), "providers": providers}


# ---------------------------------------------------------------------------
# Provider matching (used by the litellm counting hook)
# ---------------------------------------------------------------------------

def match_provider(model: str, api_base: Optional[str] = None) -> Optional[str]:
    """Map a litellm model string / api_base to a provider id.

    api_base wins over model prefix; no match -> None (do not record —
    an invented "unknown" bucket would pollute the budgets).
    """
    registry = load_registry()
    base = (api_base or "").lower()
    if base:
        if any(m in base for m in _LOCAL_URL_MARKERS):
            return "local"
        for pid, cfg in registry.items():
            marker = (cfg.get("base_url_match") or "").lower()
            if not marker and cfg.get("base_url"):
                marker = (urlparse(cfg["base_url"]).hostname or "").lower()
            if marker and marker in base:
                return pid
    model_l = (model or "").lower()
    for pid, cfg in registry.items():
        for prefix in cfg.get("litellm_prefixes") or []:
            if prefix and model_l.startswith(prefix.lower()):
                return pid
    return None


# ---------------------------------------------------------------------------
# Routing (recommend-only) + decisions log
# ---------------------------------------------------------------------------

def _pending_tokens(provider_id: str, now: float) -> int:
    entries = [(exp, tok) for exp, tok in _reservations.get(provider_id, []) if exp > now]
    _reservations[provider_id] = entries
    return sum(tok for _, tok in entries)


def _reserve(provider_id: str, tokens: int, now: float) -> str:
    global _res_counter
    _res_counter += 1
    _reservations.setdefault(provider_id, []).append((now + RESERVATION_TTL_S, tokens))
    return f"r-{int(now)}-{_res_counter}"


def _budget_blocks(provider_id: str, cfg: dict, need: int, now: float) -> bool:
    """True when any declared window can't absorb `need` more tokens / 1 more request."""
    pending = _pending_tokens(provider_id, now)
    for lim in cfg.get("limits") or []:
        try:
            seconds = parse_window(lim.get("window", "1d"))
        except ValueError:
            continue
        totals = usage_ledger.window_totals(provider_id, seconds, now=now)
        max_tokens = lim.get("max_tokens")
        max_requests = lim.get("max_requests")
        if max_tokens and totals["tokens"] + pending + need > int(max_tokens):
            return True
        if max_requests and totals["requests"] + 1 > int(max_requests):
            return True
    return False


def _tightest_window(provider_id: str, cfg: dict, now: float, extra_tokens: int = 0) -> Optional[dict]:
    """Summary of the most-consumed window, optionally projected with extra tokens."""
    tightest = None
    for lim in cfg.get("limits") or []:
        max_tokens = lim.get("max_tokens")
        if not max_tokens:
            continue
        try:
            seconds = parse_window(lim.get("window", "1d"))
        except ValueError:
            continue
        used = usage_ledger.window_totals(provider_id, seconds, now=now)["tokens"] + extra_tokens
        pct = used / int(max_tokens) * 100
        if tightest is None or pct > tightest["pct"]:
            tightest = {
                "tightest_window": lim.get("window"),
                "used": used,
                "limit": int(max_tokens),
                "remaining": max(int(max_tokens) - used, 0),
                "pct": round(pct, 1),
            }
    return tightest


def route_task(
    task: str = "",
    role: str = "chat",
    est_input_tokens: int = 0,
    est_output_tokens: int = 0,
    quality: str = "best_available",
) -> dict:
    """Deterministic routing packet: which provider/model the caller should use.

    Recommend-only — the calling agent executes the request itself.
    """
    now = time.time()
    registry = load_registry()
    need = (int(est_input_tokens) + int(est_output_tokens)) or DEFAULT_NEED_TOKENS

    candidates = [(pid, cfg) for pid, cfg in registry.items() if cfg.get("enabled", True) is not False]
    if quality in ("fast", "cheap"):
        candidates.sort(key=lambda kv: (0 if kv[1].get("kind") == "local" else 1,
                                        kv[1].get("priority", 50), kv[0]))
    else:
        candidates.sort(key=lambda kv: (kv[1].get("priority", 50), kv[0]))

    cooldown = _cooldown_ids()
    survivors: List[tuple] = []  # (pid, cfg, local_slot_or_None)
    local_cap: Optional[dict] = None
    for pid, cfg in candidates:
        if pid in cooldown:
            continue
        if cfg.get("kind") == "local":
            if local_cap is None:
                try:
                    local_cap = local_capacity()
                except Exception:
                    continue
            slot = next(
                (s for s in local_cap["slots"] if s["role"] == role and s["healthy"]),
                None,
            )
            if slot is None or local_cap["free_vram_mb"] < MIN_LOCAL_FREE_VRAM_MB:
                continue
            survivors.append((pid, cfg, slot))
        else:
            if _budget_blocks(pid, cfg, need, now):
                continue
            survivors.append((pid, cfg, None))

    notes = ""
    if not survivors:
        packet: Dict[str, Any] = {
            "provider_id": None,
            "model": "",
            "base_url": "",
            "invoke": "",
            "expected_tokens": {"input": int(est_input_tokens), "output": int(est_output_tokens)},
            "budget_before": None,
            "budget_after": None,
            "fallback": [{"provider_id": pid, "model": cfg.get("default_model", "")}
                         for pid, cfg in candidates],
            "decision_reason": "all providers exhausted or cooling down",
            "notes": "wait for a window to roll over, or raise a declared limit",
            "reservation_id": None,
        }
    else:
        pid, cfg, slot = survivors[0]
        if cfg.get("kind") == "local":
            model = (slot or {}).get("model_id", "")
            port = (slot or {}).get("port")
            base_url = f"http://localhost:{port}/v1" if port else ""
        else:
            model = cfg.get("default_model", "")
            base_url = cfg.get("base_url", "")
        budget_before = _tightest_window(pid, cfg, now)
        budget_after = _tightest_window(pid, cfg, now, extra_tokens=need)
        if budget_after and budget_after["pct"] >= WARN_PCT:
            notes = (f"warning: this task lands the {budget_after['tightest_window']} "
                     f"window at {budget_after['pct']}%")
        packet = {
            "provider_id": pid,
            "model": model,
            "base_url": base_url,
            "invoke": cfg.get("invoke", "openai_compatible"),
            "expected_tokens": {"input": int(est_input_tokens), "output": int(est_output_tokens)},
            "budget_before": budget_before,
            "budget_after": budget_after,
            "fallback": [{"provider_id": p, "model": c.get("default_model", "")}
                         for p, c, _ in survivors[1:4]],
            "decision_reason": (
                f"priority {cfg.get('priority', 50)} provider with headroom in all windows"
                if cfg.get("kind") != "local"
                else f"healthy local slot for role '{role}' with free VRAM"
            ),
            "notes": notes,
            "reservation_id": _reserve(pid, need, now),
        }

    log_decision({
        "ts": now,
        "task": (task or "")[:200],
        "role": role,
        "quality": quality,
        "est_in": int(est_input_tokens),
        "est_out": int(est_output_tokens),
        "chosen": packet["provider_id"],
        "model": packet["model"],
        "reason": packet["decision_reason"],
        "fallback": [f["provider_id"] for f in packet["fallback"]],
    })
    return packet


def log_decision(decision: dict) -> None:
    """Append to data/route_decisions.jsonl — the seed for future observability
    (LangSmith/Harbor) integration. Single writer: the MCP process."""
    try:
        path = usage_ledger.DATA_DIR / "route_decisions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("decision log failed: %s", e)
