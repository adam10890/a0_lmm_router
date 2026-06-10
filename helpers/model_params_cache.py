"""Local cache for per-model llama.cpp parameters (Router Mode preset planning).

On first run (or when the GGUF / hardware fingerprint changes), plans are
computed via ``context_planner.plan_model_context``. Results are stored under
``data/model_params_cache.json`` so ``render_router_preset.py`` and the host
helper do not re-plan from scratch on every ignite.

``last_success_by_role`` keeps the last model+params that worked per role; when
planning a new GGUF for that role, global llama options (KV quant, batch, etc.)
are seeded from that entry before recomputing context size for the new weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    from usr.plugins.a0_lmm_router.helpers.context_planner import (
        ContextPlan,
        available_vram_from_env,
        container_path_to_host,
        effective_ratio_for_role,
        min_ctx_for_role,
        plan_model_context,
        response_reserve_for_role,
        vram_safety_margin_gb,
    )
    from usr.plugins.a0_lmm_router.helpers.context_calculator import read_gguf_metadata
except ImportError:  # pragma: no cover
    from context_planner import (  # type: ignore
        ContextPlan,
        available_vram_from_env,
        container_path_to_host,
        effective_ratio_for_role,
        min_ctx_for_role,
        plan_model_context,
        response_reserve_for_role,
        vram_safety_margin_gb,
    )
    from context_calculator import read_gguf_metadata  # type: ignore


CACHE_VERSION = 1
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "model_params_cache.json"
VRAM_FP_TOLERANCE_GB = 1.5


def default_cache_path() -> Path:
    raw = (os.environ.get("A0_LMM_PARAMS_CACHE_PATH") or "").strip()
    return Path(raw) if raw else DEFAULT_CACHE_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hardware_fingerprint(available_vram_gb: Optional[float]) -> dict[str, Any]:
    vram = round(float(available_vram_gb or 0.0), 1)
    return {"total_vram_gb": vram}


def _vram_compatible(cached_hw: Mapping[str, Any], current_hw: Mapping[str, Any]) -> bool:
    old = float(cached_hw.get("total_vram_gb") or 0.0)
    new = float(current_hw.get("total_vram_gb") or 0.0)
    if old <= 0 or new <= 0:
        return True
    return abs(old - new) <= VRAM_FP_TOLERANCE_GB


def _file_fingerprint(host_model_path: str) -> dict[str, Any]:
    path = Path(host_model_path)
    if not path.is_file():
        return {"exists": False, "path": host_model_path}
    stat = path.stat()
    meta = read_gguf_metadata(str(path))
    return {
        "exists": True,
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "n_ctx_train": meta.get("n_ctx_train"),
        "n_layer": meta.get("n_layer"),
        "n_embd": meta.get("n_embd"),
        "file_size_gb": meta.get("file_size_gb"),
    }


def model_cache_key(host_model_path: str, fingerprint: Optional[Mapping[str, Any]] = None) -> str:
    fp = dict(fingerprint or _file_fingerprint(host_model_path))
    blob = json.dumps(fp, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _plan_to_dict(plan: ContextPlan) -> dict[str, Any]:
    return plan.as_dict()


def _global_options_from_env(env: Mapping[str, str]) -> dict[str, str]:
    return {
        "n-gpu-layers": env.get("ROUTER_GPU_LAYERS", "999"),
        "batch-size": env.get("ROUTER_BATCH_SIZE", "512"),
        "ubatch-size": env.get("ROUTER_UBATCH_SIZE", "1024"),
        "cache-type-k": env.get("ROUTER_CACHE_TYPE_K", "q8_0"),
        "cache-type-v": env.get("ROUTER_CACHE_TYPE_V", "q8_0"),
        "flash-attn": env.get("ROUTER_FLASH_ATTN", "on"),
    }


def _per_model_options_from_env(env: Mapping[str, str], role: str) -> dict[str, str]:
    """Role-specific llama.cpp overrides (optional env vars)."""
    role_key = role.upper().replace("EMBED", "EMBED")
    if role == "embed":
        role_key = "EMBED"
    opts: dict[str, str] = {}
    for opt, env_suffix in (
        ("batch-size", "BATCH_SIZE"),
        ("ubatch-size", "UBATCH_SIZE"),
        ("cache-type-k", "CACHE_TYPE_K"),
        ("cache-type-v", "CACHE_TYPE_V"),
        ("flash-attn", "FLASH_ATTN"),
        ("n-gpu-layers", "GPU_LAYERS"),
    ):
        val = (env.get(f"ROUTER_{role_key}_{env_suffix}") or "").strip()
        if val:
            opts[opt] = val
    return opts


def load_cache(path: Optional[Path] = None) -> dict[str, Any]:
    cache_path = path or default_cache_path()
    if not cache_path.is_file():
        return {"version": CACHE_VERSION, "hardware_fingerprint": {}, "last_success_by_role": {}, "models": {}}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": CACHE_VERSION, "hardware_fingerprint": {}, "last_success_by_role": {}, "models": {}}
    if not isinstance(data, dict):
        return {"version": CACHE_VERSION, "hardware_fingerprint": {}, "last_success_by_role": {}, "models": {}}
    data.setdefault("version", CACHE_VERSION)
    data.setdefault("last_success_by_role", {})
    data.setdefault("models", {})
    return data


def save_cache(data: dict[str, Any], path: Optional[Path] = None) -> Path:
    cache_path = path or default_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(cache_path)
    return cache_path


def _dict_to_plan(data: Mapping[str, Any], *, container_model_path: str) -> ContextPlan:
    return ContextPlan(
        alias=str(data.get("alias") or ""),
        role=str(data.get("role") or "chat"),
        model_path=container_model_path,
        min_ctx=int(data.get("min_ctx") or 8192),
        hard_ctx=int(data.get("hard_ctx") or 8192),
        effective_ctx=int(data.get("effective_ctx") or 8192),
        response_reserve=int(data.get("response_reserve") or 2048),
        effective_ratio=float(data.get("effective_ratio") or 0.7),
        n_ctx_train=data.get("n_ctx_train"),
        planned_vram_gb=data.get("planned_vram_gb"),
        kv_cache_gb=data.get("kv_cache_gb"),
        no_capacity=bool(data.get("no_capacity")),
        reason=str(data.get("reason") or ""),
    )


def get_cached_entry(
    *,
    alias: str,
    role: str,
    host_model_path: str,
    container_model_path: str,
    env: Mapping[str, str],
    available_vram_gb: Optional[float],
    cache: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Return a full cache record if still valid, else None."""
    data = cache if cache is not None else load_cache()
    fp = _file_fingerprint(host_model_path)
    if not fp.get("exists"):
        return None
    key = model_cache_key(host_model_path, fp)
    entry = data.get("models", {}).get(key)
    if not entry:
        return None
    if entry.get("alias") != alias or entry.get("role") != role:
        return None
    if entry.get("container_model_path") != container_model_path:
        return None
    stored_fp = entry.get("file_fingerprint") or {}
    if stored_fp.get("size") != fp.get("size") or stored_fp.get("mtime_ns") != fp.get("mtime_ns"):
        return None
    hw = _hardware_fingerprint(available_vram_gb)
    if not _vram_compatible(entry.get("hardware_fingerprint") or {}, hw):
        return None
    role_min = min_ctx_for_role(role, env)
    if int(entry.get("min_ctx_at_plan") or 0) != role_min:
        return None
    return entry


def plan_model_with_cache(
    *,
    alias: str,
    role: str,
    container_model_path: str,
    env: Mapping[str, str],
    available_vram_gb: Optional[float] = None,
    other_resident_vram_gb: float = 0.0,
    force_refresh: bool = False,
    cache: Optional[dict[str, Any]] = None,
) -> tuple[ContextPlan, dict[str, str], dict[str, str], str]:
    """Plan context; use cache when valid.

    Returns (plan, global_options, per_model_options, source).
    source is one of: cache, computed, seeded_from_last_success.
    """
    env = env or os.environ
    if available_vram_gb is None:
        available_vram_gb = available_vram_from_env(env)

    models_dir = (env.get("LLAMA_MODELS_DIR") or "").strip()
    host_model = container_path_to_host(container_model_path, models_dir)
    data = cache if cache is not None else load_cache()
    hw = _hardware_fingerprint(available_vram_gb)

    global_options = _global_options_from_env(env)
    per_model_options = _per_model_options_from_env(env, role)
    source = "computed"

    if not force_refresh:
        hit = get_cached_entry(
            alias=alias,
            role=role,
            host_model_path=host_model,
            container_model_path=container_model_path,
            env=env,
            available_vram_gb=available_vram_gb,
            cache=data,
        )
        if hit:
            plan = _dict_to_plan(hit.get("plan") or {}, container_model_path=container_model_path)
            return (
                plan,
                dict(hit.get("global_options") or global_options),
                dict(hit.get("per_model_options") or per_model_options),
                "cache",
            )

    last = (data.get("last_success_by_role") or {}).get(role) or (data.get("last_success_by_role") or {}).get(
        "embedding" if role == "embed" else role
    )
    if isinstance(last, dict) and last.get("global_options"):
        global_options = dict(last["global_options"])
        per_model_options = dict(last.get("per_model_options") or per_model_options)
        source = "seeded_from_last_success"

    plan = plan_model_context(
        alias=alias,
        role=role,
        model_path=host_model,
        min_ctx=min_ctx_for_role(role, env),
        available_vram_gb=available_vram_gb,
        other_resident_vram_gb=other_resident_vram_gb,
        cache_type_k=global_options.get("cache-type-k", "q8_0"),
        cache_type_v=global_options.get("cache-type-v", "q8_0"),
        parallel=int(env.get("ROUTER_PARALLEL", "1") or "1"),
        effective_ratio=effective_ratio_for_role(role, env),
        response_reserve=response_reserve_for_role(role, env),
        vram_margin_gb=vram_safety_margin_gb(env),
    )

    plan = ContextPlan(
        alias=plan.alias,
        role=plan.role,
        model_path=container_model_path,
        min_ctx=plan.min_ctx,
        hard_ctx=plan.hard_ctx,
        effective_ctx=plan.effective_ctx,
        response_reserve=plan.response_reserve,
        effective_ratio=plan.effective_ratio,
        n_ctx_train=plan.n_ctx_train,
        planned_vram_gb=plan.planned_vram_gb,
        kv_cache_gb=plan.kv_cache_gb,
        no_capacity=plan.no_capacity,
        reason=plan.reason,
    )
    return plan, global_options, per_model_options, source


def store_plan_in_cache(
    *,
    alias: str,
    role: str,
    container_model_path: str,
    host_model_path: str,
    plan: ContextPlan,
    global_options: Mapping[str, str],
    per_model_options: Mapping[str, str],
    source: str,
    env: Mapping[str, str],
    available_vram_gb: Optional[float],
    cache: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    data = cache if cache is not None else load_cache()
    fp = _file_fingerprint(host_model_path)
    key = model_cache_key(host_model_path, fp)
    hw = _hardware_fingerprint(available_vram_gb)
    data["hardware_fingerprint"] = hw
    data["updated_at"] = _utc_now()
    data.setdefault("models", {})
    data["models"][key] = {
        "alias": alias,
        "role": role,
        "container_model_path": container_model_path,
        "host_model_path": host_model_path,
        "file_fingerprint": fp,
        "hardware_fingerprint": hw,
        "min_ctx_at_plan": min_ctx_for_role(role, env),
        "plan": _plan_to_dict(plan),
        "global_options": dict(global_options),
        "per_model_options": dict(per_model_options),
        "planned_at": _utc_now(),
        "source": source,
    }
    save_cache(data)
    return data


def record_role_success(
    role: str,
    *,
    container_model_path: str,
    env: Mapping[str, str],
    available_vram_gb: Optional[float] = None,
    cache: Optional[dict[str, Any]] = None,
) -> None:
    """Remember params for the model that last ran successfully on this role."""
    env = env or os.environ
    if available_vram_gb is None:
        available_vram_gb = available_vram_from_env(env)
    models_dir = (env.get("LLAMA_MODELS_DIR") or "").strip()
    host_model = container_path_to_host(container_model_path, models_dir)

    data = cache if cache is not None else load_cache()
    key = model_cache_key(host_model)
    entry = (data.get("models") or {}).get(key)
    if not entry:
        plan, global_options, per_model_options, source = plan_model_with_cache(
            alias=role if role != "embed" else "embedding",
            role=role,
            container_model_path=container_model_path,
            env=env,
            available_vram_gb=available_vram_gb,
            force_refresh=False,
            cache=data,
        )
        store_plan_in_cache(
            alias=plan.alias,
            role=plan.role,
            container_model_path=container_model_path,
            host_model_path=host_model,
            plan=plan,
            global_options=global_options,
            per_model_options=per_model_options,
            source=source,
            env=env,
            available_vram_gb=available_vram_gb,
            cache=data,
        )
        entry = (data.get("models") or {}).get(key) or {}

    data.setdefault("last_success_by_role", {})
    data["last_success_by_role"][role] = {
        "container_model_path": container_model_path,
        "host_model_path": host_model,
        "model_cache_key": key,
        "recorded_at": _utc_now(),
        "plan": entry.get("plan"),
        "global_options": entry.get("global_options"),
        "per_model_options": entry.get("per_model_options"),
    }
    save_cache(data)


# Standard Router Mode aliases from docker-compose.lmm.env
FLEET_ALIASES: tuple[tuple[str, str, str], ...] = (
    ("chat", "chat", "CHAT_MODEL_PATH"),
    ("utility", "utility", "UTILITY_MODEL_PATH"),
    ("embedding", "embed", "EMBED_MODEL_PATH"),
)


def _truthy_env(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = str(env.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _router_models_max(env: Mapping[str, str]) -> int:
    try:
        return max(1, int(str(env.get("ROUTER_MODELS_MAX") or "2").strip()))
    except ValueError:
        return 2


def _utility_follows_chat(env: Mapping[str, str]) -> bool:
    return _truthy_env(env, "A0_LMM_UTILITY_FOLLOWS_CHAT", False)


def warm_fleet_params(
    env: Mapping[str, str],
    *,
    force_refresh: bool = False,
    write_cache: bool = True,
) -> dict[str, Any]:
    """Plan (or load from cache) all fixed fleet slots; optionally persist cache."""
    env = dict(env)
    if _utility_follows_chat(env) and not str(env.get("UTILITY_CTX_SIZE") or "").strip():
        chat_ctx = str(env.get("CHAT_CTX_SIZE") or "").strip()
        if chat_ctx:
            env["UTILITY_CTX_SIZE"] = chat_ctx

    available_vram_gb = available_vram_from_env(env)
    data = load_cache()
    entries: list[ContextPlan] = []
    per_entry_options: dict[str, dict[str, str]] = {}
    stats = {"cached": 0, "computed": 0, "seeded_from_last_success": 0}
    global_options_merged: dict[str, str] = _global_options_from_env(env)

    serial_router = _router_models_max(env) <= 1
    resident_model_paths: set[str] = set()
    resident_vram = 0.0
    for alias, role, model_var in FLEET_ALIASES:
        effective_model_var = model_var
        if alias == "utility" and _utility_follows_chat(env):
            effective_model_var = "CHAT_MODEL_PATH"
        container_model = (env.get(effective_model_var) or "").strip()
        if not container_model:
            continue
        plan, g_opts, p_opts, source = plan_model_with_cache(
            alias=alias,
            role=role,
            container_model_path=container_model,
            env=env,
            available_vram_gb=available_vram_gb,
            other_resident_vram_gb=0.0 if serial_router else resident_vram,
            force_refresh=force_refresh,
            cache=data,
        )
        if source == "cache":
            stats["cached"] += 1
        elif source == "seeded_from_last_success":
            stats["seeded_from_last_success"] += 1
        else:
            stats["computed"] += 1
        entries.append(plan)
        if p_opts:
            per_entry_options[alias] = dict(p_opts)
        if alias == "chat":
            global_options_merged = dict(g_opts)

        if write_cache:
            models_dir = (env.get("LLAMA_MODELS_DIR") or "").strip()
            host_model = container_path_to_host(container_model, models_dir)
            data = store_plan_in_cache(
                alias=alias,
                role=role,
                container_model_path=container_model,
                host_model_path=host_model,
                plan=plan,
                global_options=g_opts,
                per_model_options=p_opts,
                source=source,
                env=env,
                available_vram_gb=available_vram_gb,
                cache=data,
            )

        if plan.planned_vram_gb:
            planned = float(plan.planned_vram_gb)
            if serial_router:
                if container_model not in resident_model_paths:
                    resident_vram = max(resident_vram, planned)
                    resident_model_paths.add(container_model)
            else:
                resident_vram += planned

    return {
        "ok": True,
        "entries": entries,
        "global_options": global_options_merged,
        "per_entry_options": per_entry_options,
        "stats": stats,
        "cache_path": str(default_cache_path()),
        "hardware_fingerprint": _hardware_fingerprint(available_vram_gb),
        "resident_vram_gb": round(resident_vram, 2),
        "router_models_max": _router_models_max(env),
        "utility_follows_chat": _utility_follows_chat(env),
    }


def render_fleet_preset_to_file(
    env: Mapping[str, str],
    output_path: Path,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Warm cache, render models_preset.ini, return warm stats."""
    try:
        from usr.plugins.a0_lmm_router.helpers.context_planner import render_preset
    except ImportError:
        from context_planner import render_preset  # type: ignore

    warm = warm_fleet_params(env, force_refresh=force_refresh, write_cache=True)
    preset = render_preset(
        warm["entries"],
        global_options=warm["global_options"],
        per_entry_options=warm["per_entry_options"],
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(preset, encoding="utf-8")
    warm["preset_path"] = str(output_path)
    return warm
