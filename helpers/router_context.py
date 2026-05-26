"""Resolve llama.cpp router context limits for Agent Zero prompt budgeting."""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger("a0_lmm_router.router_context")

DEFAULT_ROUTER_CTX = 65536
RESPONSE_TOKEN_RESERVE = 8192
EXTRAS_TEMPLATE_RESERVE = 2048
PROMPT_SAFETY_RATIO = 0.90


def _normalize_api_base(api_base: str) -> str:
    base = (api_base or "http://host.docker.internal:8080/v1").rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1" if base.endswith("/") else f"{base}/v1"
    return base


def fetch_router_model_ctx(model_name: str, api_base: str, timeout: float = 3.0) -> int | None:
    """Read n_ctx from router /v1/models for the active alias (chat/utility/embedding)."""
    url = f"{_normalize_api_base(api_base)}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        log.debug("router /v1/models unreachable: %s", exc)
        return None

    for row in payload.get("data") or []:
        if str(row.get("id", "")).lower() != str(model_name).lower():
            continue
        meta = row.get("meta") or {}
        n_ctx = meta.get("n_ctx")
        if isinstance(n_ctx, int) and n_ctx > 0:
            return n_ctx
    return None


def read_slot_context_size() -> int | None:
    """Fallback: slot context_size from llama_cpp_servers.yaml."""
    try:
        from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager
    except ImportError:
        return None
    try:
        mgr = BackendManager.get_instance()
        for cfg in getattr(mgr, "_slot_configs", {}).values():
            if cfg.get("router_mode") and cfg.get("enabled", True):
                size = int(cfg.get("context_size") or 0)
                if size > 0:
                    return size
    except Exception as exc:
        log.debug("slot context_size lookup failed: %s", exc)
    return None


def resolve_router_ctx_limit(model_cfg: dict[str, Any]) -> int:
    """Best-effort router n_ctx for the configured chat/utility model."""
    cfg_ctx = int(model_cfg.get("ctx_length") or 0)
    api_base = str(model_cfg.get("api_base") or os.environ.get("A0_LMM_ROUTER_API", ""))
    model_name = str(model_cfg.get("name") or "chat")

    live = fetch_router_model_ctx(model_name, api_base)
    if live:
        return live

    slot_ctx = read_slot_context_size()
    if slot_ctx:
        return slot_ctx

    if cfg_ctx > 0:
        return cfg_ctx

    env_ctx = int(os.environ.get("ROUTER_CTX_SIZE", "0") or "0")
    if env_ctx > 0:
        return env_ctx

    return DEFAULT_ROUTER_CTX


def estimate_extras_tokens(loop_data: Any) -> int:
    """Approximate tokens for prompt extras injected after this extension."""
    try:
        from helpers import dirty_json, tokens as tok
    except ImportError:
        return EXTRAS_TEMPLATE_RESERVE

    persistent = getattr(loop_data, "extras_persistent", None) or {}
    temporary = getattr(loop_data, "extras_temporary", None) or {}
    merged = {**persistent, **temporary}
    if not merged:
        return EXTRAS_TEMPLATE_RESERVE
    return EXTRAS_TEMPLATE_RESERVE + tok.approximate_prompt_tokens(
        dirty_json.stringify(merged)
    )


def history_token_budget(
    model_cfg: dict[str, Any],
    system_tokens: int,
    *,
    extras_tokens: int = 0,
) -> int:
    """Tokens available for conversation history after system, extras, and completion reserve."""
    router_ctx = resolve_router_ctx_limit(model_cfg)
    effective = int(router_ctx * PROMPT_SAFETY_RATIO)
    budget = effective - int(system_tokens) - int(extras_tokens) - RESPONSE_TOKEN_RESERVE
    return max(budget, 4096)
