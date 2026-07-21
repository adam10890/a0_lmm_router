"""
API endpoint: /plugins/a0_lmm_router/lmm_providers

Compute-provider registry + live budget state for the dashboard.

Actions (POST body):
    {action: "list"}                              -> providers with budget/capacity
    {action: "save", provider: {...}}             -> add/update in provider_limits.yaml
    {action: "delete", provider_id: "..."}        -> remove from provider_limits.yaml
    {action: "toggle", provider_id, enabled}      -> persist enabled flag
"""
from __future__ import annotations

import logging
import re

from flask import Request
from helpers.api import ApiHandler

try:
    from usr.plugins.a0_lmm_router.helpers import budget_engine, usage_ledger
except ImportError:
    import os
    import sys
    _plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _plugin_root not in sys.path:
        sys.path.insert(0, _plugin_root)
    from helpers import budget_engine, usage_ledger

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,39}$")

# Whitelist of keys persisted to provider_limits.yaml (trust boundary).
_CFG_KEYS = ("name", "kind", "invoke", "base_url", "api_key_env", "default_model",
             "priority", "enabled", "litellm_prefixes", "base_url_match", "limits")


def _validate_provider(pid: str, provider: dict) -> str:
    """Return an error message, or '' when valid."""
    if not _ID_RE.match(pid):
        return "id must be a slug: lowercase letters/digits/_/- (max 40 chars)"
    if pid == "local":
        return "id 'local' is reserved for the built-in local fleet"
    limits = provider.get("limits") or []
    if not isinstance(limits, list):
        return "limits must be a list"
    max_window_seconds = usage_ledger.MAX_AGE_DAYS * 86400
    for lim in limits:
        if not isinstance(lim, dict):
            return "each limit must be an object"
        try:
            seconds = budget_engine.parse_window(lim.get("window", ""))
        except ValueError as e:
            return str(e)
        if seconds > max_window_seconds:
            return (f"window '{lim.get('window')}' exceeds the "
                     f"{usage_ledger.MAX_AGE_DAYS}-day usage ledger retention")
        max_tokens = lim.get("max_tokens")
        max_requests = lim.get("max_requests")
        if max_tokens is None and max_requests is None:
            return f"limit '{lim.get('window')}' needs max_tokens and/or max_requests"
        for key, val in (("max_tokens", max_tokens), ("max_requests", max_requests)):
            if val is not None and (not isinstance(val, int) or val < 1):
                return f"{key} must be a positive integer (>= 1)"
    priority = provider.get("priority", 50)
    if not isinstance(priority, int) or not (1 <= priority <= 99):
        return "priority must be an integer between 1 and 99"
    return ""


def _clean_cfg(provider: dict) -> dict:
    cfg = {k: provider[k] for k in _CFG_KEYS if k in provider and provider[k] is not None}
    cfg.setdefault("kind", "subscription")
    # Drop nulls inside limits so the YAML stays tidy
    cfg["limits"] = [
        {k: v for k, v in (lim or {}).items() if v is not None}
        for lim in (cfg.get("limits") or [])
    ]
    return cfg


class LmmProviders(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            action = str(input.get("action") or "list").strip()

            if action == "list":
                return {"ok": True, **budget_engine.compute_budget()}

            if action == "save":
                provider = input.get("provider") or {}
                pid = str(provider.get("id", "")).strip().lower()
                err = _validate_provider(pid, provider)
                if err:
                    return {"ok": False, "error": err}
                budget_engine.save_provider(pid, _clean_cfg(provider))
                return {"ok": True, "id": pid}

            if action == "delete":
                pid = str(input.get("provider_id", "")).strip()
                if not budget_engine.delete_provider(pid):
                    return {"ok": False,
                            "error": f"provider '{pid}' not found in provider_limits.yaml"}
                return {"ok": True, "id": pid}

            if action == "toggle":
                pid = str(input.get("provider_id", "")).strip()
                enabled = bool(input.get("enabled", True))
                registry = budget_engine.load_registry()
                if pid not in registry:
                    return {"ok": False, "error": f"unknown provider '{pid}'"}
                cfg = registry[pid]
                cfg["enabled"] = enabled
                budget_engine.save_provider(pid, cfg)
                return {"ok": True, "id": pid, "enabled": enabled}

            return {"ok": False, "error": f"unknown action '{action}'"}
        except Exception as e:
            logger.exception("lmm_providers failed")
            return {"ok": False, "error": str(e)}
