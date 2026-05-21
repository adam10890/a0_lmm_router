"""
api/assign_model.py — Assign model to slot endpoint.

Unified flow:
  1. Calls host helper to rewrite .env + restart container via docker compose.
  2. Also updates llama_cpp_servers.yaml so Start/Stop/Status stay in sync.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from flask import Request
from helpers.api import ApiHandler

try:
    from usr.plugins.a0_lmm_router.helpers import fleet_models
except ImportError:
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _plugin_root = os.path.dirname(os.path.dirname(_here))
    if _plugin_root not in sys.path:
        sys.path.insert(0, _plugin_root)
    from helpers import fleet_models

logger = logging.getLogger(__name__)

# Role → slot_id mapping (matches docker-compose service names and YAML ids)
_ROLE_TO_SLOT_ID = {
    "chat": "slot_chat",
    "utility": "slot_utility",
    "embedding": "slot_embedding",
    "embed": "slot_embedding",
    "vision": "slot_vision",
    "reasoning": "slot_reasoning",
}


def _resolve_yaml_path() -> str:
    """Find llama_cpp_servers.yaml using same logic as compute_monitor."""
    env_conf = os.environ.get("A0_LMM_ROUTER_CONFIG", "").strip()
    if env_conf and os.path.exists(env_conf):
        return env_conf
    here = Path(__file__).resolve()
    plugin_conf = str(here.parent.parent / "conf" / "llama_cpp_servers.yaml")
    root_conf = str(here.parents[4] / "conf" / "llama_cpp_servers.yaml")
    if os.path.exists(root_conf):
        return root_conf
    return plugin_conf


def _update_yaml_slot(yaml_path: str, slot_id: str, model_id: str) -> bool:
    """Update model_id for a slot in llama_cpp_servers.yaml."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        found = False
        for slot in config.get("active_slots", []):
            if slot and slot.get("id") == slot_id:
                slot["model_id"] = model_id
                found = True
                break

        if not found:
            logger.warning("Slot '%s' not found in %s", slot_id, yaml_path)
            return False

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        logger.info("Updated %s → model_id=%s in %s", slot_id, model_id, yaml_path)
        return True
    except Exception as e:
        logger.error("Failed to update YAML: %s", e)
        return False


class AssignModel(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        """Assign a model to a slot — unified: .env + YAML + container restart."""
        try:
            slot = input.get("slot", "").strip()
            model_id = input.get("model_id", "").strip()
            apply_now = input.get("apply_now", True)

            if not slot:
                return {"ok": False, "error": "slot is required"}
            if not model_id:
                return {"ok": False, "error": "model_id is required"}

            # 1) Call host helper (rewrites .env + restarts container)
            result = fleet_models.assign_model(slot, model_id, apply_now)

            if result.get("_router_unreachable"):
                return {
                    "ok": False,
                    "error": (
                        "Host helper unreachable. Is lmm_host_helper.py running on the host? "
                        "Run: python usr/plugins/a0_lmm_router/tools/lmm_host_helper.py"
                    ),
                }

            if not result.get("ok"):
                return {"ok": False, "error": result.get("error", "Unknown error")}

            # 2) Also update llama_cpp_servers.yaml so status display stays in sync
            slot_id = _ROLE_TO_SLOT_ID.get(slot.lower(), f"slot_{slot}")
            yaml_path = _resolve_yaml_path()
            yaml_updated = _update_yaml_slot(yaml_path, slot_id, model_id)

            return {
                "ok": True,
                "message": "Model assigned",
                "slot": slot,
                "slot_id": slot_id,
                "model_id": model_id,
                "restarted": result.get("restarted", False),
                "yaml_updated": yaml_updated,
                "context_calculation": result.get("context_calculation"),
            }
        except Exception as e:
            logger.exception("assign_model failed")
            return {"ok": False, "error": str(e)}
