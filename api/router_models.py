"""
api/router_models.py — List models available for a router-mode slot.

Reads the slot's models_preset .ini file and returns per-model alias info,
including which model is currently set as the default (pre-loaded on startup).
"""
from __future__ import annotations

import configparser
import os

from flask import Request
from helpers.api import ApiHandler

_IMPORT_DONE = False
try:
    from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager
    _IMPORT_DONE = True
except ImportError:
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _plugin_root = os.path.dirname(_here)
    if _plugin_root not in sys.path:
        sys.path.insert(0, _plugin_root)
    from helpers.llama_cpp_manager import BackendManager
    _IMPORT_DONE = True


def _parse_preset(preset_path: str, models_dir: str) -> list[dict]:
    """Return a list of model dicts parsed from a .ini preset file."""
    if not preset_path or not os.path.exists(preset_path):
        return []
    cp = configparser.ConfigParser()
    cp.read(preset_path, encoding="utf-8")
    models = []
    for section in cp.sections():
        alias = cp.get(section, "alias", fallback=section)
        rel   = cp.get(section, "model", fallback="")
        ctx   = cp.get(section, "ctx_size", fallback="")
        is_embed = cp.getboolean(section, "embedding", fallback=False)
        ngl   = cp.get(section, "n_gpu_layers", fallback="")
        ck    = cp.get(section, "cache_type_k", fallback="")
        full_path = os.path.join(models_dir, rel) if (models_dir and rel) else rel
        models.append({
            "alias":      alias,
            "model_rel":  rel,
            "full_path":  full_path,
            "ctx_size":   ctx,
            "gpu_layers": ngl,
            "cache_type_k": ck,
            "is_embedding": is_embed,
        })
    return models


def _scan_dir(models_dir: str) -> list[dict]:
    """Fallback: scan directory for .gguf files."""
    if not models_dir or not os.path.isdir(models_dir):
        return []
    return [
        {
            "alias":        os.path.splitext(f)[0],
            "model_rel":    f,
            "full_path":    os.path.join(models_dir, f),
            "ctx_size":     "",
            "gpu_layers":   "",
            "cache_type_k": "",
            "is_embedding": False,
        }
        for f in sorted(os.listdir(models_dir))
        if f.endswith(".gguf")
    ]


class RouterModels(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        slot_id = input.get("slot_id", "")
        if not slot_id:
            return {"ok": False, "error": "slot_id is required"}

        mgr = BackendManager.get_instance()
        slot_cfg = mgr._slot_configs.get(slot_id)
        if not slot_cfg:
            return {"ok": False, "error": f"Slot '{slot_id}' not found"}
        if not slot_cfg.get("router_mode"):
            return {"ok": False, "error": f"Slot '{slot_id}' is not in router mode"}

        preset_path = slot_cfg.get("router_models_preset", "")
        models_dir  = slot_cfg.get("router_models_dir", "")
        current_default = slot_cfg.get("router_default_model", "")

        models = _parse_preset(preset_path, models_dir) or _scan_dir(models_dir)
        for m in models:
            m["is_default"] = m["alias"] == current_default

        return {
            "ok": True,
            "slot_id": slot_id,
            "current_default": current_default,
            "preset_path": preset_path,
            "models_dir": models_dir,
            "models": models,
        }
