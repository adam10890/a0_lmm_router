"""api/router_aliases.py - Live role bindings for llama.cpp Router Mode."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import configparser
from pathlib import Path, PurePosixPath

import yaml

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from flask import Request
try:
    from helpers.api import ApiHandler
except ImportError:
    class ApiHandler:  # type: ignore[no-redef]
        pass

try:
    from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager
    from usr.plugins.a0_lmm_router.helpers import fleet_models
    from usr.plugins.a0_lmm_router.helpers.router_probe import detect_fleet_http
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    _plugin_root = os.path.dirname(_here)
    if _plugin_root not in sys.path:
        sys.path.insert(0, _plugin_root)
    from helpers.llama_cpp_manager import BackendManager
    from helpers import fleet_models
    from helpers.router_probe import detect_fleet_http

ROLES = ("chat", "utility", "embedding")


def _resolve_conf_path() -> str:
    """Locate llama_cpp_servers.yaml (env override → root → plugin fallback)."""
    env_conf = os.environ.get("A0_LMM_ROUTER_CONFIG", "").strip()
    if env_conf and os.path.exists(env_conf):
        return env_conf
    here = Path(__file__).resolve()
    plugin_conf = str(here.parents[1] / "conf" / "llama_cpp_servers.yaml")
    root_conf = str(here.parents[4] / "conf" / "llama_cpp_servers.yaml")
    return root_conf if os.path.exists(root_conf) else plugin_conf


def _load_config() -> dict:
    try:
        with open(_resolve_conf_path(), "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _synthesize_router_slot_cfg() -> dict | None:
    """When the config has no router slot, detect a live router over HTTP and
    build an ad-hoc slot_cfg so the dashboard still gets live role bindings.

    Returns None when no router is reachable.
    """
    cfg = _load_config()
    lmm_hosts = (cfg.get("global", {}) or {}).get("lmm_hosts", {}) or {}
    detected = detect_fleet_http(lmm_hosts or None)
    if detected.get("mode") != "router" or not detected.get("router"):
        return None
    defaults = cfg.get("slot_defaults", {}) or {}
    return {
        "id": "slot_router",
        "port": detected.get("primary_port", 8080),
        "router_mode": True,
        "router_models_dir": str(defaults.get("router_models_dir", "") or ""),
        "router_models_preset": str(defaults.get("router_models_preset", "") or ""),
        "_detected_via": "http",
    }


def _parse_preset_file(preset_path: str) -> list[dict]:
    if not preset_path or not os.path.exists(preset_path):
        return []
    cp = configparser.ConfigParser()
    cp.read(preset_path, encoding="utf-8")
    rows = []
    for section in cp.sections():
        rows.append({
            "alias": cp.get(section, "alias", fallback=section),
            "model_path": cp.get(section, "model", fallback=""),
            "ctx_size": cp.get(section, "ctx_size", fallback=cp.get(section, "ctx-size", fallback="")),
        })
    return rows


def _extract_arg(args: list, name: str) -> str:
    for idx, value in enumerate(args or []):
        if value == name and idx + 1 < len(args):
            return str(args[idx + 1])
    return ""


def _extract_preset_value(preset: str, key: str) -> str:
    prefix = f"{key.lower()} "
    for line in (preset or "").splitlines():
        text = line.strip()
        if not text or text.startswith(";") or "=" not in text:
            continue
        k, v = text.split("=", 1)
        if k.strip().lower() == key.lower():
            return v.strip()
        if text.lower().startswith(prefix):
            return text[len(prefix):].strip()
    return ""


def _binding(alias: str, status: dict | None = None) -> dict:
    status = status or {}
    args = status.get("args") or []
    preset = str(status.get("preset") or "")
    model_path = _extract_arg(args, "--model") or _extract_preset_value(preset, "model")
    ctx = _extract_arg(args, "--ctx-size") or _extract_arg(args, "-c") or _extract_preset_value(preset, "ctx-size")
    try:
        ctx_size = int(ctx) if ctx else None
    except ValueError:
        ctx_size = None
    port = status.get("port")
    try:
        port = int(port) if port is not None else None
    except (TypeError, ValueError):
        port = None
    return {
        "alias": alias,
        "loaded": str(status.get("value") or "").lower() == "loaded",
        "port": port,
        "model_path": model_path,
        "model_filename": PurePosixPath(model_path).name if model_path else "",
        "ctx_size": ctx_size,
        "preset": preset,
    }


def _apply_slot_defaults(bindings: dict[str, dict], slot_cfg: dict) -> None:
    """Fill missing router port when alias is loaded but /v1/models omits it."""
    try:
        default_port = int(slot_cfg.get("port") or 8080)
    except (TypeError, ValueError):
        default_port = 8080
    for row in bindings.values():
        if row.get("loaded") and row.get("port") is None:
            row["port"] = default_port


def parse_router_models_payload(payload: dict) -> dict[str, dict]:
    by_alias = {role: _binding(role) for role in ROLES}
    for item in payload.get("data", []) if isinstance(payload, dict) else []:
        alias = str(item.get("id") or item.get("alias") or "")
        if alias not in ROLES:
            continue
        by_alias[alias] = _binding(alias, item.get("status") or {})
    return by_alias


def _router_url(slot_cfg: dict) -> str:
    port = int(slot_cfg.get("port") or 8080)
    return f"http://host.docker.internal:{port}/v1/models"


def _fetch_router_models(slot_cfg: dict) -> dict:
    req = urllib.request.Request(_router_url(slot_cfg))
    with urllib.request.urlopen(req, timeout=3.0) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _fallback_bindings(slot_cfg: dict) -> dict[str, dict]:
    preset_path = slot_cfg.get("router_models_preset", "")
    models_dir = slot_cfg.get("router_models_dir", "")
    bindings = {role: _binding(role) for role in ROLES}
    for item in _parse_preset_file(preset_path):
        alias = item.get("alias")
        if alias in bindings:
            model_path = item.get("model_path") or ""
            bindings[alias].update({
                "model_path": model_path,
                "model_filename": PurePosixPath(model_path).name if model_path else "",
                "ctx_size": int(item["ctx_size"]) if str(item.get("ctx_size") or "").isdigit() else None,
            })
    return bindings


class RouterAliases(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        slot_id = input.get("slot_id", "slot_router")

        # 1) Prefer the configured slot (BackendManager view of the YAML).
        slot_cfg = None
        try:
            mgr = BackendManager.get_instance()
            slot_cfg = mgr._slot_configs.get(slot_id)
        except Exception:
            slot_cfg = None

        detected_via_http = False
        # 2) Fall back to HTTP reality: if the config has no router slot (or it
        #    is not flagged router_mode), but a router is actually answering on
        #    the chat port, synthesize a slot_cfg so the dashboard still works.
        #    This is what lets the plugin "see" a router started out-of-band.
        if not slot_cfg or not slot_cfg.get("router_mode"):
            synthesized = _synthesize_router_slot_cfg()
            if synthesized is not None:
                slot_cfg = synthesized
                detected_via_http = True

        if not slot_cfg:
            return {
                "ok": False,
                "error": f"Slot '{slot_id}' not found and no live router detected",
            }
        if not slot_cfg.get("router_mode"):
            return {"ok": False, "error": f"Slot '{slot_id}' is not in router mode"}

        source = "live"
        try:
            bindings = parse_router_models_payload(_fetch_router_models(slot_cfg))
        except Exception:
            source = "preset"
            bindings = _fallback_bindings(slot_cfg)

        _apply_slot_defaults(bindings, slot_cfg)

        models_result = fleet_models.list_models()
        return {
            "ok": True,
            "slot_id": slot_id,
            "source": source,
            "detected_via_http": detected_via_http,
            "roles": [bindings[role] for role in ROLES],
            "bindings": bindings,
            "models": models_result.get("models", {}) if models_result.get("ok") else {},
            "models_dir": models_result.get("models_dir", ""),
        }
