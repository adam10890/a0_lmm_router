"""
API endpoint: /plugins/a0_lmm_router/lmm_slot_recommendations

Per-slot, role-aware model recommendations driven by the LIVE compute monitor.

This is the unified entrypoint that replaces the old recommend / hardware-scan
duplication. It pulls together:

    1. Live compute_monitor snapshot — gives us GPUs + RAM + slot definitions
    2. derive_tier_from_stats() — converts live stats to EIM + Tier (no scan)
    3. fleet_models.list_models() — installed-model manifest from host helper
    4. slot_recommender.suggest_for_all_slots() — picks per role × tier

Returns:
    {
        "ok": true,
        "tier": "T4",
        "eim_gb": 24.0,
        "eim_basis": "vram",
        "gpu_summary": "NVIDIA RTX 4090 (24.0 GB)",
        "ram_gb": 63.0,
        "models_dir": "C:/Users/.../models",
        "installed_count": 12,
        "installed_diagnostics": {                # surfaced when models = empty
            "host_reachable": true,
            "manifest_present": false,
            "scanned_dir": "...",
            "message": "..."
        },
        "slots": [
            {
                "slot_id": "slot_chat",
                "role": "chat",
                "current_model_id": "qwen3_5_9b",
                "current_size_gb": 5.7,
                "current_status": "UNDERSIZED",
                "current_status_reason": "...",
                "suggestions": [
                    { name, source, model_id?, repo_id?, filename?, install,
                      engine, quant, size_gb, aa_score, speed_class, reason },
                    ...
                ]
            },
            ...
        ]
    }
"""

from flask import Request

from helpers.api import ApiHandler


class LmmSlotRecommendations(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            from usr.plugins.a0_lmm_router.helpers.compute_monitor import (
                get_compute_snapshot,
            )
            from usr.plugins.a0_lmm_router.helpers.hardware_inspector import (
                derive_tier_from_stats,
            )
            from usr.plugins.a0_lmm_router.helpers.slot_recommender import (
                suggest_for_all_slots,
            )
            from usr.plugins.a0_lmm_router.helpers.fleet_models import list_models

            # 1. Live compute snapshot (already host-helper-aware)
            snap = get_compute_snapshot()

            # 2. Derive tier from live stats — no extra scan
            tier_info = derive_tier_from_stats(snap)
            tier = tier_info["tier"]

            # 3. Installed models from host helper. Surface diagnostics when empty.
            models_payload = list_models() or {}
            installed_models = models_payload.get("models", {}) or {}
            installed_diagnostics = None
            if not installed_models:
                installed_diagnostics = {
                    "host_reachable": not models_payload.get("_router_unreachable"),
                    "scanned_dir": models_payload.get("models_dir", ""),
                    "message": models_payload.get(
                        "message",
                        "Host helper returned no installed models. "
                        "Verify LLAMA_MODELS_DIR on the host and that .gguf "
                        "files exist there.",
                    ),
                }

            # 4. Slot suggestions
            slot_suggestions = suggest_for_all_slots(
                slots=snap.get("slots", []),
                tier=tier,
                installed_models=installed_models,
                max_per_slot=3,
            )

            # GPU summary string for header
            gpu_summary = ""
            if tier_info["gpus"]:
                parts = []
                for g in tier_info["gpus"]:
                    vram_gb = g.get("total_vram_mb", 0) / 1024.0
                    if vram_gb > 0:
                        parts.append(f'{g.get("name", "GPU")} ({vram_gb:.1f} GB)')
                    else:
                        parts.append(g.get("name", "GPU"))
                gpu_summary = ", ".join(parts)

            return {
                "ok": True,
                "tier": tier,
                "eim_gb": tier_info["eim_gb"],
                "eim_basis": tier_info["eim_basis"],
                "gpu_summary": gpu_summary,
                "ram_gb": tier_info["ram_gb"],
                "models_dir": models_payload.get("models_dir", ""),
                "installed_count": len(installed_models),
                "installed_diagnostics": installed_diagnostics,
                "slots": slot_suggestions,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "slots": [],
            }
