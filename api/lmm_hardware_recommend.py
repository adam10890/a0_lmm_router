"""
API endpoint: /plugins/a0_lmm_router/lmm_hardware_recommend

Implements the local-llm-recommender skill (snapshot 11-05-2026) as a single
JSON endpoint:

    POST /api/plugins/a0_lmm_router/lmm_hardware_recommend
    body: {}  (no params)

Returns:
    {
        "ok": true,
        "snapshot": "11-05-2026",
        "hardware": { ... HardwareReport ... },
        "picks": {
            "tier": "T6",
            "comfortable": { name, params_b, quant, size_gb, aa_score,
                             speed_class, install, reason, ... } | null,
            "balanced":    { ... } | null,
            "stretch":     { ... } | null
        },
        "notes": [ "string", ... ]    # caveats + warnings (host + tier)
    }

The endpoint orchestrates two helpers:
    helpers.hardware_inspector.scan_hardware()   → calls host helper
    helpers.tier_catalog.pick_three(tier)         → maps tier to 3 picks
"""

from flask import Request

from helpers.api import ApiHandler


class LmmHardwareRecommend(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            from usr.plugins.a0_lmm_router.helpers.hardware_inspector import (
                scan_hardware, report_to_dict,
            )
            from usr.plugins.a0_lmm_router.helpers.tier_catalog import pick_three

            report = scan_hardware()
            if not report.ok:
                return {
                    "ok": False,
                    "error": report.error or "Hardware scan failed",
                    "hardware": report_to_dict(report),
                }

            picks = pick_three(report.tier, disk_free_gb=report.disk_free_gb)

            # Build final notes: combine inspector notes + per-pick warnings
            notes = list(report.notes)
            for slot_name in ("comfortable", "balanced", "stretch"):
                pick = picks.get(slot_name)
                if pick and pick.get("disk_warning"):
                    notes.append(f"[{slot_name.upper()}] {pick['disk_warning']}")
                if pick and pick.get("note"):
                    notes.append(f"[{slot_name.upper()}] {pick['note']}")
                if pick and pick.get("note_stretch"):
                    notes.append(f"[{slot_name.upper()}] {pick['note_stretch']}")

            # Standard caveats that should ALWAYS surface (per skill)
            notes.append(
                "AA Intelligence Index is an aggregate; task-specific scores "
                "differ (e.g. Qwen3.6-27B beats Qwen3.6-35B-A3B on coding)."
            )
            notes.append(
                "KV cache grows linearly with context; at 256K ctx add 20-40 GB."
            )

            return {
                "ok": True,
                "snapshot": report.snapshot,
                "hardware": report_to_dict(report),
                "picks": picks,
                "notes": notes,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }
