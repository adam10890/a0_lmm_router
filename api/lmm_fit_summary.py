"""API endpoint: /api/plugins/a0_lmm_router/lmm_fit_summary.

Read-only Hardware / Model Fit Summary. Existing Agent Zero plugin APIs are
POST handlers under ``/api/plugins/<plugin>/<handler>``; this follows that
convention while implementing the requested read-only fit-summary concept.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

try:
    from helpers.api import ApiHandler
except ModuleNotFoundError as exc:  # pragma: no cover - minimal test venv without Flask
    if exc.name not in {"flask", "helpers.api"}:
        raise

    class ApiHandler:  # type: ignore[no-redef]
        """Import-time shim for py_compile/import checks outside A0's Flask runtime."""

        pass

if TYPE_CHECKING:  # pragma: no cover - import-time compatibility for test shells without Flask
    from flask import Request
else:
    Request = Any


class LmmFitSummary(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            from usr.plugins.a0_lmm_router.helpers.compute_monitor import get_compute_snapshot
            from usr.plugins.a0_lmm_router.helpers.fit_summary import build_fit_summary

            return build_fit_summary(get_compute_snapshot())
        except Exception as exc:
            return {
                "ok": False,
                "mode": "read_only",
                "error": f"{type(exc).__name__}: {exc}",
                "hardware": {
                    "gpu_count": 0,
                    "gpu_summary": "unknown",
                    "gpu_source": "none",
                    "ram_total_mb": None,
                    "ram_free_mb": None,
                    "ram_used_mb": None,
                    "cpu_load_pct": None,
                },
                "slots": [],
                "risk_notes": ["Read-only summary failed before producing telemetry data."],
                "recommendations": ["Check existing compute monitor configuration and plugin imports."],
                "warnings": ["Fit summary unavailable."],
                "data_sources": {"hardware": "unknown", "slots": "unknown", "fleet_mode": "unknown"},
            }
