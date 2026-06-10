from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_build_fit_summary_returns_stable_shape_with_empty_snapshot():
    from usr.plugins.a0_lmm_router.helpers.fit_summary import build_fit_summary

    summary = build_fit_summary({})

    assert summary["ok"] is True
    assert summary["mode"] == "read_only"
    assert summary["hardware"]["gpu_count"] == 0
    assert summary["hardware"]["gpu_summary"] == "unknown"
    assert summary["hardware"]["ram_total_mb"] is None
    assert summary["slots"] == []
    assert "No slot data available" in summary["warnings"]
    assert summary["data_sources"]["hardware"] == "unknown"
    assert set(summary) >= {
        "ok",
        "mode",
        "hardware",
        "slots",
        "risk_notes",
        "recommendations",
        "warnings",
        "data_sources",
    }


def test_build_fit_summary_classifies_disabled_and_missing_hardware_without_crash():
    from usr.plugins.a0_lmm_router.helpers.fit_summary import build_fit_summary

    snapshot = {
        "gpu_source": "none",
        "cpu": {"ram_total_mb": 0, "ram_used_mb": 0, "ram_free_mb": 0},
        "slots": [
            {
                "id": "slot_chat",
                "role": "chat",
                "model_id": "qwen.gguf",
                "port": 8080,
                "running": False,
                "healthy": False,
                "enabled": False,
                "router_mode": False,
                "source": "config",
            }
        ],
    }

    summary = build_fit_summary(snapshot)

    assert summary["hardware"]["gpu_summary"] == "unknown"
    assert summary["slots"][0]["slot_id"] == "slot_chat"
    assert summary["slots"][0]["enabled"] is False
    assert summary["slots"][0]["fit_status"] == "unknown"
    assert any("disabled" in note.lower() for note in summary["slots"][0]["risk_notes"])
    assert any("hardware" in warning.lower() for warning in summary["warnings"])


def test_build_fit_summary_reports_good_and_caution_slots_from_existing_snapshot():
    from usr.plugins.a0_lmm_router.helpers.fit_summary import build_fit_summary

    snapshot = {
        "gpu_source": "local",
        "gpus": [
            {
                "id": 0,
                "name": "RTX 4090",
                "total_vram_mb": 24576,
                "used_vram_mb": 4096,
                "free_vram_mb": 20480,
                "utilization_pct": 8,
                "temperature_c": 45,
            }
        ],
        "cpu": {"ram_total_mb": 65536, "ram_used_mb": 12000, "ram_free_mb": 52000},
        "slots": [
            {
                "id": "slot_chat",
                "role": "chat",
                "model_id": "qwen3-9b",
                "port": 8080,
                "running": True,
                "healthy": True,
                "router_mode": False,
                "context_size": 32768,
                "source": "config",
            },
            {
                "id": "slot_utility",
                "role": "utility",
                "model_id": "phi-mini",
                "port": 8088,
                "running": True,
                "healthy": False,
                "router_mode": False,
                "source": "config",
            },
        ],
    }

    summary = build_fit_summary(snapshot)

    assert summary["hardware"]["gpu_count"] == 1
    assert "RTX 4090" in summary["hardware"]["gpu_summary"]
    by_id = {slot["slot_id"]: slot for slot in summary["slots"]}
    assert by_id["slot_chat"]["fit_status"] == "good"
    assert by_id["slot_chat"]["context_size"] == 32768
    assert by_id["slot_utility"]["fit_status"] == "caution"
    assert any("unhealthy" in note.lower() for note in by_id["slot_utility"]["risk_notes"])
    assert summary["data_sources"]["hardware"] == "compute_monitor:local"


def test_api_handler_returns_fit_summary_shape_when_compute_monitor_is_monkeypatched(monkeypatch):
    from usr.plugins.a0_lmm_router.api.lmm_fit_summary import LmmFitSummary
    import usr.plugins.a0_lmm_router.helpers.compute_monitor as compute_monitor

    monkeypatch.setattr(
        compute_monitor,
        "get_compute_snapshot",
        lambda: {"gpu_source": "none", "cpu": {}, "slots": []},
    )

    result = asyncio.run(LmmFitSummary().process({}, None))

    assert result["ok"] is True
    assert result["mode"] == "read_only"
    assert result["slots"] == []


def test_fit_summary_does_not_expose_mutating_actions():
    from usr.plugins.a0_lmm_router.helpers.fit_summary import build_fit_summary

    summary = build_fit_summary({"slots": []})
    serialized = repr(summary).lower()

    for forbidden in ("download", "serve", "start_slot", "stop_slot", "ssh", "shell"):
        assert forbidden not in serialized
