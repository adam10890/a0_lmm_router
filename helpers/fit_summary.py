"""Read-only hardware/model fit summary for the LMM Router dashboard.

Phase 1 intentionally consumes only data that the router already exposes via
``helpers.compute_monitor.get_compute_snapshot``. It does not start/stop slots,
install models, manage credentials, or perform independent deep hardware probes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

FIT_STATUSES = {"unknown", "good", "caution", "poor"}


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _gpu_label(gpu: Dict[str, Any]) -> str:
    name = str(gpu.get("name") or "GPU")
    total_mb = _as_int(gpu.get("total_vram_mb")) or 0
    if total_mb > 0:
        return f"{name} ({total_mb / 1024.0:.1f} GB VRAM)"
    return name


def _hardware_summary(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    gpus = snapshot.get("gpus") if isinstance(snapshot.get("gpus"), list) else []
    cpu = snapshot.get("cpu") if isinstance(snapshot.get("cpu"), dict) else {}
    gpu_labels = [_gpu_label(g) for g in gpus if isinstance(g, dict)]
    return {
        "gpu_count": len(gpu_labels),
        "gpu_summary": ", ".join(gpu_labels) if gpu_labels else "unknown",
        "gpu_source": str(snapshot.get("gpu_source") or "none"),
        "ram_total_mb": _as_int(cpu.get("ram_total_mb")) if cpu else None,
        "ram_free_mb": _as_int(cpu.get("ram_free_mb")) if cpu else None,
        "ram_used_mb": _as_int(cpu.get("ram_used_mb")) if cpu else None,
        "cpu_load_pct": cpu.get("load_pct") if cpu else None,
    }


def _context_size(slot: Dict[str, Any]) -> Optional[int]:
    for key in ("context_size", "ctx_size", "n_ctx", "n_ctx_train"):
        value = _as_int(slot.get(key))
        if value is not None:
            return value
    return None


def _slot_fit(slot: Dict[str, Any], hardware_known: bool) -> tuple[str, List[str], List[str]]:
    enabled = bool(slot.get("enabled", True))
    running = bool(slot.get("running", False))
    healthy = bool(slot.get("healthy", False))
    model_id = str(slot.get("model_id") or "").strip()
    risk_notes: List[str] = []
    recommendations: List[str] = []

    if not enabled:
        risk_notes.append("Slot is disabled; fit cannot be evaluated from runtime state.")
        recommendations.append("Enable the slot only after validating its role and capacity requirements.")
        return "unknown", risk_notes, recommendations

    if not hardware_known:
        risk_notes.append("Hardware capacity is unknown; model fit is conservative.")
        recommendations.append("Add a future safe read-only detector for host GPU/RAM inventory.")
        return "unknown", risk_notes, recommendations

    if not model_id:
        risk_notes.append("No current model_id is available for this slot.")
        recommendations.append("Assign a model through the existing router workflow after reviewing capacity.")
        return "poor", risk_notes, recommendations

    if running and healthy:
        risk_notes.append("Slot is running and healthy according to existing telemetry data.")
        return "good", risk_notes, recommendations

    if running and not healthy:
        risk_notes.append("Slot is running but unhealthy; fit may be blocked by load, startup, or endpoint issues.")
        recommendations.append("Use existing diagnostics to inspect health before changing model assignment.")
        return "caution", risk_notes, recommendations

    risk_notes.append("Slot is configured but not currently running; runtime fit is not verified.")
    recommendations.append("Start from existing health diagnostics before changing model assignment.")
    return "caution", risk_notes, recommendations


def _slot_summary(slot: Dict[str, Any], hardware_known: bool, default_backend: str) -> Dict[str, Any]:
    fit_status, risk_notes, recommendations = _slot_fit(slot, hardware_known)
    if fit_status not in FIT_STATUSES:
        fit_status = "unknown"
    return {
        "slot_id": str(slot.get("id") or slot.get("slot_id") or "unknown"),
        "role": str(slot.get("role") or ""),
        "model_id": str(slot.get("model_id") or ""),
        "backend_type": str(slot.get("backend_type") or default_backend or "unknown"),
        "context_size": _context_size(slot),
        "enabled": bool(slot.get("enabled", True)),
        "running": bool(slot.get("running", False)),
        "healthy": bool(slot.get("healthy", False)),
        "router_mode": bool(slot.get("router_mode", False)),
        "fit_status": fit_status,
        "risk_notes": risk_notes,
        "recommendations": recommendations,
        "data_sources": {
            "slot": str(slot.get("source") or "compute_monitor"),
            "fit_status": "fit_summary:phase_1_heuristic",
        },
    }


def build_fit_summary(snapshot: Dict[str, Any] | None) -> Dict[str, Any]:
    """Build a stable read-only fit summary from an existing compute snapshot."""
    snap = snapshot if isinstance(snapshot, dict) else {}
    hardware = _hardware_summary(snap)
    hardware_known = bool(hardware["gpu_count"] > 0 or (hardware.get("ram_total_mb") or 0) > 0)
    slots_raw = snap.get("slots") if isinstance(snap.get("slots"), list) else []
    fleet_mode = snap.get("fleet_mode") if isinstance(snap.get("fleet_mode"), dict) else {}
    default_backend = str(fleet_mode.get("backend") or fleet_mode.get("mode") or "unknown")

    slots = [_slot_summary(slot, hardware_known, default_backend) for slot in slots_raw if isinstance(slot, dict)]

    warnings: List[str] = []
    recommendations: List[str] = []
    risk_notes: List[str] = [
        "Read-only summary: no model lifecycle, host-control, credential, or command actions are exposed.",
        "Fit status is a phase-1 heuristic based only on existing compute/slot telemetry data.",
    ]

    if not hardware_known:
        warnings.append("Hardware inventory is unknown or incomplete; per-slot fit_status may remain unknown.")
        recommendations.append("Future phase: add a safe read-only host hardware detector with explicit data-source labeling.")
    if not slots:
        warnings.append("No slot data available")
        recommendations.append("Verify the llama.cpp fleet config and existing fleet telemetry endpoints.")
    if any(slot["fit_status"] in ("poor", "caution") for slot in slots):
        recommendations.append("Review caution/poor slots before changing routing defaults.")

    return {
        "ok": True,
        "mode": "read_only",
        "hardware": hardware,
        "slots": slots,
        "risk_notes": risk_notes,
        "recommendations": recommendations,
        "warnings": warnings,
        "data_sources": {
            "hardware": f"compute_monitor:{hardware['gpu_source']}" if hardware_known else "unknown",
            "slots": "compute_monitor.slots" if slots_raw else "unknown",
            "fleet_mode": "compute_monitor.fleet_mode" if fleet_mode else "unknown",
        },
    }


__all__ = ["FIT_STATUSES", "build_fit_summary"]
