from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_detect_fleet_mode_idle():
    from usr.plugins.a0_lmm_router.helpers.fleet_mode import detect_fleet_mode

    mode = detect_fleet_mode(lambda: {})

    assert mode["mode"] == "idle"
    assert mode["router_running"] is False
    assert mode["three_slot_running"] is False


def test_detect_fleet_mode_router():
    from usr.plugins.a0_lmm_router.helpers.fleet_mode import detect_fleet_mode

    mode = detect_fleet_mode(lambda: {"a0-llama-router": "Up 1 minute"})

    assert mode["mode"] == "router"
    assert mode["router_running"] is True
    assert mode["three_slot_running"] is False


def test_detect_fleet_mode_three_slot():
    from usr.plugins.a0_lmm_router.helpers.fleet_mode import detect_fleet_mode

    mode = detect_fleet_mode(lambda: {"a0-llama-chat": "Up", "a0-llama-utility": "Up"})

    assert mode["mode"] == "three_slot"
    assert mode["router_running"] is False
    assert mode["three_slot_running"] is True


def test_detect_fleet_mode_conflict():
    from usr.plugins.a0_lmm_router.helpers.fleet_mode import detect_fleet_mode

    mode = detect_fleet_mode(lambda: {"a0-llama-router": "Up", "a0-llama-chat": "Up"})

    assert mode["mode"] == "conflict"
    assert mode["router_running"] is True
    assert mode["three_slot_running"] is True


def test_compose_target_mode_detects_router_stack():
    from usr.plugins.a0_lmm_router.helpers.fleet_mode import compose_target_mode

    assert compose_target_mode("docker/docker-compose.lmm.router.yml") == "router"
    assert compose_target_mode("docker/docker-compose.lmm.yml") == "three_slot"


def test_compute_snapshot_uses_http_router_slot_when_docker_socket_is_absent(monkeypatch):
    from usr.plugins.a0_lmm_router.helpers import compute_monitor

    router_slot = compute_monitor.SlotInfo(
        id="slot_router",
        role="router",
        model_id="loaded-model",
        port=8080,
        running=True,
        healthy=True,
        router_mode=True,
        source="http",
    )
    monkeypatch.setattr(compute_monitor, "_query_gpus", lambda: [])
    monkeypatch.setattr(
        compute_monitor,
        "_query_cpu",
        lambda: compute_monitor.CPUStats(
            load_pct=0.0,
            ram_total_mb=1,
            ram_used_mb=0,
            ram_free_mb=1,
        ),
    )
    monkeypatch.setattr(compute_monitor, "_query_slots", lambda: [router_slot])
    monkeypatch.setattr(
        compute_monitor,
        "detect_fleet_mode",
        lambda: {
            "mode": "idle",
            "router_running": False,
            "three_slot_running": False,
            "containers": {},
        },
    )

    snapshot = compute_monitor.get_compute_snapshot()

    assert snapshot["fleet_mode"]["mode"] == "router"
    assert snapshot["fleet_mode"]["router_running"] is True
    assert snapshot["fleet_mode"]["detected_via"] == "http"
