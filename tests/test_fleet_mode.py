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
