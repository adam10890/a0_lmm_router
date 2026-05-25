"""Fleet-mode detection shared by APIs, host helper, and dashboard state."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

ROUTER_CONTAINERS = ("a0-llama-router",)
THREE_SLOT_CONTAINERS = ("a0-llama-chat", "a0-llama-utility", "a0-llama-embed")


def _docker_container_status() -> dict[str, str]:
    """Return docker container statuses keyed by name."""
    names = [*ROUTER_CONTAINERS, *THREE_SLOT_CONTAINERS]
    containers: dict[str, str] = {}
    for name in names:
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}\t{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception:
            continue
        for line in result.stdout.splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[0] == name:
                containers[name] = parts[1]
    return containers


def detect_fleet_mode(status_provider: Callable[[], dict[str, str]] | None = None) -> dict:
    """Detect whether router, 3-slot, both, or neither fleet is running."""
    containers = (status_provider or _docker_container_status)()
    router_running = any(name in containers for name in ROUTER_CONTAINERS)
    three_slot_running = any(name in containers for name in THREE_SLOT_CONTAINERS)
    if router_running and three_slot_running:
        mode = "conflict"
    elif router_running:
        mode = "router"
    elif three_slot_running:
        mode = "three_slot"
    else:
        mode = "idle"
    return {
        "mode": mode,
        "router_running": router_running,
        "three_slot_running": three_slot_running,
        "containers": containers,
    }


def compose_target_mode(compose_path: str) -> str:
    """Infer intended fleet mode from compose filename."""
    name = Path(compose_path).name.lower()
    return "router" if "router" in name else "three_slot"


def is_conflicting_mode(active_mode: str, target_mode: str) -> bool:
    if active_mode == "conflict":
        return True
    if active_mode == "idle":
        return False
    return active_mode != target_mode
