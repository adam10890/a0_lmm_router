"""a0_lmm_router — dedicated launcher.

Purpose
-------
Self-contained CLI entry point for bringing the plugin's llama.cpp fleet
up / down without going through the Agent Zero message loop. Intended to
be called from `start_agent_zero.bat` after the A0 container is up:

    docker exec agent-zero-2 /opt/venv-a0/bin/python \
        /a0/usr/plugins/a0_lmm_router/launcher.py start

Config discovery order (first match wins):
  1. $A0_LMM_ROUTER_CONFIG (if set and readable)
  2. /a0/conf/llama_cpp_servers.yaml
  3. /a0/usr/plugins/a0_lmm_router/conf/llama_cpp_servers.yaml  (mounted)

If no config is found OR `global.auto_start` is false, the launcher exits
with code 0 and a "disabled" banner — safe to wire into startup scripts.

CLI
---
    launcher.py status      show discovered config + per-slot state
    launcher.py start       start all enabled slots
    launcher.py start <id>  start one slot by id
    launcher.py stop        stop all running slots
    launcher.py stop  <id>  stop one slot by id
    launcher.py restart     stop-then-start all enabled slots

All subcommands are idempotent. Exit codes: 0 success, 2 config missing,
3 partial failure (some slots failed), 4 unhandled error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


# Allow running both as a module (`python -m usr.plugins.a0_lmm_router.launcher`)
# and as a file path (`python launcher.py`). In the latter case we need /a0 on
# sys.path so `usr.plugins.*` imports resolve.
_A0_ROOT = "/a0"
if _A0_ROOT not in sys.path and os.path.isdir(_A0_ROOT):
    sys.path.insert(0, _A0_ROOT)


PLUGIN_DIR = Path(__file__).resolve().parent
CONFIG_CANDIDATES = [
    os.environ.get("A0_LMM_ROUTER_CONFIG", ""),
    "/a0/conf/llama_cpp_servers.yaml",
    str(PLUGIN_DIR / "conf" / "llama_cpp_servers.yaml"),
]


def _find_config() -> str | None:
    for candidate in CONFIG_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _load_global(config_path: str) -> dict[str, Any]:
    import yaml

    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("global", {}) if isinstance(data, dict) else {}


def _get_manager(config_path: str):
    from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager

    # Reset singleton so repeated CLI calls re-read the config on disk.
    BackendManager._instance = None  # noqa: SLF001
    return BackendManager(config_path=config_path)


def _print_banner(title: str, config_path: str | None) -> None:
    print(f"[a0_lmm_router] {title}")
    print(f"  config: {config_path or '<none>'}")


def cmd_status(config_path: str | None) -> int:
    _print_banner("status", config_path)
    if not config_path:
        print("  state: disabled (no config found)")
        return 0

    global_cfg = _load_global(config_path)
    auto_start = bool(global_cfg.get("auto_start", False))
    backend = global_cfg.get("backend", "auto")
    print(f"  auto_start: {auto_start}")
    print(f"  backend: {backend}")

    try:
        manager = _get_manager(config_path)
    except Exception as exc:
        print(f"  ERROR: failed to build manager: {type(exc).__name__}: {exc}")
        return 4

    slots = getattr(manager, "_slot_configs", {})
    if not slots:
        print("  slots: <none configured>")
        return 0

    print("  slots:")
    for name, cfg in slots.items():
        print(
            f"    - {name:<20} "
            f"port={cfg.get('port', '')!s:<6} "
            f"role={cfg.get('role', '')!s:<10} "
            f"enabled={cfg.get('enabled', True)!s:<5}"
        )
    return 0


async def _start_all(manager, slot: str | None) -> tuple[list[str], list[str]]:
    started: list[str] = []
    failed: list[str] = []
    if slot:
        result = await manager.start_slot(slot)
        ok = bool(result.get("healthy") or result.get("running")) and not result.get("error")
        (started if ok else failed).append(slot)
        return started, failed

    results = await manager.start_all()
    for name, result in results.items():
        ok = bool(result.get("healthy") or result.get("running")) and not result.get("error")
        (started if ok else failed).append(name)
    return started, failed


async def _stop_all(manager, slot: str | None) -> tuple[list[str], list[str]]:
    stopped: list[str] = []
    failed: list[str] = []
    if slot:
        ok = await manager.stop_slot(slot)
        (stopped if ok else failed).append(slot)
        return stopped, failed

    names = list(getattr(manager, "_slot_configs", {}).keys())
    await manager.stop_all()
    stopped.extend(names)
    return stopped, failed


def cmd_start(config_path: str | None, slot: str | None) -> int:
    _print_banner("start", config_path)
    if not config_path:
        print("  state: disabled (no config)")
        return 2

    global_cfg = _load_global(config_path)
    if not slot and not global_cfg.get("auto_start", False):
        print("  auto_start=false — nothing to do (pass a slot id to force)")
        return 0

    manager = _get_manager(config_path)
    if not getattr(manager, "_slot_configs", {}):
        print("  slots: <none configured> — nothing to start")
        return 0

    started, failed = asyncio.run(_start_all(manager, slot))
    print(f"  started: {started or '<none>'}")
    if failed:
        print(f"  failed:  {failed}")
        return 3
    return 0


def cmd_stop(config_path: str | None, slot: str | None) -> int:
    _print_banner("stop", config_path)
    if not config_path:
        print("  state: disabled (no config)")
        return 2

    manager = _get_manager(config_path)
    if not getattr(manager, "_slot_configs", {}):
        print("  slots: <none configured> — nothing to stop")
        return 0

    stopped, failed = asyncio.run(_stop_all(manager, slot))
    print(f"  stopped: {stopped or '<none>'}")
    if failed:
        print(f"  failed:  {failed}")
        return 3
    return 0


def cmd_restart(config_path: str | None) -> int:
    rc_stop = cmd_stop(config_path, slot=None)
    rc_start = cmd_start(config_path, slot=None)
    # restart succeeds as long as start succeeded; stop failure on a
    # stopped server is non-fatal.
    return rc_start if rc_start else rc_stop if rc_stop in (0, 3) else rc_stop


def cmd_mcp(config_path: str | None) -> int:
    """Start the MCP server (Streamable HTTP, default port 8095)."""
    _print_banner("mcp", config_path)
    # Resolve sys.path so mcp_server.* imports work from any cwd.
    plugin_dir = str(PLUGIN_DIR)
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)

    from mcp_server.server import main as mcp_main  # noqa: PLC0415
    mcp_main()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a0_lmm_router.launcher",
        description="Start/stop/inspect llama.cpp slots managed by a0_lmm_router.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="print discovered config and slot state")

    p_start = sub.add_parser("start", help="start all or one slot")
    p_start.add_argument("slot", nargs="?", help="slot id (omit to start all)")

    p_stop = sub.add_parser("stop", help="stop all or one slot")
    p_stop.add_argument("slot", nargs="?", help="slot id (omit to stop all)")

    sub.add_parser("restart", help="stop then start all enabled slots")

    sub.add_parser("mcp", help="start the MCP server (Streamable HTTP, default :8095)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = _find_config()

    try:
        if args.cmd == "status":
            return cmd_status(config_path)
        if args.cmd == "start":
            return cmd_start(config_path, args.slot)
        if args.cmd == "stop":
            return cmd_stop(config_path, args.slot)
        if args.cmd == "restart":
            return cmd_restart(config_path)
        if args.cmd == "mcp":
            return cmd_mcp(config_path)
    except KeyboardInterrupt:
        print("aborted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[a0_lmm_router] unhandled error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    # Unreachable; argparse enforces required subcommand.
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
