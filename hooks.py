"""a0_lmm_router plugin hooks.

Runs in the Agent Zero framework runtime when the plugin is installed
or updated. Ensures Python dependencies declared in requirements.txt
are present in the A0 venv so the MCP server (and any other tools that
import mcp / aiohttp / yaml) can boot.

Idempotent: if every required dist is already importable, skips pip
entirely so the install is fast and offline-safe.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path

_PLUGIN_DIR = Path(__file__).parent
_REQUIREMENTS = _PLUGIN_DIR / "requirements.txt"


def _parse_requirements(path: Path) -> list[tuple[str, str]]:
    """Return [(dist_name, full_spec), ...] from a simple requirements.txt.

    Handles `name`, `name>=ver`, `name==ver`, `name~=ver`, and skips
    blanks / comments. Does not handle URLs or extras — keep the file
    simple.
    """
    out: list[tuple[str, str]] = []
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for sep in (">=", "==", "~=", "<=", "<", ">"):
            if sep in line:
                name = line.split(sep, 1)[0].strip()
                break
        else:
            name = line
        out.append((name, line))
    return out


def _missing(reqs: list[tuple[str, str]]) -> list[str]:
    """Return the full specs of distributions not currently installed."""
    missing: list[str] = []
    for dist_name, spec in reqs:
        try:
            importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(spec)
    return missing


def install(**kwargs):
    """Called by Agent Zero after the plugin is copied into place.

    Ensures requirements.txt is satisfied. Safe to call repeatedly.
    """
    reqs = _parse_requirements(_REQUIREMENTS)
    if not reqs:
        print("[a0_lmm_router] No requirements.txt entries — nothing to install.")
        return

    missing = _missing(reqs)
    if not missing:
        present = ", ".join(name for name, _ in reqs)
        print(f"[a0_lmm_router] All deps already installed: {present}")
        return

    print(f"[a0_lmm_router] Installing missing deps: {missing}")
    cmd = [sys.executable, "-m", "pip", "install", *missing]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        print("[a0_lmm_router] pip install timed out after 180s.")
        return

    if result.returncode == 0:
        print("[a0_lmm_router] ✓ Deps installed.")
    else:
        print(f"[a0_lmm_router] ✗ pip install failed (exit {result.returncode}).")
        if result.stderr:
            print(result.stderr.strip()[:800])


def uninstall(**kwargs):
    """Called by Agent Zero on plugin removal — intentionally a no-op.

    We don't uninstall shared deps (mcp, aiohttp, pyyaml) since other
    plugins or the A0 core may depend on them.
    """
    print("[a0_lmm_router] Uninstall hook: leaving shared deps in place.")


if __name__ == "__main__":
    # Allow manual: python hooks.py
    install()
