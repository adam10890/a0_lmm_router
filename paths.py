"""
lmm-router — path resolution layer.

Centralizes the three hardcoded /a0/* paths inherited from the Agent Zero
plugin layout so the same codebase can run in three modes:

    1. As an A0 plugin inside the container
         → all defaults resolve to /a0/...           (no env vars set)

    2. As a standalone install on the host (Windows / Linux native)
         → set LMM_HOME, LMM_CONFIG, LMM_TOKEN_PATH  via env or .env

    3. From a fresh checkout for development
         → defaults to <repo_root>/conf, <repo_root>/tmp etc.
           (auto-detected from this file's location)

Usage (replacing hardcoded paths):

    # before:
    PLUGIN_DIR = "/a0/usr/plugins/a0_lmm_router"
    TOKEN_PATH = "/a0/tmp/lmm_host_token"
    CONFIG     = "/a0/conf/llama_cpp_servers.yaml"

    # after:
    from paths import PLUGIN_DIR, TOKEN_PATH, CONFIG_FILE
"""
from __future__ import annotations

import os
from pathlib import Path


# ─── auto-detect repo root (the directory containing this file) ──────────
_REPO_ROOT = Path(__file__).resolve().parent


def _resolve(env_var: str, a0_default: str, dev_default: Path) -> Path:
    """Resolve a path in priority order: env var → A0 path (if exists) → dev default."""
    if (val := os.environ.get(env_var)):
        return Path(val)
    a0_path = Path(a0_default)
    if a0_path.exists() or a0_path.parent.exists():
        return a0_path
    return dev_default


# ─── PLUGIN_DIR / project root ───────────────────────────────────────────
# Where the code itself lives. In A0: /a0/usr/plugins/a0_lmm_router
# Standalone: the repo root.
PLUGIN_DIR: Path = _resolve(
    env_var="LMM_HOME",
    a0_default="/a0/usr/plugins/a0_lmm_router",
    dev_default=_REPO_ROOT,
)


# ─── CONFIG file (llama_cpp_servers.yaml) ────────────────────────────────
# Lookup order: $LMM_CONFIG → /a0/conf/llama_cpp_servers.yaml
#              → <plugin>/conf/llama_cpp_servers.yaml
CONFIG_FILE: Path = _resolve(
    env_var="LMM_CONFIG",
    a0_default="/a0/conf/llama_cpp_servers.yaml",
    dev_default=PLUGIN_DIR / "conf" / "llama_cpp_servers.yaml",
)


# ─── HOST helper token path ──────────────────────────────────────────────
# Token written by tools/lmm_host_helper.py so containerized code can call
# the host bridge. In A0: /a0/tmp/lmm_host_token (mounted from container)
# Standalone: <plugin>/tmp/host_token (lives next to the code)
TOKEN_PATH: Path = _resolve(
    env_var="LMM_TOKEN_PATH",
    a0_default="/a0/tmp/lmm_host_token",
    dev_default=PLUGIN_DIR / "tmp" / "host_token",
)

# Alternative locations to try (legacy fallback used by some helpers)
TOKEN_CANDIDATES: tuple[Path, ...] = (
    TOKEN_PATH,
    Path("/a0/tmp/lmm_host_token"),
    Path("/host/a0_lmm_host.key"),
)


# ─── Derived paths ────────────────────────────────────────────────────────
CONF_DIR: Path = CONFIG_FILE.parent
MCP_LOG_DIR: Path = Path(os.environ.get("LMM_LOG_DIR", str(PLUGIN_DIR / "logs")))
MODELS_MANIFEST: Path = PLUGIN_DIR / "conf" / "model_providers.yaml"


def describe() -> dict[str, str]:
    """Return current path resolution for debugging / `lmm-router status`."""
    return {
        "PLUGIN_DIR":      str(PLUGIN_DIR),
        "CONFIG_FILE":     str(CONFIG_FILE),
        "TOKEN_PATH":      str(TOKEN_PATH),
        "MCP_LOG_DIR":     str(MCP_LOG_DIR),
        "MODELS_MANIFEST": str(MODELS_MANIFEST),
        "_repo_root":      str(_REPO_ROOT),
        "_env_LMM_HOME":   os.environ.get("LMM_HOME", "<unset>"),
        "_env_LMM_CONFIG": os.environ.get("LMM_CONFIG", "<unset>"),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
