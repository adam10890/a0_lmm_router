"""
Shared pytest configuration for a0_lmm_router tests.

Sets up two things required for running outside an Agent Zero container:

1. sys.path — so that `from usr.plugins.a0_lmm_router.X import Y` resolves.
   The repo root (/usr/plugins/a0_lmm_router symlink) is already in place;
   we ensure '/' (parents[4]) is in sys.path as a safety guard.

2. Agent Zero stub modules — workflow_registry.py imports Agent Zero's
   `helpers.files` and `helpers.yaml` at module level (via smart_router/__init__).
   Those don't exist outside the A0 container. We inject minimal stubs into
   sys.modules before any test imports trigger the chain.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

# ── 1. Ensure REPO_ROOT is on sys.path ──────────────────────────────────────
# tests/ is at parents[0], plugin root at parents[1], REPO_ROOT = parents[4]
# which resolves to '/' in this dev environment.  All test files do this too,
# but conftest runs first so it's safe to do it here once.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── 2. Stub Agent Zero helpers ───────────────────────────────────────────────
# These are Agent Zero core modules that the plugin's workflow_registry imports
# at module level.  They are NOT part of this repo; stub them so tests that
# don't exercise workflow routing can still import the plugin without crashing.

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []   # looks like a package to Python
    sys.modules[name] = mod
    return mod


if "helpers" not in sys.modules:
    _helpers = _make_stub("helpers")
else:
    _helpers = sys.modules["helpers"]

# helpers.files stub — only the attributes workflow_registry.py uses
if "helpers.files" not in sys.modules:
    _files = _make_stub("helpers.files")
    _files.get_abs_path = lambda *a, **kw: ""
    _helpers.files = _files

# helpers.yaml stub — only the attributes workflow_registry.py uses
if "helpers.yaml" not in sys.modules:
    import yaml as _real_yaml  # pyyaml is a real dep; just re-export it

    _yaml_stub = _make_stub("helpers.yaml")
    _yaml_stub.load_file = lambda path, **kw: {}
    _yaml_stub.save_file = lambda path, data, **kw: None
    _helpers.yaml = _yaml_stub
