from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_parse_router_models_extracts_role_bindings():
    from usr.plugins.a0_lmm_router.api.router_aliases import parse_router_models_payload

    payload = {
        "data": [
            {
                "id": "chat",
                "status": {
                    "value": "loaded",
                    "port": 41959,
                    "args": ["--model", "/models/chat/gemma.gguf", "--ctx-size", "65536"],
                },
            },
            {
                "id": "embedding",
                "status": {
                    "value": "unloaded",
                    "preset": "alias = embedding\nmodel = /models/embed/nomic.gguf\nembedding = true\n",
                },
            },
        ]
    }

    bindings = parse_router_models_payload(payload)

    assert bindings["chat"]["loaded"] is True
    assert bindings["chat"]["port"] == 41959
    assert bindings["chat"]["model_path"] == "/models/chat/gemma.gguf"
    assert bindings["chat"]["model_filename"] == "gemma.gguf"
    assert bindings["chat"]["ctx_size"] == 65536
    assert bindings["embedding"]["loaded"] is False
    assert bindings["embedding"]["model_path"] == "/models/embed/nomic.gguf"


def test_rewrite_preset_alias_model_preserves_other_lines():
    from usr.plugins.a0_lmm_router.tools import lmm_host_helper as helper

    original = """[chat]
alias = chat
model = /models/old.gguf
ctx-size = 65536

[utility]
alias = utility
model = /models/utility.gguf
"""

    updated, snippet = helper._rewrite_preset_alias_model(original, "chat", "/models/new.gguf")

    assert "model = /models/new.gguf" in snippet
    assert "ctx-size = 65536" in snippet
    assert "[utility]" in updated
    assert "model = /models/utility.gguf" in updated


def test_atomic_preset_write_creates_backup(tmp_path):
    from usr.plugins.a0_lmm_router.tools import lmm_host_helper as helper

    preset = tmp_path / "models_preset.ini"
    preset.write_text("[chat]\nmodel = /models/old.gguf\n", encoding="utf-8")

    helper._write_preset_ini_atomic(preset, "[chat]\nmodel = /models/new.gguf\n")

    assert preset.read_text(encoding="utf-8") == "[chat]\nmodel = /models/new.gguf\n"
    assert preset.with_suffix(".ini.bak").read_text(encoding="utf-8") == "[chat]\nmodel = /models/old.gguf\n"


def test_router_url_rejects_low_ports():
    from usr.plugins.a0_lmm_router.api.router_aliases import _router_url

    try:
        _router_url({"port": 80})
        assert False, "low router port should be rejected"
    except ValueError as exc:
        assert "outside allowed range" in str(exc)


def test_router_url_accepts_router_range():
    from usr.plugins.a0_lmm_router.api.router_aliases import _router_url

    assert _router_url({"port": 8080}) == "http://host.docker.internal:8080/v1/models"


def test_scribe_role_maps_to_scribe_alias():
    """The scribe (super-ego) role is a first-class router role.

    See a0_scribe plugin design: the scribe runs a local CPU model behind the
    'scribe' role and must keep its own router alias rather than collapsing to
    'chat'. The 'documentation' task type routes to the scribe role.
    """
    pytest.importorskip("pydantic")
    from usr.plugins.a0_lmm_router.service.routing_intent import (
        _role_from_task_type,
        _router_alias_from_role,
    )

    # The scribe role keeps its own alias (case-insensitive).
    assert _router_alias_from_role("scribe") == "scribe"
    assert _router_alias_from_role("SCRIBE") == "scribe"

    # The documentation task type routes to the scribe role.
    assert _role_from_task_type("documentation") == "scribe"

    # Existing roles are unaffected by the new branch.
    assert _router_alias_from_role("chat") == "chat"
    assert _router_alias_from_role("utility") == "utility"
    assert _router_alias_from_role("embedding") == "embedding"
    assert _router_alias_from_role(None) == "chat"
