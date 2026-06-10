"""Tests for model_params_cache."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HELPERS_ROOT = PLUGIN_ROOT / "helpers"
if str(HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPERS_ROOT))

from model_params_cache import (  # noqa: E402
    ContextPlan,
    get_cached_entry,
    load_cache,
    model_cache_key,
    plan_model_with_cache,
    save_cache,
    warm_fleet_params,
)

_META = {
    "n_ctx_train": 131072,
    "n_layer": 48,
    "n_embd": 3840,
    "file_size_gb": 6.63,
}


def test_cache_hit_after_store(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr("model_params_cache.default_cache_path", lambda: cache_path)
    monkeypatch.setattr(
        "model_params_cache.read_gguf_metadata",
        lambda _p: dict(_META),
    )
    monkeypatch.setattr(
        "model_params_cache._file_fingerprint",
        lambda _p: {"exists": True, "size": 100, "mtime_ns": 1, **_META},
    )

    env = {
        "LLAMA_MODELS_DIR": "C:/models",
        "CHAT_CTX_SIZE": "65536",
        "ROUTER_CACHE_TYPE_K": "q8_0",
        "ROUTER_CACHE_TYPE_V": "q8_0",
        "ROUTER_PARALLEL": "1",
        "A0_LMM_AVAILABLE_VRAM_GB": "24",
    }

    plan1, _, _, src1 = plan_model_with_cache(
        alias="chat",
        role="chat",
        container_model_path="/models/chat/test.gguf",
        env=env,
        available_vram_gb=24.0,
        force_refresh=True,
    )
    assert src1 in ("computed", "seeded_from_last_success")

    from model_params_cache import store_plan_in_cache

    store_plan_in_cache(
        alias="chat",
        role="chat",
        container_model_path="/models/chat/test.gguf",
        host_model_path="C:/models/chat/test.gguf",
        plan=plan1,
        global_options={"cache-type-k": "q8_0"},
        per_model_options={},
        source=src1,
        env=env,
        available_vram_gb=24.0,
    )

    plan2, _, _, src2 = plan_model_with_cache(
        alias="chat",
        role="chat",
        container_model_path="/models/chat/test.gguf",
        env=env,
        available_vram_gb=24.0,
        force_refresh=False,
    )
    assert src2 == "cache"
    assert plan2.hard_ctx == plan1.hard_ctx


def test_warm_fleet_stats_keys(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr("model_params_cache.default_cache_path", lambda: cache_path)
    monkeypatch.setattr(
        "model_params_cache.read_gguf_metadata",
        lambda _p: dict(_META),
    )
    monkeypatch.setattr(
        "model_params_cache._file_fingerprint",
        lambda _p: {"exists": True, "size": 100, "mtime_ns": 1, **_META},
    )
    monkeypatch.setattr(
        "model_params_cache.container_path_to_host",
        lambda c, _d: "C:/models/host.gguf" if c.endswith(".gguf") else c,
    )

    env = {
        "LLAMA_MODELS_DIR": "C:/models",
        "CHAT_MODEL_PATH": "/models/chat/test.gguf",
        "UTILITY_MODEL_PATH": "/models/utility/u.gguf",
        "EMBED_MODEL_PATH": "/models/embed/e.gguf",
        "CHAT_CTX_SIZE": "65536",
        "UTILITY_CTX_SIZE": "16384",
        "EMBED_CTX_SIZE": "8192",
        "ROUTER_PARALLEL": "1",
        "A0_LMM_AVAILABLE_VRAM_GB": "24",
    }
    warm = warm_fleet_params(env, force_refresh=True, write_cache=True)
    assert warm["ok"] is True
    assert "cached" in warm["stats"]
    assert len(warm["entries"]) == 3
    data = load_cache(cache_path)
    assert data.get("models")


def test_utility_can_follow_chat_without_duplicate_resident_vram(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr("model_params_cache.default_cache_path", lambda: cache_path)
    monkeypatch.setattr(
        "model_params_cache.read_gguf_metadata",
        lambda _p: dict(_META),
    )
    monkeypatch.setattr(
        "model_params_cache._file_fingerprint",
        lambda _p: {"exists": True, "size": 100, "mtime_ns": 1, **_META},
    )
    monkeypatch.setattr(
        "model_params_cache.container_path_to_host",
        lambda c, _d: "C:/models/chat/test.gguf" if c.endswith(".gguf") else c,
    )
    monkeypatch.setattr(
        "model_params_cache.plan_model_context",
        lambda **kwargs: ContextPlan(
            alias=str(kwargs["alias"]),
            role=str(kwargs["role"]),
            model_path=str(kwargs["model_path"]),
            min_ctx=int(kwargs["min_ctx"]),
            hard_ctx=65536,
            effective_ctx=45875,
            response_reserve=int(kwargs["response_reserve"]),
            effective_ratio=float(kwargs["effective_ratio"]),
            n_ctx_train=131072,
            planned_vram_gb=10.0,
            kv_cache_gb=4.0,
            no_capacity=False,
            reason="test planner",
        ),
    )

    env = {
        "LLAMA_MODELS_DIR": "C:/models",
        "CHAT_MODEL_PATH": "/models/chat/test.gguf",
        "UTILITY_MODEL_PATH": "/models/chat/test.gguf",
        "EMBED_MODEL_PATH": "",
        "CHAT_CTX_SIZE": "65536",
        "UTILITY_CTX_SIZE": "65536",
        "ROUTER_MODELS_MAX": "1",
        "ROUTER_PARALLEL": "1",
        "A0_LMM_UTILITY_FOLLOWS_CHAT": "1",
        "A0_LMM_AVAILABLE_VRAM_GB": "24",
    }

    warm = warm_fleet_params(env, force_refresh=True, write_cache=False)
    by_alias = {entry.alias: entry for entry in warm["entries"]}

    assert by_alias["chat"].model_path == "/models/chat/test.gguf"
    assert by_alias["utility"].model_path == "/models/chat/test.gguf"
    assert by_alias["utility"].hard_ctx == by_alias["chat"].hard_ctx
    assert warm["resident_vram_gb"] == by_alias["chat"].planned_vram_gb
