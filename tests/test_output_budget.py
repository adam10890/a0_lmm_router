"""Tests for Local Fleet output budget enforcement."""
from __future__ import annotations

import sys
import types
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HELPERS_ROOT = PLUGIN_ROOT / "helpers"
if str(HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPERS_ROOT))

from output_budget import (  # noqa: E402
    apply_output_budget,
    limits_for_role,
    output_budget_enabled,
    should_apply_to_model,
)


def test_injects_default_when_missing():
    kwargs, decision = apply_output_budget({}, role="chat")

    assert kwargs["max_tokens"] == 2048
    assert decision.applied
    assert decision.reason == "default_injected"


def test_preserves_smaller_requested_limit():
    kwargs, decision = apply_output_budget({"max_tokens": 512}, role="chat")

    assert kwargs["max_tokens"] == 512
    assert not decision.applied
    assert decision.reason == "within_limit"


def test_caps_excessive_requested_limit():
    kwargs, decision = apply_output_budget({"max_tokens": 10000}, role="chat")

    assert kwargs["max_tokens"] == 4096
    assert decision.applied
    assert decision.requested_tokens == 10000
    assert decision.reason == "hard_cap_applied"


def test_utility_uses_smaller_defaults():
    kwargs, decision = apply_output_budget({}, role="utility")

    assert kwargs["max_tokens"] == 768
    assert decision.hard_max_tokens == 1024


def test_respects_config_overrides():
    kwargs, decision = apply_output_budget(
        {},
        role="utility",
        config={
            "output_budget": {
                "utility": {
                    "default_max_tokens": 256,
                    "hard_max_tokens": 512,
                }
            }
        },
    )

    assert kwargs["max_tokens"] == 256
    assert decision.hard_max_tokens == 512


def test_disabled_budget_leaves_kwargs_unchanged():
    kwargs, decision = apply_output_budget(
        {},
        role="chat",
        config={"output_budget": {"enabled": False}},
    )

    assert kwargs == {}
    assert not decision.enabled
    assert not decision.applied


def test_limits_default_cannot_exceed_hard_cap():
    limits = limits_for_role(
        "chat",
        {"output_budget": {"chat": {"default_max_tokens": 9999, "hard_max_tokens": 1000}}},
    )

    assert limits == {"default_max_tokens": 1000, "hard_max_tokens": 1000}


def test_should_apply_to_local_fleet_model_config():
    model = types.SimpleNamespace(
        model_name="lmm_router/chat",
        a0_model_conf=types.SimpleNamespace(provider="lmm_router", api_base=""),
    )

    assert should_apply_to_model(model)


def test_should_not_apply_to_external_model():
    model = types.SimpleNamespace(
        model_name="openai/gpt-4.1",
        a0_model_conf=types.SimpleNamespace(provider="openai", api_base="https://api.openai.com/v1"),
    )

    assert not should_apply_to_model(model)


def test_enabled_default():
    assert output_budget_enabled({}) is True
