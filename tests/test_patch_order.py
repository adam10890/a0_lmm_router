"""Patch-order unit tests for rate_limit_retry + tiny_router interaction.

Verifies that:
1. patch_litellm() correctly wraps litellm.acompletion and Model.unified_call.
2. Patches are idempotent (safe to call multiple times).
3. is_rate_limit_error() correctly identifies retryable errors.
4. The with_retry decorator performs exponential backoff on RateLimitError.
5. Preset switching (which reassigns Agent.config.chat_model) does NOT undo
   the class-level Model.unified_call patch — this is the critical invariant
   because tiny_router swaps presets mid-session.

Run from repo root:
    python -m pytest usr/plugins/a0_lmm_router/tests/test_patch_order.py -v
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make repo root importable so `usr.plugins...` resolves
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Fake litellm + models modules (so tests run without A0 container)
# ---------------------------------------------------------------------------


class FakeRateLimitError(Exception):
    """Mimics litellm.exceptions.RateLimitError."""
    pass


FakeRateLimitError.__module__ = "litellm.exceptions"


def _install_fake_litellm(monkeypatch, acompletion_impl):
    mod = types.ModuleType("litellm")
    mod.acompletion = acompletion_impl
    exc_mod = types.ModuleType("litellm.exceptions")
    exc_mod.RateLimitError = FakeRateLimitError
    mod.exceptions = exc_mod
    monkeypatch.setitem(sys.modules, "litellm", mod)
    monkeypatch.setitem(sys.modules, "litellm.exceptions", exc_mod)
    return mod


def _install_fake_models(monkeypatch, unified_call_impl):
    mod = types.ModuleType("models")

    class Model:
        pass

    Model.unified_call = unified_call_impl  # type: ignore[method-assign]
    mod.Model = Model
    monkeypatch.setitem(sys.modules, "models", mod)
    return mod, Model


def _fresh_import_retry_module(monkeypatch):
    """Re-import helpers.rate_limit_retry cleanly for each test."""
    # Remove any cached version
    for key in list(sys.modules.keys()):
        if "rate_limit_retry" in key:
            del sys.modules[key]
    from usr.plugins.a0_lmm_router.helpers import rate_limit_retry  # noqa: WPS433
    return rate_limit_retry


# ---------------------------------------------------------------------------
# is_rate_limit_error
# ---------------------------------------------------------------------------


class TestIsRateLimitError:
    def test_litellm_rate_limit_error_detected(self, monkeypatch):
        rlr = _fresh_import_retry_module(monkeypatch)
        assert rlr.is_rate_limit_error(FakeRateLimitError("rate limited"))

    def test_http_429_detected(self, monkeypatch):
        rlr = _fresh_import_retry_module(monkeypatch)
        assert rlr.is_rate_limit_error(Exception("Provider returned error 429"))

    def test_too_many_requests_detected(self, monkeypatch):
        rlr = _fresh_import_retry_module(monkeypatch)
        assert rlr.is_rate_limit_error(Exception("Too Many Requests"))

    def test_non_rate_limit_not_detected(self, monkeypatch):
        rlr = _fresh_import_retry_module(monkeypatch)
        assert not rlr.is_rate_limit_error(ValueError("bad input"))

    def test_none_not_detected(self, monkeypatch):
        rlr = _fresh_import_retry_module(monkeypatch)
        assert not rlr.is_rate_limit_error(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# with_retry decorator
# ---------------------------------------------------------------------------


class TestWithRetry:
    def test_succeeds_immediately_no_retry(self, monkeypatch):
        rlr = _fresh_import_retry_module(monkeypatch)
        calls = {"n": 0}

        @rlr.with_retry(max_retries=3, base_delay=0.01)
        async def ok():
            calls["n"] += 1
            return "ok"

        result = asyncio.run(ok())
        assert result == "ok"
        assert calls["n"] == 1

    def test_retries_on_rate_limit_then_succeeds(self, monkeypatch):
        rlr = _fresh_import_retry_module(monkeypatch)
        calls = {"n": 0}

        @rlr.with_retry(max_retries=5, base_delay=0.001, max_delay=0.005)
        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise FakeRateLimitError("429 rate limit")
            return "ok"

        result = asyncio.run(flaky())
        assert result == "ok"
        assert calls["n"] == 3

    def test_non_retryable_propagates_immediately(self, monkeypatch):
        rlr = _fresh_import_retry_module(monkeypatch)
        calls = {"n": 0}

        @rlr.with_retry(max_retries=5, base_delay=0.001)
        async def bad():
            calls["n"] += 1
            raise ValueError("invalid input")

        with pytest.raises(ValueError):
            asyncio.run(bad())
        assert calls["n"] == 1  # No retries for non-rate-limit errors

    def test_exhausts_retries_raises_last_exception(self, monkeypatch):
        rlr = _fresh_import_retry_module(monkeypatch)
        calls = {"n": 0}

        @rlr.with_retry(max_retries=3, base_delay=0.001, max_delay=0.002)
        async def always_throttled():
            calls["n"] += 1
            raise FakeRateLimitError("persistent 429")

        with pytest.raises(FakeRateLimitError):
            asyncio.run(always_throttled())
        assert calls["n"] == 3


# ---------------------------------------------------------------------------
# patch_litellm — core patch-order invariant
# ---------------------------------------------------------------------------


class TestPatchLitellm:
    def test_patch_acompletion_wraps_original(self, monkeypatch):
        async def orig_acompletion(*a, **kw):
            return {"ok": True}

        litellm = _install_fake_litellm(monkeypatch, orig_acompletion)
        rlr = _fresh_import_retry_module(monkeypatch)

        assert rlr.patch_acompletion() is True
        assert hasattr(litellm, "_original_acompletion")
        assert litellm._original_acompletion is orig_acompletion
        assert litellm.acompletion is not orig_acompletion  # wrapped

    def test_patch_idempotent(self, monkeypatch):
        """Calling patch twice should not double-wrap."""
        async def orig_acompletion(*a, **kw):
            return None

        litellm = _install_fake_litellm(monkeypatch, orig_acompletion)
        rlr = _fresh_import_retry_module(monkeypatch)

        rlr.patch_acompletion()
        wrapped_once = litellm.acompletion
        rlr.patch_acompletion()
        wrapped_twice = litellm.acompletion
        assert wrapped_once is wrapped_twice, "Double-patching detected"

    def test_patch_model_unified_call_wraps_original(self, monkeypatch):
        async def orig_unified(self, *a, **kw):
            return "result"

        _, Model = _install_fake_models(monkeypatch, orig_unified)
        rlr = _fresh_import_retry_module(monkeypatch)

        assert rlr.patch_model_unified_call() is True
        assert hasattr(Model, "_original_unified_call")
        assert Model._original_unified_call is orig_unified
        assert Model.unified_call is not orig_unified

    def test_patched_acompletion_retries_on_429(self, monkeypatch):
        call_count = {"n": 0}

        async def flaky_acompletion(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise FakeRateLimitError("rate limit")
            return {"choice": "final"}

        litellm = _install_fake_litellm(monkeypatch, flaky_acompletion)
        rlr = _fresh_import_retry_module(monkeypatch)

        # Shrink delays for fast test
        monkeypatch.setattr(rlr, "DEFAULT_BASE_DELAY", 0.001)
        monkeypatch.setattr(rlr, "DEFAULT_MAX_DELAY", 0.005)

        rlr.patch_acompletion()
        result = asyncio.run(litellm.acompletion())
        assert result == {"choice": "final"}
        assert call_count["n"] == 2

    def test_patch_litellm_umbrella_returns_true_when_any_patch_applied(self, monkeypatch):
        async def orig(*a, **kw):
            return None

        _install_fake_litellm(monkeypatch, orig)
        # models NOT installed — simulating running outside A0 container
        for key in list(sys.modules.keys()):
            if key == "models" or key.startswith("models."):
                del sys.modules[key]

        rlr = _fresh_import_retry_module(monkeypatch)
        assert rlr.patch_litellm() is True  # at least acompletion was patched


# ---------------------------------------------------------------------------
# Critical: preset switching should NOT undo the class-level patch
# ---------------------------------------------------------------------------


class TestPresetSwitchingPreservesPatch:
    """tiny_router overrides Agent.config.chat_model to swap presets mid-session.

    The rate_limit_retry patch is applied at the *class* level
    (Model.unified_call). Replacing config.chat_model with a new Model()
    instance must NOT bypass the patch because class-level methods are
    inherited by all instances, including newly-created ones.
    """

    def test_new_model_instance_inherits_patched_method(self, monkeypatch):
        async def orig_unified(self, *a, **kw):
            return "original"

        _, Model = _install_fake_models(monkeypatch, orig_unified)
        rlr = _fresh_import_retry_module(monkeypatch)
        rlr.patch_model_unified_call()

        # Simulate tiny_router swapping to a new Model instance mid-session
        model_instance_old = Model()
        preset_switch = Model()  # simulated "Local Fleet" preset
        model_instance_new = Model()

        # Both old and new instances should resolve to the patched method
        assert model_instance_old.unified_call.__func__ is Model.unified_call
        assert model_instance_new.unified_call.__func__ is Model.unified_call
        assert preset_switch.unified_call.__func__ is Model.unified_call

    def test_patched_method_still_retries_after_preset_switch(self, monkeypatch):
        call_count = {"n": 0}

        async def orig_unified(self, *a, **kw):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise FakeRateLimitError("429")
            return "success"

        _, Model = _install_fake_models(monkeypatch, orig_unified)
        rlr = _fresh_import_retry_module(monkeypatch)
        monkeypatch.setattr(rlr, "DEFAULT_BASE_DELAY", 0.001)
        monkeypatch.setattr(rlr, "DEFAULT_MAX_DELAY", 0.005)
        rlr.patch_model_unified_call()

        # Simulate a preset switch: tiny_router assigns a new Model instance
        # to agent.config.chat_model
        agent_config = types.SimpleNamespace(chat_model=Model())

        # First call (initial preset)
        result1 = asyncio.run(agent_config.chat_model.unified_call())
        assert result1 == "success"

        # Switch preset (new Model instance)
        agent_config.chat_model = Model()
        call_count["n"] = 0  # reset to exercise retry again

        # Second call (new preset, same patched class) — retry still active
        result2 = asyncio.run(agent_config.chat_model.unified_call())
        assert result2 == "success"
        assert call_count["n"] == 2, "Retry logic was not active after preset switch"
