"""
Pure unit tests for helpers/smart_router/failover.py.
No I/O, no BackendManager, no network — covers the classification and
chain-traversal logic that select_slot_with_failover() depends on.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# classify_failover_reason
# ---------------------------------------------------------------------------

class _FakeExc(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        if status_code is not None:
            self.status_code = status_code


def test_classify_by_status_429():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import (
        classify_failover_reason, FailoverReason,
    )
    result = classify_failover_reason(_FakeExc("too many", status_code=429))
    assert result["reason"] == FailoverReason.RATE_LIMIT
    assert result["status_code"] == 429


def test_classify_by_status_500():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import (
        classify_failover_reason, FailoverReason,
    )
    result = classify_failover_reason(_FakeExc("server error", status_code=500))
    assert result["reason"] == FailoverReason.PROVIDER_ERROR
    assert result["status_code"] == 500


def test_classify_by_status_408():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import (
        classify_failover_reason, FailoverReason,
    )
    result = classify_failover_reason(_FakeExc("timeout", status_code=408))
    assert result["reason"] == FailoverReason.TIMEOUT


def test_classify_by_status_non_failover():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import (
        classify_failover_reason, FailoverReason,
    )
    result = classify_failover_reason(_FakeExc("bad request", status_code=400))
    assert result["reason"] == FailoverReason.HTTP_ERROR


def test_classify_by_text_timeout():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import (
        classify_failover_reason, FailoverReason,
    )
    result = classify_failover_reason(Exception("read timed out after 30s"))
    assert result["reason"] == FailoverReason.TIMEOUT
    assert result["status_code"] is None


def test_classify_by_text_connection_refused():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import (
        classify_failover_reason, FailoverReason,
    )
    result = classify_failover_reason(Exception("connection refused"))
    assert result["reason"] == FailoverReason.PROVIDER_ERROR


def test_classify_by_text_rate_limit():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import (
        classify_failover_reason, FailoverReason,
    )
    result = classify_failover_reason(Exception("weekly usage limit exceeded"))
    assert result["reason"] == FailoverReason.RATE_LIMIT


def test_classify_by_text_quota():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import (
        classify_failover_reason, FailoverReason,
    )
    result = classify_failover_reason(Exception("insufficient_quota: no credits left"))
    assert result["reason"] == FailoverReason.QUOTA_EXHAUSTED


def test_classify_unknown():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import (
        classify_failover_reason, FailoverReason,
    )
    result = classify_failover_reason(Exception("something completely weird"))
    assert result["reason"] == FailoverReason.UNKNOWN_ERROR


# ---------------------------------------------------------------------------
# should_failover
# ---------------------------------------------------------------------------

def test_should_failover_on_503():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import should_failover
    assert should_failover(_FakeExc("service unavailable", status_code=503)) is True


def test_should_not_failover_on_400():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import should_failover
    assert should_failover(_FakeExc("bad request", status_code=400)) is False


def test_should_failover_on_connection_error_text():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import should_failover
    assert should_failover(Exception("connection reset by peer")) is True


# ---------------------------------------------------------------------------
# get_next_in_chain
# ---------------------------------------------------------------------------

def test_get_next_in_chain_normal():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import get_next_in_chain
    assert get_next_in_chain("chat", ["chat", "utility", "openrouter_fallback"]) == "utility"


def test_get_next_in_chain_middle():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import get_next_in_chain
    assert get_next_in_chain("utility", ["chat", "utility", "openrouter_fallback"]) == "openrouter_fallback"


def test_get_next_in_chain_exhausted():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import get_next_in_chain
    assert get_next_in_chain("openrouter_fallback", ["chat", "utility", "openrouter_fallback"]) is None


def test_get_next_in_chain_not_in_chain_starts_from_head():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import get_next_in_chain
    assert get_next_in_chain("unknown_slot", ["chat", "utility"]) == "chat"


def test_get_next_in_chain_empty_chain():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import get_next_in_chain
    assert get_next_in_chain("chat", []) is None


# ---------------------------------------------------------------------------
# get_chain_for_role
# ---------------------------------------------------------------------------

def test_get_chain_for_role_default_chat():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import (
        get_chain_for_role, DEFAULT_CHAINS,
    )
    chain = get_chain_for_role("chat")
    assert chain == DEFAULT_CHAINS["chat"]


def test_get_chain_for_role_custom_overrides_default():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import get_chain_for_role
    custom = {"chat": ["gpu_primary", "gpu_secondary"]}
    assert get_chain_for_role("chat", custom) == ["gpu_primary", "gpu_secondary"]


def test_get_chain_for_role_unknown_role_returns_role_itself():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import get_chain_for_role
    assert get_chain_for_role("vision") == ["vision"]


# ---------------------------------------------------------------------------
# CooldownTracker
# ---------------------------------------------------------------------------

def test_cooldown_tracker_mark_and_track():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import CooldownTracker
    tracker = CooldownTracker()
    tracker.mark_error("slot_chat", "connection refused")
    assert "slot_chat" in tracker.get_error_slots()


def test_cooldown_tracker_mark_recovered_removes_slot():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import CooldownTracker
    tracker = CooldownTracker()
    tracker.mark_error("slot_chat")
    tracker.mark_recovered("slot_chat")
    assert "slot_chat" not in tracker.get_error_slots()


def test_cooldown_tracker_duplicate_mark_error_is_idempotent():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import CooldownTracker
    tracker = CooldownTracker()
    tracker.mark_error("slot_chat", "first error")
    tracker.mark_error("slot_chat", "second error")
    # probe_count should remain 0 (not doubled)
    status = tracker.get_status("slot_chat")
    assert status["probe_count"] == 0


def test_cooldown_tracker_should_probe_respects_max_attempts():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import (
        CooldownTracker, CooldownProbe,
    )
    tracker = CooldownTracker()
    tracker.mark_error("slot_chat")
    probe_cfg = CooldownProbe(enabled=True, interval_seconds=0, max_attempts=2, probe_timeout=1)

    # First two probes: allowed
    tracker.record_probe("slot_chat")
    tracker.record_probe("slot_chat")
    # Third: max_attempts reached
    assert tracker.should_probe("slot_chat", probe_cfg) is False


def test_cooldown_tracker_get_status_unknown_slot_returns_none():
    from usr.plugins.a0_lmm_router.helpers.smart_router.failover import CooldownTracker
    tracker = CooldownTracker()
    assert tracker.get_status("nonexistent") is None
