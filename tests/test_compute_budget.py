"""Tests for the token-economy layer: usage_ledger + budget_engine +
the litellm counting hook (compute-budget feature, Phases A-C)."""
from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from helpers import budget_engine, usage_ledger  # noqa: E402

# api/lmm_providers.py imports flask + helpers.api, which only exist inside the
# Agent Zero host process. Stub them so the module imports standalone for tests.
if "flask" not in sys.modules:
    _flask_stub = types.ModuleType("flask")
    _flask_stub.Request = object
    sys.modules["flask"] = _flask_stub
if "helpers.api" not in sys.modules:
    _helpers_api_stub = types.ModuleType("helpers.api")
    _helpers_api_stub.ApiHandler = object
    sys.modules["helpers.api"] = _helpers_api_stub

from api import lmm_providers  # noqa: E402


NOW = time.time()

LIMITS_YAML = """
version: 1
providers:
  codex:
    name: Codex
    kind: subscription
    invoke: codex_cli
    default_model: gpt-5-codex
    priority: 1
    enabled: true
    limits:
      - {window: "5h", max_tokens: 1000}
      - {window: "7d", max_tokens: 10000}
  ollama_cloud:
    name: Ollama Cloud
    kind: subscription
    invoke: openai_compatible
    base_url: "https://ollama.com/v1"
    default_model: big-model
    priority: 2
    enabled: true
    litellm_prefixes: ["ollama_cloud/", "ollama/"]
    base_url_match: "ollama.com"
    limits:
      - {window: "1d", max_tokens: 5000}
"""

MANIFEST_YAML = """
providers:
  - id: llamacpp_local_chat
    name: local chat
    type: openai_compatible
    base_url: "http://host.docker.internal:8080/v1"
  - id: openrouter
    name: OpenRouter
    type: openai_compatible
    base_url: "https://openrouter.ai/api/v1"
"""


def _ev(ts, provider="codex", tin=100, tout=50, req=1):
    return {"ts": ts, "provider_id": provider, "model": "m", "tokens_in": tin,
            "tokens_out": tout, "requests": req, "source": "test", "requester": ""}


def _write_events(path: Path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_ledger, "DATA_DIR", tmp_path)
    monkeypatch.setattr(usage_ledger, "PROC_TAG", "a0")
    # Disable opportunistic pruning so tests control file contents exactly.
    monkeypatch.setattr(usage_ledger, "_last_prune_day", time.strftime("%Y-%m-%d"))
    usage_ledger._cache.clear()
    return tmp_path


@pytest.fixture()
def registry(tmp_path, monkeypatch, ledger):
    limits = tmp_path / "provider_limits.yaml"
    manifest = tmp_path / "model_providers.yaml"
    limits.write_text(LIMITS_YAML, encoding="utf-8")
    manifest.write_text(MANIFEST_YAML, encoding="utf-8")
    monkeypatch.setattr(budget_engine, "LIMITS_PATH", limits)
    monkeypatch.setattr(budget_engine, "MANIFEST_PATH", manifest)
    budget_engine._reservations.clear()
    return limits, manifest


# ---------------------------------------------------------------------------
# Phase A — ledger + budget engine
# ---------------------------------------------------------------------------

class TestParseWindow:
    def test_hours_and_days(self):
        assert budget_engine.parse_window("5h") == 5 * 3600
        assert budget_engine.parse_window("7d") == 7 * 86400

    @pytest.mark.parametrize("bad", ["", "5x", "h5", "1.5h", "5 h"])
    def test_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            budget_engine.parse_window(bad)


class TestLedger:
    def test_rolling_windows(self, ledger):
        _write_events(ledger / "usage_a0.jsonl", [
            _ev(NOW - 4 * 3600),          # inside 5h
            _ev(NOW - 6 * 3600),          # outside 5h, inside 1d
            _ev(NOW - 2 * 86400),         # inside 7d only
        ])
        assert usage_ledger.window_totals("codex", 5 * 3600, now=NOW)["tokens"] == 150
        assert usage_ledger.window_totals("codex", 86400, now=NOW)["tokens"] == 300
        totals_7d = usage_ledger.window_totals("codex", 7 * 86400, now=NOW)
        assert totals_7d["tokens"] == 450
        assert totals_7d["requests"] == 3

    def test_merges_process_files(self, ledger):
        _write_events(ledger / "usage_a0.jsonl", [_ev(NOW - 60)])
        _write_events(ledger / "usage_mcp.jsonl", [_ev(NOW - 60, tin=10, tout=0)])
        totals = usage_ledger.window_totals("codex", 3600, now=NOW)
        assert totals["tokens"] == 160
        assert totals["requests"] == 2

    def test_record_usage_appends_parseable_line(self, ledger):
        usage_ledger.record_usage("codex", tokens_in=10, tokens_out=5, model="x", source="test")
        lines = (ledger / "usage_a0.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        ev = json.loads(lines[0])
        assert ev["provider_id"] == "codex"
        assert usage_ledger.window_totals("codex", 3600)["tokens"] == 15

    def test_prune_drops_old_events(self, ledger):
        _write_events(ledger / "usage_a0.jsonl", [_ev(NOW - 40 * 86400), _ev(NOW - 60)])
        usage_ledger.prune()
        events = usage_ledger.all_events()
        assert len(events) == 1
        assert events[0]["ts"] == pytest.approx(NOW - 60)


class TestRegistry:
    def test_merge_folds_local_and_adds_external(self, registry):
        reg = budget_engine.load_registry()
        assert "codex" in reg and "ollama_cloud" in reg
        assert "llamacpp_local_chat" not in reg          # folds into `local`
        assert reg["openrouter"]["kind"] == "external"   # tracked, unlimited
        assert reg["openrouter"]["limits"] == []

    def test_save_provider_never_touches_manifest(self, registry):
        limits, manifest = registry
        manifest_bytes = manifest.read_bytes()
        budget_engine.save_provider("newprov", {"name": "New", "kind": "subscription",
                                                "priority": 5, "limits": []})
        assert manifest.read_bytes() == manifest_bytes
        assert "newprov" in budget_engine.load_registry()

    def test_delete_provider(self, registry):
        budget_engine.save_provider("tmpprov", {"name": "T", "limits": []})
        assert budget_engine.delete_provider("tmpprov") is True
        assert budget_engine.delete_provider("tmpprov") is False
        assert budget_engine.delete_provider("openrouter") is False  # manifest-derived


class TestProviderBudget:
    def _codex_cfg(self):
        return budget_engine.load_registry()["codex"]

    def test_status_ok_when_unused(self, registry):
        b = budget_engine.provider_budget("codex", self._codex_cfg(), now=NOW)
        assert b["status"] == "ok"
        assert b["windows"][0]["pct"] == 0.0
        assert b["windows"][0]["remaining"] == 1000

    def test_status_warn_at_80pct(self, registry, ledger):
        _write_events(ledger / "usage_a0.jsonl", [_ev(NOW - 60, tin=800, tout=50)])
        b = budget_engine.provider_budget("codex", self._codex_cfg(), now=NOW)
        assert b["status"] == "warn"
        assert b["worst_pct"] == 85.0

    def test_status_exhausted_at_100pct(self, registry, ledger):
        _write_events(ledger / "usage_a0.jsonl", [_ev(NOW - 60, tin=1000, tout=0)])
        b = budget_engine.provider_budget("codex", self._codex_cfg(), now=NOW)
        assert b["status"] == "exhausted"
        assert b["windows"][0]["remaining"] == 0

    def test_burn_rate_and_eta(self, registry, ledger):
        _write_events(ledger / "usage_a0.jsonl", [_ev(NOW - 600, tin=500, tout=0)])
        b = budget_engine.provider_budget("codex", self._codex_cfg(), now=NOW)
        w5h = b["windows"][0]
        assert w5h["burn_per_hour"] == 500
        assert w5h["exhausts_in_hours"] == 1.0  # 500 remaining / 500 per hour

    def test_compute_budget_shape_and_order(self, registry):
        out = budget_engine.compute_budget()
        ids = [p["id"] for p in out["providers"]]
        assert ids[0] == "codex"  # priority 1 first
        codex = out["providers"][0]
        assert codex["status"] in ("ok", "warn", "exhausted")
        assert len(codex["windows"]) == 2


# ---------------------------------------------------------------------------
# Phase B — route_task + decisions log
# ---------------------------------------------------------------------------

class TestRouteTask:
    def test_picks_priority_one_with_headroom(self, registry, ledger):
        packet = budget_engine.route_task("summarize repo", est_input_tokens=100,
                                          est_output_tokens=100)
        assert packet["provider_id"] == "codex"
        assert packet["model"] == "gpt-5-codex"
        assert packet["invoke"] == "codex_cli"
        assert packet["reservation_id"]
        assert [f["provider_id"] for f in packet["fallback"]][0] == "ollama_cloud"
        # decision logged and parseable
        lines = (ledger / "route_decisions.jsonl").read_text(encoding="utf-8").splitlines()
        decision = json.loads(lines[-1])
        assert decision["chosen"] == "codex"

    def test_skips_exhausted_provider(self, registry, ledger):
        _write_events(ledger / "usage_a0.jsonl", [_ev(NOW - 60, tin=1000, tout=0)])
        packet = budget_engine.route_task("t", est_input_tokens=100, est_output_tokens=0)
        assert packet["provider_id"] == "ollama_cloud"

    def test_reservation_blocks_next_call(self, registry):
        p1 = budget_engine.route_task("t1", est_input_tokens=400, est_output_tokens=200)
        assert p1["provider_id"] == "codex"
        # 600 reserved + 600 needed > 1000 limit -> codex skipped
        p2 = budget_engine.route_task("t2", est_input_tokens=400, est_output_tokens=200)
        assert p2["provider_id"] == "ollama_cloud"

    def test_deterministic_for_same_inputs(self, registry):
        p1 = budget_engine.route_task("same", est_input_tokens=10, est_output_tokens=10)
        budget_engine._reservations.clear()
        p2 = budget_engine.route_task("same", est_input_tokens=10, est_output_tokens=10)
        p1.pop("reservation_id"), p2.pop("reservation_id")
        assert p1 == p2

    def test_local_preferred_when_quality_fast(self, registry, monkeypatch):
        budget_engine.save_provider("local", {"name": "Local", "kind": "local", "priority": 3})
        monkeypatch.setattr(budget_engine, "local_capacity", lambda: {
            "gross_vram_mb": 16000, "free_vram_mb": 8000,
            "ram_total_mb": 32000, "ram_free_mb": 16000,
            "slots": [{"id": "slot_chat", "role": "chat", "model_id": "qwen3.5-9b",
                       "port": 8080, "running": True, "healthy": True,
                       "est_tok_per_sec": None}],
        })
        packet = budget_engine.route_task("t", role="chat", quality="fast")
        assert packet["provider_id"] == "local"
        assert packet["model"] == "qwen3.5-9b"
        assert packet["base_url"] == "http://localhost:8080/v1"

    def test_all_exhausted_returns_null_packet(self, registry, ledger):
        limits, manifest = registry
        manifest.write_text("providers: []\n", encoding="utf-8")  # drop unlimited openrouter
        _write_events(ledger / "usage_a0.jsonl", [
            _ev(NOW - 60, provider="codex", tin=1000, tout=0),
            _ev(NOW - 60, provider="ollama_cloud", tin=5000, tout=0),
        ])
        packet = budget_engine.route_task("t", est_input_tokens=100, est_output_tokens=0)
        assert packet["provider_id"] is None
        assert "exhausted" in packet["decision_reason"]
        assert len(packet["fallback"]) == 2  # full ordered list so the agent can wait


# ---------------------------------------------------------------------------
# Phase C — litellm counting hook (no double counting)
# ---------------------------------------------------------------------------

def _install_fake_litellm(monkeypatch, acompletion_impl):
    mod = types.ModuleType("litellm")
    mod.acompletion = acompletion_impl
    monkeypatch.setitem(sys.modules, "litellm", mod)
    return mod


class TestLitellmHook:
    def test_records_exactly_one_event_by_prefix(self, registry, ledger, monkeypatch):
        from helpers import rate_limit_retry as rlr

        async def fake_acompletion(*a, **kw):
            return types.SimpleNamespace(usage=types.SimpleNamespace(
                prompt_tokens=100, completion_tokens=40))

        mod = _install_fake_litellm(monkeypatch, fake_acompletion)
        assert rlr.patch_acompletion() is True
        asyncio.run(mod.acompletion(model="ollama_cloud/qwen3:480b"))

        events = usage_ledger.all_events()
        assert len(events) == 1
        assert events[0]["provider_id"] == "ollama_cloud"
        assert events[0]["tokens_in"] == 100
        assert events[0]["tokens_out"] == 40
        assert events[0]["source"] == "litellm"

    def test_api_base_match_wins(self, registry, ledger, monkeypatch):
        from helpers import rate_limit_retry as rlr

        async def fake_acompletion(*a, **kw):
            return {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        mod = _install_fake_litellm(monkeypatch, fake_acompletion)
        assert rlr.patch_acompletion() is True
        asyncio.run(mod.acompletion(model="openai/whatever",
                                    api_base="https://ollama.com/v1"))
        events = usage_ledger.all_events()
        assert len(events) == 1
        assert events[0]["provider_id"] == "ollama_cloud"

    def test_unmatched_model_not_recorded(self, registry, ledger, monkeypatch):
        from helpers import rate_limit_retry as rlr

        async def fake_acompletion(*a, **kw):
            return {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        mod = _install_fake_litellm(monkeypatch, fake_acompletion)
        assert rlr.patch_acompletion() is True
        asyncio.run(mod.acompletion(model="gemini/flash"))
        assert usage_ledger.all_events() == []

    def test_unified_call_does_not_double_count(self, registry, ledger, monkeypatch):
        from helpers import rate_limit_retry as rlr

        async def fake_acompletion(*a, **kw):
            return {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        _install_fake_litellm(monkeypatch, fake_acompletion)

        models_mod = types.ModuleType("models")

        class Model:
            async def unified_call(self, *a, **kw):
                import litellm
                return await litellm.acompletion(model="ollama_cloud/x", **kw)

        models_mod.Model = Model
        monkeypatch.setitem(sys.modules, "models", models_mod)

        assert rlr.patch_acompletion() is True
        assert rlr.patch_model_unified_call() is True
        asyncio.run(Model().unified_call())

        events = usage_ledger.all_events()
        assert len(events) == 1  # acompletion layer only — no double count

    def test_recording_failure_does_not_break_call(self, registry, ledger, monkeypatch):
        from helpers import rate_limit_retry as rlr

        async def fake_acompletion(*a, **kw):
            return {"choices": [{"message": {"content": "hi"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        mod = _install_fake_litellm(monkeypatch, fake_acompletion)
        assert rlr.patch_acompletion() is True
        monkeypatch.setattr(rlr, "_record_litellm_usage",
                            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        result = asyncio.run(mod.acompletion(model="ollama_cloud/x"))
        assert result["choices"][0]["message"]["content"] == "hi"


# ---------------------------------------------------------------------------
# Provider validation (_validate_provider input-guard fixes)
# ---------------------------------------------------------------------------
class TestProviderValidation:
    def test_max_tokens_zero_rejected(self):
        err = lmm_providers._validate_provider(
            "openrouter", {"limits": [{"window": "1d", "max_tokens": 0}]})
        assert err

    def test_max_requests_zero_rejected(self):
        err = lmm_providers._validate_provider(
            "openrouter", {"limits": [{"window": "1d", "max_requests": 0}]})
        assert err

    def test_negative_max_tokens_rejected(self):
        err = lmm_providers._validate_provider(
            "openrouter", {"limits": [{"window": "1d", "max_tokens": -5}]})
        assert err

    def test_valid_positive_limit_accepted(self):
        err = lmm_providers._validate_provider(
            "openrouter", {"limits": [{"window": "1d", "max_tokens": 1000}]})
        assert err == ""

    def test_window_over_retention_rejected(self):
        err = lmm_providers._validate_provider(
            "openrouter", {"limits": [{"window": "60d", "max_tokens": 1000}]})
        assert err

    def test_window_30d_accepted(self):
        err = lmm_providers._validate_provider(
            "openrouter", {"limits": [{"window": "30d", "max_tokens": 1000}]})
        assert err == ""

    def test_window_7d_accepted(self):
        err = lmm_providers._validate_provider(
            "openrouter", {"limits": [{"window": "7d", "max_tokens": 1000}]})
        assert err == ""

    def test_reserved_id_local_rejected(self):
        err = lmm_providers._validate_provider("local", {"limits": []})
        assert err

    def test_normal_pid_accepted(self):
        err = lmm_providers._validate_provider("openrouter", {"limits": []})
        assert err == ""
