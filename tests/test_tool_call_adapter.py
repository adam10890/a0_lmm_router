"""Tests for Local Fleet tool-call normalization."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HELPERS_ROOT = PLUGIN_ROOT / "helpers"
if str(HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPERS_ROOT))

from tool_call_adapter import (  # noqa: E402
    DEFAULT_WIKI_TOOL_ALLOWLIST,
    adapt_final_model_response,
    normalize_tool_call_output,
    wrap_markdown_as_response,
)


def _decode_normalized(raw: str) -> dict:
    normalized = normalize_tool_call_output(raw, allowed_tools=DEFAULT_WIKI_TOOL_ALLOWLIST)
    assert normalized is not None
    return json.loads(normalized)


def test_normalizes_local_call_with_unquoted_key():
    payload = _decode_normalized('<|tool_call>call:wiki_query{question: "מה יש ב-commons?"}')

    assert payload == {
        "tool_name": "wiki_query",
        "tool_args": {"question": "מה יש ב-commons?"},
    }


def test_normalizes_local_call_with_array_argument():
    payload = _decode_normalized(
        '<|tool_call>call:wiki_query{question: "what do we know?", wikis: ["commons"]}'
    )

    assert payload == {
        "tool_name": "wiki_query",
        "tool_args": {"question": "what do we know?", "wikis": ["commons"]},
    }


def test_does_not_quote_key_like_text_inside_string_values():
    payload = _decode_normalized(
        '<|tool_call>call:wiki_query{question: "compare foo, wikis: should stay text"}'
    )

    assert payload == {
        "tool_name": "wiki_query",
        "tool_args": {"question": "compare foo, wikis: should stay text"},
    }


def test_plain_markdown_is_not_normalized():
    assert normalize_tool_call_output("## Answer\n\nNo tool needed.") is None


def test_wraps_plain_markdown_as_response_tool():
    wrapped = wrap_markdown_as_response("### Answer\n\nזו תשובה ארוכה מספיק כדי לעבור את סף האורך המינימלי.")
    payload = json.loads(wrapped)

    assert payload["tool_name"] == "response"
    assert "תשובה ארוכה" in payload["tool_args"]["text"]


def test_adapt_final_model_response_prefers_tool_call_over_markdown():
    raw = '<|tool_call>call:wiki_list{verbose:false}'
    adapted = adapt_final_model_response(raw, allowed_tools=DEFAULT_WIKI_TOOL_ALLOWLIST)
    payload = json.loads(adapted)

    assert payload["tool_name"] == "wiki_list"


def test_multiple_tool_calls_are_not_normalized():
    raw = (
        '<|tool_call>call:wiki_query{question: "one"}\n'
        '<|tool_call>call:wiki_query{question: "two"}'
    )

    assert normalize_tool_call_output(raw, allowed_tools=DEFAULT_WIKI_TOOL_ALLOWLIST) is None


def test_disallowed_tool_is_not_normalized_for_wiki_librarian():
    raw = '<|tool_call>call:shell{command: "whoami"}'

    assert normalize_tool_call_output(raw, allowed_tools=DEFAULT_WIKI_TOOL_ALLOWLIST) is None


def test_normalizes_tokenized_wikis_array_from_live_logs():
    payload = _decode_normalized(
        '<|tool_call>call:wiki_query{question:"test", wikis:[<|"|>commons<|"|>]}<tool_call|>'
    )

    assert payload == {
        "tool_name": "wiki_query",
        "tool_args": {"question": "test", "wikis": ["commons"]},
    }


def test_normalizes_colon_prefixed_tool_path_to_allowlisted_name():
    payload = _decode_normalized(
        '<tool_call>call:google:search:wiki_list{verbose:false}</tool_call>'
    )

    assert payload == {
        "tool_name": "wiki_list",
        "tool_args": {"verbose": False},
    }


def test_extension_wraps_response_callback_for_local_fleet(monkeypatch):
    helpers = types.ModuleType("helpers")
    extension_mod = types.ModuleType("helpers.extension")

    class Extension:
        def __init__(self, agent=None):
            self.agent = agent

    extension_mod.Extension = Extension
    monkeypatch.setitem(sys.modules, "helpers", helpers)
    monkeypatch.setitem(sys.modules, "helpers.extension", extension_mod)

    import importlib.util

    ext_path = (
        PLUGIN_ROOT
        / "extensions"
        / "python"
        / "chat_model_call_before"
        / "_20_tool_call_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("tool_call_adapter_extension", ext_path)
    assert spec is not None and spec.loader is not None
    ext_module = importlib.util.module_from_spec(spec)
    sys.modules["tool_call_adapter_extension"] = ext_module
    spec.loader.exec_module(ext_module)

    runtime_path = PLUGIN_ROOT / "helpers" / "tool_call_adapter_runtime.py"
    runtime_spec = importlib.util.spec_from_file_location("tool_call_adapter_runtime", runtime_path)
    assert runtime_spec is not None and runtime_spec.loader is not None
    runtime_module = importlib.util.module_from_spec(runtime_spec)
    sys.modules["tool_call_adapter_runtime"] = runtime_module
    runtime_spec.loader.exec_module(runtime_module)
    monkeypatch.setattr(runtime_module, "is_local_fleet_chat_active", lambda agent: True)

    logged: list[dict] = []

    class FakeLog:
        def log(self, **kwargs):
            logged.append(kwargs)

    class FakeContext:
        log = FakeLog()

    class FakeConfig:
        profile = "wiki_librarian"

    class FakeAgent:
        config = FakeConfig()
        context = FakeContext()

    seen: list[tuple[str, str]] = []

    async def original_callback(chunk: str, full: str):
        seen.append((chunk, full))
        return full

    call_data = {"response_callback": original_callback}
    adapter = ext_module.LocalFleetToolCallAdapter(FakeAgent())

    import asyncio

    asyncio.run(adapter.execute(call_data=call_data))
    normalized = asyncio.run(
        call_data["response_callback"](
            '<|tool_call>call:wiki_query{question: "secret question"}',
            '<|tool_call>call:wiki_query{question: "secret question"}',
        )
    )

    assert json.loads(normalized) == {
        "tool_name": "wiki_query",
        "tool_args": {"question": "secret question"},
    }
    assert seen == [(normalized, normalized)]
    assert logged
    assert "secret question" not in json.dumps(logged)


def test_extension_does_not_wrap_outside_local_fleet_and_librarian(monkeypatch):
    helpers = types.ModuleType("helpers")
    extension_mod = types.ModuleType("helpers.extension")

    class Extension:
        def __init__(self, agent=None):
            self.agent = agent

    extension_mod.Extension = Extension
    monkeypatch.setitem(sys.modules, "helpers", helpers)
    monkeypatch.setitem(sys.modules, "helpers.extension", extension_mod)

    import importlib.util

    ext_path = (
        PLUGIN_ROOT
        / "extensions"
        / "python"
        / "chat_model_call_before"
        / "_20_tool_call_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("inactive_tool_call_adapter_extension", ext_path)
    assert spec is not None and spec.loader is not None
    ext_module = importlib.util.module_from_spec(spec)
    sys.modules["inactive_tool_call_adapter_extension"] = ext_module
    spec.loader.exec_module(ext_module)

    runtime_path = PLUGIN_ROOT / "helpers" / "tool_call_adapter_runtime.py"
    runtime_spec = importlib.util.spec_from_file_location("inactive_tool_call_adapter_runtime", runtime_path)
    assert runtime_spec is not None and runtime_spec.loader is not None
    runtime_module = importlib.util.module_from_spec(runtime_spec)
    sys.modules["inactive_tool_call_adapter_runtime"] = runtime_module
    runtime_spec.loader.exec_module(runtime_module)
    monkeypatch.setattr(runtime_module, "is_local_fleet_chat_active", lambda agent: False)

    class FakeConfig:
        profile = "agent0"

    class FakeAgent:
        config = FakeConfig()

    async def original_callback(chunk: str, full: str):
        return None

    call_data = {"response_callback": original_callback}
    adapter = ext_module.LocalFleetToolCallAdapter(FakeAgent())

    import asyncio

    asyncio.run(adapter.execute(call_data=call_data))

    assert call_data["response_callback"] is original_callback
    assert "_a0_lmm_tool_call_adapter_wrapped" not in call_data


def test_model_proxy_injects_local_fleet_output_budget(monkeypatch):
    import importlib

    runtime_module = importlib.import_module("tool_call_adapter_runtime")
    monkeypatch.setattr(runtime_module, "should_enable_tool_call_adapter", lambda agent: False)

    class FakeLog:
        def log(self, **kwargs):
            pass

    class FakeContext:
        log = FakeLog()

    class FakeAgent:
        context = FakeContext()

        def set_data(self, key, value):
            self.telemetry = (key, value)

    class FakeModelConfig:
        provider = "lmm_router"
        api_base = "http://host.docker.internal:8080/v1"

    class FakeModel:
        model_name = "lmm_router/chat"
        a0_model_conf = FakeModelConfig()

        async def unified_call(self, *args, **kwargs):
            self.kwargs = kwargs
            return "ok", ""

    call_data = {"model": FakeModel()}
    runtime_module.prepare_local_fleet_call_data(FakeAgent(), call_data, role="chat")

    import asyncio

    response, _ = asyncio.run(call_data["model"].unified_call())

    assert response == "ok"
    assert call_data["model"]._model.kwargs["max_tokens"] == 2048


def test_utility_proxy_can_route_to_chat_alias_without_losing_utility_budget(monkeypatch):
    import importlib

    runtime_module = importlib.import_module("tool_call_adapter_runtime")
    monkeypatch.setattr(runtime_module, "should_enable_tool_call_adapter", lambda agent: False)
    monkeypatch.setattr(
        runtime_module,
        "_plugin_config",
        lambda agent: {
            "local_fleet": {
                "route_utility_to_chat_alias": True,
                "utility_chat_alias": "chat",
            }
        },
    )

    class FakeLog:
        def log(self, **kwargs):
            pass

    class FakeContext:
        log = FakeLog()

    class FakeAgent:
        context = FakeContext()

        def set_data(self, key, value):
            self.telemetry = (key, value)

    class FakeModelConfig:
        provider = "lmm_router"
        api_base = "http://host.docker.internal:8080/v1"

    class FakeModel:
        model_name = "lmm_router/utility"
        a0_model_conf = FakeModelConfig()

        async def unified_call(self, *args, **kwargs):
            self.seen_model_name = self.model_name
            self.kwargs = kwargs
            return "ok", ""

    model = FakeModel()
    call_data = {"model": model}
    runtime_module.prepare_local_fleet_call_data(FakeAgent(), call_data, role="utility")

    import asyncio

    response, _ = asyncio.run(call_data["model"].unified_call())

    assert response == "ok"
    assert model.seen_model_name == "lmm_router/chat"
    assert model.model_name == "lmm_router/utility"
    assert call_data["model"]._model.kwargs["max_tokens"] == 768
