"""
Tests for Phase 6 OpenAI-compatible provider shell.

GET  /v1/models            — model list from slots
POST /v1/chat/completions  — routing decision; no inference; no fake output

All tests use stub health checkers; no real llama.cpp servers needed.
Slot IDs must match DEFAULT_CHAINS keys for routing to succeed.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from starlette.testclient import TestClient  # noqa: E402

# Slot IDs match DEFAULT_CHAINS so routing succeeds without custom failover_chains.
_ROUTING_CONFIG = """\
active_slots:
  - id: chat
    port: 8080
    host: localhost
    role: chat
    enabled: true
    model_id: mistral-7b-q4
  - id: utility
    port: 8088
    host: localhost
    role: utility
    enabled: true
global:
  backend: remote
"""

_EMPTY_CONFIG = "active_slots: []\nglobal:\n  backend: remote\n"

_SECRETS_CONFIG = """\
active_slots:
  - id: chat
    port: 8080
    host: localhost
    role: chat
    enabled: true
global:
  backend: remote
  api_key: "should-be-redacted"
  some_token: "also-redacted"
"""


def _make_client(tmp_path, yaml_content=_ROUTING_CONFIG, health_result="healthy", post_fn=None):
    """Return a TestClient with stub health checker and /v1/* routes wired up.

    post_fn: injectable async callable(url, payload, timeout) → (status, dict).
    Default raises aiohttp.ClientError, simulating a refused connection so tests
    never accidentally hit a real llama.cpp server.
    """
    import json as _json

    from pydantic import ValidationError
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from service.observer import ObserverBackend
    from service.openai_compat import OpenAIChatRequest, OpenAICompatHandler
    from service.routing_intent import RoutingIntentHandler
    from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager

    cfg = tmp_path / "llama_cpp_servers.yaml"
    cfg.write_text(yaml_content)

    BackendManager._instance = None
    mgr = BackendManager(str(cfg))

    _hr = health_result

    class StubChecker:
        async def check_async(self, config):
            return _hr
        def check(self, config):
            return _hr

    mgr._health_checker = StubChecker()

    obs = ObserverBackend(str(cfg))
    obs._make_manager = lambda: mgr

    intent_handler = RoutingIntentHandler(obs)

    # Default: simulate a connection-refused error so no real server is needed.
    if post_fn is None:
        import aiohttp as _aiohttp
        async def post_fn(url, payload, timeout=120):  # noqa: E306
            raise _aiohttp.ClientError(f"Connection refused (test stub): {url}")

    compat_handler = OpenAICompatHandler(obs, intent_handler, _post_fn=post_fn)

    async def v1_models(request: Request) -> JSONResponse:
        return JSONResponse(compat_handler.get_models().model_dump())

    async def v1_chat_completions(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid_json"}, status_code=400)
        try:
            req = OpenAIChatRequest.model_validate(body)
        except ValidationError as exc:
            return JSONResponse(
                {"error": "validation_error", "detail": _json.loads(exc.json())},
                status_code=422,
            )
        status, result = await compat_handler.handle_chat_completion(req)
        return JSONResponse(result, status_code=status)

    stub_app = Starlette(routes=[
        Route("/v1/models", v1_models, methods=["GET"]),
        Route("/v1/chat/completions", v1_chat_completions, methods=["POST"]),
    ])
    return TestClient(stub_app, raise_server_exceptions=True), mgr


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

class TestModelsEndpoint:
    def test_returns_list_object(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.get("/v1/models").json()
        assert body["object"] == "list"

    def test_data_is_list(self, tmp_path):
        client, _ = _make_client(tmp_path)
        assert isinstance(client.get("/v1/models").json()["data"], list)

    def test_each_model_has_required_openai_fields(self, tmp_path):
        client, _ = _make_client(tmp_path)
        for m in client.get("/v1/models").json()["data"]:
            assert "id" in m
            assert m["object"] == "model"
            assert "owned_by" in m

    def test_owned_by_local(self, tmp_path):
        client, _ = _make_client(tmp_path)
        for m in client.get("/v1/models").json()["data"]:
            assert m["owned_by"] == "local"

    def test_includes_configured_slot_ids(self, tmp_path):
        client, _ = _make_client(tmp_path)
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
        assert "chat" in ids
        assert "utility" in ids

    def test_metadata_includes_role(self, tmp_path):
        client, _ = _make_client(tmp_path)
        chat = next(m for m in client.get("/v1/models").json()["data"] if m["id"] == "chat")
        assert chat["metadata"]["role"] == "chat"

    def test_metadata_includes_model_id(self, tmp_path):
        client, _ = _make_client(tmp_path)
        chat = next(m for m in client.get("/v1/models").json()["data"] if m["id"] == "chat")
        assert chat["metadata"]["model_id"] == "mistral-7b-q4"

    def test_no_secrets_in_response(self, tmp_path):
        client, _ = _make_client(tmp_path, _SECRETS_CONFIG)
        text = str(client.get("/v1/models").json())
        assert "should-be-redacted" not in text
        assert "also-redacted" not in text

    def test_empty_config_returns_empty_data(self, tmp_path):
        client, _ = _make_client(tmp_path, _EMPTY_CONFIG)
        body = client.get("/v1/models").json()
        assert body["data"] == []


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — status codes and structure
# ---------------------------------------------------------------------------

class TestChatCompletionsStatus:
    def test_healthy_slot_connection_error_returns_502(self, tmp_path):
        # Default post_fn raises ClientError (no real server) → 502.
        client, _ = _make_client(tmp_path)
        resp = client.post("/v1/chat/completions", json={"model": "chat", "messages": []})
        assert resp.status_code == 502

    def test_stream_true_returns_400(self, tmp_path):
        client, _ = _make_client(tmp_path)
        resp = client.post("/v1/chat/completions", json={
            "model": "chat", "messages": [], "stream": True,
        })
        assert resp.status_code == 400

    def test_stream_error_code(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={
            "model": "chat", "messages": [], "stream": True,
        }).json()
        assert body["error"]["code"] == "streaming_not_implemented"

    def test_malformed_json_returns_400(self, tmp_path):
        client, _ = _make_client(tmp_path)
        resp = client.post(
            "/v1/chat/completions",
            content=b"{bad json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_connection_error_code_is_upstream_connection_error(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={"model": "chat", "messages": []}).json()
        assert body["error"]["code"] == "upstream_connection_error"


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — contract guarantees (no fake inference)
# ---------------------------------------------------------------------------

class TestNoFakeInference:
    def test_no_choices_field(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={
            "model": "chat",
            "messages": [{"role": "user", "content": "hello"}],
        }).json()
        assert "choices" not in body

    def test_no_usage_field(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={
            "model": "chat",
            "messages": [{"role": "user", "content": "hello"}],
        }).json()
        assert "usage" not in body

    def test_object_is_not_chat_completion(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={"model": "chat", "messages": []}).json()
        assert body.get("object") != "chat.completion"

    def test_response_contains_routing_decision(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={"model": "chat", "messages": []}).json()
        assert "routing_decision" in body
        assert body["routing_decision"]["dry_run"] is True

    def test_provider_shell_in_502_error(self, tmp_path):
        # On connection error, the 502 body includes provider_shell metadata.
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={"model": "chat", "messages": []}).json()
        assert body["provider_shell"]["phase"] == 7
        assert body["provider_shell"]["streaming"] is False


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — routing behaviour
# ---------------------------------------------------------------------------

class TestChatCompletionsRouting:
    def test_model_maps_to_preferred_slot(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={
            "model": "utility", "messages": [],
        }).json()
        assert body["routing_decision"]["selected_slot_id"] == "utility"

    def test_model_id_resolves_to_slot(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={
            "model": "mistral-7b-q4", "messages": [],
        }).json()
        assert body["routing_decision"]["selected_slot_id"] == "chat"

    def test_unknown_model_adds_translation_warning(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={
            "model": "gpt-4o", "messages": [],
        }).json()
        assert any("unknown_model" in w for w in body["translation_warnings"])

    def test_unknown_model_still_routes(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={
            "model": "gpt-4o", "messages": [],
        }).json()
        assert body["routing_decision"]["no_slot_available"] is False

    def test_all_unhealthy_returns_no_slot(self, tmp_path):
        client, _ = _make_client(tmp_path, health_result="unhealthy")
        body = client.post("/v1/chat/completions", json={"model": "chat", "messages": []}).json()
        assert body["routing_decision"]["no_slot_available"] is True

    def test_extra_openai_fields_accepted(self, tmp_path):
        # Unknown/extra fields must not cause a 422 — they are silently accepted.
        # Default stub has no real server → 502, but NOT 400/422.
        client, _ = _make_client(tmp_path)
        resp = client.post("/v1/chat/completions", json={
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.0,
        })
        assert resp.status_code not in (400, 422)


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — metadata passthrough
# ---------------------------------------------------------------------------

class TestChatCompletionsMetadata:
    def test_privacy_mode_local_only_enforced(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={
            "model": "chat",
            "messages": [],
            "metadata": {"privacy_mode": "local_only"},
        }).json()
        assert body["routing_decision"]["local_only_enforced"] is True

    def test_local_only_flag_enforced(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={
            "model": "chat",
            "messages": [],
            "metadata": {"local_only": True},
        }).json()
        assert body["routing_decision"]["local_only_enforced"] is True

    def test_tools_present_triggers_requires_tools_warning(self, tmp_path):
        client, _ = _make_client(tmp_path)
        body = client.post("/v1/chat/completions", json={
            "model": "chat",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }).json()
        assert any("tool_routing" in w for w in body["routing_decision"]["warnings"])


# ---------------------------------------------------------------------------
# Translation unit tests (no HTTP)
# ---------------------------------------------------------------------------

class TestTranslation:
    def test_slot_id_match_sets_preferred_slot(self):
        from service.openai_compat import OpenAIChatRequest, chat_request_to_routing_intent
        slots = [{"id": "chat", "model_id": None}, {"id": "utility", "model_id": None}]
        req = OpenAIChatRequest.model_validate({"model": "utility", "messages": []})
        intent, warnings = chat_request_to_routing_intent(req, slots)
        assert intent.preferred_slot == "utility"
        assert not any("unknown_model" in w for w in warnings)

    def test_model_id_match_resolves_to_slot(self):
        from service.openai_compat import OpenAIChatRequest, chat_request_to_routing_intent
        slots = [{"id": "chat", "model_id": "mistral-7b-q4"}]
        req = OpenAIChatRequest.model_validate({"model": "mistral-7b-q4", "messages": []})
        intent, warnings = chat_request_to_routing_intent(req, slots)
        assert intent.preferred_slot == "chat"
        assert not any("unknown_model" in w for w in warnings)

    def test_unknown_model_warning(self):
        from service.openai_compat import OpenAIChatRequest, chat_request_to_routing_intent
        req = OpenAIChatRequest.model_validate({"model": "gpt-4o", "messages": []})
        _, warnings = chat_request_to_routing_intent(req, [])
        assert any("unknown_model" in w for w in warnings)

    def test_token_estimate_from_messages(self):
        from service.openai_compat import OpenAIChatRequest, chat_request_to_routing_intent
        req = OpenAIChatRequest.model_validate({
            "model": "chat",
            "messages": [{"role": "user", "content": "a" * 400}],
        })
        intent, _ = chat_request_to_routing_intent(req, [])
        assert intent.estimated_tokens == 100   # 400 // 4

    def test_task_type_from_metadata(self):
        from service.openai_compat import OpenAIChatRequest, chat_request_to_routing_intent
        req = OpenAIChatRequest.model_validate({
            "model": "chat",
            "messages": [],
            "metadata": {"task_type": "coding"},
        })
        intent, _ = chat_request_to_routing_intent(req, [])
        assert intent.task_type == "coding"

    def test_default_model_no_warning(self):
        from service.openai_compat import OpenAIChatRequest, chat_request_to_routing_intent
        req = OpenAIChatRequest.model_validate({"model": "default", "messages": []})
        _, warnings = chat_request_to_routing_intent(req, [])
        assert not any("unknown_model" in w for w in warnings)


# ---------------------------------------------------------------------------
# Forwarding tests (Phase 7a) — inject stub post_fn to avoid real network
# ---------------------------------------------------------------------------

def _success_post(response_body=None):
    """Return a post_fn stub that simulates a successful upstream response."""
    if response_body is None:
        response_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    async def _post(url, payload, timeout=120):
        return 200, response_body

    return _post


class TestForwarding:
    def test_successful_response_passed_through(self, tmp_path):
        client, _ = _make_client(tmp_path, post_fn=_success_post())
        resp = client.post("/v1/chat/completions", json={
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert "choices" in body

    def test_no_slot_returns_503(self, tmp_path):
        client, _ = _make_client(tmp_path, health_result="unhealthy",
                                  post_fn=_success_post())
        resp = client.post("/v1/chat/completions", json={"model": "chat", "messages": []})
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "no_slot_available"

    def test_connection_error_returns_502(self, tmp_path):
        # Default _make_client uses connection-error stub.
        client, _ = _make_client(tmp_path)
        resp = client.post("/v1/chat/completions", json={"model": "chat", "messages": []})
        assert resp.status_code == 502
        assert resp.json()["error"]["code"] == "upstream_connection_error"

    def test_upstream_error_passed_through(self, tmp_path):
        async def error_post(url, payload, timeout=120):
            return 500, {"error": {"message": "Internal error", "code": "internal_error"}}

        client, _ = _make_client(tmp_path, post_fn=error_post)
        resp = client.post("/v1/chat/completions", json={"model": "chat", "messages": []})
        assert resp.status_code == 500

    def test_stream_still_rejected_before_forwarding(self, tmp_path):
        forwarding_calls = []

        async def tracking_post(url, payload, timeout=120):
            forwarding_calls.append(url)
            return 200, {}

        client, _ = _make_client(tmp_path, post_fn=tracking_post)
        resp = client.post("/v1/chat/completions", json={
            "model": "chat", "messages": [], "stream": True,
        })
        assert resp.status_code == 400
        assert forwarding_calls == []   # never called

    def test_forwarded_payload_includes_messages(self, tmp_path):
        received: dict = {}

        async def capturing_post(url, payload, timeout=120):
            received.update(payload)
            return 200, {"object": "chat.completion", "choices": []}

        client, _ = _make_client(tmp_path, post_fn=capturing_post)
        client.post("/v1/chat/completions", json={
            "model": "chat",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert "messages" in received
        assert received["messages"][0]["content"] == "hello"

    def test_forwarded_payload_excludes_metadata(self, tmp_path):
        received: dict = {}

        async def capturing_post(url, payload, timeout=120):
            received.update(payload)
            return 200, {"object": "chat.completion", "choices": []}

        client, _ = _make_client(tmp_path, post_fn=capturing_post)
        client.post("/v1/chat/completions", json={
            "model": "chat",
            "messages": [],
            "metadata": {"privacy_mode": "local_only", "agent_id": "hermes"},
        })
        assert "metadata" not in received
        assert "privacy_mode" not in received
        assert "agent_id" not in received

    def test_forwarded_payload_excludes_tools(self, tmp_path):
        received: dict = {}

        async def capturing_post(url, payload, timeout=120):
            received.update(payload)
            return 200, {"object": "chat.completion", "choices": []}

        client, _ = _make_client(tmp_path, post_fn=capturing_post)
        client.post("/v1/chat/completions", json={
            "model": "chat",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "search"}}],
        })
        assert "tools" not in received
        assert "tool_choice" not in received

    def test_stream_forced_false_in_forwarded_payload(self, tmp_path):
        received: dict = {}

        async def capturing_post(url, payload, timeout=120):
            received.update(payload)
            return 200, {"object": "chat.completion", "choices": []}

        client, _ = _make_client(tmp_path, post_fn=capturing_post)
        client.post("/v1/chat/completions", json={"model": "chat", "messages": []})
        assert received.get("stream") is False

    def test_temperature_and_max_tokens_forwarded(self, tmp_path):
        received: dict = {}

        async def capturing_post(url, payload, timeout=120):
            received.update(payload)
            return 200, {"object": "chat.completion", "choices": []}

        client, _ = _make_client(tmp_path, post_fn=capturing_post)
        client.post("/v1/chat/completions", json={
            "model": "chat",
            "messages": [],
            "temperature": 0.7,
            "max_tokens": 256,
        })
        assert received.get("temperature") == 0.7
        assert received.get("max_tokens") == 256

    def test_forwarding_url_has_no_double_v1(self, tmp_path):
        captured: dict = {}

        async def url_capturing_post(url, payload, timeout=120):
            captured["url"] = url
            return 200, {"object": "chat.completion", "choices": []}

        client, _ = _make_client(tmp_path, post_fn=url_capturing_post)
        client.post("/v1/chat/completions", json={"model": "chat", "messages": []})
        url = captured.get("url", "")
        assert url.endswith("/v1/chat/completions")
        assert "/v1/v1/" not in url

    def test_routing_performed_before_forwarding(self, tmp_path):
        captured: dict = {}

        async def tracking_post(url, payload, timeout=120):
            captured["url"] = url
            return 200, {"object": "chat.completion", "choices": []}

        client, _ = _make_client(tmp_path, post_fn=tracking_post)
        client.post("/v1/chat/completions", json={
            "model": "utility",   # should select utility slot (port 8088)
            "messages": [],
        })
        assert "8088" in captured.get("url", "")
