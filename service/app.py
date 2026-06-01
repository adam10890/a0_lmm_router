"""
Starlette app factory for the lmm-router read-only observer service.

Usage:
    from service.app import create_app
    app = create_app("/path/to/llama_cpp_servers.yaml")

Or run via python -m service (see __main__.py).
"""
from __future__ import annotations

import hmac
import os
from typing import Any, Awaitable, Callable, Dict, Optional

import aiohttp
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .observer import ObserverBackend
from .routing_intent import RoutingIntentHandler, RoutingIntentRequest

_VERSION = "0.1.0"
_SERVICE_NAME = "lmm-router-observer"
_FORWARD_TIMEOUT_SECONDS = 120
_API_KEY_ENV = "A0_LMM_ROUTER_API_KEY"

_ROUTING_ONLY_KEYS = {
    "routing",
    "preferred_slot",
    "role",
    "agent_id",
    "agent_type",
    "task_type",
    "privacy_mode",
    "local_only",
    "cloud_allowed",
    "requires_long_context",
    "requires_tools",
    "requires_code_execution",
    "latency_preference",
    "quality_preference",
    "cost_preference",
    "estimated_tokens",
    "input_classification",
}


def _openai_error(
    message: str,
    code: str,
    status_code: int,
    *,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    body: Dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
    if extra:
        body.update(extra)
    return JSONResponse(body, status_code=status_code)


def _configured_api_key() -> str:
    return os.environ.get(_API_KEY_ENV, "").strip()


def _bearer_token(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return ""
    return token.strip()


def _authorized(request: Request, api_key: str) -> bool:
    if not api_key:
        return True
    return hmac.compare_digest(_bearer_token(request), api_key)


def _unauthorized_response(request: Request) -> JSONResponse:
    if request.url.path.startswith("/v1/"):
        return _openai_error(
            "missing or invalid bearer token",
            "unauthorized",
            401,
            error_type="authentication_error",
        )
    return JSONResponse(
        {"error": "unauthorized", "detail": "missing or invalid bearer token"},
        status_code=401,
    )


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _pick_routing_value(
    body: Dict[str, Any],
    routing: Dict[str, Any],
    metadata: Dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if key in routing:
        return routing[key]
    if key in body:
        return body[key]
    if key in metadata:
        return metadata[key]
    return default


def _role_from_chat_body(body: Dict[str, Any], routing: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    explicit_role = _pick_routing_value(body, routing, metadata, "role")
    if explicit_role:
        return str(explicit_role)

    model = str(body.get("model", "") or "").lower()
    if "utility" in model or model in {"util", "coding", "debugging"}:
        return "utility"
    return "chat"


def _intent_from_chat_body(body: Dict[str, Any]) -> RoutingIntentRequest:
    routing = _dict_or_empty(body.get("routing"))
    metadata = _dict_or_empty(body.get("metadata"))
    role = _role_from_chat_body(body, routing, metadata)

    payload = {
        "agent_id": _pick_routing_value(
            body, routing, metadata, "agent_id", body.get("user") or "openai_compatible_client"
        ),
        "agent_type": _pick_routing_value(body, routing, metadata, "agent_type", "custom"),
        "role": role,
        "task_type": _pick_routing_value(body, routing, metadata, "task_type", "chat"),
        "privacy_mode": _pick_routing_value(body, routing, metadata, "privacy_mode", "unknown"),
        "local_only": _pick_routing_value(body, routing, metadata, "local_only", False),
        "cloud_allowed": _pick_routing_value(body, routing, metadata, "cloud_allowed", True),
        "requires_long_context": _pick_routing_value(
            body, routing, metadata, "requires_long_context", False
        ),
        "requires_tools": _pick_routing_value(body, routing, metadata, "requires_tools", False),
        "requires_code_execution": _pick_routing_value(
            body, routing, metadata, "requires_code_execution", False
        ),
        "latency_preference": _pick_routing_value(body, routing, metadata, "latency_preference", "normal"),
        "quality_preference": _pick_routing_value(body, routing, metadata, "quality_preference", "normal"),
        "cost_preference": _pick_routing_value(body, routing, metadata, "cost_preference", "normal"),
        "estimated_tokens": _pick_routing_value(body, routing, metadata, "estimated_tokens"),
        "preferred_slot": _pick_routing_value(body, routing, metadata, "preferred_slot"),
        "input_classification": _pick_routing_value(body, routing, metadata, "input_classification"),
        "metadata": metadata,
    }
    return RoutingIntentRequest.model_validate(payload)


def _forward_payload(body: Dict[str, Any], selected_model: Optional[str], *, stream: bool) -> Dict[str, Any]:
    payload = {k: v for k, v in body.items() if k not in _ROUTING_ONLY_KEYS}
    payload["stream"] = stream
    if not payload.get("model"):
        payload["model"] = selected_model or "local"
    return payload


async def _stream_upstream_response(resp: aiohttp.ClientResponse, session: aiohttp.ClientSession):
    try:
        async for chunk in resp.content.iter_chunked(8192):
            if chunk:
                yield chunk
    except aiohttp.ClientError as exc:
        import json as _json

        payload = {
            "error": {
                "message": f"upstream llama.cpp stream failed: {exc}",
                "type": "server_error",
                "param": None,
                "code": "upstream_stream_error",
            }
        }
        yield f"data: {_json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")
    finally:
        resp.release()
        await session.close()


def create_app(config_path: Optional[str] = None) -> Starlette:
    """Return a configured Starlette app.  Safe to call multiple times (no side-effects)."""
    observer = ObserverBackend(config_path)
    intent_handler = RoutingIntentHandler(observer)
    api_key = _configured_api_key()

    def protected(handler: Callable[[Request], Awaitable[Response]]) -> Callable[[Request], Awaitable[Response]]:
        async def wrapper(request: Request) -> Response:
            if not _authorized(request, api_key):
                return _unauthorized_response(request)
            return await handler(request)

        return wrapper

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "service": _SERVICE_NAME,
            "version": _VERSION,
            "config_path": observer.config_path,
        })

    async def slots(request: Request) -> JSONResponse:
        return JSONResponse(observer.get_slots())

    async def config_preview(request: Request) -> JSONResponse:
        return JSONResponse(observer.get_config_preview())

    async def routing_preview(request: Request) -> JSONResponse:
        role = request.query_params.get("role", "chat")
        result = await observer.get_routing_preview(role)
        return JSONResponse(result)

    async def health_slots(request: Request) -> JSONResponse:
        results = await observer.get_slots_health()
        return JSONResponse(results)

    async def routing_request(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid_json", "detail": "request body is not valid JSON"}, status_code=400)

        try:
            intent = RoutingIntentRequest.model_validate(body)
        except ValidationError as exc:
            import json as _json
            return JSONResponse({"error": "validation_error", "detail": _json.loads(exc.json())}, status_code=422)

        result = await intent_handler.handle(intent)
        return JSONResponse(result.model_dump())

    async def chat_completions(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return _openai_error("request body is not valid JSON", "invalid_json", 400)

        if not isinstance(body, dict):
            return _openai_error("request body must be a JSON object", "invalid_request_body", 400)

        if "messages" not in body:
            return _openai_error("missing required field: messages", "missing_messages", 400, param="messages")

        try:
            intent = _intent_from_chat_body(body)
        except ValidationError as exc:
            import json as _json
            return _openai_error(
                "routing metadata failed validation",
                "routing_validation_error",
                422,
                extra={"detail": _json.loads(exc.json())},
            )

        decision = await intent_handler.handle(intent)
        decision_body = decision.model_dump()
        if decision.no_slot_available or not decision.selected_url:
            return _openai_error(
                "no healthy local llama.cpp slot is available for this request",
                "no_slot_available",
                503,
                error_type="server_error",
                extra={"routing": decision_body},
            )

        url = decision.selected_url.rstrip("/") + "/chat/completions"
        wants_stream = body.get("stream") is True
        payload = _forward_payload(body, decision.selected_model, stream=wants_stream)

        try:
            if wants_stream:
                session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=_FORWARD_TIMEOUT_SECONDS))
                try:
                    resp = await session.post(url, json=payload, headers={"Content-Type": "application/json"})
                except Exception:
                    await session.close()
                    raise

                headers = {
                    "x-a0-router-slot-id": decision.selected_slot_id or "",
                    "x-a0-router-backend": decision.selected_backend_type or "",
                    "cache-control": "no-cache",
                    "x-accel-buffering": "no",
                }
                if decision.selected_model:
                    headers["x-a0-router-model"] = decision.selected_model

                if resp.status >= 400:
                    try:
                        upstream_json = await resp.json(content_type=None)
                    except Exception:
                        upstream_json = await resp.text()
                    resp.release()
                    await session.close()

                    message = "upstream llama.cpp stream request failed"
                    if isinstance(upstream_json, dict):
                        upstream_error = upstream_json.get("error")
                        if isinstance(upstream_error, dict) and upstream_error.get("message"):
                            message = str(upstream_error["message"])
                        elif isinstance(upstream_error, str):
                            message = upstream_error
                    return _openai_error(
                        message,
                        "upstream_error",
                        resp.status,
                        error_type="server_error",
                        extra={"upstream": upstream_json, "routing": decision_body},
                    )

                return StreamingResponse(
                    _stream_upstream_response(resp, session),
                    status_code=resp.status,
                    media_type="text/event-stream",
                    headers=headers,
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=_FORWARD_TIMEOUT_SECONDS),
                ) as resp:
                    try:
                        upstream_json = await resp.json(content_type=None)
                    except Exception:
                        upstream_text = await resp.text()
                        return _openai_error(
                            "upstream llama.cpp response was not valid JSON",
                            "upstream_invalid_json",
                            502,
                            error_type="server_error",
                            extra={"upstream_status": resp.status, "upstream_body": upstream_text[:1000]},
                        )

                    headers = {
                        "x-a0-router-slot-id": decision.selected_slot_id or "",
                        "x-a0-router-backend": decision.selected_backend_type or "",
                    }
                    if decision.selected_model:
                        headers["x-a0-router-model"] = decision.selected_model

                    if resp.status >= 400:
                        message = "upstream llama.cpp request failed"
                        if isinstance(upstream_json, dict):
                            upstream_error = upstream_json.get("error")
                            if isinstance(upstream_error, dict) and upstream_error.get("message"):
                                message = str(upstream_error["message"])
                            elif isinstance(upstream_error, str):
                                message = upstream_error
                        return _openai_error(
                            message,
                            "upstream_error",
                            resp.status,
                            error_type="server_error",
                            extra={"upstream": upstream_json, "routing": decision_body},
                        )

                    return JSONResponse(upstream_json, status_code=resp.status, headers=headers)
        except aiohttp.ClientError as exc:
            return _openai_error(
                f"could not reach selected llama.cpp slot: {exc}",
                "upstream_unreachable",
                502,
                error_type="server_error",
                extra={"routing": decision_body},
            )
        except TimeoutError:
            return _openai_error(
                "selected llama.cpp slot timed out",
                "upstream_timeout",
                504,
                error_type="server_error",
                extra={"routing": decision_body},
            )

    routes = [
        Route("/health", health),
        Route("/slots", protected(slots)),
        Route("/config/preview", protected(config_preview)),
        Route("/routing/preview", protected(routing_preview)),
        Route("/health/slots", protected(health_slots)),
        Route("/routing/request", protected(routing_request), methods=["POST"]),
        Route("/v1/chat/completions", protected(chat_completions), methods=["POST"]),
    ]

    return Starlette(routes=routes)
