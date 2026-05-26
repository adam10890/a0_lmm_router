"""
Starlette app factory for the lmm-router read-only observer service.

Usage:
    from service.app import create_app
    app = create_app("/path/to/llama_cpp_servers.yaml")

Or run via python -m service (see __main__.py).
"""
from __future__ import annotations

from typing import Optional

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .observer import ObserverBackend
from .openai_compat import OpenAICompatHandler, OpenAIChatRequest
from .routing_intent import RoutingIntentHandler, RoutingIntentRequest

_VERSION = "0.1.0"
_SERVICE_NAME = "lmm-router-observer"


def create_app(config_path: Optional[str] = None) -> Starlette:
    """Return a configured Starlette app.  Safe to call multiple times (no side-effects)."""
    observer = ObserverBackend(config_path)
    intent_handler = RoutingIntentHandler(observer)
    compat_handler = OpenAICompatHandler(observer, intent_handler)

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

    async def v1_models(request: Request) -> JSONResponse:
        return JSONResponse(compat_handler.get_models().model_dump())

    async def v1_chat_completions(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid_json", "detail": "request body is not valid JSON"}, status_code=400)

        try:
            req = OpenAIChatRequest.model_validate(body)
        except ValidationError as exc:
            import json as _json
            return JSONResponse({"error": "validation_error", "detail": _json.loads(exc.json())}, status_code=422)

        status, result = await compat_handler.handle_chat_completion(req)
        return JSONResponse(result, status_code=status)

    routes = [
        Route("/health", health),
        Route("/slots", slots),
        Route("/config/preview", config_preview),
        Route("/routing/preview", routing_preview),
        Route("/health/slots", health_slots),
        Route("/routing/request", routing_request, methods=["POST"]),
        Route("/v1/models", v1_models),
        Route("/v1/chat/completions", v1_chat_completions, methods=["POST"]),
    ]

    return Starlette(routes=routes)
