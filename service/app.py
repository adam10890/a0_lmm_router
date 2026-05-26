"""
Starlette app factory for the lmm-router read-only observer service.

Usage:
    from service.app import create_app
    app = create_app("/path/to/llama_cpp_servers.yaml")

Or run via python -m service (see __main__.py).
"""
from __future__ import annotations

from typing import Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .observer import ObserverBackend

_VERSION = "0.1.0"
_SERVICE_NAME = "lmm-router-observer"


def create_app(config_path: Optional[str] = None) -> Starlette:
    """Return a configured Starlette app.  Safe to call multiple times (no side-effects)."""
    observer = ObserverBackend(config_path)

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

    routes = [
        Route("/health", health),
        Route("/slots", slots),
        Route("/config/preview", config_preview),
        Route("/routing/preview", routing_preview),
        Route("/health/slots", health_slots),
    ]

    return Starlette(routes=routes)
