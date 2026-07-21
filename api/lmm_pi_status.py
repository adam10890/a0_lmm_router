"""API: /plugins/a0_lmm_router/lmm_pi_status — pi harness status for the GUI."""
from __future__ import annotations

from flask import Request

from helpers.api import ApiHandler

try:
    from usr.plugins.a0_lmm_router.helpers.pi_runner import get_pi_status
except ImportError:
    from helpers.pi_runner import get_pi_status


class LmmPiStatus(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        prefer_host = input.get("prefer_host")
        if prefer_host is not None:
            prefer_host = bool(prefer_host)
        return await get_pi_status(prefer_host=prefer_host)
