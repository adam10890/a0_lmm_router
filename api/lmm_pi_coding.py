"""API: /plugins/a0_lmm_router/lmm_pi_coding — run pi from the GUI."""
from __future__ import annotations

from flask import Request

from helpers.api import ApiHandler

try:
    from usr.plugins.a0_lmm_router.helpers.pi_runner import (
        CHAT_MODEL,
        DEFAULT_MODEL,
        run_pi_coding,
    )
except ImportError:
    from helpers.pi_runner import CHAT_MODEL, DEFAULT_MODEL, run_pi_coding


class LmmPiCoding(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        prompt = str(input.get("prompt", "")).strip()
        if not prompt:
            return {"ok": False, "error": "prompt is required"}

        role = str(input.get("role", "coding")).strip().lower()
        model = str(input.get("model", "")).strip()
        if not model:
            model = CHAT_MODEL if role == "chat" else DEFAULT_MODEL

        working_directory = input.get("working_directory") or input.get("cwd")
        read_only = bool(input.get("read_only", False))
        timeout = int(input.get("timeout", 600) or 600)

        return await run_pi_coding(
            prompt,
            working_directory=working_directory,
            model=model,
            timeout=timeout,
            read_only=read_only,
        )
