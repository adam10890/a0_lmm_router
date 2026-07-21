"""API: /plugins/a0_lmm_router/lmm_pi_install — install pi + deploy router config."""
from __future__ import annotations

from flask import Request

from helpers.api import ApiHandler

try:
    from usr.plugins.a0_lmm_router.helpers.pi_runner import deploy_pi_config, install_pi
except ImportError:
    from helpers.pi_runner import deploy_pi_config, install_pi


class LmmPiInstall(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        action = str(input.get("action", "install")).strip().lower()
        if action == "deploy":
            paths = deploy_pi_config()
            return {"ok": True, "action": "deploy", "config_paths": paths}
        return await install_pi()
