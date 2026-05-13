"""
API endpoint: /plugins/a0_lmm_router/lmm_model_recommend

Returns model recommendations based on hardware capacity and installed models.

Input (optional):
  { "role": "chat" }   — filter by role
"""
import os
from flask import Request
from helpers.api import ApiHandler
from helpers import files


class LmmModelRecommend(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            from usr.plugins.a0_lmm_router.helpers.model_recommender import get_recommendations

            role = input.get("role")
            recs = get_recommendations(role_filter=role)
            return {"ok": True, "recommendations": recs}
        except Exception as e:
            return {"ok": False, "error": str(e), "recommendations": []}
