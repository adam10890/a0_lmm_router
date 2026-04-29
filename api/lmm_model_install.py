"""
API endpoint: /plugins/a0_lmm_router/lmm_model_install

Download a GGUF model from HuggingFace.

Input:
  { "repo_id": "...", "filename": "...", "target_dir": "..." }
"""
import os
from flask import Request
from helpers.api import ApiHandler
from helpers import files


class LmmModelInstall(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        repo_id = input.get("repo_id", "")
        filename = input.get("filename", "")
        target_dir = input.get("target_dir", "")

        if not repo_id or not filename:
            return {"ok": False, "error": "repo_id and filename are required"}

        # Default target_dir: models_path from installed_models.yaml or /models
        if not target_dir:
            try:
                import yaml
                conf_path = files.get_abs_path("conf/installed_models.yaml")
                with open(conf_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                target_dir = data.get("models_path", "/models")
            except Exception:
                target_dir = "/models"

        try:
            from usr.plugins.a0_lmm_router.helpers.model_recommender import install_model
            result = install_model(repo_id=repo_id, filename=filename, target_dir=target_dir)
            return result
        except Exception as e:
            return {"ok": False, "error": str(e)}
