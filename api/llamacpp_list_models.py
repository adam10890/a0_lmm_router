"""
List locally available .gguf model files for the a0_lmm_router plugin.
Scans the configured models_dir and returns metadata per file.
"""
import os
from flask import Request
from helpers.api import ApiHandler
from helpers import files


class LlamacppListModels(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            import yaml
            plugin_conf = files.get_abs_path("usr/plugins/a0_lmm_router/conf/llama_cpp_servers.yaml")
            root_conf = files.get_abs_path("conf/llama_cpp_servers.yaml")
            conf_path = plugin_conf if os.path.exists(plugin_conf) else root_conf

            models_dir = ""
            if os.path.exists(conf_path):
                with open(conf_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                models_dir = cfg.get("global", {}).get("models_dir", "")

            if not models_dir or not os.path.isdir(models_dir):
                models_dir = files.get_abs_path("usr/models")

            models = []
            if os.path.isdir(models_dir):
                for fname in sorted(os.listdir(models_dir)):
                    if fname.lower().endswith(".gguf"):
                        fpath = os.path.join(models_dir, fname)
                        size_gb = round(os.path.getsize(fpath) / (1024 ** 3), 2)
                        models.append({"name": fname, "path": fpath, "size_gb": size_gb})

            return {"ok": True, "models_dir": models_dir, "models": models}
        except Exception as e:
            return {"ok": False, "error": str(e), "models": []}
