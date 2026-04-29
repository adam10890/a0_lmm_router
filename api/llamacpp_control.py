"""
HTTP API wrapper for llama.cpp server control operations (start/stop/start_all/stop_all).
Delegates to LlamaCppManager.
"""
import os
from flask import Request
from helpers.api import ApiHandler
from helpers import files


class LlamacppControl(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        data = input.get("data", input)
        operation = data.get("operation", "status")
        server_name = data.get("server", "")
        try:
            from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import LlamaCppManager

            plugin_conf = files.get_abs_path("usr/plugins/a0_lmm_router/conf/llama_cpp_servers.yaml")
            root_conf = files.get_abs_path("conf/llama_cpp_servers.yaml")
            conf_path = plugin_conf if os.path.exists(plugin_conf) else root_conf

            manager = LlamaCppManager.get_instance(conf_path)

            if operation == "start":
                result = await manager.start_server(server_name)
            elif operation == "stop":
                result = await manager.stop_server(server_name)
            elif operation == "start_all":
                result = await manager.start_all()
            elif operation == "stop_all":
                result = await manager.stop_all()
            else:
                return {"ok": False, "error": f"Unknown operation: {operation}"}

            return {"ok": True, "message": str(result) if result else f"{operation} complete"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
