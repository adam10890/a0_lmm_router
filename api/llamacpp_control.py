import os
from flask import Request
from helpers.api import ApiHandler
from helpers import files


def _resolve_conf_path() -> str:
    env_conf = os.environ.get("A0_LMM_ROUTER_CONFIG", "")
    plugin_conf = files.get_abs_path("usr/plugins/a0_lmm_router/conf/llama_cpp_servers.yaml")
    root_conf = files.get_abs_path("conf/llama_cpp_servers.yaml")
    if env_conf and os.path.exists(env_conf):
        return env_conf
    return root_conf if os.path.exists(root_conf) else plugin_conf


def _result_ok(result: dict) -> bool:
    return bool(result.get("healthy") or result.get("running")) and not result.get("error")


class LlamacppControl(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        data = input.get("data", input)
        operation = data.get("operation", "status")
        server_name = data.get("server", "")
        try:
            from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager

            conf_path = _resolve_conf_path()
            BackendManager._instance = None  # noqa: SLF001
            manager = BackendManager.get_instance(conf_path)

            if operation == "start":
                result = await manager.start_slot(server_name)
                return {
                    "ok": _result_ok(result),
                    "message": result.get("error") or f"Remote slot '{server_name}' is reachable",
                    "result": result,
                    "backend": manager.backend_type,
                }
            elif operation == "stop":
                result = await manager.stop_slot(server_name)
                return {
                    "ok": bool(result),
                    "message": (
                        f"Unregistered remote slot '{server_name}'"
                        if result else f"Slot '{server_name}' was not registered"
                    ),
                    "backend": manager.backend_type,
                }
            elif operation == "start_all":
                result = await manager.start_all()
                failed = [name for name, slot in result.items() if slot.get("error")]
                return {
                    "ok": not failed,
                    "message": (
                        "All configured remote slots are reachable"
                        if not failed else f"Unreachable slots: {', '.join(failed)}"
                    ),
                    "result": result,
                    "backend": manager.backend_type,
                }
            elif operation == "stop_all":
                await manager.stop_all()
                return {
                    "ok": True,
                    "message": "Cleared remote slot tracking; external containers were not stopped",
                    "backend": manager.backend_type,
                }
            elif operation == "status":
                result = await manager.status()
                return {
                    "ok": True,
                    "message": "Status loaded",
                    "result": result,
                    "backend": manager.backend_type,
                }
            else:
                return {"ok": False, "error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
