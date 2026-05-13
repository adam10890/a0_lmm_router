"""
llama.cpp Control Tool for Agent Zero

Allows the agent to manage local llama.cpp servers:
- Start/stop servers
- Check server status
- Switch between model profiles
"""

from helpers.tool import Tool, Response
from helpers import files
import os


class LlamaCppControl(Tool):
    """Tool for controlling llama.cpp servers."""
    
    async def execute(self, **kwargs) -> Response:
        """Execute llama.cpp control operations."""
        
        operation = kwargs.get("operation", "status")
        server_name = kwargs.get("server")
        
        try:
            from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager
            
            plugin_conf = files.get_abs_path("usr/plugins/a0_lmm_router/conf/llama_cpp_servers.yaml")
            root_conf = files.get_abs_path("conf/llama_cpp_servers.yaml")
            env_conf = os.environ.get("A0_LMM_ROUTER_CONFIG", "")
            config_path = env_conf if env_conf and os.path.exists(env_conf) else (
                root_conf if os.path.exists(root_conf) else plugin_conf
            )
            BackendManager._instance = None  # noqa: SLF001
            manager = BackendManager.get_instance(config_path)
            
        except ImportError:
            return Response(
                message="llama.cpp manager not available",
                break_loop=False
            )
        except Exception as e:
            return Response(
                message=f"Error initializing llama.cpp manager: {str(e)}",
                break_loop=False
            )
        
        # Execute operation
        if operation == "status":
            return await self._get_status(manager)
        
        elif operation == "start":
            return await self._start_server(manager, server_name or "")
        
        elif operation == "stop":
            return await self._stop_server(manager, server_name or "")
        
        elif operation == "start_all":
            return await self._start_all(manager)
        
        elif operation == "stop_all":
            return await self._stop_all(manager)
        
        elif operation == "endpoints":
            return self._get_endpoints(manager)
        
        else:
            return Response(
                message=f"Unknown operation: {operation}. "
                        f"Available: status, start, stop, start_all, stop_all, endpoints",
                break_loop=False
            )
    
    async def _get_status(self, manager) -> Response:
        """Get status of all servers."""
        status = getattr(manager, "_slot_configs", {})
        
        if not status:
            return Response(
                message="No llama.cpp servers configured",
                break_loop=False
            )
        
        lines = ["## llama.cpp Server Status\n"]
        
        for name, info in status.items():
            lines.append(f"### {name}")
            lines.append(f"- **Backend:** {manager.backend_type}")
            lines.append(f"- **Role:** {info.get('role', 'chat')}")
            lines.append(f"- **Port:** {info.get('port', '')}")
            lines.append(f"- **Model ID:** {info.get('model_id', 'unknown')}")
            lines.append("- **Lifecycle:** external LMM container/service")
            lines.append("")
        
        return Response(
            message="\n".join(lines),
            break_loop=False
        )
    
    async def _start_server(self, manager, name: str) -> Response:
        """Start a specific server."""
        if not name:
            return Response(
                message="Please specify server name. Use 'status' to see available servers.",
                break_loop=False
            )
        
        slots = getattr(manager, "_slot_configs", {})
        if name not in slots:
            available = ", ".join(slots.keys())
            return Response(
                message=f"Server '{name}' not found. Available: {available}",
                break_loop=False
            )
        
        result = await manager.start_slot(name)
        success = bool(result.get("healthy") or result.get("running")) and not result.get("error")
        
        if success:
            port = result.get("port", slots[name].get("port", ""))
            return Response(
                message=f"Remote slot '{name}' is reachable on port {port}",
                break_loop=False
            )
        else:
            error = result.get("error") or "Unknown error"
            return Response(
                message=f"Remote slot '{name}' is not reachable: {error}",
                break_loop=False
            )
    
    async def _stop_server(self, manager, name: str) -> Response:
        """Stop a specific server."""
        if not name:
            return Response(
                message="Please specify server name",
                break_loop=False
            )
        
        if name not in getattr(manager, "_slot_configs", {}):
            return Response(
                message=f"Server '{name}' not found",
                break_loop=False
            )
        
        success = await manager.stop_slot(name)
        
        if success:
            return Response(
                message=f"Unregistered remote slot '{name}'; external container was not stopped",
                break_loop=False
            )
        else:
            return Response(
                message=f"Slot '{name}' was not registered",
                break_loop=False
            )
    
    async def _start_all(self, manager) -> Response:
        """Start all enabled servers."""
        results = await manager.start_all()
        
        started = [
            n for n, s in results.items()
            if bool(s.get("healthy") or s.get("running")) and not s.get("error")
        ]
        failed = [n for n, s in results.items() if s.get("error")]
        
        lines = []
        if started:
            lines.append(f"Reachable: {', '.join(started)}")
        if failed:
            lines.append(f"Not reachable: {', '.join(failed)}")
        
        if not lines:
            lines.append("No servers to start")
        
        return Response(
            message="\n".join(lines),
            break_loop=False
        )
    
    async def _stop_all(self, manager) -> Response:
        """Stop all running servers."""
        slots = list(getattr(manager, "_slot_configs", {}).keys())
        await manager.stop_all()
        
        return Response(
            message=(
                f"Cleared remote slot tracking for: {', '.join(slots)}. "
                "External containers were not stopped."
            ) if slots else "No servers were configured",
            break_loop=False
        )
    
    def _get_endpoints(self, manager) -> Response:
        """Get API endpoints for running servers."""
        endpoints = {
            'chat': manager.get_endpoint('chat'),
            'utility': manager.get_endpoint('utility'),
            'embedding': manager.get_endpoint('embedding'),
            'router': manager.get_endpoint('router'),
        }
        
        lines = ["## llama.cpp API Endpoints\n"]
        
        for role, endpoint in endpoints.items():
            if endpoint:
                lines.append(f"- **{role}:** `{endpoint}`")
            else:
                lines.append(f"- **{role}:** Not available")
        
        lines.append("\n### Usage in Settings")
        lines.append("Set these as `api_base` in model configuration:")
        lines.append("- Provider: `llama_cpp` or `other`")
        lines.append("- Model name: Any (e.g., `local-model`)")
        
        return Response(
            message="\n".join(lines),
            break_loop=False
        )
