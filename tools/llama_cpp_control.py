"""
llama.cpp Control Tool for Agent Zero

Allows the agent to manage local llama.cpp servers:
- Start/stop servers
- Check server status
- Switch between model profiles
"""

from helpers.tool import Tool, Response
from helpers import files
import asyncio


class LlamaCppControl(Tool):
    """Tool for controlling llama.cpp servers."""
    
    async def execute(self, **kwargs) -> Response:
        """Execute llama.cpp control operations."""
        
        operation = kwargs.get("operation", "status")
        server_name = kwargs.get("server")
        
        try:
            from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import LlamaCppManager
            
            # Prefer plugin conf, fall back to root conf
            plugin_conf = files.get_abs_path("usr/plugins/a0_lmm_router/conf/llama_cpp_servers.yaml")
            root_conf = files.get_abs_path("conf/llama_cpp_servers.yaml")
            import os as _os
            config_path = plugin_conf if _os.path.exists(plugin_conf) else root_conf
            manager = LlamaCppManager.get_instance(config_path)
            
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
        status = manager.get_status()
        
        if not status:
            return Response(
                message="No llama.cpp servers configured",
                break_loop=False
            )
        
        lines = ["## llama.cpp Server Status\n"]
        
        for name, info in status.items():
            status_emoji = {
                'running': '≡ƒƒó',
                'stopped': 'ΓÜ½',
                'starting': '≡ƒƒí',
                'stopping': '≡ƒƒí',
                'error': '≡ƒö┤',
            }.get(info['status'], 'ΓÜ¬')
            
            lines.append(f"### {name} {status_emoji}")
            lines.append(f"- **Status:** {info['status']}")
            lines.append(f"- **Role:** {info['role']}")
            lines.append(f"- **Port:** {info['port']}")
            lines.append(f"- **Model:** {info['model']}")
            
            if info['status'] == 'running':
                uptime_mins = int(info['uptime'] / 60)
                lines.append(f"- **Uptime:** {uptime_mins} minutes")
                lines.append(f"- **PID:** {info['pid']}")
            
            if info['error']:
                lines.append(f"- **Error:** {info['error']}")
            
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
        
        if name not in manager.servers:
            available = ", ".join(manager.servers.keys())
            return Response(
                message=f"Server '{name}' not found. Available: {available}",
                break_loop=False
            )
        
        success = await manager.start_server(name)
        
        if success:
            port = manager.servers[name].config.port
            return Response(
                message=f"Γ£à Server '{name}' started successfully on port {port}",
                break_loop=False
            )
        else:
            error = manager.servers[name].error_message or "Unknown error"
            return Response(
                message=f"Γ¥î Failed to start '{name}': {error}",
                break_loop=False
            )
    
    async def _stop_server(self, manager, name: str) -> Response:
        """Stop a specific server."""
        if not name:
            return Response(
                message="Please specify server name",
                break_loop=False
            )
        
        if name not in manager.servers:
            return Response(
                message=f"Server '{name}' not found",
                break_loop=False
            )
        
        success = await manager.stop_server(name)
        
        if success:
            return Response(
                message=f"Γ£à Server '{name}' stopped",
                break_loop=False
            )
        else:
            return Response(
                message=f"Γ¥î Failed to stop '{name}'",
                break_loop=False
            )
    
    async def _start_all(self, manager) -> Response:
        """Start all enabled servers."""
        results = await manager.start_all()
        
        started = [n for n, s in results.items() if s]
        failed = [n for n, s in results.items() if not s]
        
        lines = []
        if started:
            lines.append(f"Γ£à Started: {', '.join(started)}")
        if failed:
            lines.append(f"Γ¥î Failed: {', '.join(failed)}")
        
        if not lines:
            lines.append("No servers to start")
        
        return Response(
            message="\n".join(lines),
            break_loop=False
        )
    
    async def _stop_all(self, manager) -> Response:
        """Stop all running servers."""
        results = await manager.stop_all()
        
        stopped = [n for n, s in results.items() if s]
        failed = [n for n, s in results.items() if not s]
        
        lines = []
        if stopped:
            lines.append(f"Γ£à Stopped: {', '.join(stopped)}")
        if failed:
            lines.append(f"Γ¥î Failed to stop: {', '.join(failed)}")
        
        return Response(
            message="\n".join(lines) or "No servers were running",
            break_loop=False
        )
    
    def _get_endpoints(self, manager) -> Response:
        """Get API endpoints for running servers."""
        endpoints = {
            'chat': manager.get_chat_endpoint(),
            'utility': manager.get_utility_endpoint(),
            'embedding': manager.get_embedding_endpoint(),
            'router': manager.get_router_endpoint(),
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
