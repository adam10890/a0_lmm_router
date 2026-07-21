"""
server.py — FastMCP server entry point for lmm-router.

Exposes tools and resources over Streamable HTTP (MCP spec 2025-06-18).
Endpoint: POST/GET http://<host>:<port>/mcp

Usage:
    python -m mcp_server.server           # uses defaults (0.0.0.0:8095)
    MCP_PORT=9000 python -m mcp_server.server

Config override via llama_cpp_servers.yaml:
    mcp_server:
      enabled: true
      host: "0.0.0.0"
      port: 8095
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow import resolution both inside the container and for local testing.
_A0_ROOT = "/a0"
if _A0_ROOT not in sys.path and os.path.isdir(_A0_ROOT):
    sys.path.insert(0, _A0_ROOT)

# Ensure the plugin root is on sys.path so `mcp_server.*` imports resolve
# when this file is executed directly from the plugin directory.
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp_server.tools import register_tools  # noqa: E402
from mcp_server.resources import register_resources  # noqa: E402
from helpers import usage_ledger  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("lmm_router.mcp")


def _load_mcp_config() -> dict:
    """Read mcp_server section from llama_cpp_servers.yaml if present."""
    candidates = [
        os.environ.get("A0_LMM_ROUTER_CONFIG", ""),
        "/a0/conf/llama_cpp_servers.yaml",
        str(_PLUGIN_DIR / "conf" / "llama_cpp_servers.yaml"),
    ]
    for path in candidates:
        if path and Path(path).is_file():
            try:
                import yaml  # noqa: PLC0415
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                return data.get("mcp_server", {})
            except Exception:
                pass
    return {}


def create_app(host: str = "0.0.0.0", port: int = 8095) -> FastMCP:
    # host/port are passed to FastMCP at construction time; mcp>=1.10 dropped
    # them from FastMCP.run() kwargs and reads them from settings instead.
    # The MCP server is its own process — write usage to data/usage_mcp.jsonl
    # so it never contends with the A0 process's usage_a0.jsonl.
    usage_ledger.PROC_TAG = "mcp"

    mcp = FastMCP(
        "lmm-router",
        instructions=(
            "Local LLM router. Use chat_completion / utility_completion for inference, "
            "pi_coding for autonomous code edits via the pi agent, "
            "get_embeddings for vectors, fleet_status to inspect slots, "
            "and start_fleet / stop_slot to manage containers. "
            "Before heavy or multi-step work call route_task to get a routing packet "
            "(which provider/model to use given current budgets); call compute_budget "
            "when planning parallel work."
        ),
        host=host,
        port=port,
    )
    register_tools(mcp)
    register_resources(mcp)
    return mcp


def main() -> None:
    cfg = _load_mcp_config()
    host = os.environ.get("MCP_HOST", cfg.get("host", "0.0.0.0"))
    port = int(os.environ.get("MCP_PORT", cfg.get("port", 8095)))

    if not cfg.get("enabled", True):
        logger.info("MCP server disabled in config (mcp_server.enabled=false). Exiting.")
        return

    mcp = create_app(host=host, port=port)

    logger.info("Starting lmm-router MCP server on %s:%d/mcp (Streamable HTTP)", host, port)
    logger.info("Inspector: npx @modelcontextprotocol/inspector http://%s:%d/mcp", host, port)

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
