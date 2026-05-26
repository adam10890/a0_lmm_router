"""
Entry point for the lmm-router standalone observer service.

    python -m service                          # 127.0.0.1:8096
    OBSERVER_PORT=9000 python -m service       # custom port
    A0_LMM_ROUTER_CONFIG=/path python -m service  # custom config
"""
from __future__ import annotations

import os

import uvicorn

from .app import create_app


def main() -> None:
    host = os.environ.get("OBSERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("OBSERVER_PORT", "8096"))
    config_path = os.environ.get("A0_LMM_ROUTER_CONFIG", "").strip() or None

    app = create_app(config_path)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
