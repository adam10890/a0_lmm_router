"""Compatibility shim for Agent Zero's short _12_mcp_prompt module path."""
from __future__ import annotations

from usr.plugins.a0_lmm_router.extensions.python._functions.extensions.python.system_prompt._12_mcp_prompt.build_prompt.end._99_mcp_exposure_filter import (  # noqa: E501
    MCPExposureFilter,
)

__all__ = ["MCPExposureFilter"]
