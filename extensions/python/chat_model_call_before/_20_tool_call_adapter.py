"""Normalize clear Local Fleet tool-call syntax before A0 parses responses."""
from __future__ import annotations

from typing import Any

from helpers.extension import Extension

try:
    from usr.plugins.a0_lmm_router.helpers.tool_call_adapter_runtime import prepare_local_fleet_call_data
except ImportError:  # pragma: no cover
    from tool_call_adapter_runtime import prepare_local_fleet_call_data  # type: ignore[no-redef]


class LocalFleetToolCallAdapter(Extension):
    async def execute(self, call_data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        prepare_local_fleet_call_data(self.agent, call_data, role="chat")
