"""Fit Agent Zero history to the router's effective context window.

A0's built-in history compression only compares history tokens to
ctx_length * ctx_history.  The system prompt is assembled afterwards, so long
chats can exceed the llama.cpp window or drift beyond the quality-safe part of
that window.  This extension runs after the system prompt is known and
compresses history to the Fleet context policy budget.
"""
from __future__ import annotations

from typing import Callable

from agent import LoopData
from helpers import tokens
from helpers.extension import Extension

try:
    from usr.plugins.a0_lmm_router.helpers.router_context import (
        context_budget_details,
        estimate_extras_tokens,
        is_local_fleet_chat_active,
    )
    from usr.plugins.a0_lmm_router.helpers.mcp_exposure import TELEMETRY_KEY as MCP_TELEMETRY_KEY
    from usr.plugins.a0_lmm_router.helpers.tool_exposure import TELEMETRY_KEY
except ImportError:
    from helpers.router_context import (  # type: ignore[no-redef]
        context_budget_details,
        estimate_extras_tokens,
        is_local_fleet_chat_active,
    )
    from helpers.mcp_exposure import TELEMETRY_KEY as MCP_TELEMETRY_KEY  # type: ignore[no-redef]
    from helpers.tool_exposure import TELEMETRY_KEY  # type: ignore[no-redef]

MAX_PASSES = 64
CONTEXT_TELEMETRY_KEY = "a0_lmm_router_context_guard"


class RouterContextGuard(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not self.agent:
            return

        try:
            from plugins._model_config.helpers.model_config import get_chat_model_config
        except ImportError:
            return

        if not is_local_fleet_chat_active(self.agent):
            return

        cfg = get_chat_model_config(self.agent)
        system_text = "\n\n".join(loop_data.system or [])
        system_tokens = tokens.approximate_prompt_tokens(system_text)
        extras_tokens = estimate_extras_tokens(loop_data)

        history = self.agent.history
        before = history.get_tokens()
        budget_details = context_budget_details(
            cfg,
            system_tokens,
            extras_tokens=extras_tokens,
            history_tokens=before,
            role="chat",
        )
        budget = int(budget_details["history_budget"])
        router_ctx = int(budget_details["hard_ctx"])

        tool_note = ""
        tool_telemetry = self.agent.get_data(TELEMETRY_KEY) or {}
        if isinstance(tool_telemetry, dict) and tool_telemetry.get("total_tools"):
            tool_note = (
                f" tools={int(tool_telemetry.get('kept_tools', 0))}/"
                f"{int(tool_telemetry.get('total_tools', 0))}, "
                f"tool_tokens~{int(tool_telemetry.get('tool_tokens_after', 0)):,} "
                f"(saved~{int(tool_telemetry.get('tool_tokens_saved', 0)):,}),"
            )

        mcp_note = ""
        mcp_telemetry = self.agent.get_data(MCP_TELEMETRY_KEY) or {}
        if isinstance(mcp_telemetry, dict) and mcp_telemetry.get("total_tools"):
            mcp_note = (
                f" mcp_tools={int(mcp_telemetry.get('kept_tools', 0))}/"
                f"{int(mcp_telemetry.get('total_tools', 0))}, "
                f"mcp_tokens~{int(mcp_telemetry.get('mcp_tokens_after', 0)):,} "
                f"(saved~{int(mcp_telemetry.get('mcp_tokens_saved', 0)):,}),"
            )

        try:
            self.agent.set_data(
                CONTEXT_TELEMETRY_KEY,
                {
                    **budget_details,
                    "history_budget": budget,
                    "compression_needed": before > budget,
                    "tool_exposure": tool_telemetry if isinstance(tool_telemetry, dict) else {},
                    "mcp_exposure": mcp_telemetry if isinstance(mcp_telemetry, dict) else {},
                },
            )
        except Exception:
            pass

        if before <= budget:
            return

        self.agent.context.log.log(
            type="info",
            heading="LMM Router context guard",
            content=(
                f"History {before:,} tokens exceeds router budget {budget:,} "
                f"(hard_ctx={router_ctx:,}, "
                f"effective_ctx={int(budget_details['effective_ctx']):,}, "
                f"ratio={float(budget_details['effective_ratio']):.2f}, "
                f"system~{system_tokens:,}, extras~{extras_tokens:,}, "
                f"{tool_note}{mcp_note} "
                f"projected_occupancy={float(budget_details['projected_occupancy']):.1%}). "
                "Compressing..."
            ),
        )

        original_limit: Callable[[], int] = history._get_ctx_size_for_history  # type: ignore[attr-defined]

        def _router_history_limit() -> int:
            return budget

        history._get_ctx_size_for_history = _router_history_limit  # type: ignore[method-assign]
        try:
            passes = 0
            while passes < MAX_PASSES and history.get_tokens() > budget:
                passes += 1
                prev = history.get_tokens()
                compressed = await history.compress()
                now = history.get_tokens()
                if not compressed or now >= prev:
                    self.agent.context.log.log(
                        type="warning",
                        heading="LMM Router context guard",
                        content=(
                            f"History still {now:,} tokens (budget {budget:,}). "
                            "Start a new chat or reduce system prompt / memories."
                        ),
                    )
                    break
        finally:
            history._get_ctx_size_for_history = original_limit  # type: ignore[method-assign]

        loop_data.history_output = history.output()
        after = history.get_tokens()
        if after < before:
            self.agent.context.log.log(
                type="info",
                heading="LMM Router context guard",
                content=(
                    f"History reduced {before:,} -> {after:,} tokens "
                    f"(budget {budget:,}, hard_ctx {router_ctx:,})."
                ),
            )
