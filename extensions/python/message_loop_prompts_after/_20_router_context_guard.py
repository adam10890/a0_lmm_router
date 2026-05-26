"""Fit Agent Zero history to llama.cpp router n_ctx (system prompt included).

A0's built-in history compression only compares history tokens to
ctx_length * ctx_history. The system prompt is assembled afterwards, so
long chats can exceed the router's hard n_ctx and fail with
exceed_context_size_error.

This extension runs after the system prompt is known and re-compresses
history until it fits the router budget, then refreshes loop_data.history_output.
"""
from __future__ import annotations

from typing import Callable

from helpers import dirty_json, tokens
from helpers.extension import Extension
from agent import LoopData

try:
    from usr.plugins.a0_lmm_router.helpers.router_context import (
        estimate_extras_tokens,
        history_token_budget,
        resolve_router_ctx_limit,
    )
except ImportError:
    from helpers.router_context import (  # type: ignore[no-redef]
        estimate_extras_tokens,
        history_token_budget,
        resolve_router_ctx_limit,
    )

MAX_PASSES = 64


class RouterContextGuard(Extension):
    async def execute(self, loop_data: LoopData = LoopData(), **kwargs):
        if not self.agent:
            return

        try:
            from plugins._model_config.helpers.model_config import get_chat_model_config
        except ImportError:
            return

        cfg = get_chat_model_config(self.agent)
        if str(cfg.get("provider", "")).lower() != "lmm_router":
            return

        system_text = "\n\n".join(loop_data.system or [])
        system_tokens = tokens.approximate_prompt_tokens(system_text)
        extras_tokens = estimate_extras_tokens(loop_data)
        budget = history_token_budget(cfg, system_tokens, extras_tokens=extras_tokens)
        router_ctx = resolve_router_ctx_limit(cfg)

        history = self.agent.history
        before = history.get_tokens()
        if before <= budget:
            return

        self.agent.context.log.log(
            type="info",
            heading="LMM Router context guard",
            content=(
                f"History {before:,} tokens exceeds router budget {budget:,} "
                f"(n_ctx={router_ctx:,}, system≈{system_tokens:,}, extras≈{extras_tokens:,}). "
                "Compressing…"
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
                content=f"History reduced {before:,} → {after:,} tokens (budget {budget:,}).",
            )
