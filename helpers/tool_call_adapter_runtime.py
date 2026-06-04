"""Shared runtime hook for wrapping model response callbacks."""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

try:
    from usr.plugins.a0_lmm_router.helpers.output_budget import (
        TELEMETRY_KEY as OUTPUT_BUDGET_TELEMETRY_KEY,
        apply_output_budget,
        should_apply_to_model,
    )
    from usr.plugins.a0_lmm_router.helpers.router_context import is_local_fleet_chat_active
    from usr.plugins.a0_lmm_router.helpers.tool_call_adapter import (
        DEFAULT_WIKI_TOOL_ALLOWLIST,
        adapt_final_model_response,
        normalize_tool_call_output,
    )
except ImportError:  # pragma: no cover - direct test/import path variance
    from output_budget import (  # type: ignore[no-redef]
        TELEMETRY_KEY as OUTPUT_BUDGET_TELEMETRY_KEY,
        apply_output_budget,
        should_apply_to_model,
    )
    from router_context import is_local_fleet_chat_active  # type: ignore[no-redef]
    from tool_call_adapter import (  # type: ignore[no-redef]
        DEFAULT_WIKI_TOOL_ALLOWLIST,
        adapt_final_model_response,
        normalize_tool_call_output,
    )

_LIBRARIAN_PROFILES = frozenset(
    {
        "wiki_librarian",
        "wiki-librarian",
        "wikilibrarian",
        "librarian",
    }
)

_WRAPPED_FLAG = "_a0_lmm_tool_call_adapter_wrapped"
_MODEL_PROXY_FLAG = "_a0_lmm_tool_call_adapter_model_proxy"


class _ModelCallAdapterProxy:
    """Proxy one model call so the final response can be normalized post-stream."""

    def __init__(self, model: Any, agent: Any, role: str) -> None:
        self._model = model
        self._agent = agent
        self._role = role

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    async def unified_call(self, *args: Any, **kwargs: Any) -> tuple[str, str]:
        if should_apply_to_model(self._model):
            kwargs, decision = apply_output_budget(
                kwargs,
                role=self._role,
                config=_plugin_config(self._agent),
            )
            _record_output_budget(self._agent, decision)
        response, reasoning = await self._model.unified_call(*args, **kwargs)
        adapted = adapt_final_model_response(
            response,
            allowed_tools=DEFAULT_WIKI_TOOL_ALLOWLIST,
        )
        if adapted != response:
            _log_adapted_final(self._agent, adapted)
        return adapted, reasoning


def prepare_local_fleet_call_data(
    agent: Any,
    call_data: dict[str, Any] | None,
    *,
    role: str = "chat",
) -> None:
    """Wrap streaming callback and final model output for Local Fleet chats."""
    wrap_response_callback(agent, call_data)
    wrap_model_call(agent, call_data, role=role)


def wrap_model_call(agent: Any, call_data: dict[str, Any] | None, *, role: str = "chat") -> None:
    if not agent or not isinstance(call_data, dict):
        return
    if call_data.get(_MODEL_PROXY_FLAG):
        return

    model = call_data.get("model")
    if model is None or isinstance(model, _ModelCallAdapterProxy):
        return
    if not should_enable_tool_call_adapter(agent) and not should_apply_to_model(model):
        return

    call_data["model"] = _ModelCallAdapterProxy(model, agent, role)
    call_data[_MODEL_PROXY_FLAG] = True


def should_enable_tool_call_adapter(agent: Any) -> bool:
    try:
        if is_local_fleet_chat_active(agent):
            return True
    except Exception:
        pass
    return _profile(agent) in _LIBRARIAN_PROFILES


def wrap_response_callback(agent: Any, call_data: dict[str, Any] | None) -> None:
    """Wrap ``response_callback`` once to normalize clear Local Fleet tool calls."""
    if not agent or not isinstance(call_data, dict):
        return
    if call_data.get(_WRAPPED_FLAG):
        return
    if not should_enable_tool_call_adapter(agent):
        return

    original_callback = call_data.get("response_callback")
    if original_callback is not None and not callable(original_callback):
        return

    normalized_once = {"done": False}

    async def _adapted_response_callback(chunk: str, full: str) -> str | None:
        normalized = normalize_tool_call_output(
            full,
            allowed_tools=DEFAULT_WIKI_TOOL_ALLOWLIST,
        )
        if normalized is None:
            if original_callback is None:
                return None
            return await original_callback(chunk, full)

        if not normalized_once["done"]:
            normalized_once["done"] = True
            _log_normalized(agent, normalized)

        if original_callback is None:
            return normalized

        stop_response = await original_callback(normalized, normalized)
        return stop_response or normalized

    call_data["response_callback"] = _adapted_response_callback
    call_data[_WRAPPED_FLAG] = True


def _profile(agent: Any) -> str:
    try:
        cfg = getattr(agent, "config", None)
        profile = getattr(cfg, "profile", "")
        if isinstance(profile, str):
            return profile.lower()
    except Exception:
        pass
    return ""


def _plugin_config(agent: Any) -> dict[str, Any]:
    try:
        from helpers import plugins

        config = plugins.get_plugin_config("a0_lmm_router", agent=agent)
        return config if isinstance(config, dict) else {}
    except Exception:
        return {}


def _record_output_budget(agent: Any, decision: Any) -> None:
    telemetry = decision.telemetry()
    try:
        agent.set_data(OUTPUT_BUDGET_TELEMETRY_KEY, telemetry)
    except Exception:
        pass
    if not telemetry.get("applied"):
        return
    try:
        agent.context.log.log(
            type="info",
            heading="LMM Router output budget",
            content=(
                f"{telemetry['role']} output budget {telemetry['reason']}: "
                f"{telemetry['key']}={telemetry['final_tokens']} "
                f"(hard cap {telemetry['hard_max_tokens']})."
            ),
        )
    except Exception:
        pass


def _log_normalized(agent: Any, normalized: str) -> None:
    tool_name = ""
    try:
        data = json.loads(normalized)
        tool_name = str(data.get("tool_name") or "")
    except Exception:
        tool_name = "unknown"

    try:
        agent.context.log.log(
            type="info",
            heading="LMM Router tool-call adapter",
            content=(
                "Normalized a clear Local Fleet tool-call variant "
                f"(tool={tool_name}, profile={_profile(agent) or 'unknown'})."
            ),
        )
    except Exception:
        pass


def _log_adapted_final(agent: Any, adapted: str) -> None:
    tool_name = ""
    try:
        data = json.loads(adapted)
        tool_name = str(data.get("tool_name") or "")
    except Exception:
        tool_name = "unknown"

    try:
        agent.context.log.log(
            type="info",
            heading="LMM Router tool-call adapter",
            content=(
                "Wrapped final Local Fleet output into canonical tool JSON "
                f"(tool={tool_name}, profile={_profile(agent) or 'unknown'})."
            ),
        )
    except Exception:
        pass
