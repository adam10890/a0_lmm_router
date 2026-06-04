"""Filter Agent Zero tool prompts for Local Fleet profiles."""
from __future__ import annotations

from typing import Any

from helpers.extension import Extension

try:
    from usr.plugins.a0_lmm_router.helpers.router_context import is_local_fleet_chat_active
    from usr.plugins.a0_lmm_router.helpers.tool_exposure import (
        TELEMETRY_KEY,
        exposure_enabled,
        manifest_signature,
        persist_manifest,
        render_filtered_tools_prompt,
        should_apply_for_local_fleet,
    )
except ImportError:  # pragma: no cover
    from helpers.router_context import is_local_fleet_chat_active  # type: ignore[no-redef]
    from helpers.tool_exposure import (  # type: ignore[no-redef]
        TELEMETRY_KEY,
        exposure_enabled,
        manifest_signature,
        persist_manifest,
        render_filtered_tools_prompt,
        should_apply_for_local_fleet,
    )


class ToolExposureFilter(Extension):
    """Keep the main local-model prompt lean by exposing only routed tools."""

    def execute(self, data: dict[str, Any] = {}, **kwargs: Any) -> None:
        if self.agent is None or not isinstance(data, dict):
            return
        if not isinstance(data.get("result"), str):
            return

        config = _plugin_config(self.agent)
        if not exposure_enabled(config):
            return

        local_fleet_active = False
        try:
            local_fleet_active = bool(is_local_fleet_chat_active(self.agent))
        except Exception:
            local_fleet_active = False

        if not should_apply_for_local_fleet(config, local_fleet_active=local_fleet_active):
            return

        rendered = render_filtered_tools_prompt(self.agent, config)
        if rendered is None:
            return

        prompt, plan = rendered
        data["result"] = prompt
        telemetry = plan.telemetry()
        telemetry["local_fleet_active"] = local_fleet_active
        self.agent.set_data(TELEMETRY_KEY, telemetry)
        persist_manifest(plan, config=config)
        _log_once(self.agent, plan)


def _plugin_config(agent: Any) -> dict[str, Any]:
    try:
        from helpers import plugins

        config = plugins.get_plugin_config("a0_lmm_router", agent=agent)
        return config if isinstance(config, dict) else {}
    except Exception:
        return {}


def _log_once(agent: Any, plan: Any) -> None:
    if not getattr(plan, "changed", False):
        return
    signature = manifest_signature(plan)
    previous = None
    try:
        previous = agent.get_data("_a0_lmm_tool_exposure_signature")
    except Exception:
        previous = None
    if previous == signature:
        return
    try:
        agent.set_data("_a0_lmm_tool_exposure_signature", signature)
        agent.context.log.log(
            type="info",
            heading="LMM Router tool exposure",
            content=(
                f"Profile {plan.profile!r} uses policy {plan.policy!r}: "
                f"kept {plan.kept_count}/{plan.total_tools} tool prompts, "
                f"saved about {max(0, plan.tokens_before - plan.tokens_after):,} "
                "tool-prompt tokens."
            ),
        )
    except Exception:
        return
