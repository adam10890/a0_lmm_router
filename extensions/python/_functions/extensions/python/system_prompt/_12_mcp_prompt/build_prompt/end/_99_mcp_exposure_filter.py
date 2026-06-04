"""Filter Agent Zero MCP prompts for Local Fleet profiles."""
from __future__ import annotations

from typing import Any

from helpers.extension import Extension

try:
    from usr.plugins.a0_lmm_router.helpers.mcp_exposure import (
        TELEMETRY_KEY,
        exposure_enabled,
        filter_mcp_prompt,
        manifest_signature,
        persist_manifest,
        should_apply_for_local_fleet,
    )
    from usr.plugins.a0_lmm_router.helpers.router_context import is_local_fleet_chat_active
except ImportError:  # pragma: no cover
    from helpers.mcp_exposure import (  # type: ignore[no-redef]
        TELEMETRY_KEY,
        exposure_enabled,
        filter_mcp_prompt,
        manifest_signature,
        persist_manifest,
        should_apply_for_local_fleet,
    )
    from helpers.router_context import is_local_fleet_chat_active  # type: ignore[no-redef]


class MCPExposureFilter(Extension):
    """Keep MCP prompt surface small for the main Local Fleet agent."""

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

        prompt, plan = filter_mcp_prompt(
            data["result"],
            profile=_agent_profile(self.agent),
            config=config,
        )
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


def _agent_profile(agent: Any) -> str:
    try:
        cfg = getattr(agent, "config", None)
        profile = getattr(cfg, "profile", "")
        if isinstance(profile, str) and profile.strip():
            return profile.strip()
    except Exception:
        pass
    return "agent0"


def _log_once(agent: Any, plan: Any) -> None:
    if not getattr(plan, "changed", False):
        return
    signature = manifest_signature(plan)
    previous = None
    try:
        previous = agent.get_data("_a0_lmm_mcp_exposure_signature")
    except Exception:
        previous = None
    if previous == signature:
        return
    try:
        agent.set_data("_a0_lmm_mcp_exposure_signature", signature)
        agent.context.log.log(
            type="info",
            heading="LMM Router MCP exposure",
            content=(
                f"Profile {plan.profile!r} uses policy {plan.policy!r}: "
                f"kept {len(plan.kept_tools)}/{plan.total_tools} MCP tools "
                f"across {len(plan.kept_servers)}/{plan.total_servers} servers, "
                f"saved about {max(0, plan.tokens_before - plan.tokens_after):,} "
                "MCP-prompt tokens."
            ),
        )
    except Exception:
        return
