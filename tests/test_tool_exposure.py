"""Tests for Local Fleet tool prompt exposure policy."""
from __future__ import annotations

from usr.plugins.a0_lmm_router.helpers.tool_exposure import (
    ToolPrompt,
    build_tool_exposure_plan,
    extract_tool_candidates,
    is_allowed,
    should_apply_for_local_fleet,
)


def _tool(file_name: str, heading: str, tool_name: str | None = None) -> ToolPrompt:
    usage = ""
    if tool_name:
        usage = f'\n~~~json\n{{"tool_name": "{tool_name}", "tool_args": {{}}}}\n~~~\n'
    return ToolPrompt(
        file_name=f"agent.system.tool.{file_name}.md",
        text=f"### {heading}\nshort prompt{usage}",
        source_path=f"/fake/{file_name}.md",
    )


def test_extract_tool_candidates_uses_file_heading_and_json_name():
    prompt = _tool("call_sub", "call_subordinate", "call_subordinate")

    assert extract_tool_candidates(prompt.file_name, prompt.text) == [
        "call_sub",
        "call_subordinate",
    ]


def test_agent0_policy_keeps_router_tools_and_removes_heavy_tools():
    tools = [
        _tool("response", "response", "response"),
        _tool("call_sub", "call_subordinate", "call_subordinate"),
        _tool("delegate_profile", "delegate_profile: One-shot", "delegate_profile"),
        _tool("pen_paper", "pen_paper", "pen_paper"),
        _tool("gmail_send", "gmail_send", "gmail_send"),
        _tool("browser", "browser", "browser"),
        _tool("code_exe", "code_execution_tool", "code_execution_tool"),
    ]

    plan = build_tool_exposure_plan(tools, profile="agent0")
    kept = {decision.candidates[0] for decision in plan.decisions if decision.kept}
    removed = {decision.candidates[0] for decision in plan.decisions if not decision.kept}

    assert {"response", "call_sub", "delegate_profile", "pen_paper"} <= kept
    assert {"gmail_send", "browser", "code_exe"} <= removed
    assert plan.tokens_after < plan.tokens_before


def test_wiki_policy_keeps_wiki_glob_and_response_only():
    tools = [
        _tool("response", "response", "response"),
        _tool("wiki_query", "wiki_query", "wiki_query"),
        _tool("wiki_commit", "wiki_commit", "wiki_commit"),
        _tool("gmail_send", "gmail_send", "gmail_send"),
    ]

    plan = build_tool_exposure_plan(tools, profile="wiki_librarian")
    kept_names = {decision.candidates[0] for decision in plan.decisions if decision.kept}

    assert kept_names == {"response", "wiki_query", "wiki_commit"}


def test_google_policy_matches_google_tool_globs():
    tools = [
        _tool("response", "response", "response"),
        _tool("gmail_send", "gmail_send", "gmail_send"),
        _tool("calendar_read", "calendar_read", "calendar_read"),
        _tool("wiki_query", "wiki_query", "wiki_query"),
    ]

    plan = build_tool_exposure_plan(tools, profile="gmail")
    kept_names = {decision.candidates[0] for decision in plan.decisions if decision.kept}

    assert kept_names == {"response", "gmail_send", "calendar_read"}


def test_custom_allowlist_can_override_profile_policy():
    tools = [
        _tool("response", "response", "response"),
        _tool("browser", "browser", "browser"),
    ]
    config = {
        "tool_exposure": {
            "profile_allowlists": {
                "agent0": ["response", "browser"],
            }
        }
    }

    plan = build_tool_exposure_plan(tools, profile="agent0", config=config)

    assert all(decision.kept for decision in plan.decisions)


def test_apply_gate_respects_enabled_mode_and_local_fleet_state():
    assert should_apply_for_local_fleet(
        {"tool_exposure": {"enabled": True, "mode": "local_fleet"}},
        local_fleet_active=True,
    )
    assert not should_apply_for_local_fleet(
        {"tool_exposure": {"enabled": True, "mode": "local_fleet"}},
        local_fleet_active=False,
    )
    assert should_apply_for_local_fleet(
        {"tool_exposure": {"enabled": True, "mode": "always"}},
        local_fleet_active=False,
    )
    assert not should_apply_for_local_fleet(
        {"tool_exposure": {"enabled": False, "mode": "always"}},
        local_fleet_active=True,
    )


def test_allowed_supports_globs_against_any_candidate():
    assert is_allowed(["agent.system.tool.gmail_send", "gmail_send"], ["gmail_*"])
