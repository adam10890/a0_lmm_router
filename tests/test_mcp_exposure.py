"""Tests for profile-aware MCP prompt exposure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HELPERS_ROOT = PLUGIN_ROOT / "helpers"
if str(HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPERS_ROOT))

from mcp_exposure import (  # noqa: E402
    filter_mcp_prompt,
    parse_mcp_prompt,
    persist_manifest,
    should_apply_for_local_fleet,
)


PROMPT = '''## "Remote (MCP Server) Agent Tools" available:

### context7
Library documentation lookup.

### context7.resolve-library-id:
Resolve a package name.

#### Input schema for tool_args:
{"type":"object"}

### git
Local git operations.

### git.git_status:
Show status.

#### Input schema for tool_args:
{"type":"object"}

### github
GitHub operations.

### github.search_repositories:
Search repos.

#### Input schema for tool_args:
{"type":"object"}

### lmm_router
Local fleet tools.

### lmm_router.fleet_status:
Show fleet status.

#### Input schema for tool_args:
{"type":"object"}

### lmm_router.list_slots:
List local model slots.

#### Input schema for tool_args:
{"type":"object"}
'''


def test_parse_mcp_prompt_groups_servers_and_tools():
    prefix, servers = parse_mcp_prompt(PROMPT)

    assert "Remote (MCP Server)" in prefix
    assert [server.name for server in servers] == ["context7", "git", "github", "lmm_router"]
    assert [tool.qualified_name for tool in servers[-1].tools] == [
        "lmm_router.fleet_status",
        "lmm_router.list_slots",
    ]


def test_agent0_keeps_only_router_status_tools():
    filtered, plan = filter_mcp_prompt(PROMPT, profile="agent0")

    assert "lmm_router.fleet_status" in filtered
    assert "lmm_router.list_slots" in filtered
    assert "github.search_repositories" not in filtered
    assert "context7.resolve-library-id" not in filtered
    assert plan.total_tools == 5
    assert plan.kept_tools == ("lmm_router.fleet_status", "lmm_router.list_slots")
    assert plan.tokens_after < plan.tokens_before


def test_coding_profile_keeps_dev_mcp_tools():
    filtered, plan = filter_mcp_prompt(PROMPT, profile="developer")

    assert "git.git_status" in filtered
    assert "github.search_repositories" in filtered
    assert "context7.resolve-library-id" in filtered
    assert "lmm_router.fleet_status" in filtered
    assert plan.policy == "coding"


def test_configured_allowlist_can_remove_all_tools():
    filtered, plan = filter_mcp_prompt(
        PROMPT,
        profile="agent0",
        config={
            "mcp_exposure": {
                "profile_allowlists": {
                    "agent0": ["does_not_exist.*"],
                }
            }
        },
    )

    assert filtered == ""
    assert plan.kept_tools == ()
    assert len(plan.removed_tools) == 5


def test_persist_manifest_writes_decisions(tmp_path):
    _, plan = filter_mcp_prompt(PROMPT, profile="agent0")

    persist_manifest(
        plan,
        config={"mcp_exposure": {"manifest_path": "manifest.json"}},
        plugin_root=tmp_path,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kept_tools"] == 2
    assert manifest["removed_tools"] == 3
    assert manifest["tools"][0]["qualified_name"] == "context7.resolve_library_id"


def test_should_apply_modes():
    assert should_apply_for_local_fleet({"mcp_exposure": {"enabled": True}}, local_fleet_active=True)
    assert not should_apply_for_local_fleet({"mcp_exposure": {"enabled": True}}, local_fleet_active=False)
    assert should_apply_for_local_fleet(
        {"mcp_exposure": {"enabled": True, "mode": "always"}},
        local_fleet_active=False,
    )
    assert not should_apply_for_local_fleet(
        {"mcp_exposure": {"enabled": False, "mode": "always"}},
        local_fleet_active=True,
    )
