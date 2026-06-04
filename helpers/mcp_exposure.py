"""Profile-aware MCP prompt exposure for Local Fleet chats."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PLUGIN_NAME = "a0_lmm_router"
TELEMETRY_KEY = "a0_lmm_router_mcp_exposure"

_HEADING_RE = re.compile(r"(?m)^###\s+(.+?)\s*$")
_MCP_HEADER = '## "Remote (MCP Server) Agent Tools" available:'


DEFAULT_PROFILE_ALIASES: dict[str, str] = {
    "agent0": "agent0",
    "default": "agent0",
    "main": "agent0",
    "chat": "agent0",
    "supervisor": "agent0",
    "subagent": "agent0",
    "developer": "coding",
    "dev": "coding",
    "coder": "coding",
    "code": "coding",
    "hacker": "coding",
    "researcher": "research",
    "research": "research",
    "browser": "browser",
    "web": "browser",
    "wiki_librarian": "wiki",
    "wiki-librarian": "wiki",
    "wikilibrarian": "wiki",
    "librarian": "wiki",
    "llm_wiki": "wiki",
    "google": "google",
    "gmail": "google",
    "calendar": "google",
    "drive": "google",
    "documents": "documents",
    "document": "documents",
    "document_worker": "documents",
    "meta_supervisor": "meta",
    "meta": "meta",
    "swarm": "swarm",
    "parallel": "swarm",
}


DEFAULT_PROFILE_ALLOWLISTS: dict[str, list[str]] = {
    "agent0": [
        "lmm_router.fleet_status",
        "lmm_router.list_slots",
    ],
    "coding": [
        "context7.*",
        "fetch.*",
        "git.*",
        "github.*",
        "lmm_router.*",
        "memory.*",
        "sequential_thinking.*",
    ],
    "research": [
        "context7.*",
        "fetch.*",
        "github.search_*",
        "github.get_*",
        "perplexity_ask.*",
        "memory.*",
        "sequential_thinking.*",
    ],
    "browser": [
        "context7.*",
        "fetch.*",
        "perplexity_ask.*",
    ],
    "wiki": [
        "memory.*",
        "lmm_router.fleet_status",
        "lmm_router.list_slots",
    ],
    "google": [],
    "documents": [
        "fetch.*",
    ],
    "meta": [
        "lmm_router.fleet_status",
        "lmm_router.list_slots",
        "memory.*",
        "sequential_thinking.*",
    ],
    "swarm": [
        "lmm_router.fleet_status",
        "lmm_router.list_slots",
        "memory.*",
        "sequential_thinking.*",
    ],
}


@dataclass(frozen=True)
class MCPToolSection:
    server: str
    name: str
    heading: str
    text: str
    tokens: int

    @property
    def qualified_name(self) -> str:
        return f"{self.server}.{self.name}" if self.name else self.server


@dataclass(frozen=True)
class MCPServerSection:
    name: str
    heading: str
    text: str
    tokens: int
    tools: tuple[MCPToolSection, ...]


@dataclass(frozen=True)
class MCPToolDecision:
    server: str
    name: str
    qualified_name: str
    tokens: int
    kept: bool


@dataclass(frozen=True)
class MCPExposurePlan:
    profile: str
    policy: str
    allowlist: tuple[str, ...]
    total_servers: int
    kept_servers: tuple[str, ...]
    removed_servers: tuple[str, ...]
    total_tools: int
    kept_tools: tuple[str, ...]
    removed_tools: tuple[str, ...]
    decisions: tuple[MCPToolDecision, ...]
    tokens_before: int
    tokens_after: int

    @property
    def changed(self) -> bool:
        return bool(self.removed_tools or self.removed_servers)

    def telemetry(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "policy": self.policy,
            "allowlist": list(self.allowlist),
            "total_servers": self.total_servers,
            "kept_servers": len(self.kept_servers),
            "removed_servers": len(self.removed_servers),
            "total_tools": self.total_tools,
            "kept_tools": len(self.kept_tools),
            "removed_tools": len(self.removed_tools),
            "mcp_tokens_before": self.tokens_before,
            "mcp_tokens_after": self.tokens_after,
            "mcp_tokens_saved": max(0, self.tokens_before - self.tokens_after),
            "kept": list(self.kept_tools),
            "removed": list(self.removed_tools),
            "kept_server_names": list(self.kept_servers),
            "removed_server_names": list(self.removed_servers),
        }


def exposure_enabled(config: dict[str, Any] | None) -> bool:
    exposure = _exposure_config(config)
    if not _truthy(exposure.get("enabled", True)):
        return False
    env = os.getenv("A0_LMM_MCP_EXPOSURE_ENABLED", "").strip().lower()
    if env:
        return env not in {"0", "false", "no", "off"}
    return True


def exposure_mode(config: dict[str, Any] | None) -> str:
    env = os.getenv("A0_LMM_MCP_EXPOSURE_MODE", "").strip().lower()
    if env:
        return env
    return str(_exposure_config(config).get("mode") or "local_fleet").strip().lower()


def should_apply_for_local_fleet(config: dict[str, Any] | None, *, local_fleet_active: bool) -> bool:
    if not exposure_enabled(config):
        return False
    mode = exposure_mode(config)
    if mode in {"off", "disabled", "none"}:
        return False
    if mode in {"always", "on", "enabled"}:
        return True
    return bool(local_fleet_active)


def filter_mcp_prompt(
    prompt: str,
    *,
    profile: str,
    config: dict[str, Any] | None = None,
) -> tuple[str, MCPExposurePlan]:
    """Filter a rendered Agent Zero MCP prompt by profile allowlist."""
    exposure = _exposure_config(config)
    policy = resolve_policy(profile, exposure)
    allowlist = tuple(resolve_allowlist(policy, exposure))
    prefix, servers = parse_mcp_prompt(prompt)
    if not servers:
        token_count = estimate_tokens(prompt or "")
        plan = MCPExposurePlan(
            profile=profile or "agent0",
            policy=policy,
            allowlist=allowlist,
            total_servers=0,
            kept_servers=(),
            removed_servers=(),
            total_tools=0,
            kept_tools=(),
            removed_tools=(),
            decisions=(),
            tokens_before=token_count,
            tokens_after=token_count,
        )
        return prompt, plan

    kept_blocks: list[str] = []
    kept_servers: list[str] = []
    removed_servers: list[str] = []
    kept_tools: list[str] = []
    removed_tools: list[str] = []
    decisions: list[MCPToolDecision] = []
    tokens_before = estimate_tokens(prefix)
    tokens_after = estimate_tokens(prefix)

    for server in servers:
        server_kept_tools: list[MCPToolSection] = []
        server_token_before = server.tokens + sum(tool.tokens for tool in server.tools)
        tokens_before += server_token_before
        for tool in server.tools:
            keep = is_allowed(
                [tool.qualified_name, tool.server, tool.name],
                allowlist,
            )
            if keep:
                server_kept_tools.append(tool)
                kept_tools.append(tool.qualified_name)
            else:
                removed_tools.append(tool.qualified_name)
            decisions.append(
                MCPToolDecision(
                    server=tool.server,
                    name=tool.name,
                    qualified_name=tool.qualified_name,
                    tokens=tool.tokens,
                    kept=keep,
                )
            )

        if server_kept_tools:
            kept_servers.append(server.name)
            kept_blocks.append(server.text.rstrip())
            kept_blocks.extend(tool.text.rstrip() for tool in server_kept_tools)
            tokens_after += server.tokens + sum(tool.tokens for tool in server_kept_tools)
        else:
            removed_servers.append(server.name)

    rendered = ""
    if kept_blocks:
        header = prefix.strip() or _MCP_HEADER
        rendered = header + "\n\n" + "\n\n".join(block for block in kept_blocks if block).strip()

    plan = MCPExposurePlan(
        profile=profile or "agent0",
        policy=policy,
        allowlist=allowlist,
        total_servers=len(servers),
        kept_servers=tuple(_dedupe(kept_servers)),
        removed_servers=tuple(_dedupe(removed_servers)),
        total_tools=sum(len(server.tools) for server in servers),
        kept_tools=tuple(_dedupe(kept_tools)),
        removed_tools=tuple(_dedupe(removed_tools)),
        decisions=tuple(decisions),
        tokens_before=tokens_before,
        tokens_after=tokens_after if rendered else 0,
    )
    return rendered, plan


def parse_mcp_prompt(prompt: str) -> tuple[str, tuple[MCPServerSection, ...]]:
    """Parse Agent Zero's rendered MCP tools prompt into server/tool sections."""
    text = prompt or ""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return text, ()

    prefix = text[: matches[0].start()].rstrip()
    servers: list[MCPServerSection] = []
    current_name = ""
    current_heading = ""
    current_text = ""
    current_tokens = 0
    current_tools: list[MCPToolSection] = []

    def flush_server() -> None:
        nonlocal current_name, current_heading, current_text, current_tokens, current_tools
        if not current_name:
            return
        servers.append(
            MCPServerSection(
                name=current_name,
                heading=current_heading,
                text=current_text,
                tokens=current_tokens,
                tools=tuple(current_tools),
            )
        )
        current_name = ""
        current_heading = ""
        current_text = ""
        current_tokens = 0
        current_tools = []

    for idx, match in enumerate(matches):
        raw_heading = match.group(1).strip()
        section_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[match.start() : section_end].strip()
        parsed = _parse_tool_heading(raw_heading)
        if parsed is None:
            flush_server()
            current_name = _clean_name(raw_heading)
            current_heading = raw_heading
            current_text = section_text
            current_tokens = estimate_tokens(section_text)
            current_tools = []
            continue

        server_name, tool_name = parsed
        if not current_name or current_name != server_name:
            flush_server()
            current_name = server_name
            current_heading = server_name
            current_text = f"### {server_name}"
            current_tokens = estimate_tokens(current_text)
            current_tools = []

        current_tools.append(
            MCPToolSection(
                server=server_name,
                name=tool_name,
                heading=raw_heading,
                text=section_text,
                tokens=estimate_tokens(section_text),
            )
        )

    flush_server()
    return prefix, tuple(servers)


def resolve_policy(profile: str, exposure: dict[str, Any] | None = None) -> str:
    exposure = exposure or {}
    normalized = _clean_name(profile)
    aliases = dict(DEFAULT_PROFILE_ALIASES)
    aliases.update(_string_map(exposure.get("profile_aliases")))
    if normalized in aliases:
        return aliases[normalized]
    default_profile = str(exposure.get("default_profile") or "agent0")
    return _clean_name(default_profile) or "agent0"


def resolve_allowlist(policy: str, exposure: dict[str, Any] | None = None) -> list[str]:
    exposure = exposure or {}
    allowlists = {key: list(value) for key, value in DEFAULT_PROFILE_ALLOWLISTS.items()}
    configured = exposure.get("profile_allowlists")
    if isinstance(configured, dict):
        for key, value in configured.items():
            normalized_key = _clean_name(str(key))
            configured_values = _string_list(value)
            if configured_values:
                allowlists[normalized_key] = configured_values
    selected = allowlists.get(_clean_name(policy), allowlists["agent0"])
    return _dedupe([_clean_pattern(item) for item in selected if _clean_pattern(item)])


def is_allowed(candidates: Iterable[str], allowlist: Iterable[str]) -> bool:
    names = [_clean_pattern(name) for name in candidates]
    patterns = [_clean_pattern(pattern) for pattern in allowlist]
    for name in names:
        for pattern in patterns:
            if fnmatch.fnmatchcase(name, pattern):
                return True
    return False


def persist_manifest(
    plan: MCPExposurePlan,
    *,
    config: dict[str, Any] | None = None,
    plugin_root: Path | None = None,
) -> None:
    exposure = _exposure_config(config)
    raw_path = str(exposure.get("manifest_path") or "data/mcp_exposure_manifest.json")
    if not raw_path:
        return

    root = plugin_root or Path(__file__).resolve().parents[1]
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path

    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **plan.telemetry(),
        "tools": [
            {
                "server": decision.server,
                "name": decision.name,
                "qualified_name": decision.qualified_name,
                "tokens": decision.tokens,
                "kept": decision.kept,
            }
            for decision in plan.decisions
        ],
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if _same_manifest_without_timestamp(path, manifest):
            return
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        return


def manifest_signature(plan: MCPExposurePlan) -> str:
    payload = {
        "profile": plan.profile,
        "policy": plan.policy,
        "kept": list(plan.kept_tools),
        "removed": list(plan.removed_tools),
        "tokens_before": plan.tokens_before,
        "tokens_after": plan.tokens_after,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    try:
        from helpers import tokens

        return int(tokens.approximate_prompt_tokens(text))
    except Exception:
        return max(1, len(text or "") // 4)


def _parse_tool_heading(raw_heading: str) -> tuple[str, str] | None:
    heading = raw_heading.strip().rstrip(":")
    if "." not in heading:
        return None
    server, tool = heading.split(".", 1)
    server_name = _clean_name(server)
    tool_name = _clean_name(tool)
    if not server_name or not tool_name:
        return None
    return server_name, tool_name


def _exposure_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    exposure = config.get("mcp_exposure")
    return exposure if isinstance(exposure, dict) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _clean_name(value: str) -> str:
    cleaned = str(value or "").strip().lower().replace("-", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    return cleaned.strip("_")


def _clean_pattern(value: str) -> str:
    parts = str(value or "").strip().lower().replace("-", "_").split(".", 1)
    cleaned = [_clean_name(part).replace("_", "_") for part in parts]
    if len(parts) == 2:
        right = re.sub(r"[^a-z0-9_*]+", "_", parts[1].strip().lower().replace("-", "_"))
        return f"{cleaned[0]}.{right.strip('_')}"
    raw = str(value or "").strip().lower().replace("-", "_")
    return re.sub(r"[^a-z0-9_.*]+", "_", raw).strip("_")


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,\s]+", value) if part.strip()]
    return []


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key, item in value.items():
        normalized_key = _clean_name(str(key))
        normalized_value = _clean_name(str(item))
        if normalized_key and normalized_value:
            result[normalized_key] = normalized_value
    return result


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _same_manifest_without_timestamp(path: Path, manifest: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    current.pop("generated_at", None)
    proposed = dict(manifest)
    proposed.pop("generated_at", None)
    return current == proposed
