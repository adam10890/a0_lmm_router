"""Profile-aware Agent Zero tool prompt exposure for local fleet chats."""
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
TELEMETRY_KEY = "a0_lmm_router_tool_exposure"
TOOL_KWARGS_KEY = "_tool_prompt_kwargs"

_TOOL_FILE_PREFIX = "agent.system.tool."
_TOOL_FILE_SUFFIX = ".md"
_HEADING_RE = re.compile(r"^\s*#{2,4}\s+([A-Za-z_][A-Za-z0-9_:-]*)", re.M)
_TOOL_JSON_RE = re.compile(r'"tool_name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"')


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
        "response",
        "call_subordinate",
        "delegate_profile",
        "pen_paper",
        "skills",
        "skills_tool",
        "memory",
        "memory_load",
        "memory_save",
        "memory_delete",
        "memory_forget",
        "notify_user",
        "wait",
    ],
    "coding": [
        "response",
        "call_subordinate",
        "delegate_profile",
        "text_editor",
        "text_editor_remote",
        "code_exe",
        "code_execution_tool",
        "code_execution_remote",
        "input",
        "document_query",
        "search_engine",
        "skills",
        "skills_tool",
        "memory",
        "memory_*",
    ],
    "research": [
        "response",
        "call_subordinate",
        "delegate_profile",
        "search_engine",
        "browser",
        "document_query",
        "autoresearch",
        "wiki_list",
        "wiki_query",
        "skills",
        "skills_tool",
        "memory",
        "memory_*",
    ],
    "browser": [
        "response",
        "call_subordinate",
        "delegate_profile",
        "browser",
        "search_engine",
        "document_query",
        "skills",
        "skills_tool",
        "memory",
        "memory_*",
    ],
    "wiki": [
        "response",
        "wiki_*",
        "skills",
        "skills_tool",
        "memory",
        "memory_*",
    ],
    "google": [
        "response",
        "delegate_profile",
        "calendar_*",
        "contacts_*",
        "drive_*",
        "gmail_*",
        "sheets_*",
        "tasks_*",
        "skills",
        "skills_tool",
        "memory",
        "memory_*",
    ],
    "documents": [
        "response",
        "delegate_profile",
        "document_query",
        "office_artifact",
        "text_editor",
        "skills",
        "skills_tool",
        "memory",
        "memory_*",
    ],
    "meta": [
        "response",
        "meta_*",
        "delegate_profile",
        "skills",
        "skills_tool",
        "memory",
        "memory_*",
    ],
    "swarm": [
        "response",
        "delegate_profile",
        "delegate_parallel",
        "swarm_message",
        "pen_paper",
        "skills",
        "skills_tool",
        "memory",
        "memory_*",
    ],
}


@dataclass(frozen=True)
class ToolPrompt:
    file_name: str
    text: str
    source_path: str = ""


@dataclass(frozen=True)
class ToolPromptDecision:
    file_name: str
    source_path: str
    candidates: tuple[str, ...]
    capabilities: tuple[str, ...]
    tokens: int
    kept: bool


@dataclass(frozen=True)
class ToolExposurePlan:
    profile: str
    policy: str
    allowlist: tuple[str, ...]
    total_tools: int
    kept_tools: tuple[ToolPrompt, ...]
    removed_tools: tuple[ToolPrompt, ...]
    decisions: tuple[ToolPromptDecision, ...]
    tokens_before: int
    tokens_after: int

    @property
    def removed_count(self) -> int:
        return len(self.removed_tools)

    @property
    def kept_count(self) -> int:
        return len(self.kept_tools)

    @property
    def changed(self) -> bool:
        return self.removed_count > 0

    def telemetry(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "policy": self.policy,
            "allowlist": list(self.allowlist),
            "total_tools": self.total_tools,
            "kept_tools": self.kept_count,
            "removed_tools": self.removed_count,
            "tool_tokens_before": self.tokens_before,
            "tool_tokens_after": self.tokens_after,
            "tool_tokens_saved": max(0, self.tokens_before - self.tokens_after),
            "kept": [tool_name(d) for d in self.decisions if d.kept],
            "removed": [tool_name(d) for d in self.decisions if not d.kept],
        }


def exposure_enabled(config: dict[str, Any] | None) -> bool:
    exposure = _exposure_config(config)
    if not _truthy(exposure.get("enabled", True)):
        return False
    env = os.getenv("A0_LMM_TOOL_EXPOSURE_ENABLED", "").strip().lower()
    if env:
        return env not in {"0", "false", "no", "off"}
    return True


def exposure_mode(config: dict[str, Any] | None) -> str:
    env = os.getenv("A0_LMM_TOOL_EXPOSURE_MODE", "").strip().lower()
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


def build_tool_exposure_plan(
    tools: Iterable[ToolPrompt],
    *,
    profile: str,
    config: dict[str, Any] | None = None,
) -> ToolExposurePlan:
    """Select tool prompt files that should be visible to one profile."""
    tool_list = tuple(tools)
    exposure = _exposure_config(config)
    policy = resolve_policy(profile, exposure)
    allowlist = tuple(resolve_allowlist(policy, exposure))

    kept: list[ToolPrompt] = []
    removed: list[ToolPrompt] = []
    decisions: list[ToolPromptDecision] = []
    tokens_before = 0
    tokens_after = 0

    for prompt in tool_list:
        candidates = tuple(extract_tool_candidates(prompt.file_name, prompt.text))
        capabilities = tuple(infer_capabilities(candidates))
        token_count = estimate_tokens(prompt.text)
        tokens_before += token_count
        keep = is_allowed(candidates, allowlist)
        if keep:
            kept.append(prompt)
            tokens_after += token_count
        else:
            removed.append(prompt)
        decisions.append(
            ToolPromptDecision(
                file_name=prompt.file_name,
                source_path=prompt.source_path,
                candidates=candidates,
                capabilities=capabilities,
                tokens=token_count,
                kept=keep,
            )
        )

    if not kept and tool_list:
        fallback = _fallback_tool(tool_list)
        if fallback is not None:
            kept.append(fallback)
            removed = [tool for tool in removed if tool != fallback]
            tokens_after = sum(estimate_tokens(tool.text) for tool in kept)
            decisions = [
                _replace_decision(decision, kept=True)
                if decision.file_name == fallback.file_name
                else decision
                for decision in decisions
            ]

    return ToolExposurePlan(
        profile=profile or "agent0",
        policy=policy,
        allowlist=allowlist,
        total_tools=len(tool_list),
        kept_tools=tuple(kept),
        removed_tools=tuple(removed),
        decisions=tuple(decisions),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
    )


def render_filtered_tools_prompt(agent: Any, config: dict[str, Any] | None = None) -> tuple[str, ToolExposurePlan] | None:
    """Rebuild Agent Zero's tool prompt with profile-aware filtering.

    This mirrors core ``extensions/python/system_prompt/_11_tools_prompt.py`` but
    selects prompt files before rendering the final ``agent.system.tools.md``.
    """
    if agent is None:
        return None

    try:
        from helpers import files, subagents
    except Exception:
        return None

    try:
        prompt_dirs = subagents.get_paths(agent, "prompts")
        tool_files = files.get_unique_filenames_in_dirs(
            prompt_dirs,
            "agent.system.tool.*.md",
        )
    except Exception:
        return None

    all_tool_kwargs: dict[str, dict[str, Any]] = {}
    try:
        raw_kwargs = agent.get_data(TOOL_KWARGS_KEY) or {}
        if isinstance(raw_kwargs, dict):
            all_tool_kwargs = raw_kwargs
    except Exception:
        all_tool_kwargs = {}

    prompts: list[ToolPrompt] = []
    for tool_file in tool_files:
        try:
            file_name = os.path.basename(str(tool_file))
            extra = all_tool_kwargs.get(file_name, {})
            text = agent.read_prompt(file_name, **extra)
            prompts.append(ToolPrompt(file_name=file_name, text=text, source_path=str(tool_file)))
        except Exception:
            continue

    profile = _agent_profile(agent)
    plan = build_tool_exposure_plan(prompts, profile=profile, config=config)
    tools_str = "\n\n".join(tool.text for tool in plan.kept_tools)
    try:
        prompt = agent.read_prompt("agent.system.tools.md", tools=tools_str)
    except Exception:
        prompt = "## available tools\n" + tools_str

    if _chat_model_has_vision(agent):
        try:
            prompt += "\n\n" + agent.read_prompt("agent.system.tools_vision.md")
        except Exception:
            pass

    return prompt, plan


def extract_tool_candidates(file_name: str, text: str) -> list[str]:
    candidates: list[str] = []

    file_id = _file_tool_id(file_name)
    if file_id:
        candidates.append(file_id)

    for pattern in (_HEADING_RE, _TOOL_JSON_RE):
        for match in pattern.finditer(text or ""):
            value = _clean_candidate(match.group(1))
            if value:
                candidates.append(value)

    return _dedupe(candidates)


def resolve_policy(profile: str, exposure: dict[str, Any] | None = None) -> str:
    exposure = exposure or {}
    normalized = _normalize_profile(profile)
    aliases = dict(DEFAULT_PROFILE_ALIASES)
    aliases.update(_string_map(exposure.get("profile_aliases")))
    if normalized in aliases:
        return aliases[normalized]
    default_profile = str(exposure.get("default_profile") or "agent0")
    return _normalize_profile(default_profile) or "agent0"


def resolve_allowlist(policy: str, exposure: dict[str, Any] | None = None) -> list[str]:
    exposure = exposure or {}
    allowlists = {key: list(value) for key, value in DEFAULT_PROFILE_ALLOWLISTS.items()}
    configured = exposure.get("profile_allowlists")
    if isinstance(configured, dict):
        for key, value in configured.items():
            normalized_key = _normalize_profile(str(key))
            configured_values = _string_list(value)
            if configured_values:
                allowlists[normalized_key] = configured_values
    selected = allowlists.get(_normalize_profile(policy), allowlists["agent0"])
    return _dedupe([_clean_candidate(item) for item in selected if _clean_candidate(item)])


def is_allowed(candidates: Iterable[str], allowlist: Iterable[str]) -> bool:
    names = list(candidates)
    patterns = list(allowlist)
    for name in names:
        for pattern in patterns:
            if fnmatch.fnmatchcase(name, pattern):
                return True
    return False


def persist_manifest(
    plan: ToolExposurePlan,
    *,
    config: dict[str, Any] | None = None,
    plugin_root: Path | None = None,
) -> None:
    exposure = _exposure_config(config)
    raw_path = str(exposure.get("manifest_path") or "data/tool_exposure_manifest.json")
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
                "name": tool_name(decision),
                "file_name": decision.file_name,
                "source_path": decision.source_path,
                "candidates": list(decision.candidates),
                "capabilities": list(decision.capabilities),
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


def manifest_signature(plan: ToolExposurePlan) -> str:
    payload = {
        "profile": plan.profile,
        "policy": plan.policy,
        "kept": [tool_name(d) for d in plan.decisions if d.kept],
        "removed": [tool_name(d) for d in plan.decisions if not d.kept],
        "tokens_before": plan.tokens_before,
        "tokens_after": plan.tokens_after,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tool_name(decision: ToolPromptDecision) -> str:
    for candidate in decision.candidates:
        if candidate and not candidate.endswith("_tool"):
            return candidate
    return decision.candidates[0] if decision.candidates else decision.file_name


def infer_capabilities(candidates: Iterable[str]) -> list[str]:
    caps: list[str] = []
    for name in candidates:
        if name.startswith(("gmail_", "calendar_", "drive_", "contacts_", "sheets_", "tasks_")):
            caps.append("google")
        elif name.startswith("wiki_"):
            caps.append("wiki")
        elif name.startswith("meta_"):
            caps.append("meta")
        elif name in {"browser", "search_engine", "autoresearch"}:
            caps.append("research")
        elif name in {"code_exe", "code_execution_tool", "text_editor", "text_editor_remote"}:
            caps.append("coding")
        elif name in {"document_query", "office_artifact"}:
            caps.append("documents")
        elif name in {"delegate_profile", "delegate_parallel", "call_subordinate", "swarm_message"}:
            caps.append("delegation")
        elif name.startswith("memory"):
            caps.append("memory")
        elif name.startswith("pen_paper"):
            caps.append("planning")
    return _dedupe(caps)


def estimate_tokens(text: str) -> int:
    try:
        from helpers import tokens

        return int(tokens.approximate_prompt_tokens(text))
    except Exception:
        return max(1, len(text or "") // 4)


def _exposure_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    exposure = config.get("tool_exposure")
    return exposure if isinstance(exposure, dict) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _file_tool_id(file_name: str) -> str:
    base = os.path.basename(str(file_name))
    if base.startswith(_TOOL_FILE_PREFIX):
        base = base[len(_TOOL_FILE_PREFIX) :]
    if base.endswith(_TOOL_FILE_SUFFIX):
        base = base[: -len(_TOOL_FILE_SUFFIX)]
    return _clean_candidate(base)


def _clean_candidate(value: str) -> str:
    cleaned = str(value or "").strip().lower().replace("-", "_")
    cleaned = cleaned.rstrip(":")
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[0]
    return cleaned


def _normalize_profile(profile: str) -> str:
    return _clean_candidate(profile)


def _agent_profile(agent: Any) -> str:
    try:
        cfg = getattr(agent, "config", None)
        profile = getattr(cfg, "profile", "")
        if isinstance(profile, str) and profile.strip():
            return profile.strip()
    except Exception:
        pass
    return "agent0"


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
        normalized_key = _normalize_profile(str(key))
        normalized_value = _normalize_profile(str(item))
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


def _fallback_tool(tools: Iterable[ToolPrompt]) -> ToolPrompt | None:
    for tool in tools:
        if "response" in extract_tool_candidates(tool.file_name, tool.text):
            return tool
    for tool in tools:
        return tool
    return None


def _replace_decision(decision: ToolPromptDecision, *, kept: bool) -> ToolPromptDecision:
    return ToolPromptDecision(
        file_name=decision.file_name,
        source_path=decision.source_path,
        candidates=decision.candidates,
        capabilities=decision.capabilities,
        tokens=decision.tokens,
        kept=kept,
    )


def _chat_model_has_vision(agent: Any) -> bool:
    try:
        from plugins._model_config.helpers.model_config import get_chat_model_config

        cfg = get_chat_model_config(agent)
        return bool(cfg.get("vision", False))
    except Exception:
        return False


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
