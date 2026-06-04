"""Narrow Local Fleet adapter for clear, single tool-call outputs.

This module does not execute tools. It only converts a small local-model syntax
variant into Agent Zero's canonical JSON tool request shape.
"""
from __future__ import annotations

import json
import re
from typing import Iterable

DEFAULT_WIKI_TOOL_ALLOWLIST = frozenset(
    {
        "wiki_list",
        "wiki_query",
        "wiki_ingest",
        "wiki_lint",
        "wiki_commit",
        "wiki_register",
        "response",
    }
)

_LOCAL_TOOL_MARKERS = ("<|tool_call>call:", "<tool_call>call:")
_TRAILING_SUFFIXES = ("<tool_call|>", "<|tool_call|>", "</tool_call>")
_TOKENIZED_STRING_RE = re.compile(r'<\|"\|>([^<]+)<\|"\|>')
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def normalize_tool_call_output(
    raw: str,
    *,
    allowed_tools: Iterable[str] | None = None,
) -> str | None:
    """Return canonical A0 tool JSON when ``raw`` is a safe single local call.

    ``None`` means "do not normalize"; callers should pass the original output
    through to Agent Zero's normal parser.
    """
    if not isinstance(raw, str):
        return None
    if sum(raw.count(marker) for marker in _LOCAL_TOOL_MARKERS) != 1:
        return None

    sanitized = _sanitize_local_tool_call(raw)
    parsed = _parse_local_tool_call(sanitized, allowed_tools=allowed_tools)
    if parsed is None:
        return None

    tool_name, tool_args = parsed
    allowlist = set(allowed_tools) if allowed_tools is not None else None
    if allowlist is not None and tool_name not in allowlist:
        return None

    return json.dumps(
        {"tool_name": tool_name, "tool_args": tool_args},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def wrap_markdown_as_response(raw: str) -> str | None:
    """Wrap plain markdown/text as A0 ``response`` tool JSON when no tool JSON exists."""
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if len(text) < 40:
        return None
    if _looks_like_tool_json(text):
        return None
    if any(marker in text for marker in _LOCAL_TOOL_MARKERS):
        return None

    return json.dumps(
        {"tool_name": "response", "tool_args": {"text": text}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def adapt_final_model_response(
    raw: str,
    *,
    allowed_tools: Iterable[str] | None = None,
) -> str:
    """Best-effort final response normalization for Local Fleet outputs."""
    normalized = normalize_tool_call_output(raw, allowed_tools=allowed_tools)
    if normalized is not None:
        return normalized

    wrapped = wrap_markdown_as_response(raw)
    if wrapped is not None:
        return wrapped

    return raw


def _looks_like_tool_json(raw: str) -> bool:
    text = raw.strip()
    if not text.startswith("{"):
        return False
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return '"tool_name"' in text or '"tool"' in text
    if not isinstance(decoded, dict):
        return False
    tool_name = decoded.get("tool_name") or decoded.get("tool")
    tool_args = decoded.get("tool_args") or decoded.get("args")
    return isinstance(tool_name, str) and bool(tool_name) and isinstance(tool_args, dict)


def _sanitize_local_tool_call(raw: str) -> str:
    """Fix common Gemma/local artifacts before strict parsing."""
    text = raw.strip()
    changed = True
    while changed:
        changed = False
        for suffix in _TRAILING_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)].strip()
                changed = True
    text = _TOKENIZED_STRING_RE.sub(r'"\1"', text)
    return text


def _parse_local_tool_call(
    raw: str,
    *,
    allowed_tools: Iterable[str] | None,
) -> tuple[str, dict] | None:
    marker = next((m for m in _LOCAL_TOOL_MARKERS if m in raw), None)
    if marker is None:
        return None

    rest = raw.split(marker, 1)[1].strip()
    brace = rest.find("{")
    if brace == -1:
        return None

    tool_path = rest[:brace].strip().rstrip(":")
    args_raw = rest[brace:].strip()
    if not tool_path or not args_raw.startswith("{"):
        return None

    tool_name = _resolve_tool_name(tool_path, allowed_tools)
    if not tool_name:
        return None

    tool_args = _parse_jsonish_object(args_raw)
    if tool_args is None:
        return None
    return tool_name, tool_args


def _resolve_tool_name(tool_path: str, allowed_tools: Iterable[str] | None) -> str:
    allowlist = set(allowed_tools) if allowed_tools is not None else None
    if allowlist is not None and tool_path in allowlist:
        return tool_path

    parts = [part for part in tool_path.split(":") if part]
    if allowlist is not None:
        for part in reversed(parts):
            if part in allowlist:
                return part

    if parts and _IDENTIFIER_RE.fullmatch(parts[-1]):
        return parts[-1]
    return ""


def _parse_jsonish_object(raw_args: str) -> dict | None:
    """Parse a restricted JSON-like object with bare keys.

    The local syntax observed in logs is JSON except for unquoted object keys,
    e.g. ``{question: "..."}``. We only quote those keys and then use
    ``json.loads``; there is no Python expression evaluation here.
    """
    quoted = _quote_bare_object_keys(raw_args.strip())
    try:
        decoded = json.loads(quoted)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded


def _quote_bare_object_keys(raw_args: str) -> str:
    """Quote bare object keys while leaving string values untouched."""
    out: list[str] = []
    i = 0
    in_string = False
    escaped = False

    while i < len(raw_args):
        ch = raw_args[i]

        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch in "{,":
            out.append(ch)
            i += 1
            whitespace_start = i
            while i < len(raw_args) and raw_args[i].isspace():
                i += 1

            match = _IDENTIFIER_RE.match(raw_args, i)
            if match:
                key_end = match.end()
                colon_at = key_end
                while colon_at < len(raw_args) and raw_args[colon_at].isspace():
                    colon_at += 1
                if colon_at < len(raw_args) and raw_args[colon_at] == ":":
                    out.append(raw_args[whitespace_start:i])
                    out.append(f'"{match.group(0)}"')
                    out.append(raw_args[key_end:colon_at])
                    out.append(":")
                    i = colon_at + 1
                    continue

            out.append(raw_args[whitespace_start:i])
            continue

        out.append(ch)
        i += 1

    return "".join(out)
