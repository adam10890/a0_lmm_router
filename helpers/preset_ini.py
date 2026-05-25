"""Shared helpers for rewriting llama.cpp router models_preset.ini."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def default_preset_path(project_dir: str | None = None) -> Path:
    env_path = os.environ.get("LLAMACPP_PRESET_HOST", "").strip()
    if env_path:
        return Path(env_path).resolve()
    if project_dir:
        return (
            Path(project_dir) / "usr" / "plugins" / "a0_lmm_router" / "conf" / "models_preset.ini"
        ).resolve()
    return (_PLUGIN_ROOT / "conf" / "models_preset.ini").resolve()


def resolve_preset_path(requested: str | None = None, project_dir: str | None = None) -> Path:
    allowed = default_preset_path(project_dir)
    if not requested:
        return allowed
    candidate = Path(requested).resolve()
    if candidate != allowed:
        raise ValueError(f"preset path not allowed: {candidate}")
    return candidate


def rewrite_alias_model(content: str, alias: str, model_path: str) -> tuple[str, str]:
    if not alias or not model_path:
        raise ValueError("alias and model_path are required")
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    in_section = False
    found_section = False
    replaced = False
    snippet: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_section:
                in_section = False
            section = stripped[1:-1].strip()
            if section == alias:
                in_section = True
                found_section = True
        if in_section and stripped.lower().startswith("model") and "=" in stripped:
            newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            line = f"model = {model_path}{newline}"
            replaced = True
        out.append(line)
        if in_section:
            snippet.append(line)

    if not found_section:
        raise ValueError(f"alias section not found: {alias}")
    if not replaced:
        raise ValueError(f"model line not found in alias section: {alias}")
    return "".join(out), "".join(snippet)


def write_atomic(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    return backup


def write_alias_model(alias: str, model_path: str, preset_path: str | Path | None = None) -> dict:
    path = Path(preset_path).resolve() if preset_path else default_preset_path()
    if not path.is_file():
        return {"ok": False, "error": f"preset file not found: {path}"}
    try:
        current = path.read_text(encoding="utf-8")
        updated, snippet = rewrite_alias_model(current, alias, model_path)
        backup = write_atomic(path, updated)
        return {
            "ok": True,
            "preset_path": str(path),
            "backup_path": str(backup),
            "snippet": snippet,
            "via": "local",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
