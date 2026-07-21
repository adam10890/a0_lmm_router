"""
pi_runner — invoke the pi coding agent against local LMM Router slots.

Deploys conf/pi/* into ~/.pi/agent/, then runs pi in print mode for
headless coding tasks. Falls back to the host helper when pi is not on
PATH (e.g. MCP server inside the A0 container).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import aiohttp

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from paths import PLUGIN_DIR, TOKEN_CANDIDATES

DEFAULT_MODEL = "lmm-coding/utility"
CHAT_MODEL = "lmm-router/chat"
DEFAULT_TIMEOUT = 600
HOST_DEFAULT_PORT = 55501


def _pi_config_dir() -> Path:
    override = os.environ.get("PI_CODING_AGENT_DIR", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".pi" / "agent"


def _source_pi_conf() -> Path:
    return PLUGIN_DIR / "conf" / "pi"


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def deploy_pi_config() -> dict[str, str]:
    """Merge bundled pi config into ~/.pi/agent. Idempotent."""
    src = _source_pi_conf()
    dest = _pi_config_dir()
    dest.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for name in ("models.json", "settings.json"):
        src_file = src / name
        if not src_file.is_file():
            continue
        dest_file = dest / name
        payload = json.loads(src_file.read_text(encoding="utf-8"))
        if dest_file.is_file():
            try:
                existing = json.loads(dest_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            if isinstance(existing, dict) and isinstance(payload, dict):
                payload = _deep_merge_dict(existing, payload)
        dest_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written[name] = str(dest_file)

    return written


def find_pi_binary() -> str | None:
    custom = os.environ.get("PI_BIN", "").strip()
    if custom and Path(custom).is_file():
        return custom
    return shutil.which("pi")


def get_pi_version() -> str | None:
    pi_bin = find_pi_binary()
    if not pi_bin:
        return None
    try:
        result = subprocess.run(
            [pi_bin, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
        text = (result.stdout or result.stderr or "").strip()
        return text or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def load_router_pi_models() -> list[dict[str, Any]]:
    """Return lmm-router provider models from bundled or deployed config."""
    candidates = [_source_pi_conf() / "models.json", _pi_config_dir() / "models.json"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        providers = data.get("providers") or {}
        models: list[dict[str, Any]] = []
        for provider_id, provider in providers.items():
            if not str(provider_id).startswith("lmm"):
                continue
            base_url = str(provider.get("baseUrl", ""))
            for model in provider.get("models") or []:
                model_id = str(model.get("id", ""))
                if not model_id:
                    continue
                models.append({
                    "provider": provider_id,
                    "model": model_id,
                    "full_id": f"{provider_id}/{model_id}",
                    "name": model.get("name") or model_id,
                    "base_url": base_url,
                    "role": "chat" if "chat" in model_id or provider_id == "lmm-router" else "coding",
                    "context_window": model.get("contextWindow"),
                    "max_tokens": model.get("maxTokens"),
                })
        return models
    return []


async def _probe_slot(base_url: str) -> dict[str, Any]:
    health_url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    try:
        timeout = aiohttp.ClientTimeout(total=4)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(health_url) as resp:
                return {
                    "reachable": resp.status == 200,
                    "http_status": resp.status,
                    "health_url": health_url,
                }
    except Exception as exc:
        return {
            "reachable": False,
            "http_status": None,
            "health_url": health_url,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def get_pi_status_local() -> dict[str, Any]:
    pi_bin = find_pi_binary()
    models = load_router_pi_models()
    slots: dict[str, Any] = {}
    for entry in models:
        base_url = entry.get("base_url", "")
        if base_url:
            slots[entry["full_id"]] = await _probe_slot(str(base_url))

    config_paths = deploy_pi_config() if _source_pi_conf().is_dir() else {}
    return {
        "ok": True,
        "pi_installed": bool(pi_bin),
        "pi_binary": pi_bin,
        "pi_version": get_pi_version(),
        "config_dir": str(_pi_config_dir()),
        "config_paths": config_paths,
        "models": models,
        "slots": slots,
        "backend": "local",
    }


async def _host_helper_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout_sec: int = 180,
) -> dict[str, Any]:
    token = _resolve_token()
    if not token:
        return {
            "ok": False,
            "error": "host helper token missing",
            "hint": "Start tools/lmm_host_helper.py on the host.",
        }

    url = f"{_resolve_host_url()}{path}"
    headers = {"Content-Type": "application/json", "X-Token": token}
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout_sec)
        async with aiohttp.ClientSession(timeout=client_timeout, headers=headers) as session:
            if method.upper() == "GET":
                async with session.get(url) as resp:
                    text = await resp.text()
            else:
                async with session.post(url, json=json_body or {}) as resp:
                    text = await resp.text()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"ok": False, "error": text[:3000]}
            data.setdefault("backend", "host_helper")
            return data
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Host helper request failed: {type(exc).__name__}: {exc}",
            "backend": "host_helper",
        }


async def get_pi_status(prefer_host: bool | None = None) -> dict[str, Any]:
    use_host = prefer_host if prefer_host is not None else find_pi_binary() is None
    if use_host:
        remote = await _host_helper_request("GET", "/pi/status", timeout_sec=30)
        if remote.get("ok"):
            return remote
        if find_pi_binary():
            local = await get_pi_status_local()
            local["host_helper_error"] = remote.get("error")
            return local
        return remote
    return await get_pi_status_local()


async def install_pi_local() -> dict[str, Any]:
    npm = shutil.which("npm")
    if not npm:
        return {"ok": False, "error": "npm not found on PATH"}

    def _install() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [npm, "install", "-g", "--ignore-scripts", "@earendil-works/pi-coding-agent"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )

    try:
        result = await asyncio.to_thread(_install)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "npm install timed out after 300s"}

    if result.returncode != 0:
        return {
            "ok": False,
            "error": "npm install failed",
            "stderr": (result.stderr or "")[:4000],
            "stdout": (result.stdout or "")[:2000],
        }

    config_paths = deploy_pi_config()
    status = await get_pi_status_local()
    status["install_stdout"] = (result.stdout or "")[:2000]
    status["config_paths"] = config_paths
    return status


async def install_pi() -> dict[str, Any]:
    if find_pi_binary():
        config_paths = deploy_pi_config()
        status = await get_pi_status_local()
        status["message"] = "pi already installed; config redeployed"
        status["config_paths"] = config_paths
        return status
    remote = await _host_helper_request("POST", "/pi/install", timeout_sec=360)
    if remote.get("ok"):
        return remote
    return await install_pi_local()


def _resolve_host_url() -> str:
    url = os.environ.get("A0_LMM_HOST_URL", "").strip()
    if url:
        return url.rstrip("/")
    host = os.environ.get("A0_LMM_HOST_HOST", "host.docker.internal").strip()
    port = os.environ.get("A0_LMM_HOST_PORT", str(HOST_DEFAULT_PORT)).strip()
    return f"http://{host}:{port}"


def _resolve_token() -> str:
    tok = os.environ.get("A0_LMM_HOST_TOKEN", "").strip()
    if tok:
        return tok
    for candidate in TOKEN_CANDIDATES:
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return ""


def _normalize_working_directory(working_directory: str | None) -> Path:
    if working_directory:
        path = Path(working_directory).expanduser()
    else:
        path = Path.cwd()
    if not path.is_absolute():
        path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"working_directory does not exist: {path}")
    return path


def _build_pi_command(
    prompt: str,
    model: str,
    *,
    read_only: bool = False,
) -> list[str]:
    pi_bin = find_pi_binary()
    if not pi_bin:
        raise FileNotFoundError(
            "pi CLI not found on PATH. Install with: "
            "npm install -g --ignore-scripts @earendil-works/pi-coding-agent"
        )
    cmd = [
        pi_bin,
        "-p",
        "--model",
        model,
        "--no-session",
        "--approve",
        prompt,
    ]
    if read_only:
        cmd.insert(-1, "--tools")
        cmd.insert(-1, "read,grep,find,ls")
    return cmd


async def _run_local(
    prompt: str,
    working_directory: Path,
    model: str,
    timeout: int,
    read_only: bool,
) -> dict[str, Any]:
    deploy_pi_config()
    cmd = _build_pi_command(prompt, model, read_only=read_only)

    def _run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            cwd=str(working_directory),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )

    try:
        result = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"pi timed out after {timeout}s",
            "model": model,
            "working_directory": str(working_directory),
        }

    ok = result.returncode == 0
    return {
        "ok": ok,
        "model": model,
        "working_directory": str(working_directory),
        "exit_code": result.returncode,
        "output": (result.stdout or "").strip(),
        "stderr": (result.stderr or "").strip(),
        "backend": "local",
    }


async def _run_via_host_helper(
    prompt: str,
    working_directory: Path,
    model: str,
    timeout: int,
    read_only: bool,
) -> dict[str, Any]:
    token = _resolve_token()
    if not token:
        return {
            "ok": False,
            "error": (
                "pi is not on PATH and host helper token is missing. "
                "Install pi on the host or start tools/lmm_host_helper.py."
            ),
        }

    payload = {
        "prompt": prompt,
        "working_directory": str(working_directory),
        "model": model,
        "timeout": timeout,
        "read_only": read_only,
    }
    url = f"{_resolve_host_url()}/pi/coding"
    headers = {"Content-Type": "application/json", "X-Token": token}
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout + 30)
        async with aiohttp.ClientSession(timeout=client_timeout, headers=headers) as session:
            async with session.post(url, json=payload) as resp:
                text = await resp.text()
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = {"ok": False, "error": text[:3000]}
                data.setdefault("backend", "host_helper")
                return data
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Host helper pi call failed: {type(exc).__name__}: {exc}",
            "backend": "host_helper",
        }


async def run_pi_coding(
    prompt: str,
    *,
    working_directory: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    read_only: bool = False,
    prefer_host: bool | None = None,
) -> dict[str, Any]:
    """Run a pi coding task against the configured local model."""
    prompt = (prompt or "").strip()
    if not prompt:
        return {"ok": False, "error": "prompt is required"}

    try:
        cwd = _normalize_working_directory(working_directory)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    use_host = prefer_host
    if use_host is None:
        use_host = find_pi_binary() is None

    if use_host:
        return await _run_via_host_helper(prompt, cwd, model, timeout, read_only)
    return await _run_local(prompt, cwd, model, timeout, read_only)


def run_pi_coding_sync(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(run_pi_coding(**kwargs))


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "deploy"
    if action == "deploy":
        print(json.dumps(deploy_pi_config(), indent=2))
    elif action == "list-models":
        deploy_pi_config()
        pi_bin = find_pi_binary()
        if not pi_bin:
            print("pi not found", file=sys.stderr)
            sys.exit(1)
        subprocess.run([pi_bin, "--list-models", "lmm"], check=False)
    else:
        print("Usage: python helpers/pi_runner.py [deploy|list-models]", file=sys.stderr)
        sys.exit(2)
