"""API endpoint: /plugins/a0_lmm_router/lmm_test_prompt

Send a test prompt to a specific llama.cpp slot and return the structured
response, including any `reasoning_content` field returned by reasoning
models (e.g. Gemma 4, DeepSeek-R1). This drives the Model Test panel in
the dashboard and is useful for quickly comparing model behavior without
routing through the full A0 agent stack.

Request body:
    {
      "slot": "chat" | "utility" | "embedding" | "<slot_id>",
      "prompt": "string",
      "max_tokens": 512,          # optional, default 512
      "temperature": 0.7,         # optional
      "system": "string"          # optional system prompt
    }

Response:
    {
      "ok": true,
      "slot_id": "slot_chat",
      "host": "host.docker.internal:8080",
      "content": "...answer...",
      "reasoning_content": "...thought process..." | null,
      "finish_reason": "stop",
      "usage": { "prompt_tokens": 24, "completion_tokens": 180, ... },
      "timings": { "predicted_per_second": 163.5, "prompt_per_second": 262.1, ... },
      "duration_ms": 1340,
      "model_alias": "chat"
    }
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import aiohttp
from flask import Request

from helpers.api import ApiHandler


DEFAULT_MAX_TOKENS = 512
REQUEST_TIMEOUT_SEC = 120


def _resolve_conf_path() -> str:
    """Same self-contained resolver as llamacpp_status — kept local to avoid
    depending on `helpers.files` (which imports the fragile simpleeval)."""
    here = Path(__file__).resolve()
    plugin_conf = str(here.parents[1] / "conf" / "llama_cpp_servers.yaml")
    root_conf = str(here.parents[4] / "conf" / "llama_cpp_servers.yaml")
    return plugin_conf if os.path.exists(plugin_conf) else root_conf


def _resolve_slot(manager, slot_key: str):
    """Accept either a slot id (`slot_chat`) or a role (`chat`). Returns
    the matching ServerInstance or None."""
    if not slot_key:
        return None
    # Direct id match first
    if slot_key in manager.servers:
        return manager.servers[slot_key]
    # Role match
    for srv in manager.servers.values():
        role = getattr(srv.config.role, "value", str(srv.config.role))
        if role == slot_key:
            return srv
    return None


class LmmTestPrompt(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        slot_key = str(input.get("slot", "chat")).strip()
        prompt = input.get("prompt", "")
        system = input.get("system")
        max_tokens = int(input.get("max_tokens", DEFAULT_MAX_TOKENS) or DEFAULT_MAX_TOKENS)
        temperature = input.get("temperature")

        if not prompt or not isinstance(prompt, str):
            return {"ok": False, "error": "prompt is required (non-empty string)"}

        try:
            from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import LlamaCppManager

            conf_path = _resolve_conf_path()
            LlamaCppManager._instance = None  # noqa: SLF001
            manager = LlamaCppManager.get_instance(conf_path)

            srv = _resolve_slot(manager, slot_key)
            if srv is None:
                available = sorted(manager.servers.keys())
                return {
                    "ok": False,
                    "error": f"unknown slot: {slot_key!r}",
                    "available": available,
                }

            cfg = srv.config
            lmm_hosts = (manager.global_config or {}).get("lmm_hosts", {}) or {}
            role_val = getattr(cfg.role, "value", str(cfg.role))
            host_cfg = lmm_hosts.get(role_val, "host.docker.internal")
            host_only = host_cfg.split(":")[0] if ":" in host_cfg else host_cfg
            base_url = f"http://{host_only}:{cfg.port}"
        except Exception as exc:
            return {"ok": False, "error": f"slot resolution failed: {type(exc).__name__}: {exc}"}

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": role_val,  # llama-server accepts any alias here
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            try:
                payload["temperature"] = float(temperature)
            except (TypeError, ValueError):
                pass

        url = f"{base_url}/v1/chat/completions"
        started = time.monotonic()
        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"}) as resp:
                    text = await resp.text()
                    duration_ms = int((time.monotonic() - started) * 1000)
                    if resp.status >= 400:
                        return {
                            "ok": False,
                            "error": f"slot HTTP {resp.status}",
                            "body": text[:500],
                            "host": f"{host_only}:{cfg.port}",
                            "duration_ms": duration_ms,
                        }
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        return {
                            "ok": False,
                            "error": "non-JSON response from slot",
                            "body": text[:500],
                            "duration_ms": duration_ms,
                        }
        except aiohttp.ClientError as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "host": f"{host_only}:{cfg.port}",
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {
            "ok": True,
            "slot_id": next((k for k, v in manager.servers.items() if v is srv), slot_key),
            "slot_role": role_val,
            "host": f"{host_only}:{cfg.port}",
            "model_alias": data.get("model"),
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage") or {},
            "timings": data.get("timings") or {},
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
