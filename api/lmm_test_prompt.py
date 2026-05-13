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
import yaml
from flask import Request

from helpers.api import ApiHandler


DEFAULT_MAX_TOKENS = 512
REQUEST_TIMEOUT_SEC = 120


def _resolve_conf_path() -> str:
    """Same self-contained resolver as llamacpp_status — kept local to avoid
    depending on `helpers.files` (which imports the fragile simpleeval)."""
    env_path = os.environ.get("A0_LMM_ROUTER_CONFIG", "").strip()
    if env_path:
        return env_path
    here = Path(__file__).resolve()
    plugin_conf = str(here.parents[1] / "conf" / "llama_cpp_servers.yaml")
    root_conf = str(here.parents[4] / "conf" / "llama_cpp_servers.yaml")
    return plugin_conf if os.path.exists(plugin_conf) else root_conf


def _resolve_slot(config: dict, slot_key: str):
    if not slot_key:
        return None, None
    active_slots = config.get("active_slots", []) or []
    for slot in active_slots:
        if not slot or not slot.get("enabled", True):
            continue
        slot_id = slot.get("id", f"slot_{slot.get('port', 'unknown')}")
        if slot_id == slot_key:
            return slot_id, slot
    for slot in active_slots:
        if not slot or not slot.get("enabled", True):
            continue
        slot_id = slot.get("id", f"slot_{slot.get('port', 'unknown')}")
        if str(slot.get("role", "")) == slot_key:
            return slot_id, slot
    return None, None


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
            conf_path = _resolve_conf_path()
            with open(conf_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            slot_id, cfg = _resolve_slot(config, slot_key)
            if cfg is None:
                available = sorted(
                    slot.get("id", f"slot_{slot.get('port', 'unknown')}")
                    for slot in config.get("active_slots", []) or []
                    if slot and slot.get("enabled", True)
                )
                return {
                    "ok": False,
                    "error": f"unknown slot: {slot_key!r}",
                    "available": available,
                }

            global_config = config.get("global", {}) or {}
            lmm_hosts = global_config.get("lmm_hosts", {}) or {}
            role_val = str(cfg.get("role", ""))
            port = int(cfg.get("port", 0) or 0)
            host_cfg = str(lmm_hosts.get(role_val, f"host.docker.internal:{port}"))
            host_cfg = host_cfg.replace("http://", "").replace("https://", "").split("/", 1)[0]
            host_only = host_cfg
            if ":" in host_cfg:
                host_only, port_text = host_cfg.rsplit(":", 1)
                if port_text.isdigit():
                    port = int(port_text)
            base_url = f"http://{host_only}:{port}"
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
                            "host": f"{host_only}:{port}",
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
                "host": f"{host_only}:{port}",
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {
            "ok": True,
            "slot_id": slot_id,
            "slot_role": role_val,
            "host": f"{host_only}:{port}",
            "model_alias": data.get("model"),
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage") or {},
            "timings": data.get("timings") or {},
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
