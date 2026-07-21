"""Agent tool: delegate coding tasks to pi (local llama.cpp utility slot).

Usage from the agent:
    {
      "thoughts": ["I need pi to implement this change"],
      "tool_name": "pi_coding",
      "tool_args": {
        "prompt": "Add error handling to helpers/pi_runner.py",
        "working_directory": "C:/Users/frant/lmm-router",
        "read_only": false
      }
    }

Runs pi in print mode against the lmm-coding provider (utility slot on :8088).
Chat completions remain available via chat_completion / the chat slot.
"""
from __future__ import annotations

from helpers.tool import Tool, Response

try:
    from usr.plugins.a0_lmm_router.helpers.pi_runner import (
        CHAT_MODEL,
        DEFAULT_MODEL,
        run_pi_coding,
    )
except ImportError:
    from helpers.pi_runner import CHAT_MODEL, DEFAULT_MODEL, run_pi_coding


class PiCoding(Tool):
    """Run the pi coding agent against the local LMM Router utility slot."""

    async def execute(self, **kwargs) -> Response:
        prompt = str(kwargs.get("prompt", "")).strip()
        if not prompt:
            return Response(
                message="`prompt` is required — describe the coding task for pi.",
                break_loop=False,
            )

        role = str(kwargs.get("role", "coding")).strip().lower()
        model = str(kwargs.get("model", "")).strip()
        if not model:
            model = CHAT_MODEL if role == "chat" else DEFAULT_MODEL

        working_directory = kwargs.get("working_directory") or kwargs.get("cwd")
        read_only = bool(kwargs.get("read_only", False))
        timeout = int(kwargs.get("timeout", 600))

        result = await run_pi_coding(
            prompt,
            working_directory=working_directory,
            model=model,
            timeout=timeout,
            read_only=read_only,
        )

        lines = [
            f"## pi coding — {'OK' if result.get('ok') else 'FAILED'}",
            f"- **model:** `{result.get('model', model)}`",
            f"- **backend:** {result.get('backend', 'unknown')}",
        ]
        if result.get("working_directory"):
            lines.append(f"- **cwd:** `{result['working_directory']}`")
        if result.get("error"):
            lines.append(f"\n**error:** {result['error']}")
        if result.get("stderr"):
            lines.append(f"\n### stderr\n```\n{str(result['stderr'])[:4000]}\n```")
        output = str(result.get("output", "") or "").strip()
        if output:
            lines.append(f"\n### output\n{output[:12000]}")
        elif not result.get("ok") and not result.get("error"):
            lines.append("\nNo output returned from pi.")

        return Response(message="\n".join(lines), break_loop=False)
