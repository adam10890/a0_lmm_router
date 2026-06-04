"""Local Fleet output token budget enforcement."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


TELEMETRY_KEY = "a0_lmm_router_output_budget"

DEFAULT_LIMITS: dict[str, dict[str, int]] = {
    "chat": {"default_max_tokens": 2048, "hard_max_tokens": 4096},
    "utility": {"default_max_tokens": 768, "hard_max_tokens": 1024},
}

OUTPUT_KEYS = ("max_tokens", "max_completion_tokens")


@dataclass(frozen=True)
class OutputBudgetDecision:
    role: str
    enabled: bool
    applied: bool
    key: str
    requested_tokens: int | None
    final_tokens: int | None
    default_max_tokens: int
    hard_max_tokens: int
    reason: str

    def telemetry(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "enabled": self.enabled,
            "applied": self.applied,
            "key": self.key,
            "requested_tokens": self.requested_tokens,
            "final_tokens": self.final_tokens,
            "default_max_tokens": self.default_max_tokens,
            "hard_max_tokens": self.hard_max_tokens,
            "reason": self.reason,
        }


def output_budget_enabled(config: Mapping[str, Any] | None) -> bool:
    budget = _budget_config(config)
    if not _truthy(budget.get("enabled", True)):
        return False
    env = os.getenv("A0_LMM_OUTPUT_BUDGET_ENABLED", "").strip().lower()
    if env:
        return env not in {"0", "false", "no", "off"}
    return True


def apply_output_budget(
    kwargs: Mapping[str, Any] | None,
    *,
    role: str,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], OutputBudgetDecision]:
    """Return kwargs with bounded max output tokens for one Local Fleet call."""
    role_key = normalize_role(role)
    values = dict(kwargs or {})
    limits = limits_for_role(role_key, config)
    enabled = output_budget_enabled(config)
    if not enabled:
        return values, OutputBudgetDecision(
            role=role_key,
            enabled=False,
            applied=False,
            key="",
            requested_tokens=None,
            final_tokens=None,
            default_max_tokens=limits["default_max_tokens"],
            hard_max_tokens=limits["hard_max_tokens"],
            reason="disabled",
        )

    selected_key = _existing_output_key(values)
    if selected_key is None:
        values["max_tokens"] = limits["default_max_tokens"]
        return values, OutputBudgetDecision(
            role=role_key,
            enabled=True,
            applied=True,
            key="max_tokens",
            requested_tokens=None,
            final_tokens=limits["default_max_tokens"],
            default_max_tokens=limits["default_max_tokens"],
            hard_max_tokens=limits["hard_max_tokens"],
            reason="default_injected",
        )

    requested = _positive_int(values.get(selected_key))
    if requested is None:
        values[selected_key] = limits["default_max_tokens"]
        return values, OutputBudgetDecision(
            role=role_key,
            enabled=True,
            applied=True,
            key=selected_key,
            requested_tokens=None,
            final_tokens=limits["default_max_tokens"],
            default_max_tokens=limits["default_max_tokens"],
            hard_max_tokens=limits["hard_max_tokens"],
            reason="invalid_replaced",
        )

    hard = limits["hard_max_tokens"]
    if requested > hard:
        values[selected_key] = hard
        return values, OutputBudgetDecision(
            role=role_key,
            enabled=True,
            applied=True,
            key=selected_key,
            requested_tokens=requested,
            final_tokens=hard,
            default_max_tokens=limits["default_max_tokens"],
            hard_max_tokens=hard,
            reason="hard_cap_applied",
        )

    return values, OutputBudgetDecision(
        role=role_key,
        enabled=True,
        applied=False,
        key=selected_key,
        requested_tokens=requested,
        final_tokens=requested,
        default_max_tokens=limits["default_max_tokens"],
        hard_max_tokens=hard,
        reason="within_limit",
    )


def should_apply_to_model(model: Any) -> bool:
    """Best-effort detection of Agent Zero's Local Fleet model wrapper."""
    cfg = getattr(model, "a0_model_conf", None)
    provider = str(getattr(cfg, "provider", "") or "").lower()
    if provider in {"lmm_router", "local_fleet"}:
        return True
    model_name = str(getattr(model, "model_name", "") or "").lower()
    if model_name.startswith(("lmm_router/", "local_fleet/")):
        return True
    api_base = str(getattr(cfg, "api_base", "") or "").lower()
    return "host.docker.internal:8080" in api_base or "localhost:8080" in api_base


def limits_for_role(role: str, config: Mapping[str, Any] | None = None) -> dict[str, int]:
    role_key = normalize_role(role)
    defaults = DEFAULT_LIMITS.get(role_key, DEFAULT_LIMITS["chat"])
    budget = _budget_config(config)
    role_cfg = budget.get(role_key)
    if not isinstance(role_cfg, Mapping):
        role_cfg = {}
    default_max = _env_int(
        (
            f"A0_LMM_OUTPUT_DEFAULT_MAX_TOKENS_{role_key.upper()}",
            f"A0_LMM_{role_key.upper()}_MAX_TOKENS",
        ),
        _int_from_config(role_cfg, "default_max_tokens", defaults["default_max_tokens"]),
    )
    hard_max = _env_int(
        (
            f"A0_LMM_OUTPUT_HARD_MAX_TOKENS_{role_key.upper()}",
            f"A0_LMM_{role_key.upper()}_HARD_MAX_TOKENS",
        ),
        _int_from_config(role_cfg, "hard_max_tokens", defaults["hard_max_tokens"]),
    )
    hard_max = max(1, hard_max)
    default_max = min(max(1, default_max), hard_max)
    return {
        "default_max_tokens": default_max,
        "hard_max_tokens": hard_max,
    }


def normalize_role(role: str) -> str:
    value = str(role or "chat").strip().lower()
    if value in {"util", "utility_model"}:
        return "utility"
    if value not in DEFAULT_LIMITS:
        return "chat"
    return value


def _budget_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    budget = config.get("output_budget")
    return budget if isinstance(budget, Mapping) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _existing_output_key(values: Mapping[str, Any]) -> str | None:
    for key in OUTPUT_KEYS:
        if key in values:
            return key
    return None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _int_from_config(config: Mapping[str, Any], key: str, default: int) -> int:
    parsed = _positive_int(config.get(key))
    return parsed if parsed is not None else default


def _env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        parsed = _positive_int(raw)
        if parsed is not None:
            return parsed
    return default
