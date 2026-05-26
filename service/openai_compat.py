"""
OpenAI-compatible provider contract shell (Phase 6).

GET  /v1/models            — list available slots as OpenAI model objects
POST /v1/chat/completions  — accept OpenAI-style request; resolve routing
                             decision; return 501 (no inference forwarding yet)

Invariants:
  - No prompts are forwarded to any model.
  - No fake completions are returned.
  - No streaming is implemented.
  - No state is mutated.
  - All routing decisions flow through RoutingIntentHandler.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .routing_intent import (
    RoutingDecisionResponse,
    RoutingIntentHandler,
    RoutingIntentRequest,
)

# ---------------------------------------------------------------------------
# OpenAI request schema — minimum viable subset
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Any] = None      # str or list (multimodal); stored but not forwarded


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # unknown fields accepted silently

    model: str = "default"
    messages: List[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# OpenAI model-list response schema
# ---------------------------------------------------------------------------

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelObject]


# ---------------------------------------------------------------------------
# Translation: OpenAI request → RoutingIntentRequest
# ---------------------------------------------------------------------------

def _estimate_tokens(messages: List[ChatMessage]) -> int:
    """Rough token estimate: total content chars / 4, minimum 1."""
    total = sum(
        len(m.content) if isinstance(m.content, str) else 0
        for m in messages
    )
    return max(1, total // 4)


def _resolve_preferred_slot(
    model: str,
    observer_slots: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Map an OpenAI model name to a slot id.

    Priority: exact slot-id match → then model_id field match → None.
    """
    if not model or model == "default":
        return None
    slot_ids = {s["id"] for s in observer_slots}
    if model in slot_ids:
        return model
    for slot in observer_slots:
        if slot.get("model_id") and slot["model_id"] == model:
            return slot["id"]
    return None


def chat_request_to_routing_intent(
    req: OpenAIChatRequest,
    observer_slots: List[Dict[str, Any]],
) -> Tuple[RoutingIntentRequest, List[str]]:
    """
    Translate an OpenAI-style chat request into a RoutingIntentRequest.

    Returns (routing_intent, translation_warnings).
    Fields unknown to RoutingIntentRequest are stashed in intent.metadata
    so they survive the pipeline without loss.
    """
    translation_warnings: List[str] = []
    meta = req.metadata

    # ── model → preferred_slot ────────────────────────────────────────────
    preferred_slot = _resolve_preferred_slot(req.model, observer_slots)
    if req.model not in ("default", "", None) and preferred_slot is None:
        translation_warnings.append(
            f"unknown_model:{req.model} — model does not match any configured "
            "slot id or model_id; routing to default chain"
        )

    # ── pull routing hints from metadata ─────────────────────────────────
    privacy_mode    = str(meta.get("privacy_mode", "unknown"))
    local_only      = bool(meta.get("local_only", False))
    cloud_allowed   = bool(meta.get("cloud_allowed", True))
    agent_id        = str(meta.get("agent_id", "unknown"))
    agent_type      = str(meta.get("agent_type", "unknown"))
    task_type       = str(meta.get("task_type", "chat"))
    role            = meta.get("role") or None   # explicit role override

    # ── capability flags ──────────────────────────────────────────────────
    requires_tools = bool(req.tools)

    # ── token estimate ───────────────────────────────────────────────────
    estimated_tokens = _estimate_tokens(req.messages) if req.messages else None

    intent = RoutingIntentRequest(
        agent_id=agent_id,
        agent_type=agent_type,
        role=role,
        task_type=task_type,
        privacy_mode=privacy_mode,
        local_only=local_only,
        cloud_allowed=cloud_allowed,
        requires_tools=requires_tools,
        estimated_tokens=estimated_tokens,
        preferred_slot=preferred_slot,
        # Stash forwarding-only fields for downstream reference.
        metadata={
            "openai_model": req.model,
            "message_count": len(req.messages),
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream_requested": req.stream,
            "tools_present": bool(req.tools),
            "tool_choice": req.tool_choice,
        },
    )
    return intent, translation_warnings


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class OpenAICompatHandler:
    """
    Implements GET /v1/models and POST /v1/chat/completions.

    Collaborates with ObserverBackend (slot list) and RoutingIntentHandler
    (routing decisions).  Never touches BackendManager start/stop paths.
    """

    def __init__(self, observer: Any, intent_handler: RoutingIntentHandler) -> None:
        self._observer = observer
        self._intent_handler = intent_handler

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------

    def get_models(self) -> ModelList:
        """Return configured slots as an OpenAI-style model list."""
        data = [
            ModelObject(
                id=slot["id"],
                owned_by="local",
                metadata={
                    "slot_id":      slot["id"],
                    "role":         slot.get("role"),
                    "model_id":     slot.get("model_id"),
                    "backend_type": slot.get("backend_type"),
                    "enabled":      slot.get("enabled", True),
                    "base_url":     slot.get("base_url"),
                    "context_size": slot.get("context_size"),
                },
            )
            for slot in self._observer.get_slots()
        ]
        return ModelList(data=data)

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------

    async def handle_chat_completion(
        self, req: OpenAIChatRequest
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Process an OpenAI-style chat request.

        Returns (http_status_code, response_body).

        Contract:
          - stream=True  → 400  streaming_not_implemented
          - otherwise    → 501  forwarding_not_implemented + routing_decision
          - Never returns fake inference output.
          - Never forwards the prompt to any model.
        """
        if req.stream:
            return 400, {
                "object": "error",
                "error": {
                    "message": (
                        "Streaming is not implemented in the Phase 6 provider shell. "
                        "Set stream=false."
                    ),
                    "type": "not_implemented",
                    "code": "streaming_not_implemented",
                },
            }

        slots = self._observer.get_slots()
        intent, translation_warnings = chat_request_to_routing_intent(req, slots)
        decision: RoutingDecisionResponse = await self._intent_handler.handle(intent)

        return 501, {
            "object": "error",
            "error": {
                "message": (
                    "Inference forwarding is not implemented in the Phase 6 provider shell. "
                    "Inspect routing_decision to see which slot would serve this request."
                ),
                "type": "not_implemented",
                "code": "forwarding_not_implemented",
            },
            "routing_decision": decision.model_dump(),
            "translation_warnings": translation_warnings,
            "provider_shell": {
                "phase": 6,
                "forwarding": False,
                "streaming": False,
                "note": (
                    "This endpoint resolves routing decisions for OpenAI-compatible "
                    "requests but does not forward prompts to any model."
                ),
            },
        }
