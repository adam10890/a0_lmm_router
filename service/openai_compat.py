"""
OpenAI-compatible provider (Phase 7a: non-streaming forwarding).

GET  /v1/models            — list available slots as OpenAI model objects
POST /v1/chat/completions  — accept OpenAI request; resolve routing; forward
                             non-streaming request to selected llama.cpp slot

Forwarding invariants:
  - Only whitelisted OpenAI inference fields are forwarded.
  - metadata / privacy flags / routing hints are stripped before forwarding.
  - stream=true is rejected (400); streaming is not implemented.
  - No state is mutated.
  - All routing flows through RoutingIntentHandler.
  - Successful upstream responses are passed through without modification.
  - Errors include routing_decision for diagnostics.
"""
from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .routing_intent import (
    RoutingDecisionResponse,
    RoutingIntentHandler,
    RoutingIntentRequest,
)

# ---------------------------------------------------------------------------
# Fields forwarded to llama.cpp — explicit whitelist.
# Nothing outside this set reaches the model.
# ---------------------------------------------------------------------------

_FORWARDED_SCALAR_FIELDS = frozenset({
    "model", "temperature", "max_tokens", "top_p",
    "stop", "presence_penalty", "frequency_penalty", "seed",
})


# ---------------------------------------------------------------------------
# OpenAI request schema — minimum viable subset
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Any] = None   # str or list (multimodal)


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # unknown fields silently accepted

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
# HTTP post — injectable for testing
# ---------------------------------------------------------------------------

async def _aiohttp_post(
    url: str,
    payload: Dict[str, Any],
    timeout: int = 120,
) -> Tuple[int, Dict[str, Any]]:
    """POST JSON payload to url; return (status_code, response_dict)."""
    import aiohttp  # noqa: PLC0415 — lazy import keeps module importable without aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            try:
                body = await resp.json(content_type=None)
            except Exception:
                body = {
                    "error": {
                        "message": "Upstream returned a non-JSON response.",
                        "code": "invalid_upstream_response",
                    }
                }
            return resp.status, body


# ---------------------------------------------------------------------------
# Forward payload construction
# ---------------------------------------------------------------------------

def _build_forward_payload(req: OpenAIChatRequest) -> Dict[str, Any]:
    """
    Build the payload to send to llama.cpp.

    Only whitelisted inference fields are included.
    metadata, tools, tool_choice, routing hints, and unknown extra fields
    are deliberately excluded.
    """
    payload: Dict[str, Any] = {}

    for field in _FORWARDED_SCALAR_FIELDS:
        value = getattr(req, field, None)
        if value is not None:
            payload[field] = value

    # messages are serialised from pydantic objects; extra per-message
    # fields are included (some clients attach per-message metadata).
    payload["messages"] = [m.model_dump(exclude_none=True) for m in req.messages]

    # Phase 7a: always force non-streaming to the upstream.
    payload["stream"] = False

    return payload


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

    Priority: exact slot-id match → model_id field match → None.
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
    Forwarding-only fields (temperature, max_tokens, …) are stashed in
    intent.metadata for reference but are never used for routing decisions.
    """
    translation_warnings: List[str] = []
    meta = req.metadata

    preferred_slot = _resolve_preferred_slot(req.model, observer_slots)
    if req.model not in ("default", "", None) and preferred_slot is None:
        translation_warnings.append(
            f"unknown_model:{req.model} — model does not match any configured "
            "slot id or model_id; routing to default chain"
        )

    privacy_mode  = str(meta.get("privacy_mode", "unknown"))
    local_only    = bool(meta.get("local_only", False))
    cloud_allowed = bool(meta.get("cloud_allowed", True))
    agent_id      = str(meta.get("agent_id", "unknown"))
    agent_type    = str(meta.get("agent_type", "unknown"))
    task_type     = str(meta.get("task_type", "chat"))
    role          = meta.get("role") or None

    requires_tools = bool(req.tools)
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
        metadata={
            "openai_model":    req.model,
            "message_count":   len(req.messages),
            "max_tokens":      req.max_tokens,
            "temperature":     req.temperature,
            "stream_requested": req.stream,
            "tools_present":   bool(req.tools),
            "tool_choice":     req.tool_choice,
        },
    )
    return intent, translation_warnings


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

# Type alias for the injectable HTTP post function.
_PostFn = Callable[
    [str, Dict[str, Any], int],
    Coroutine[Any, Any, Tuple[int, Dict[str, Any]]],
]


class OpenAICompatHandler:
    """
    Implements GET /v1/models and POST /v1/chat/completions.

    Collaborates with ObserverBackend (slot list) and RoutingIntentHandler
    (routing decisions).

    _post_fn is injectable for testing: tests pass a stub coroutine that
    avoids real network calls.  Production uses _aiohttp_post.
    """

    def __init__(
        self,
        observer: Any,
        intent_handler: RoutingIntentHandler,
        _post_fn: Optional[_PostFn] = None,
    ) -> None:
        self._observer = observer
        self._intent_handler = intent_handler
        self._post_fn: _PostFn = _post_fn or _aiohttp_post

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

        Flow:
          stream=True             → 400 streaming_not_implemented
          no healthy slot         → 503 no_slot_available
          connection error        → 502 upstream_connection_error
          upstream non-2xx        → upstream status + body (pass-through)
          upstream 200            → 200 + body (pass-through, no wrapper)
        """
        # ── 1. Reject streaming ───────────────────────────────────────────
        if req.stream:
            return 400, {
                "object": "error",
                "error": {
                    "message": (
                        "Streaming is not implemented. Set stream=false."
                    ),
                    "type": "not_implemented",
                    "code": "streaming_not_implemented",
                },
            }

        # ── 2. Resolve routing decision ───────────────────────────────────
        slots = self._observer.get_slots()
        intent, translation_warnings = chat_request_to_routing_intent(req, slots)
        decision: RoutingDecisionResponse = await self._intent_handler.handle(intent)

        # ── 3. No slot available ──────────────────────────────────────────
        if decision.no_slot_available:
            return 503, {
                "object": "error",
                "error": {
                    "message": "No healthy local slot is available for this request.",
                    "type": "service_unavailable",
                    "code": "no_slot_available",
                },
                "routing_decision": decision.model_dump(),
                "translation_warnings": translation_warnings,
            }

        # ── 4. Build forwarding payload (whitelist only) ──────────────────
        payload = _build_forward_payload(req)
        # selected_url is already http://host:port/v1 — append endpoint only.
        target_url = f"{decision.selected_url}/chat/completions"

        # ── 5. Forward to slot ────────────────────────────────────────────
        try:
            upstream_status, upstream_body = await self._post_fn(target_url, payload)
        except Exception as exc:
            # Connection-level failure (refused, timeout, DNS, …).
            # Do not leak the full exception; include safe error category only.
            err_type = type(exc).__name__
            return 502, {
                "object": "error",
                "error": {
                    "message": "Connection to the selected slot failed.",
                    "type": "bad_gateway",
                    "code": "upstream_connection_error",
                    "slot_id": decision.selected_slot_id,
                    "error_type": err_type,
                },
                "routing_decision": decision.model_dump(),
                "translation_warnings": translation_warnings,
                "provider_shell": {
                    "phase": 7,
                    "forwarding": True,
                    "streaming": False,
                    "note": "Forwarding was attempted but the upstream connection failed.",
                },
            }

        # ── 6. Pass upstream response through ─────────────────────────────
        # For successful responses: return body as-is (OpenAI-compatible shape).
        # For upstream errors: return their status and body without modification.
        return upstream_status, upstream_body
