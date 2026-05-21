"""
helpers/conversation_tracker.py — Per-conversation token budget tracker.

Tracks token accumulation across all sources for a conversation:
  - pen_paper workspace reads
  - wiki_query results
  - rolling chat history
  - system prompt

Fires expansion callbacks when the accumulated budget exceeds the assigned
context window, enabling the ephemeral pool to upgrade the container size.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .context_calculator import (
    ExternalTokenBudget,
    recommend_context_for_budget,
)

log = logging.getLogger("a0_lmm_router.conversation_tracker")

_DEFAULT_TTL_HOURS = 24


@dataclass
class ConversationState:
    """Token budget state for a single conversation."""

    conv_id: str
    budget: ExternalTokenBudget = field(default_factory=ExternalTokenBudget)
    assigned_context_size: int = 0
    model_n_ctx_train: int = 131072
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    expansion_callbacks: List[Callable] = field(default_factory=list, repr=False)

    def is_overflow(self) -> bool:
        return (
            self.assigned_context_size > 0
            and self.budget.total > self.assigned_context_size
        )

    def required_context(self) -> int:
        return recommend_context_for_budget(self.budget, self.model_n_ctx_train)


class ConversationTracker:
    """Thread-safe per-conversation token budget manager.

    Single shared instance per router process. Conversations are garbage-
    collected after TTL hours of inactivity.
    """

    def __init__(self, ttl_hours: float = _DEFAULT_TTL_HOURS):
        self._states: Dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()
        self._ttl_seconds = ttl_hours * 3600

    # ── Public update API ────────────────────────────────────────────

    async def update_from_pen_paper(self, conv_id: str, token_count: int) -> None:
        """Record tokens retrieved from a pen_paper workspace read."""
        async with self._lock:
            state = self._get_or_create(conv_id)
            state.budget.pen_paper += token_count
            log.debug(
                f"[tracker] {conv_id}: pen_paper +{token_count} "
                f"→ total={state.budget.total}"
            )
            await self._check_overflow(state)

    async def update_from_wiki(self, conv_id: str, token_count: int) -> None:
        """Record tokens retrieved from a wiki_query result."""
        async with self._lock:
            state = self._get_or_create(conv_id)
            state.budget.wiki += token_count
            log.debug(
                f"[tracker] {conv_id}: wiki +{token_count} "
                f"→ total={state.budget.total}"
            )
            await self._check_overflow(state)

    async def update_history(self, conv_id: str, token_count: int) -> None:
        """Accumulate chat history tokens (call on each new message)."""
        async with self._lock:
            state = self._get_or_create(conv_id)
            state.budget.history += token_count
            log.debug(
                f"[tracker] {conv_id}: history +{token_count} "
                f"→ total={state.budget.total}"
            )
            await self._check_overflow(state)

    async def set_system_tokens(self, conv_id: str, token_count: int) -> None:
        """Set (replace, not accumulate) the system prompt token count."""
        async with self._lock:
            state = self._get_or_create(conv_id)
            state.budget.system = token_count

    # ── Slot assignment ──────────────────────────────────────────────

    async def assign_context_window(
        self,
        conv_id: str,
        context_size: int,
        model_n_ctx_train: int = 131072,
    ) -> None:
        """Record which context window was assigned to this conversation."""
        async with self._lock:
            state = self._get_or_create(conv_id)
            state.assigned_context_size = context_size
            state.model_n_ctx_train = model_n_ctx_train
            log.info(
                f"[tracker] {conv_id}: assigned ctx={context_size} "
                f"(model_max={model_n_ctx_train})"
            )

    async def register_expansion_callback(
        self,
        conv_id: str,
        callback: Callable[[str, int], None],
    ) -> None:
        """Register a callback fired when the budget overflows the context window.

        callback(conv_id, required_context_size)
        """
        async with self._lock:
            state = self._get_or_create(conv_id)
            state.expansion_callbacks.append(callback)

    # ── Query API ────────────────────────────────────────────────────

    async def get_budget(self, conv_id: str) -> Optional[ExternalTokenBudget]:
        async with self._lock:
            state = self._states.get(conv_id)
            return state.budget if state else None

    async def get_required_context(
        self,
        conv_id: str,
        model_n_ctx_train: int = 131072,
    ) -> int:
        """Return the bucketed context size needed for this conversation."""
        async with self._lock:
            state = self._states.get(conv_id)
            if not state:
                return recommend_context_for_budget(
                    ExternalTokenBudget(), model_n_ctx_train
                )
            state.model_n_ctx_train = model_n_ctx_train
            return state.required_context()

    # ── Lifecycle ────────────────────────────────────────────────────

    async def close_conversation(self, conv_id: str) -> None:
        """Remove a conversation and log its final budget."""
        async with self._lock:
            state = self._states.pop(conv_id, None)
            if state:
                b = state.budget
                log.info(
                    f"[tracker] closed {conv_id} — "
                    f"pp={b.pen_paper} wiki={b.wiki} "
                    f"hist={b.history} sys={b.system} total={b.total}"
                )

    async def purge_stale(self) -> int:
        """Remove conversations idle longer than TTL. Returns count removed."""
        now = time.time()
        async with self._lock:
            stale = [
                cid
                for cid, s in self._states.items()
                if (now - s.last_updated) > self._ttl_seconds
            ]
            for cid in stale:
                del self._states[cid]
        if stale:
            log.info(f"[tracker] purged {len(stale)} stale conversation(s)")
        return len(stale)

    def get_all_summaries(self) -> Dict[str, dict]:
        """Snapshot of all live conversation budgets (no lock — informational)."""
        return {
            cid: {
                "pen_paper": s.budget.pen_paper,
                "wiki": s.budget.wiki,
                "history": s.budget.history,
                "system": s.budget.system,
                "total": s.budget.total,
                "assigned_ctx": s.assigned_context_size,
                "overflow": s.is_overflow(),
                "required_ctx": s.required_context(),
            }
            for cid, s in self._states.items()
        }

    # ── Private helpers ──────────────────────────────────────────────

    def _get_or_create(self, conv_id: str) -> ConversationState:
        """Must be called with self._lock held."""
        if conv_id not in self._states:
            self._states[conv_id] = ConversationState(conv_id=conv_id)
            log.debug(f"[tracker] new conversation {conv_id}")
        state = self._states[conv_id]
        state.last_updated = time.time()
        return state

    async def _check_overflow(self, state: ConversationState) -> None:
        """Fire expansion callbacks if tokens exceed the assigned window.
        Must be called with self._lock held.
        """
        if not state.is_overflow():
            return
        required = state.required_context()
        log.warning(
            f"[tracker] context overflow in {state.conv_id}: "
            f"tokens={state.budget.total} > window={state.assigned_context_size}, "
            f"need={required}"
        )
        for cb in state.expansion_callbacks:
            try:
                cb(state.conv_id, required)
            except Exception as exc:
                log.error(f"[tracker] expansion callback error: {exc}")
