"""
helpers/ephemeral_pool.py — Warm pool of ephemeral per-conversation containers.

Mental model:
    LLMs re-read ALL context tokens on every forward pass — stateless.
    Each terminal session is a separate, isolated LLM instance.
    ⇒ Each conversation gets its own dedicated container with an exact
      context window sized to the conversation's runtime token budget.

Architecture:
    Context size buckets  [8K, 16K, 32K, 64K, 128K] — snap required size up.
    Warm pool             N pre-loaded containers per bucket ready to assign.
    Fast path             Pop from pool → assign → replenish in background.
    Slow path             Spin up new container if pool is empty (~30-60s).
    Release               Return to pool (if room) or destroy.
    GC                    TTL-based cleanup of abandoned containers.

Port allocation:
    Ephemeral containers use ports 9100–9200 (configurable), isolated from
    the fixed shared slots (chat:8080, utility:8088, embedding:8082).
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from .context_calculator import bucket_context_size, CONTEXT_SIZE_BUCKETS

if TYPE_CHECKING:
    from .backends.docker_backend import DockerBackend

log = logging.getLogger("a0_lmm_router.ephemeral_pool")

EPHEMERAL_CONTAINER_PREFIX = "a0-lmm-conv-"
DEFAULT_PORT_RANGE_START = 9100
DEFAULT_PORT_RANGE_END = 9200


@dataclass
class EphemeralSlot:
    """A running container assigned to a specific conversation."""

    conv_id: str
    container_id: str
    port: int
    context_size: int
    model_id: str
    base_url: str
    started_at: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.started_at


@dataclass
class WarmContainer:
    """A pre-loaded container sitting idle in the warm pool."""

    container_id: str
    port: int
    context_size: int
    model_id: str
    base_url: str
    created_at: float = field(default_factory=time.time)


class EphemeralPool:
    """Manages a warm pool of pre-loaded ephemeral llama.cpp containers.

    One instance per router process, shared across all conversations.
    All state mutations go through asyncio.Lock for thread safety.
    """

    def __init__(
        self,
        docker_backend: "DockerBackend",
        model_path: str,
        model_id: str,
        gpu_layers: int = -1,
        warm_pool_size: int = 2,
        max_concurrent: int = 10,
        ttl_hours: float = 24.0,
        buckets: Optional[List[int]] = None,
        port_range_start: int = DEFAULT_PORT_RANGE_START,
        port_range_end: int = DEFAULT_PORT_RANGE_END,
    ):
        self._backend = docker_backend
        self._model_path = model_path
        self._model_id = model_id
        self._gpu_layers = gpu_layers
        self._warm_pool_size = warm_pool_size
        self._max_concurrent = max_concurrent
        self._ttl_seconds = ttl_hours * 3600
        self._buckets = sorted(buckets or CONTEXT_SIZE_BUCKETS)
        self._port_start = port_range_start
        self._port_end = port_range_end

        self._lock = asyncio.Lock()
        self._warm: Dict[int, List[WarmContainer]] = defaultdict(list)
        self._active: Dict[str, EphemeralSlot] = {}
        self._used_ports: Set[int] = set()

    # ── Public API ───────────────────────────────────────────────────

    async def acquire(self, conv_id: str, required_context: int) -> EphemeralSlot:
        """Acquire a container for conv_id sized to cover required_context tokens.

        Fast path (warm pool hit): returns immediately.
        Slow path (cold start):    waits ~30-60s for model to load.

        If conv_id already has an assigned container with sufficient context,
        returns it immediately. If the context is too small, upgrades it.
        """
        async with self._lock:
            existing = self._active.get(conv_id)
            if existing and existing.context_size >= required_context:
                return existing
            if existing:
                log.info(
                    f"[pool] {conv_id}: upgrade "
                    f"{existing.context_size}→{required_context}"
                )
                await self._release_slot(existing)

            target = bucket_context_size(required_context, self._buckets)

            if self._warm[target]:
                warm = self._warm[target].pop()
                slot = EphemeralSlot(
                    conv_id=conv_id,
                    container_id=warm.container_id,
                    port=warm.port,
                    context_size=warm.context_size,
                    model_id=warm.model_id,
                    base_url=warm.base_url,
                )
                self._active[conv_id] = slot
                self._used_ports.add(warm.port)
                log.info(
                    f"[pool] {conv_id}: warm-pool hit "
                    f"ctx={target} port={warm.port}"
                )
                asyncio.create_task(self._replenish(target))
                return slot

            # Cold start
            port = self._next_free_port()
            if port is None:
                raise RuntimeError(
                    f"[pool] No free ports in range "
                    f"{self._port_start}–{self._port_end}"
                )

        log.info(f"[pool] {conv_id}: cold start ctx={target} port={port}")
        container_id = await self._start_container(
            conv_id=conv_id, context_size=target, port=port
        )

        slot = EphemeralSlot(
            conv_id=conv_id,
            container_id=container_id,
            port=port,
            context_size=target,
            model_id=self._model_id,
            base_url=f"http://localhost:{port}",
        )
        async with self._lock:
            self._active[conv_id] = slot
            self._used_ports.add(port)
        return slot

    async def release(self, conv_id: str) -> None:
        """Release a conversation's container: return to pool or destroy."""
        async with self._lock:
            slot = self._active.pop(conv_id, None)
            if slot:
                await self._release_slot(slot)

    async def cleanup_stale(self, ttl_seconds: Optional[float] = None) -> int:
        """Destroy containers idle longer than ttl. Returns count destroyed."""
        ttl = ttl_seconds or self._ttl_seconds
        now = time.time()
        destroyed = 0

        async with self._lock:
            for bucket, pool in list(self._warm.items()):
                stale = [w for w in pool if (now - w.created_at) > ttl]
                for w in stale:
                    pool.remove(w)
                    self._used_ports.discard(w.port)
                    asyncio.create_task(self._destroy_by_name(
                        f"{EPHEMERAL_CONTAINER_PREFIX}warm-{w.port}"
                    ))
                    destroyed += 1

            stale_convs = [
                cid for cid, s in self._active.items()
                if s.age_seconds > ttl
            ]
            for cid in stale_convs:
                slot = self._active.pop(cid)
                self._used_ports.discard(slot.port)
                asyncio.create_task(
                    self._destroy_by_name(
                        f"{EPHEMERAL_CONTAINER_PREFIX}{cid[:8]}"
                    )
                )
                destroyed += 1

        if destroyed:
            log.info(f"[pool] cleaned {destroyed} stale ephemeral container(s)")
        return destroyed

    def status(self) -> dict:
        """Non-locking status snapshot for dashboards."""
        return {
            "active_conversations": len(self._active),
            "warm_pool": {b: len(p) for b, p in self._warm.items() if p},
            "used_ports": sorted(self._used_ports),
        }

    # ── Private helpers ──────────────────────────────────────────────

    async def _release_slot(self, slot: EphemeralSlot) -> None:
        """Return slot to pool or destroy it. Must be called with lock held."""
        pool = self._warm[slot.context_size]
        if len(pool) < self._warm_pool_size:
            pool.append(WarmContainer(
                container_id=slot.container_id,
                port=slot.port,
                context_size=slot.context_size,
                model_id=slot.model_id,
                base_url=slot.base_url,
            ))
            log.info(
                f"[pool] conv {slot.conv_id}: returned to pool "
                f"ctx={slot.context_size} pool_size={len(pool)}"
            )
        else:
            self._used_ports.discard(slot.port)
            asyncio.create_task(
                self._destroy_by_name(
                    f"{EPHEMERAL_CONTAINER_PREFIX}{slot.conv_id[:8]}"
                )
            )

    async def _replenish(self, bucket: int) -> None:
        """Spawn a replacement warm container for the given bucket."""
        async with self._lock:
            if len(self._warm[bucket]) >= self._warm_pool_size:
                return
            port = self._next_free_port()
            if port is None:
                log.warning(f"[pool] replenish: no free port for bucket={bucket}")
                return
            self._used_ports.add(port)

        try:
            container_id = await self._start_container(
                conv_id=f"warm-{port}", context_size=bucket, port=port
            )
            async with self._lock:
                self._warm[bucket].append(WarmContainer(
                    container_id=container_id,
                    port=port,
                    context_size=bucket,
                    model_id=self._model_id,
                    base_url=f"http://localhost:{port}",
                ))
            log.info(f"[pool] replenished warm slot ctx={bucket} port={port}")
        except Exception as exc:
            async with self._lock:
                self._used_ports.discard(port)
            log.error(f"[pool] replenish failed bucket={bucket}: {exc}")

    async def _start_container(
        self, conv_id: str, context_size: int, port: int
    ) -> str:
        """Start a new ephemeral container. Returns container_id."""
        name = f"{EPHEMERAL_CONTAINER_PREFIX}{conv_id[:8]}"
        config = {
            "port": port,
            "model_path": self._model_path,
            "model_id": self._model_id,
            "context_size": context_size,
            "gpu_layers": self._gpu_layers,
            "flash_attention": True,
        }
        status = await self._backend.start_ephemeral_slot(name, config)
        if not (status.running or status.healthy):
            raise RuntimeError(
                f"Ephemeral container {name} failed: {status.error}"
            )
        return status.container_id

    async def _destroy_by_name(self, container_name: str) -> None:
        try:
            await self._backend.stop_ephemeral_slot(container_name)
            log.info(f"[pool] destroyed {container_name}")
        except Exception as exc:
            log.error(f"[pool] destroy failed for {container_name}: {exc}")

    def _next_free_port(self) -> Optional[int]:
        """Find the next unused port in our reserved range. Lock must be held."""
        for port in range(self._port_start, self._port_end + 1):
            if port not in self._used_ports:
                return port
        return None
