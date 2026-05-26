"""
Slot health probing, isolated from BackendManager routing logic.

SlotHealthChecker accepts an injectable probe_fn so callers (tests, future
async adapters) can replace the network layer without touching routing logic.

Default probe: synchronous urllib GET against /health (same as the original
inline implementation in BackendManager._get_slot_health).

Future async path:
    checker = SlotHealthChecker()
    # Phase 3: add check_async(slot_config) that awaits an aiohttp probe_fn
    # without changing select_slot_with_failover or the routing chain logic.
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger("lmm_router.health")

HEALTHY = "healthy"
UNHEALTHY = "unhealthy"
UNKNOWN = "unknown"


def _urllib_probe(url: str, timeout: int) -> Dict:
    """Default sync probe: GET /health, return {"ok": bool}.

    Isolated here so it can be swapped without touching SlotHealthChecker.
    """
    import urllib.request  # noqa: PLC0415

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                return {"ok": data.get("status") == "ok", "http_status": resp.status}
            return {"ok": False, "http_status": resp.status}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


class SlotHealthChecker:
    """Probe a single slot's /health endpoint and return a health string.

    Parameters
    ----------
    timeout:
        Seconds to wait for a probe response.
    probe_fn:
        Callable(url: str, timeout: int) -> {"ok": bool, ...}.
        Defaults to the stdlib urllib probe.  Inject a stub in tests.
    """

    def __init__(
        self,
        timeout: int = 2,
        probe_fn: Optional[Callable[[str, int], Dict]] = None,
    ) -> None:
        self.timeout = timeout
        self._probe = probe_fn or _urllib_probe

    def check(self, slot_config: Dict) -> str:
        """Return HEALTHY, UNHEALTHY, or UNKNOWN for the given slot config.

        slot_config must contain at least 'host' and 'port' keys.
        Missing / falsy port → UNKNOWN (cannot construct a valid URL).
        """
        host = slot_config.get("host", "localhost")
        port = slot_config.get("port")
        if not port:
            return UNKNOWN

        url = f"http://{host}:{port}/health"
        try:
            result = self._probe(url, self.timeout)
            return HEALTHY if result.get("ok") else UNHEALTHY
        except Exception:
            logger.debug("Health probe raised unexpectedly for %s:%s", host, port)
            return UNHEALTHY
