"""
Usage Ledger for LMM Router — per-provider token/request accounting.

Append-only JSONL, one file per writer process (data/usage_a0.jsonl for the
Agent Zero process, data/usage_mcp.jsonl for the MCP server process) so the
two processes never contend on the same file — no cross-process locking.
Reads merge every data/usage_*.jsonl file, cached by (mtime, size).

Pattern-matched to stats_tracker.py (module lock, stdlib only).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger("lmm_router.usage")

_lock = Lock()

# "a0" in the Agent Zero process; mcp_server/server.py sets this to "mcp".
PROC_TAG = "a0"

# Overridable in tests.
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"

MAX_AGE_DAYS = 35

# path -> ((mtime, size), events) — size in the key because Windows mtime is coarse
_cache: Dict[str, tuple] = {}
_last_prune_day: Optional[str] = None


def _own_path() -> Path:
    return DATA_DIR / f"usage_{PROC_TAG}.jsonl"


def record_usage(
    provider_id: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    requests: int = 1,
    model: str = "",
    source: str = "",
    requester: str = "",
) -> None:
    """Append one usage event to this process's ledger file."""
    event = {
        "ts": time.time(),
        "provider_id": provider_id,
        "model": model,
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "requests": int(requests),
        "source": source,
        "requester": requester,
    }
    with _lock:
        path = _own_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()
    _maybe_prune()


def _load_file(path: Path) -> List[dict]:
    """Parse one ledger file, dropping events older than MAX_AGE_DAYS."""
    try:
        stat = path.stat()
    except OSError:
        return []
    key = (stat.st_mtime, stat.st_size)
    cached = _cache.get(str(path))
    if cached and cached[0] == key:
        return cached[1]

    cutoff = time.time() - MAX_AGE_DAYS * 86400
    events: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("ts", 0) >= cutoff:
                    events.append(ev)
    except OSError:
        return []
    _cache[str(path)] = (key, events)
    return events


def all_events(since_ts: float = 0.0) -> List[dict]:
    """Merged events from every process's ledger file, newest window first not guaranteed."""
    if not DATA_DIR.is_dir():
        return []
    events: List[dict] = []
    for path in sorted(DATA_DIR.glob("usage_*.jsonl")):
        events.extend(e for e in _load_file(path) if e.get("ts", 0) >= since_ts)
    return events


def window_totals(provider_id: str, window_seconds: int, now: Optional[float] = None) -> Dict[str, int]:
    """Rolling-window sums for one provider: {"tokens": n, "requests": n}."""
    now = now if now is not None else time.time()
    tokens = requests = 0
    for ev in all_events(since_ts=now - window_seconds):
        if ev.get("provider_id") != provider_id or ev.get("ts", 0) > now:
            continue
        tokens += int(ev.get("tokens_in", 0)) + int(ev.get("tokens_out", 0))
        requests += int(ev.get("requests", 0))
    return {"tokens": tokens, "requests": requests}


def prune(max_age_days: int = MAX_AGE_DAYS) -> None:
    """Rewrite ONLY this process's file, dropping events older than max_age_days."""
    path = _own_path()
    if not path.exists():
        return
    cutoff = time.time() - max_age_days * 86400
    with _lock:
        keep: List[str] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line).get("ts", 0) >= cutoff:
                        keep.append(line)
                except ValueError:
                    continue
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("ledger prune failed: %s", e)


def _maybe_prune() -> None:
    """Opportunistic prune, at most once per process per day. Called outside _lock."""
    global _last_prune_day
    day = time.strftime("%Y-%m-%d")
    if day == _last_prune_day:
        return
    _last_prune_day = day
    try:
        prune()
    except Exception as e:  # accounting must never break the caller
        logger.warning("ledger prune failed: %s", e)
