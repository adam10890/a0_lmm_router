"""Render llama.cpp Router Mode preset with max-feasible context windows."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLUGIN_ROOT.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from helpers.model_params_cache import render_fleet_preset_to_file  # noqa: E402


def _read_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.is_file():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _merged_env(env_file: Path) -> dict[str, str]:
    merged = dict(os.environ)
    merged.update(_read_env_file(env_file))
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default=str(PLUGIN_ROOT / "docker" / "docker-compose.lmm.env"),
        help="Path to docker-compose.lmm.env",
    )
    parser.add_argument(
        "--output",
        default=str(PLUGIN_ROOT / "conf" / "models_preset.ini"),
        help="Preset path to write",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print preset instead of writing")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore model_params_cache.json and recompute all slots",
    )
    args = parser.parse_args()

    env_file = Path(args.env_file).resolve()
    output = Path(args.output).resolve()
    env = _merged_env(env_file)

    if args.dry_run:
        from helpers.model_params_cache import warm_fleet_params  # noqa: E402
        from helpers.context_planner import render_preset  # noqa: E402

        warm = warm_fleet_params(env, force_refresh=args.force_refresh, write_cache=False)
        print(
            render_preset(
                warm["entries"],
                global_options=warm["global_options"],
                per_entry_options=warm["per_entry_options"],
            ),
            end="",
        )
        return 0

    warm = render_fleet_preset_to_file(env, output, force_refresh=args.force_refresh)
    for entry in warm["entries"]:
        if entry.no_capacity:
            print(f"[WARN] {entry.alias}: {entry.reason}", file=sys.stderr)
        print(
            f"{entry.alias}: min={entry.min_ctx} hard={entry.hard_ctx} "
            f"effective={entry.effective_ctx} vram={entry.planned_vram_gb}"
        )
    stats = warm.get("stats") or {}
    print(
        f"Cache: {stats.get('cached', 0)} hit, {stats.get('computed', 0)} computed, "
        f"{stats.get('seeded_from_last_success', 0)} seeded from last success"
    )
    print(f"Cache file: {warm.get('cache_path')}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
