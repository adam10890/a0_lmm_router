"""One-shot structural repair for a0_lmm_router.

All plugin files were written with a literal backslash in their filename
(e.g. 'extensions\\python\\agent_init\\_10_init_servers.py' as one file)
instead of as nested directories. This script converts every such name
into a proper nested path so Agent Zero's plugin loader can discover the
extensions, helpers, api handlers, and webui pages.

Run inside the Agent Zero container so the filenames are seen with a
literal '\\' character (which is a legal filename character on Linux).
Run from the plugin root or pass --apply to execute.

Usage:
    python3 /a0/usr/plugins/a0_lmm_router/_fix_structure.py            # dry run
    python3 /a0/usr/plugins/a0_lmm_router/_fix_structure.py --apply    # perform moves
"""
from __future__ import annotations

import os
import shutil
import sys


ROOT = "/a0/usr/plugins/a0_lmm_router"


def main() -> int:
    os.chdir(ROOT)
    entries = [e for e in os.listdir(".") if "\\" in e]
    print(f"found {len(entries)} entries with backslashes", flush=True)

    ops: list[tuple[str, str, str]] = []
    pycache: list[str] = []
    for name in sorted(entries):
        parts = name.split("\\")
        if any(p == "__pycache__" for p in parts):
            pycache.append(name)
            continue
        if parts[-1] == "":
            dir_parts = [p for p in parts if p]
            if not dir_parts:
                continue
            ops.append(("MKDIR", name, os.path.join(*dir_parts)))
        else:
            ops.append(("MOVE", name, os.path.join(*parts)))

    dry = "--apply" not in sys.argv
    print(
        f"plan: {len(ops)} move/mkdir, {len(pycache)} pycache cleanup (dry={dry})",
        flush=True,
    )
    for op, s, d in ops[:8]:
        print(f"  {op} {s!r} -> {d!r}")
    if len(ops) > 8:
        print(f"  ... and {len(ops) - 8} more")

    if dry:
        return 0

    for op, src, dst in ops:
        parent = os.path.dirname(dst) if op == "MOVE" else dst
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    moved = 0
    for op, src, dst in ops:
        if op != "MOVE":
            continue
        if not os.path.exists(src):
            print(f"  MISSING: {src}")
            continue
        if os.path.exists(dst):
            print(f"  SKIP (target exists): {dst}")
            continue
        shutil.move(src, dst)
        moved += 1
    print(f"moved {moved} files")

    for name in pycache:
        try:
            if os.path.isfile(name):
                os.remove(name)
            elif os.path.isdir(name):
                shutil.rmtree(name)
        except Exception as exc:
            print(f"  cleanup failed for {name}: {exc}")

    for name in os.listdir("."):
        if "\\" in name and os.path.isfile(name) and os.path.getsize(name) == 0:
            try:
                os.remove(name)
            except Exception:
                pass

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
