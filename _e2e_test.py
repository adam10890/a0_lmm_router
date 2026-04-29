"""End-to-end smoke test: every API handler and the launcher import cleanly."""
from __future__ import annotations

MODULES = [
    "usr.plugins.a0_lmm_router.launcher",
    "usr.plugins.a0_lmm_router.api.llamacpp_status",
    "usr.plugins.a0_lmm_router.api.llamacpp_control",
    "usr.plugins.a0_lmm_router.api.llamacpp_list_models",
    "usr.plugins.a0_lmm_router.api.lmm_compute_stats",
    "usr.plugins.a0_lmm_router.api.lmm_fleet_ignite",
    "usr.plugins.a0_lmm_router.api.lmm_host_ignite",
    "usr.plugins.a0_lmm_router.tools.fleet_ignite",
    "usr.plugins.a0_lmm_router.api.lmm_model_install",
    "usr.plugins.a0_lmm_router.api.lmm_model_recommend",
    "usr.plugins.a0_lmm_router.helpers.compute_monitor",
    "usr.plugins.a0_lmm_router.helpers.llama_cpp_manager",
    "usr.plugins.a0_lmm_router.helpers.smart_router.workflow_registry",
    "usr.plugins.a0_lmm_router.extensions.python.agent_init._10_init_servers",
    "usr.plugins.a0_lmm_router.extensions.python.message_loop_start._20_smart_router",
]


def main() -> int:
    fails = []
    for m in MODULES:
        try:
            __import__(m)
            print(f"ok   {m}")
        except Exception as exc:
            fails.append((m, type(exc).__name__, str(exc)))
            print(f"FAIL {m}: {type(exc).__name__}: {exc}")
    print("---")
    if fails:
        print(f"{len(fails)} failures")
        return 1
    print(f"all {len(MODULES)} modules OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
