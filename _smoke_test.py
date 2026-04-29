"""Smoke test: verify imports work after structure repair."""
from __future__ import annotations


def main() -> int:
    ok = True
    checks = [
        ("manager", "usr.plugins.a0_lmm_router.helpers.llama_cpp_manager", "LlamaCppManager"),
        ("compute_monitor", "usr.plugins.a0_lmm_router.helpers.compute_monitor", None),
        ("model_recommender", "usr.plugins.a0_lmm_router.helpers.model_recommender", None),
        ("backends_factory", "usr.plugins.a0_lmm_router.helpers.backends.factory", None),
        ("smart_router_registry", "usr.plugins.a0_lmm_router.helpers.smart_router.workflow_registry", "get_workflow_registry"),
        ("agent_init_ext", "usr.plugins.a0_lmm_router.extensions.python.agent_init._10_init_servers", "LlamaCppInitExtension"),
        ("message_loop_ext", "usr.plugins.a0_lmm_router.extensions.python.message_loop_start._20_smart_router", None),
    ]
    for label, module, attr in checks:
        try:
            mod = __import__(module, fromlist=["*"])
            if attr and not hasattr(mod, attr):
                print(f"FAIL {label}: missing attr {attr!r} on {module}")
                ok = False
            else:
                print(f"ok   {label}")
        except Exception as exc:
            ok = False
            print(f"FAIL {label}: {type(exc).__name__}: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
