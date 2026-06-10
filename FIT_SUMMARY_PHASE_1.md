# Fit Summary Phase 1 — Read-only Hardware / Model Fit Summary

## What was implemented

- Added clean-room helper `helpers/fit_summary.py`.
- Added Agent Zero plugin API handler `api/lmm_fit_summary.py`.
- Added dashboard state/API wiring in `webui/js/dashboard-store.js`.
- Added a compact dashboard panel in `webui/dashboard.html`.
- Added focused tests in `tests/test_fit_summary.py` and `tests/test_fit_summary_ui.py`.

The route follows the existing plugin convention:

```text
POST /api/plugins/a0_lmm_router/lmm_fit_summary
```

This is the plugin-local equivalent of the requested read-only fit summary endpoint.

## What was intentionally not implemented

- No model download.
- No model serve / runtime launch flow.
- No shell execution.
- No SSH.
- No new package dependencies.
- No HuggingFace or external catalog calls.
- No new credentials or secrets.
- No deep hardware probing beyond existing project observer data.
- No changes to unrelated plugin areas.

## Why download/serve are excluded

Odysseus-style model lifecycle actions touch host files, external model sources,
long-running processes, permissions, and potentially credentials. Phase 1 is
limited to observer data so it improves operator visibility without increasing
the control surface.

## Security assumptions

- The endpoint is read-only and should be treated as an observer endpoint.
- The helper is pure: it consumes an already-built snapshot and does not perform lifecycle actions.
- Hardware data is only as trustworthy as `helpers.compute_monitor.get_compute_snapshot()`.
- If hardware data is missing, the endpoint returns `unknown` instead of probing aggressively.
- Fit status is a phase-1 heuristic, not a capacity guarantee.

## Future phase ideas

1. Add a safe host hardware detector with explicit operator opt-in and data-source labels.
2. Add model-size estimation from installed model metadata only.
3. Add KV-cache/context budget estimation using existing context planner code.
4. Add per-router-role fit summaries for native router mode aliases.
5. Add stricter schema validation if the plugin later adopts Pydantic or dataclass validation.

## How to test

```bash
python -m py_compile helpers/fit_summary.py api/lmm_fit_summary.py
python -c "from usr.plugins.a0_lmm_router.helpers.fit_summary import build_fit_summary; print(build_fit_summary({'slots': []})['mode'])"
python -c "from usr.plugins.a0_lmm_router.api.lmm_fit_summary import LmmFitSummary; print(LmmFitSummary.__name__)"
python -m pytest -q tests/test_fit_summary.py tests/test_fit_summary_ui.py
```

If `pytest` is not installed in the current runtime, run the compile/import checks and execute the pytest command in the normal Agent Zero development/test environment.
