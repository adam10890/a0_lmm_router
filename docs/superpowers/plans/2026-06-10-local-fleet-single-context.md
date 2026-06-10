# Local Fleet Single Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop Agent Zero local-router chats from failing when utility-model compression receives prompts larger than the current 16K utility context.

**Architecture:** Keep Agent Zero's `chat` and `utility` roles intact, but make utility calls in the local preset use the same llama.cpp chat alias while preserving the utility output budget. Router Mode keeps `models-max >= 2` so separate chats and sub-agents can still run concurrently.

**Tech Stack:** Python helper tests with `pytest`, llama.cpp Router Mode preset generation, Docker Compose env defaults, Agent Zero `_model_config` role semantics.

---

### Task 1: Prove Utility Can Follow Chat Context

**Files:**
- Modify: `tests/test_model_params_cache.py`
- Modify: `helpers/model_params_cache.py`

- [x] **Step 1: Write the failing test**

Add a test that enables `A0_LMM_UTILITY_FOLLOWS_CHAT=1`, sets both roles to the same chat model path, and asserts that `utility` renders with the same hard context as `chat` in optional serial planning mode.

```python
def test_utility_can_follow_chat_without_duplicate_resident_vram(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr("model_params_cache.default_cache_path", lambda: cache_path)
    monkeypatch.setattr("model_params_cache.read_gguf_metadata", lambda _p: dict(_META))
    monkeypatch.setattr(
        "model_params_cache._file_fingerprint",
        lambda _p: {"exists": True, "size": 100, "mtime_ns": 1, **_META},
    )
    monkeypatch.setattr(
        "model_params_cache.container_path_to_host",
        lambda c, _d: "C:/models/chat/test.gguf" if c.endswith(".gguf") else c,
    )

    env = {
        "LLAMA_MODELS_DIR": "C:/models",
        "CHAT_MODEL_PATH": "/models/chat/test.gguf",
        "UTILITY_MODEL_PATH": "/models/chat/test.gguf",
        "EMBED_MODEL_PATH": "",
        "CHAT_CTX_SIZE": "65536",
        "UTILITY_CTX_SIZE": "65536",
        "ROUTER_MODELS_MAX": "1",
        "ROUTER_PARALLEL": "1",
        "A0_LMM_UTILITY_FOLLOWS_CHAT": "1",
        "A0_LMM_AVAILABLE_VRAM_GB": "24",
    }

    warm = warm_fleet_params(env, force_refresh=True, write_cache=False)
    by_alias = {entry.alias: entry for entry in warm["entries"]}

    assert by_alias["chat"].model_path == "/models/chat/test.gguf"
    assert by_alias["utility"].model_path == "/models/chat/test.gguf"
    assert by_alias["utility"].hard_ctx == by_alias["chat"].hard_ctx
    assert warm["resident_vram_gb"] == by_alias["chat"].planned_vram_gb
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model_params_cache.py::test_utility_can_follow_chat_without_duplicate_resident_vram -q`

Expected: FAIL because `resident_vram_gb` is absent and utility still uses independent planning semantics.

- [x] **Step 3: Implement the minimal helper behavior**

Add helpers in `helpers/model_params_cache.py`:

```python
def _truthy_env(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = str(env.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}

def _router_models_max(env: Mapping[str, str]) -> int:
    try:
        return max(1, int(str(env.get("ROUTER_MODELS_MAX") or "2").strip()))
    except ValueError:
        return 2

def _utility_follows_chat(env: Mapping[str, str]) -> bool:
    return _truthy_env(env, "A0_LMM_UTILITY_FOLLOWS_CHAT", False)
```

In `warm_fleet_params`, when `utility` follows chat, use `CHAT_MODEL_PATH` for the utility container model and do not add duplicate resident VRAM for the same container model when `ROUTER_MODELS_MAX <= 1`.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model_params_cache.py::test_utility_can_follow_chat_without_duplicate_resident_vram -q`

Expected: PASS.

### Task 2: Route Local Utility Calls To The Chat Alias

**Files:**
- Modify: `helpers/tool_call_adapter_runtime.py`
- Modify: `tests/test_tool_call_adapter.py`
- Modify: `default_config.yaml`

- [x] **Step 1: Write the failing proxy test**

Assert that role `utility` with `model_name = "lmm_router/utility"` is called as `lmm_router/chat`, while `max_tokens` remains the utility budget.

- [x] **Step 2: Implement temporary model-name rewrite**

In `_ModelCallAdapterProxy.unified_call`, temporarily replace `model.model_name` for the duration of the call when `local_fleet.route_utility_to_chat_alias` is enabled.

- [x] **Step 3: Verify the proxy behavior**

Run: `python -m pytest tests/test_tool_call_adapter.py::test_utility_proxy_can_route_to_chat_alias_without_losing_utility_budget -q`

Expected: PASS.

### Task 3: Make Router Mode Defaults Match The Long-Context Goal

**Files:**
- Modify: `docker/docker-compose.lmm.env`
- Modify: `conf/models_preset.ini`
- Add: `tests/test_local_fleet_config.py`

- [x] **Step 1: Write/extend config assertions**

Add tests that read the committed Docker env, router preset, and default plugin
config. Assert that `utility` follows the chat GGUF/context and that the Agent
Zero hook routes utility role calls to the chat alias by default.

- [x] **Step 2: Update defaults**

Set these defaults:

```env
UTILITY_MODEL_PATH=/models/unsloth--gemma-4-12B-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf
UTILITY_CTX_SIZE=131072
A0_LMM_UTILITY_FOLLOWS_CHAT=1
```

Update `conf/models_preset.ini` so `[utility]` uses the chat model path and `ctx-size = 131072`.

- [x] **Step 3: Run config tests**

Run: `python -m pytest tests/test_local_fleet_config.py tests/test_context_planner.py tests/test_model_params_cache.py -q`

Expected: PASS.

### Task 4: Verify And Publish

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document the quantified behavior**

Add a short Router Mode note: utility compression no longer gets a 16K window; default utility context becomes 128K, and `ROUTER_MODELS_MAX` remains available for separate chats/sub-agents.

- [x] **Step 2: Run verification**

Run:

```powershell
python -m py_compile helpers/model_params_cache.py helpers/tool_call_adapter_runtime.py helpers/context_planner.py helpers/output_budget.py
python -m pytest tests/test_model_params_cache.py tests/test_tool_call_adapter.py tests/test_context_planner.py tests/test_router_context_guard.py tests/test_output_budget.py tests/test_local_fleet_config.py -q
```

Expected: all pass.

- [ ] **Step 3: Commit and PR**

Run:

```powershell
git status -sb
git add helpers/model_params_cache.py helpers/tool_call_adapter_runtime.py tests/test_model_params_cache.py tests/test_tool_call_adapter.py tests/test_local_fleet_config.py default_config.yaml docker/docker-compose.lmm.env conf/models_preset.ini README.md docs/superpowers/plans/2026-06-10-local-fleet-single-context.md
git commit -m "fix(router): run local fleet utility on chat context"
git push -u origin HEAD
```

Then open a draft PR to `main` titled `[codex] run local fleet utility on chat context`.
