# Tool: route_task / compute_budget (lmm-router MCP)

The lmm-router MCP server (port 8095) manages the shared "token economy":
all available compute — the local llama.cpp fleet, the Codex (ChatGPT)
subscription, Ollama Cloud, and any custom providers — with declared limits,
locally counted usage, and rolling-window budget forecasts.

## When to use

- **Before any heavy or multi-step task** (long generation, code work,
  batch processing): call `route_task` first and use the returned packet.
- **When planning parallel work across agents**: call `compute_budget` and
  split tasks so heavy ones go to providers with headroom and light ones
  to the local fleet.
- Budgets are computed on demand — do not poll on a timer; ask when deciding.

## route_task

Arguments:
- `task` — one-line description of what you are about to do
- `role` — "chat" | "utility" | "embedding" (default "chat")
- `est_input_tokens` / `est_output_tokens` — your estimate; improves the
  forecast and reserves budget so concurrent agents see pending demand
- `quality` — "best_available" (default) | "fast" | "cheap" (prefer local)

The result is a **routing packet** — a recommendation, not an execution:

- `provider_id`, `model`, `base_url`, `invoke` — use exactly these for your
  own call. `invoke: openai_compatible` → call `base_url` with the model;
  `invoke: codex_cli` → run the task through the codex CLI.
- `budget_before` / `budget_after` — the tightest window's state; if `notes`
  contains a warning, consider a fallback or a smaller task.
- `fallback` — ordered alternatives if your call fails.
- `provider_id: null` — everything is exhausted or cooling down: wait for a
  window to roll over, use the fallback list, or ask the user.

Make your request deterministic: after routing, do not re-decide the model
yourself — all agents get their assignment from this single point so usage
stays counted and predictable.
