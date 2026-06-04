---
name: local-fleet-task-router
description: Route local-fleet tasks through a compact Pen & Paper task sheet and a specialist Agent Zero profile with only the relevant tools exposed.
version: 1.0.0
tags: ["local-fleet", "routing", "subagents", "context", "tools"]
trigger_patterns:
  - "local fleet"
  - "route task"
  - "delegate profile"
  - "subagent"
  - "gemma"
allowed_tools:
  - pen_paper
  - delegate_profile
  - call_subordinate
  - response
---

# Local Fleet Task Router

Use this skill when the local model should stay lean and a task may need heavy
tools, long context, or parallel subagents.

## Workflow

1. Create or update a Pen & Paper workspace for the task.
2. Capture the task sheet:
   - user goal
   - required output
   - relevant files, URLs, or context
   - required tools or data sources
   - success criteria
   - risks or missing facts
3. Pick exactly one specialist profile for the next step:
   - `developer` for code editing, tests, shell, and repo work
   - `researcher` for web/search/document discovery
   - `wiki_librarian` for SharedBrain or llm_wiki work
   - `google` for Gmail, Calendar, Drive, Sheets, Contacts, or Tasks
   - `documents` for document and office artifact work
   - `meta_supervisor` for cross-chat status and project summaries
4. Delegate the task sheet with `delegate_profile` when the profile is in the
   delegate registry; otherwise use `call_subordinate` with `reset: true`.
5. Answer from the specialist result. Do not re-run the same work in the main
   profile unless the specialist failed or asked for missing input.

Keep the main profile focused on routing, task shaping, and final synthesis.
Heavy tools belong in the specialist profile.
