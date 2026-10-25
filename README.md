# Multi-Agent Orchestration System (Local, Offline, Ollama 8B)

A complete, runnable **Research → Code → Verify** multi-agent system that runs entirely
offline using **Ollama** with an 8B-class instruct model (`llama3.1:8b`, `qwen2.5:8b-instruct`,
or `mistral:7b-instruct`).

## Architecture

```
                     ┌────────────────────┐
                     │  Orchestrator       │  ← owns the state machine
                     │ (planner + router)  │
                     └─────────┬───────────┘
                               │ reads/writes
                               ▼
                     ┌────────────────────┐
                     │  Shared State       │  ← single source of truth
                     │ (SQLite TaskGraph   │     (task graph + blackboard)
                     │  / blackboard)      │
                     └─────────┬───────────┘
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │ Research      │   │ Code          │   │ Verifier     │
   │ Agent         │   │ Agent         │   │ Agent        │
   │ (search tool) │   │ (writes/runs  │   │ (checks      │
   │               │   │  code tool)   │   │  output)     │
   └──────────────┘   └──────────────┘   └──────────────┘
```

**Agents never talk to each other directly.** They only read/write the shared state. The
orchestrator decides who runs next. This keeps state management, error recovery, and loop
prevention tractable: exactly one place for control-flow decisions, exactly one place where
knowledge lives.

## Why it solves the three hard problems

1. **State management** — a `TaskGraph` persisted to SQLite survives crashes/restarts.
   Agents are stateless functions of `(task, scoped_context)`. All memory lives in the graph.
2. **Error recovery** — three layers: JSON schema validation with self-correction retries
   (in `llm.py`), tool-exception → `status=failed` handling, and logical failure → a new
   `fix` iteration as a normal graph transition.
3. **Loop prevention** — enforced in code, never trusted to the model:
   - Max attempts per task (default 3) → `failed_permanent`
   - Global step cap (default 40)
   - Cycle detection via a hash of `(goal, proposed_action)` — identical consecutive
     proposals force a strategy change
   - Stagnation detection — verify failing twice in a row abandons that branch
   - Every code execution runs in a subprocess with a wall-clock timeout

## Prerequisites

```bash
curl -fsSL https://ollama.com/install.sh | sh     # Linux
ollama pull llama3.1:8b                            # or qwen2.5:8b-instruct
ollama serve &
curl http://localhost:11434/api/tags              # confirm it serves
pip install ollama
```

> `qwen2.5:8b-instruct` is generally more reliable at strict JSON output than llama3.1:8b.

## Run

```bash
ollama serve &
python3 main.py "Write a Python function that merges two sorted lists, with tests"
```

Set the model / step cap via env vars:

```bash
OLLAMA_MODEL=qwen2.5:8b-instruct MAX_GLOBAL_STEPS=60 python3 main.py "..."
```

## Web dashboard + desktop app

A live dashboard (FastAPI + SSE) shows each run's task graph in real time, streams the
event log, and surfaces a **human-in-the-loop** dialog when the orchestrator wants to
abandon a task (stagnation, max attempts) — you can retry with feedback or abandon.

```bash
# 1. Install the extra deps
pip install fastapi uvicorn pywebview            # pywebview only needed for the desktop wrapper

# 2. Web dashboard, open http://127.0.0.1:8247
python3 desktop_app.py --no-window

# 3. Or launch a native desktop window (GTK/WebKit on Linux)
python3 desktop_app.py
```

API surface (all JSON):
- `POST /api/runs` `{"goal": …}` — start a run, returns `run_id`
- `GET  /api/runs` — list runs + aggregate status
- `GET  /api/runs/{id}` — run detail + task list
- `GET  /api/runs/{id}/stream` — Server-Sent Events (replays history, then live)
- `POST /api/runs/{id}/decision` `{"action": "retry"|"abandon", "feedback": "…"}` — HITL answer
- `GET  /api/config` — model, hosts, available Ollama models

Each run gets its own SQLite DB under `data/runs/`, so a crash mid-run never corrupts other
runs and old runs stay inspectable in the sidebar.

## Project structure

```
.
├── orchestrator.py       # state machine, routing, loop guards (+ HITL hooks)
├── state.py              # TaskGraph, Task, SQLite persistence
├── llm.py                # Ollama client wrapper + strict JSON enforcement
├── webapp.py             # FastAPI: REST API + SSE + RunManager (run lifecycle/HITL)
├── desktop_app.py        # pywebview wrapper (or --no-window = plain web server)
├── static/index.html     # dashboard UI (vanilla JS + SSE)
├── agents/
│   ├── base.py           # BaseAgent: prompt + parse + validate
│   ├── research_agent.py # search tool
│   ├── code_agent.py     # write + sandboxed-execute code tool
│   └── verifier_agent.py # pass/fail + feedback
├── tools/
│   ├── search_tool.py    # offline-first local search (swap for real backend)
│   └── sandbox_exec.py   # isolated subprocess with timeout
├── main.py               # CLI entry point
└── data/                 # graph.sqlite (CLI), runs/*.sqlite (webapp)
```

## Swap the search tool

`tools/search_tool.py` is intentionally a placeholder: it searches local files and — when
nothing matches — returns an offline note instead of phoning home. Replace
`local_or_web_search` with a real backend (DuckDuckGo, Tavily, a local doc index) without
touching agent code.

## Security note

The sandbox is a **subprocess with a timeout — a crash boundary, not a security boundary.**
For untrusted code, run inside a container (Docker/gVisor) with no network access and a
memory cap.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ollama.ResponseError: model not found` | model not pulled | `ollama pull llama3.1:8b` |
| JSON parse failures every call | model ignoring JSON-only instructions | lower temperature, shorten prompt, or switch to `qwen2.5:8b-instruct` |
| Orchestrator loops until step cap | cycle detection not tripping | confirm `task.last_signature` is persisted — `graph.update(task)` must run every step |
| Code agent's script hangs | no timeout enforced | confirm `subprocess.run(..., timeout=...)` in `tools/sandbox_exec.py` |