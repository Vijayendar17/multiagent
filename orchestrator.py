import hashlib
import json
import os
from dataclasses import asdict
from typing import Callable, Optional

from agents.code_agent import CodeAgent
from agents.research_agent import ResearchAgent
from agents.verifier_agent import VerifierAgent
from state import Task, TaskGraph


AGENTS = {"research": ResearchAgent(), "code": CodeAgent(), "verify": VerifierAgent()}
MAX_GLOBAL_STEPS = int(os.environ.get("MAX_GLOBAL_STEPS", "40"))


def signature(task: Task, action) -> str:
    """Cycle-detection signature: hash of (task.id, what the model proposed to do).

    Unlike hashing task.goal (which breaks when feedback or notes change), this stays
    identical when the model proposes the exact same action for the same task twice
    in a row — which is exactly the loop we want to break.
    """
    payload = f"{task.id}|{json.dumps(action, sort_keys=True, default=str)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _inject_strategy_change(task: Task):
    note = (
        "(NOTE: your previous attempt was identical and failed — "
        "stop repeating yourself. Try a genuinely different approach.)"
    )
    if task.feedback:
        task.feedback = f"{task.feedback}\n{note}"
    else:
        task.feedback = note


def _next_runnable(pending, graph):
    """Returns the first task whose parent is done (or has no parent)."""
    for t in pending:
        if not t.parent_id:
            return t
        try:
            parent = graph.get(t.parent_id)
            if parent.status == "done":
                return t
        except KeyError:
            return t
    return None


def run(goal: str, graph: TaskGraph, emit: Optional[Callable] = None,
        decide: Optional[Callable] = None, max_steps: int = MAX_GLOBAL_STEPS):
    """Lockstep Research -> Code -> Verify executor.

    emit(event_type: str, payload: dict) is called on every meaningful transition
    so a live dashboard (SSE) can stream progress. decide(task, reason) is called
    before a task would be abandoned; it blocks the loop until the human answers
    (returns dict with action in {"retry", "abandon"}). When None, failures are
    abandoned automatically.
    """
    research = graph.add(goal=f"Research context for: {goal}", agent="research")
    code = graph.add(goal=f"Write code to: {goal}", agent="code", parent_id=research.id)
    verify = graph.add(goal=f"Verify result satisfies: {goal}", agent="verify", parent_id=code.id)

    if emit:
        emit("run_started", {"goal": goal})
        for t in (research, code, verify):
            emit("task_updated", asdict(t))

    steps = 0
    while steps < max_steps:
        steps += 1
        pending = graph.all_pending()
        if not pending:
            break

        # Cascade permanently failed parents to child tasks (transitively)
        while True:
            newly_blocked = [
                t for t in pending
                if t.parent_id and graph.get(t.parent_id).status == "failed_permanent"
            ]
            if not newly_blocked:
                break
            for t in newly_blocked:
                t.status = "failed_permanent"
                t.feedback = f"abandoned: parent task {t.parent_id} failed permanently"
                graph.update(t)
                if emit:
                    emit("task_updated", asdict(t))
                print(f"[ORCH] Cascading failure: {t.id} ({t.agent}) abandoned (parent dead)")
            pending = [t for t in pending if t not in newly_blocked]

        if not pending:
            print(f"[ORCH] All remaining tasks blocked by dead parent.")
            break

        # Dependency-aware FIFO: a child task may only run once its parent is done.
        task = _next_runnable(pending, graph)
        if task is None:
            print(f"[ORCH] No runnable tasks (dependencies unsatisfied): "
                  f"{[f'{t.agent}:{t.status}' for t in pending]}")
            break

        # Check attempt limit before running
        if task.attempts >= task.max_attempts:
            _abandon(task, graph, reason=f"exceeded max attempts ({task.max_attempts})",
                     emit=emit, decide=decide)
            continue

        task.status = "running"
        task.attempts += 1
        graph.update(task)
        if emit:
            emit("task_updated", asdict(task))

        agent = AGENTS[task.agent]
        try:
            outcome, raw_output = agent.run(task, graph)
        except Exception as e:  # noqa: BLE001 - any tool/LLM failure becomes a failed task
            outcome = {"status": "failed", "result": None, "feedback": str(e)}
            raw_output = None

        # --- Guard 2: cycle detection (before persisting) ---
        sig = signature(task, raw_output)
        if sig and sig == task.last_signature and outcome.get("status") == "failed":
            _inject_strategy_change(task)
            print(f"[ORCH] Cycle detected on task {task.id} — forcing strategy change.")
        task.last_signature = sig

        # --- Guard 3: retry hard tool failures (with max_attempts cap) ---
        if outcome["status"] == "failed":
            task.feedback = (task.feedback or "") + f"\nATTEMPT {task.attempts} failed: {outcome['feedback']}"
            if task.attempts >= task.max_attempts:
                _abandon(task, graph, reason=f"exceeded max attempts ({task.max_attempts})",
                         emit=emit, decide=decide)
                continue
            task.status = "pending"
            task.result = outcome["result"]
            graph.update(task)
            if emit:
                emit("task_updated", asdict(task))
            print(f"[ORCH] step={steps} task={task.id} agent={task.agent} -> "
                  f"FAILED (retry {task.attempts}/{task.max_attempts})")
            continue

        # --- Guard 4: cycle/needs_fix handling on logical failure ---
        task.status = outcome["status"]
        task.result = outcome["result"]
        task.feedback = outcome["feedback"]
        if task.agent == "code" and task.status == "done":
            task.consecutive_failures = 0
        graph.update(task)
        if emit:
            emit("task_updated", asdict(task))

        if task.status == "needs_fix":
            code_task = graph.get(task.parent_id)
            code_task.consecutive_failures = getattr(code_task, "consecutive_failures", 0) + 1

            if code_task.attempts >= code_task.max_attempts:
                print(f"[ORCH] Stagnation: giving up on fixing {code_task.id}.")
                action = _abandon(code_task, graph, reason="stagnation: verifier keeps failing",
                                  emit=emit, decide=decide)
                if action == "retry":
                    task.status = "pending"
                    task.attempts = 0
                    graph.update(task)
                    if emit:
                        emit("task_updated", asdict(task))
                continue

            if code_task.consecutive_failures >= 2:
                print(f"[ORCH] Stagnation: verify failed twice in a row for code task "
                      f"{code_task.id}. Abandoning this branch.")
                action = _abandon(code_task, graph,
                                  reason="stagnation: verifier failed twice in a row",
                                  emit=emit, decide=decide)
                if action == "retry":
                    task.status = "pending"
                    task.attempts = 0
                    graph.update(task)
                    if emit:
                        emit("task_updated", asdict(task))
                continue

            code_task.status = "needs_fix"
            code_task.feedback = task.feedback
            graph.update(code_task)
            if emit:
                emit("task_updated", asdict(code_task))
            task.status = "pending"
            task.attempts = 0  # Reset verifier attempts so new code has fresh verification allowance
            graph.update(task)
            if emit:
                emit("task_updated", asdict(task))

        print(f"[ORCH] step={steps} task={task.id} agent={task.agent} -> {task.status}")

    if steps >= max_steps:
        print(f"[ORCH] Global step cap ({max_steps}) reached — hard stop.")
    if emit:
        emit("run_finished", {"steps": steps})
    print(f"[ORCH] Finished in {steps} steps.")
    return graph


def _abandon(task, graph, reason, emit=None, decide=None):
    """Mark a task failed permanently — unless a human is in the loop.

    With a decide() callback, the task goes to awaiting_human and the loop
    pauses until the operator chooses retry or abandon. Without one, it is
    abandoned automatically (deterministic, loop-free).
    """
    if decide is None:
        task.status = "failed_permanent"
        task.feedback = reason
        graph.update(task)
        if emit:
            emit("task_updated", asdict(task))
        print(f"[ORCH] Task {task.id} ({task.agent}) abandoned: {reason}")
        return "abandon"

    task.status = "awaiting_human"
    task.feedback = reason
    graph.update(task)
    if emit:
        emit("hitl_required", {"task": asdict(task), "reason": reason})
        emit("task_updated", asdict(task))
    print(f"[ORCH] Task {task.id} ({task.agent}) awaiting human decision: {reason}")

    decision = decide(task, reason) or {"action": "abandon"}
    action = decision.get("action", "abandon")

    if action == "retry":
        task.status = "pending"
        if decision.get("feedback"):
            task.feedback = decision.get("feedback")
        task.attempts = 0
        task.consecutive_failures = 0
        graph.update(task)
        if emit:
            emit("hitl_resolved", {"task": asdict(task), "decision": "retry"})
            emit("task_updated", asdict(task))
        print(f"[ORCH] Human chose RETRY on task {task.id}.")
    else:
        task.status = "failed_permanent"
        task.feedback = f"{reason} (abandoned by human)"
        graph.update(task)
        if emit:
            emit("hitl_resolved", {"task": asdict(task), "decision": "abandon"})
            emit("task_updated", asdict(task))
        print(f"[ORCH] Human chose ABANDON on task {task.id}.")
    return action
