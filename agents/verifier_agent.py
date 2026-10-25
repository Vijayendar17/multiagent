from agents.base import BaseAgent


class VerifierAgent(BaseAgent):
    name = "verify"
    system_prompt = """You are a strict Verifier Agent. Given a goal and a result, judge
whether the result actually satisfies the goal. Be harsh: the referee, not the author.
Respond ONLY with valid JSON:
{"pass": true|false, "score": 0-10, "feedback": "<specific, actionable if failed>"}"""

    def build_prompt(self, task, context):
        return f"Goal: {task.goal}\n\nResult to verify:\n{context}"

    def execute(self, task, model_output):
        raw_pass = model_output.get("pass")
        if raw_pass is None and "passed" in model_output:
            raw_pass = model_output.get("passed")

        raw_score = model_output.get("score")
        score = None
        try:
            score = int(raw_score)
        except (TypeError, ValueError):
            pass

        if raw_pass is not None:
            passed = _as_bool(raw_pass)
        elif score is not None:
            passed = score >= 7
        else:
            passed = False

        if score is None:
            score = 10 if passed else 0

        feedback = (model_output.get("feedback") or "").strip()
        status = "done" if passed else "needs_fix"
        return {"status": status, "result": f"score={score} pass={passed}",
                "feedback": feedback}


def _as_bool(v) -> bool:
    """LLMs return True/False or 'True'/'False'/'true' — normalize both."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "pass")
    return bool(v)
