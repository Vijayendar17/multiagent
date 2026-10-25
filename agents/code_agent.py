from agents.base import BaseAgent
from tools.sandbox_exec import run_python_sandboxed


def _clean_code(code: str) -> str:
    """Strips markdown code fences (```python ... ```) if the model included them inside the JSON code field."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        # Drop leading ``` or ```python
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Drop trailing ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    return code


class CodeAgent(BaseAgent):
    name = "code"
    system_prompt = """You are a Code Agent. Given a goal and context, write a single
self-contained Python script that accomplishes it. The script MUST print its output to
stdout (the verifier checks both source code and stdout). Respond ONLY with valid JSON:
{"code": "<full python source>", "explanation": "<1-2 sentences>"}"""

    def build_prompt(self, task, context):
        extra = ""
        if task.feedback:
            extra += f"\nFeedback from previous attempt to fix:\n{task.feedback}"
            if task.result:
                # Include previous code so the model can inspect and fix it
                extra += f"\nPrevious attempt implementation:\n{task.result}"
        return f"Goal: {task.goal}\n{context}{extra}\nWrite the code now. Make it print the result to stdout."

    def execute(self, task, model_output):
        code = _clean_code(model_output.get("code") or "")
        if not code:
            return {"status": "failed", "result": None,
                    "feedback": "model returned no code field"}
        try:
            stdout, stderr, ok = run_python_sandboxed(code, timeout=20)
            # Verifier must be able to see the function itself, not just stdout —
            # pass the source along so it can judge the implementation.
            result = f"--- source ---\n{code}\n--- output ---\n{stdout}"
            if ok:
                return {"status": "done", "result": result, "feedback": None}
            feedback = stderr.strip() or f"Script failed with unknown error. Output:\n{stdout.strip()}"
            return {"status": "failed", "result": result, "feedback": feedback}
        except TimeoutError:
            return {"status": "failed", "result": None, "feedback": "execution timed out"}
        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "result": None, "feedback": f"execution error: {e}"}
