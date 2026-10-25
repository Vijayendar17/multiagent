from agents.base import BaseAgent
from tools.search_tool import local_or_web_search


class ResearchAgent(BaseAgent):
    name = "research"
    system_prompt = """You are a Research Agent. Given a goal, decide what to search for.
Respond ONLY with valid JSON: {"query": "<search query>", "reasoning": "<why>", "notes": "<useful context you already know about the goal>"}"""

    def build_prompt(self, task, context):
        extra = f"\nFeedback to incorporate:\n{task.feedback}" if task.feedback else ""
        return f"Goal: {task.goal}\n{context}{extra}\nWhat should be searched?"

    def execute(self, task, model_output):
        query = (
            model_output.get("query")
            or model_output.get("search_query")
            or model_output.get("search")
            or task.goal
        )
        notes = (
            model_output.get("notes")
            or model_output.get("reasoning")
            or model_output.get("summary")
            or ""
        )
        try:
            results = local_or_web_search(query, timeout=15)
            combined = f"Search query: {query}\n{results}"
            if notes:
                combined += f"\nAgent notes: {notes}"
            return {"status": "done", "result": combined, "feedback": None}
        except Exception as e:  # noqa: BLE001 - tool-layer failures funnel to orchestrator
            return {"status": "failed", "result": None, "feedback": f"search error: {e}"}
