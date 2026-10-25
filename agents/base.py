from abc import ABC, abstractmethod

from llm import call_llm
from state import Task, TaskGraph


class BaseAgent(ABC):
    name: str
    system_prompt: str
    model_output_key: str = None

    def run(self, task: Task, graph: TaskGraph):
        """Returns (outcome dict, raw model output dict). The raw output is
        needed by the orchestrator for cycle-detection signatures."""
        context = graph.context_for(task)
        user_prompt = self.build_prompt(task, context)
        raw = call_llm(self.system_prompt, user_prompt, expect_json=True)
        return self.execute(task, raw), raw

    @abstractmethod
    def build_prompt(self, task: Task, context: str) -> str: ...

    @abstractmethod
    def execute(self, task: Task, model_output: dict) -> dict:
        """Runs any real tool call the model asked for, returns
        {'status': 'done'|'failed', 'result': str, 'feedback': str|None}"""
        ...