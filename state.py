import sqlite3, json, time, uuid, os
from dataclasses import dataclass, field, asdict
from typing import Optional


STATUSES = ("pending", "running", "done", "failed", "failed_permanent", "needs_fix",
            "awaiting_human")


@dataclass
class Task:
    id: str
    goal: str
    agent: str                     # "research" | "code" | "verify"
    status: str = "pending"
    parent_id: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    result: Optional[str] = None
    feedback: Optional[str] = None
    last_signature: Optional[str] = None   # for cycle detection
    consecutive_failures: int = 0          # for stagnation detection
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


def _row_to_task(raw_json: str) -> Task:
    data = json.loads(raw_json)
    valid_fields = Task.__dataclass_fields__
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return Task(**filtered)


class TaskGraph:
    """Single source of truth. Every agent reads/writes ONLY through this class."""

    def __init__(self, db_path="data/graph.sqlite"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=15.0)
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def add(self, goal: str, agent: str, parent_id: str = None, max_attempts=3) -> Task:
        t = Task(id=str(uuid.uuid4())[:8], goal=goal, agent=agent,
                  parent_id=parent_id, max_attempts=max_attempts)
        self._save(t)
        return t

    def get(self, task_id: str) -> Task:
        row = self.conn.execute("SELECT data FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"task {task_id} not found")
        return _row_to_task(row[0])

    def all_tasks(self) -> list[Task]:
        rows = self.conn.execute("SELECT data FROM tasks").fetchall()
        return [_row_to_task(r[0]) for r in rows]

    def all_pending(self) -> list[Task]:
        return [t for t in self.all_tasks() if t.status in ("pending", "needs_fix")]

    def all_done(self) -> list[Task]:
        return [t for t in self.all_tasks() if t.status == "done"]

    def update(self, task: Task):
        task.updated_at = time.time()
        self._save(task)

    def _save(self, t: Task):
        self.conn.execute(
            "INSERT OR REPLACE INTO tasks (id, data) VALUES (?, ?)",
            (t.id, json.dumps(asdict(t)))
        )
        self.conn.commit()

    def context_for(self, task: Task) -> str:
        """Only pull directly relevant results, not the whole history — keeps
        the prompt small and the 8B model focused."""
        if not task.parent_id:
            return ""
        try:
            parent = self.get(task.parent_id)
            result = parent.result or "(none)"
            return f"Prior result from parent task:\n{result}"
        except KeyError:
            return ""

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
