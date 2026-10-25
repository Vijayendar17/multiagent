import glob
import json
import os
import pathlib
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

from llm import MODEL, OLLAMA_HOST
from orchestrator import MAX_GLOBAL_STEPS, run
from state import TaskGraph


@dataclass
class RunHandle:
    """One orchestrator run, executed on its own thread with its own TaskGraph DB.

    Owns three things:
      - the worker thread (never touched from the web layer)
      - a thread-safe append-only event log (consumed by SSE clients)
      - a blocking HITL mailbox (decide() parks the worker until the human answers)
    """
    run_id: str
    goal: str
    db_path: str
    status: str = "running"                 # running | awaiting_human | finished | error
    error: str = field(default=None)
    created_at: float = field(default_factory=time.time)
    finished_at: float = field(default=None)

    _events: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _thread: threading.Thread = field(default=None)
    _decide_wait: threading.Event = field(default_factory=threading.Event)
    _decide_result: dict = field(default=None)

    def emit(self, event: str, payload: dict):
        with self._lock:
            self._events.append({"event": event, "payload": payload, "ts": time.time()})

    def events_since(self, index: int):
        with self._lock:
            return list(self._events[index:]), len(self._events)

    def decide(self, task, reason) -> dict:
        """Called from the worker thread inside orchestration.

        Parks the thread until webapp.resolve_decision() answers, with a safety
        cap so a dead client can never hang a run forever.
        """
        old_status = self.status
        self.status = "awaiting_human"
        self._decide_result = None
        self._decide_wait.clear()
        try:
            self._decide_wait.wait(timeout=3600)
            return self._decide_result or {"action": "abandon"}
        finally:
            self.status = old_status

    def resolve_decision(self, decision: dict):
        self._decide_result = decision
        self._decide_wait.set()


class RunManager:
    """Registry + lifecycle for all runs. Loads persisted runs from disk and tracks live ones."""

    def __init__(self, data_dir="data/runs"):
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.runs: dict[str, RunHandle] = {}
        self._load_existing_runs()

    def _load_existing_runs(self):
        """Discovers existing sqlite DBs in data_dir so past runs stay inspectable."""
        for path in glob.glob(os.path.join(self.data_dir, "*.sqlite")):
            run_id = os.path.splitext(os.path.basename(path))[0]
            if run_id in self.runs:
                continue
            try:
                g = TaskGraph(db_path=path)
                tasks = g.all_tasks()
                g.close()
                if not tasks:
                    continue

                # Root task without parent_id defines the goal
                root = next((t for t in tasks if not t.parent_id), tasks[0])
                goal = root.goal
                prefix = "Research context for: "
                if goal.startswith(prefix):
                    goal = goal[len(prefix):]

                created_at = min(t.created_at for t in tasks)
                finished_at = max(t.updated_at for t in tasks)

                if any(t.status == "awaiting_human" for t in tasks):
                    status = "awaiting_human"
                elif all(t.status == "done" for t in tasks):
                    status = "finished"
                else:
                    status = "finished"

                runh = RunHandle(
                    run_id=run_id,
                    goal=goal,
                    db_path=path,
                    status=status,
                    created_at=created_at,
                    finished_at=finished_at,
                )
                self.runs[run_id] = runh
            except Exception:
                continue

    def start(self, goal: str) -> RunHandle:
        run_id = uuid.uuid4().hex[:8]
        runh = RunHandle(
            run_id=run_id,
            goal=goal,
            db_path=os.path.join(self.data_dir, f"{run_id}.sqlite"),
        )
        runh._thread = threading.Thread(target=self._worker, args=(runh,), daemon=True)
        self.runs[run_id] = runh
        runh.emit("run_created", {"run_id": run_id, "goal": goal})
        runh._thread.start()
        return runh

    def _worker(self, runh: RunHandle):
        graph = TaskGraph(db_path=runh.db_path)
        try:
            run(runh.goal, graph, emit=lambda e, p: runh.emit(e, p),
                decide=runh.decide, max_steps=MAX_GLOBAL_STEPS)
        except Exception as e:  # noqa: BLE001
            runh.status = "error"
            runh.error = str(e)
            runh.emit("run_error", {"error": str(e)})
            return
        finally:
            if runh.status != "error":
                runh.status = "finished"
            runh.finished_at = time.time()
            graph.close()

    def get(self, run_id: str) -> RunHandle | None:
        if run_id not in self.runs:
            # Check if created on disk out-of-band
            db_path = os.path.join(self.data_dir, f"{run_id}.sqlite")
            if os.path.exists(db_path):
                self._load_existing_runs()
        return self.runs.get(run_id)

    def graph_for(self, runh: RunHandle) -> TaskGraph | None:
        try:
            return TaskGraph(db_path=runh.db_path)
        except Exception:
            return None


MANAGER = RunManager()


def make_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    from starlette.responses import StreamingResponse

    app = FastAPI(title="Multi-Agent Orchestrator")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    class StartBody(BaseModel):
        goal: str

    class DecideBody(BaseModel):
        action: str = "abandon"   # retry | abandon
        feedback: str = ""

    @app.post("/api/runs")
    def start_run(body: StartBody):
        goal = body.goal.strip()
        if not goal:
            return {"error": "goal is empty"}
        runh = MANAGER.start(goal)
        return {"run_id": runh.run_id, "status": runh.status}

    @app.get("/api/runs")
    def list_runs():
        out = []
        for runh in MANAGER.runs.values():
            graph = MANAGER.graph_for(runh)
            tasks = []
            if graph:
                tasks = graph.all_tasks()
                graph.close()
            out.append({
                "run_id": runh.run_id,
                "goal": runh.goal,
                "status": runh.status,
                "error": runh.error,
                "created_at": runh.created_at,
                "finished_at": runh.finished_at,
                "task_count": len(tasks),
                "done": sum(1 for t in tasks if t.status == "done"),
                "failed": sum(1 for t in tasks if t.status in ("failed", "failed_permanent")),
            })
        return {"runs": sorted(out, key=lambda r: r["created_at"], reverse=True)}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        runh = MANAGER.get(run_id)
        if not runh:
            return {"error": "run not found"}
        graph = MANAGER.graph_for(runh)
        tasks = []
        if graph:
            tasks = [asdict(t) for t in graph.all_tasks()]
            graph.close()
        return {"run_id": run_id, "goal": runh.goal, "status": runh.status,
                "error": runh.error, "tasks": tasks}

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str):
        runh = MANAGER.get(run_id)
        if not runh:
            return {"error": "run not found"}
        with runh._lock:
            return {"events": runh._events[-500:]}

    @app.post("/api/runs/{run_id}/decision")
    def decide(run_id: str, body: DecideBody):
        runh = MANAGER.get(run_id)
        if not runh:
            return {"error": "run not found"}
        runh.resolve_decision({"action": body.action, "feedback": body.feedback})
        return {"ok": True}

    @app.get("/api/runs/{run_id}/stream")
    def stream(run_id: str):
        """Server-Sent Events. Replays the event log on connect, then streams live."""
        runh = MANAGER.get(run_id)
        if not runh:
            return {"error": "run not found"}

        def gen():
            idx = 0
            while True:
                events, idx = runh.events_since(idx)
                for ev in events:
                    yield f"id: {int(ev['ts']*1000)}\ndata: {json.dumps(ev)}\n\n"
                if runh.status in ("finished", "error") and idx >= len(runh._events):
                    break
                time.sleep(0.4)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/config")
    def config():
        available, err = [], None
        try:
            import ollama
            client = ollama.Client(host=OLLAMA_HOST)
            res = client.list()
            raw_models = getattr(res, "models", None)
            if raw_models is None and isinstance(res, dict):
                raw_models = res.get("models", [])
            raw_models = raw_models or []
            for m in raw_models:
                name = getattr(m, "model", None)
                if name is None and isinstance(m, dict):
                    name = m.get("model")
                if name:
                    available.append(str(name))
        except Exception as ex:  # noqa: BLE001
            err = str(ex)
        return {"model": MODEL, "host": OLLAMA_HOST, "max_steps": MAX_GLOBAL_STEPS,
                "models_available": available, "ollama_error": err}

    static_dir = pathlib.Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app
