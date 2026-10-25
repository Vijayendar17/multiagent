import shutil
import sys
import uuid
from pathlib import Path

from orchestrator import run
from state import TaskGraph


DEFAULT_GOAL = "Write a Python function that checks if a number is prime, with tests."


def main():
    goal = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")]) or DEFAULT_GOAL
    resume = "--resume" in sys.argv

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = data_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / "graph.sqlite"

    if not resume and db_path.exists():
        # Archive previous run to data/runs if not already there
        archive_id = uuid.uuid4().hex[:8]
        try:
            shutil.copy(str(db_path), str(runs_dir / f"prev_{archive_id}.sqlite"))
            db_path.unlink()
        except OSError:
            pass

    graph = TaskGraph(db_path=str(db_path))
    print(f"[MAIN] Run starts. Persisting state in {db_path}")

    # Track tasks for this run
    existing_ids = {t.id for t in graph.all_tasks()}
    graph = run(goal, graph)

    current_tasks = [t for t in graph.all_tasks() if t.id not in existing_ids or not existing_ids]

    print("\n=== Final task states ===")
    for t in current_tasks:
        print(f"  {t.agent:<9} {t.status:<16} attempts={t.attempts} | {t.goal[:60]}")
        if t.result:
            print(f"      result: {t.result[:200].replace(chr(10), ' ')}")
        if t.feedback and t.status in ("failed_permanent", "failed"):
            print(f"      feedback: {t.feedback[:200].replace(chr(10), ' ')}")

    verify_tasks = [t for t in current_tasks if t.agent == "verify"]
    done_verify = [t for t in verify_tasks if t.status == "done"]

    print("\n=== Verdict ===")
    if done_verify:
        print(f"  PASSED — {done_verify[-1].result}")
    elif any(t.status in ("failed", "failed_permanent") for t in current_tasks):
        print("  FAILED. Inspect feedback above.")
    else:
        print("  Incomplete or partial execution.")

    # Also archive this run into data/runs for the web dashboard to inspect
    run_archive = runs_dir / f"cli_{uuid.uuid4().hex[:8]}.sqlite"
    try:
        shutil.copy(str(db_path), str(run_archive))
    except OSError:
        pass


if __name__ == "__main__":
    main()
