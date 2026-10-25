import json
import os
import shutil
import tempfile
import time
import unittest

from llm import _extract_json
from state import Task, TaskGraph
from tools.search_tool import _search_local, local_or_web_search
from tools.sandbox_exec import run_python_sandboxed
from agents.code_agent import CodeAgent, _clean_code
from agents.research_agent import ResearchAgent
from agents.verifier_agent import VerifierAgent, _as_bool
from orchestrator import run, signature, _inject_strategy_change
from webapp import RunManager, RunHandle


class TestLLM(unittest.TestCase):
    def test_extract_json_plain(self):
        data = _extract_json('{"foo": "bar", "num": 42}')
        self.assertEqual(data, {"foo": "bar", "num": 42})

    def test_extract_json_fenced(self):
        text = '```json\n{"code": "print(1)", "explanation": "test"}\n```'
        data = _extract_json(text)
        self.assertEqual(data, {"code": "print(1)", "explanation": "test"})

    def test_extract_json_fenced_with_preamble(self):
        text = 'Sure! Here is the output:\n```json\n{"status": "ok"}\n```\nHope that helps!'
        data = _extract_json(text)
        self.assertEqual(data, {"status": "ok"})

    def test_extract_json_embedded_backticks(self):
        text = '```json\n{"code": "x = `echo hi`"}\n```'
        data = _extract_json(text)
        self.assertEqual(data["code"], "x = `echo hi`")

    def test_extract_json_unfenced_with_chatter(self):
        text = 'Output is: {"pass": true, "score": 9} and nothing else'
        data = _extract_json(text)
        self.assertEqual(data, {"pass": True, "score": 9})

    def test_extract_json_invalid(self):
        with self.assertRaises(Exception):
            _extract_json('Not a json at all')


class TestState(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.sqlite")
        self.graph = TaskGraph(db_path=self.db_path)

    def tearDown(self):
        self.graph.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_and_get_task(self):
        t = self.graph.add(goal="Do something", agent="code")
        fetched = self.graph.get(t.id)
        self.assertEqual(fetched.id, t.id)
        self.assertEqual(fetched.goal, "Do something")
        self.assertEqual(fetched.agent, "code")
        self.assertEqual(fetched.status, "pending")

    def test_context_for_missing_parent(self):
        t = self.graph.add(goal="Child", agent="code", parent_id="nonexistent")
        ctx = self.graph.context_for(t)
        self.assertEqual(ctx, "")

    def test_context_for_existing_parent(self):
        parent = self.graph.add(goal="Parent", agent="research")
        parent.result = "Parent findings"
        self.graph.update(parent)

        child = self.graph.add(goal="Child", agent="code", parent_id=parent.id)
        ctx = self.graph.context_for(child)
        self.assertIn("Parent findings", ctx)

    def test_tolerance_to_unknown_fields(self):
        # Simulate an external or updated DB with an extra field in the JSON
        payload = json.dumps({
            "id": "t123", "goal": "g", "agent": "code", "status": "pending",
            "extra_future_field": "future_val"
        })
        self.graph.conn.execute("INSERT INTO tasks (id, data) VALUES ('t123', ?)", (payload,))
        self.graph.conn.commit()

        t = self.graph.get("t123")
        self.assertEqual(t.id, "t123")


class TestTools(unittest.TestCase):
    def test_clean_code(self):
        fenced = "```python\ndef foo():\n    return 42\n```"
        self.assertEqual(_clean_code(fenced), "def foo():\n    return 42")

        plain = "def foo():\n    return 42"
        self.assertEqual(_clean_code(plain), plain)

    def test_sandbox_exec_success(self):
        stdout, stderr, ok = run_python_sandboxed("print('hello world')", timeout=5)
        self.assertTrue(ok)
        self.assertIn("hello world", stdout)
        self.assertEqual(stderr, "")

    def test_sandbox_exec_failure(self):
        stdout, stderr, ok = run_python_sandboxed("raise ValueError('broken')", timeout=5)
        self.assertFalse(ok)
        self.assertIn("ValueError: broken", stderr)

    def test_sandbox_exec_timeout(self):
        with self.assertRaises(TimeoutError):
            run_python_sandboxed("import time; time.sleep(10)", timeout=0.5)

    def test_search_local_ignores_binary(self):
        temp_dir = tempfile.mkdtemp()
        pyc_path = os.path.join(temp_dir, "module.cpython-313.pyc")
        with open(pyc_path, "wb") as f:
            f.write(b"\x00\x00\x00\x00bytecode_garbage")
        txt_path = os.path.join(temp_dir, "readme.txt")
        with open(txt_path, "w") as f:
            f.write("A primer on sorting algorithms and prime testing.")

        cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            hits = _search_local("prime testing")
            paths = [h[0] for h in hits]
            self.assertNotIn(pyc_path, paths)
            self.assertIn(txt_path, paths)
        finally:
            os.chdir(cwd)
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestAgents(unittest.TestCase):
    def test_verifier_bool_normalization(self):
        self.assertTrue(_as_bool(True))
        self.assertTrue(_as_bool("True"))
        self.assertTrue(_as_bool("yes"))
        self.assertTrue(_as_bool("pass"))
        self.assertFalse(_as_bool(False))
        self.assertFalse(_as_bool("false"))
        self.assertFalse(_as_bool("no"))

    def test_verifier_execute(self):
        va = VerifierAgent()
        task = Task(id="1", goal="test", agent="verify")

        out = va.execute(task, {"pass": True, "score": 9, "feedback": "good"})
        self.assertEqual(out["status"], "done")
        self.assertIn("score=9 pass=True", out["result"])

        out_fail = va.execute(task, {"passed": False, "score": 3, "feedback": "bug"})
        self.assertEqual(out_fail["status"], "needs_fix")
        self.assertEqual(out_fail["feedback"], "bug")

        # Implicit pass based on score >= 7
        out_implicit = va.execute(task, {"score": 8, "feedback": "nice"})
        self.assertEqual(out_implicit["status"], "done")

    def test_research_execute(self):
        ra = ResearchAgent()
        task = Task(id="1", goal="Search sorting algorithms", agent="research")
        out = ra.execute(task, {"query": "sorting algorithms", "notes": "bubble, merge, quick"})
        self.assertEqual(out["status"], "done")
        self.assertIn("Agent notes: bubble, merge, quick", out["result"])


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "orch_test.sqlite")
        self.graph = TaskGraph(db_path=self.db_path)

    def tearDown(self):
        self.graph.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cycle_detection_signature(self):
        task = Task(id="t1", goal="Calculate fibonacci", agent="code")
        sig1 = signature(task, {"code": "print(1)"})
        sig2 = signature(task, {"code": "print(1)"})
        sig3 = signature(task, {"code": "print(2)"})

        self.assertEqual(sig1, sig2)
        self.assertNotEqual(sig1, sig3)

        # Confirm injecting strategy change does NOT break cycle detection
        _inject_strategy_change(task)
        sig4 = signature(task, {"code": "print(1)"})
        self.assertEqual(sig1, sig4)
        self.assertIn("stop repeating yourself", task.feedback)

    def test_simulated_pipeline(self):
        """Simulate orchestrator flow with mocked agents without calling Ollama."""
        events = []
        def mock_emit(ev, payload):
            events.append((ev, payload))

        # We can test orchestrator logic using deterministic agent mocks
        from unittest.mock import patch

        research_mock = {"query": "test query", "notes": "some notes"}
        code_mock = {"code": "print('prime: 2, 3, 5')", "explanation": "test"}
        verify_mock = {"pass": True, "score": 10, "feedback": "All good"}

        def mock_call_llm(system, user, expect_json=True, **kwargs):
            if "Research Agent" in system:
                return research_mock
            if "Code Agent" in system:
                return code_mock
            if "Verifier Agent" in system:
                return verify_mock
            return {}

        with patch("agents.base.call_llm", side_effect=mock_call_llm):
            graph = run("Find prime numbers", self.graph, emit=mock_emit)

        tasks = graph.all_tasks()
        self.assertEqual(len(tasks), 3)
        self.assertTrue(all(t.status == "done" for t in tasks))

        # Ensure task_updated was emitted with task dict
        task_updates = [p for e, p in events if e == "task_updated"]
        self.assertTrue(len(task_updates) > 0)
        self.assertIn("id", task_updates[0])
        self.assertIn("status", task_updates[0])

    def test_verifier_needs_fix_loop(self):
        """Simulate verification failure triggering code fix, then verification passing."""
        from unittest.mock import patch

        verify_calls = 0
        def mock_call_llm(system, user, expect_json=True, **kwargs):
            nonlocal verify_calls
            if "Research Agent" in system:
                return {"query": "q", "notes": "n"}
            if "Code Agent" in system:
                code = "print('v2')" if "Feedback" in user else "print('v1')"
                return {"code": code, "explanation": "e"}
            if "Verifier Agent" in system:
                verify_calls += 1
                if verify_calls == 1:
                    return {"pass": False, "score": 2, "feedback": "Tests failed on edge case"}
                return {"pass": True, "score": 10, "feedback": "All passed"}
            return {}

        with patch("agents.base.call_llm", side_effect=mock_call_llm):
            graph = run("Implement sorting", self.graph)

        tasks = graph.all_tasks()
        code_t = next(t for t in tasks if t.agent == "code")
        verify_t = next(t for t in tasks if t.agent == "verify")

        self.assertEqual(code_t.attempts, 2)
        self.assertEqual(code_t.status, "done")
        self.assertEqual(verify_t.status, "done")
        self.assertIn("score=10 pass=True", verify_t.result)

    def test_cascading_permanent_failure(self):
        """If research fails permanently, code and verify cascade to failed_permanent."""
        from unittest.mock import patch

        def mock_call_llm(system, user, expect_json=True, **kwargs):
            raise RuntimeError("Model unrecoverable error")

        with patch("agents.base.call_llm", side_effect=mock_call_llm):
            graph = run("Implement unreachable", self.graph)

        tasks = graph.all_tasks()
        research_t = next(t for t in tasks if t.agent == "research")
        code_t = next(t for t in tasks if t.agent == "code")
        verify_t = next(t for t in tasks if t.agent == "verify")

        self.assertEqual(research_t.status, "failed_permanent")
        self.assertEqual(code_t.status, "failed_permanent")
        self.assertEqual(verify_t.status, "failed_permanent")
        self.assertIn("abandoned: parent task", code_t.feedback)
        self.assertIn("abandoned: parent task", verify_t.feedback)

    def test_hitl_decide_retry_resets_attempts(self):
        from unittest.mock import patch

        decide_called = []
        def mock_decide(task, reason):
            decide_called.append((task.id, reason))
            return {"action": "retry", "feedback": "try simpler logic"}

        # Simulate code task failing max attempts
        task = self.graph.add(goal="Difficult task", agent="code", max_attempts=2)
        task.attempts = 2

        with patch("agents.base.call_llm", return_value={"code": "broken", "explanation": ""}):
            with patch("tools.sandbox_exec.run_python_sandboxed", return_value=("", "syntax error", False)):
                from orchestrator import _abandon
                action = _abandon(task, self.graph, "exceeded max attempts", decide=mock_decide)

        self.assertEqual(action, "retry")
        self.assertEqual(len(decide_called), 1)
        self.assertEqual(task.status, "pending")
        self.assertEqual(task.attempts, 0)
        self.assertIn("try simpler logic", task.feedback)


class TestWebapp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = RunManager(data_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_existing_runs(self):
        # Create a run database on disk
        db_file = os.path.join(self.temp_dir, "testrun1.sqlite")
        g = TaskGraph(db_path=db_file)
        t = g.add(goal="Research context for: Test Goal", agent="research")
        t.status = "done"
        g.update(t)
        g.close()

        # Reload manager
        mgr2 = RunManager(data_dir=self.temp_dir)
        self.assertIn("testrun1", mgr2.runs)
        runh = mgr2.get("testrun1")
        self.assertEqual(runh.goal, "Test Goal")
        self.assertEqual(runh.status, "finished")

    def test_hitl_event_wait(self):
        runh = RunHandle(run_id="r1", goal="Test", db_path="dummy")
        # In a background thread, resolve after 0.1s
        def resolver():
            time.sleep(0.1)
            runh.resolve_decision({"action": "retry", "feedback": "good"})

        import threading
        t = threading.Thread(target=resolver)
        t.start()

        decision = runh.decide(None, "reason")
        t.join()

        self.assertEqual(decision["action"], "retry")
        self.assertEqual(decision["feedback"], "good")


if __name__ == "__main__":
    unittest.main()
