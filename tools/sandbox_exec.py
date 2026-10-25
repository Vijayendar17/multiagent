import os
import subprocess
import sys
import tempfile


# Optional hardening knobs — read from env if present so production can
# tighten this without code changes.
SANDBOX_TIMEOUT = float(os.environ.get("SANDBOX_TIMEOUT", "20"))
SANDBOX_CMD = [sys.executable]


def run_python_sandboxed(code: str, timeout: float = SANDBOX_TIMEOUT):
    """Run Python source in an isolated subprocess with a wall-clock timeout.

    Returns (stdout, stderr, ok: bool). A subprocess is only a crash boundary,
    not a real security boundary — for untrusted code, run inside a container
    (Docker/gVisor) with no network and a memory cap.
    """
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    try:
        tmp.write(code)
        tmp.close()
        proc = subprocess.run(
            SANDBOX_CMD + [tmp.name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = proc.returncode == 0
        stderr = proc.stderr
        if not ok and not stderr.strip():
            # If process crashed/exited with error but produced no stderr, provide context
            stderr = f"Process exited with non-zero exit code {proc.returncode}"
            if proc.stdout.strip():
                stderr += f"\nOutput before failure:\n{proc.stdout.strip()[-500:]}"
        return proc.stdout, stderr, ok
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"code execution timed out after {timeout}s")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
