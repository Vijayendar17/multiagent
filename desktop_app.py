"""Desktop launcher — wraps the FastAPI dashboard in a native window via pywebview."""

import os
import socket
import sys
import threading
import time


PORT = int(os.environ.get("ORCH_PORT", "8247"))
SERVE_HOST = os.environ.get("ORCH_HOST", "127.0.0.1")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((SERVE_HOST, 0))
        return s.getsockname()[1]


def _start_server(port: int):
    import uvicorn
    from webapp import make_app

    uvicorn.run(make_app(), host=SERVE_HOST, port=port, log_level="warning")


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # --no-window -> run as plain web server (handy for testing in browser)
    if "--no-window" in sys.argv:
        port = PORT
        print(f"Dashboard: http://{SERVE_HOST}:{port}")
        _start_server(port)
        return

    try:
        import webview  # noqa: E402
    except ImportError:
        print("pywebview is not installed. Run:  pip install pywebview")
        print("Meanwhile, open the dashboard in a browser with: python desktop_app.py --no-window")
        sys.exit(1)

    port = _free_port()

    server_thread = threading.Thread(target=_start_server, args=(port,), daemon=True)
    server_thread.start()

    # Wait up to 5 seconds for uvicorn to bind and listen
    url = f"http://{SERVE_HOST}:{port}"
    for _ in range(50):
        try:
            with socket.create_connection((SERVE_HOST, port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)

    webview.create_window(
        "Multi-Agent Orchestrator", url, width=1200, height=820
    )
    webview.start()


if __name__ == "__main__":
    main()
