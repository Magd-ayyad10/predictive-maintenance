import subprocess
import time
import sys
import os


def run_api_only():
    """Run ONLY the FastAPI server — used when MODE=api."""
    port = int(os.getenv("PORT", "10000"))
    print("=== MODE: API ONLY ===")
    print(f">> Launching API Server on port {port}...")

    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api_server:app", "--host", "0.0.0.0", "--port", str(port)
    ])


def run_dashboard_only():
    """Run ONLY the Streamlit dashboard — used when MODE=dashboard."""
    port = int(os.getenv("PORT", "10000"))
    print("=== MODE: DASHBOARD ONLY ===")

    DASHBOARD_FILE = "app.py"

    if not os.path.exists(DASHBOARD_FILE):
        if os.path.exists("ai4i_dashboard_final_notebook.py"):
            DASHBOARD_FILE = "ai4i_dashboard_final_notebook.py"
        else:
            print(f"Dashboard file '{DASHBOARD_FILE}' not found!")
            return

    print(f">> Launching Streamlit Dashboard: {DASHBOARD_FILE} on port {port}")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", DASHBOARD_FILE,
        f"--server.address=0.0.0.0",
        f"--server.port={port}"
    ])


def run_both_local():
    """Run BOTH FastAPI and Streamlit — only for local testing."""
    print("=== MODE: BOTH (LOCAL DEVELOPMENT ONLY) ===")

    api_port = 8000  # fixed local API port
    print(f">> Launching API Server on port {api_port}...")
    api_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "api_server:app", "--host", "0.0.0.0", "--port", str(api_port)
    ])

    time.sleep(2)

    DASHBOARD_FILE = "app.py"
    if not os.path.exists(DASHBOARD_FILE):
        if os.path.exists("ai4i_dashboard_final_notebook.py"):
            DASHBOARD_FILE = "ai4i_dashboard_final_notebook.py"
        else:
            print("Dashboard file not found!")
            api_process.terminate()
            return

    dashboard_port = 10000  # fixed local dashboard port
    print(f">> Launching Streamlit Dashboard: {DASHBOARD_FILE} on port {dashboard_port}")

    streamlit_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", DASHBOARD_FILE,
        f"--server.address=0.0.0.0",
        f"--server.port={dashboard_port}"
    ])

    try:
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
        api_process.terminate()
        streamlit_process.terminate()


if __name__ == "__main__":
    MODE = os.getenv("MODE", "dashboard").lower()

    if MODE == "api":
        run_api_only()

    elif MODE == "dashboard":
        run_dashboard_only()

    elif MODE == "both":
        run_both_local()

    else:
        print(f"Invalid MODE value: {MODE}")
        print("Use MODE=api or MODE=dashboard or MODE=both")
