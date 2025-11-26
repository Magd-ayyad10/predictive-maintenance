import subprocess
import time
import sys
import os

def run_services():
    """
    Launches both the FastAPI server and the Streamlit dashboard
    as separate subprocesses.
    """
    print("--- Starting Predictive Maintenance System ---")

    # 1. Start the FastAPI Server
    # We use 'uvicorn' to run the api_server file.
    # Command equivalent: uvicorn api_server:app --reload
    print(">> Launching API Server on port 8000...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_server:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=os.getcwd() # Ensure we run in the current directory
    )

    # Wait a moment for the API to initialize
    time.sleep(3)

   
    DASHBOARD_FILE = "app.py"  
    
    if not os.path.exists(DASHBOARD_FILE):
         # Fallback check for other common names used in our chat
         if os.path.exists("app.py"):
             DASHBOARD_FILE = "app.py"
         elif os.path.exists("ai4i_dashboard_final_notebook.py"):
             DASHBOARD_FILE = "ai4i_dashboard_final_notebook.py"
         else:
             print(f"Error: Could not find dashboard file '{DASHBOARD_FILE}'. Please rename it in main.py.")
             api_process.terminate()
             return

    print(f">> Launching Streamlit Dashboard ({DASHBOARD_FILE})...")
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", DASHBOARD_FILE],
        cwd=os.getcwd()
    )

    print("\n--- Services are Running ---")
    print(f"API Documentation: http://127.0.0.1:8000/docs")
    print(f"Dashboard URL:     Check the Streamlit output below")
    print("Press Ctrl+C to stop both services.\n")

    try:
        # Keep the main script alive while the subprocesses run
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\n--- Stopping Services ---")
        api_process.terminate()
        streamlit_process.terminate()
        print("Goodbye!")

if __name__ == "__main__":
    run_services()
