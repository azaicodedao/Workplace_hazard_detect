import os
import socket
import subprocess
import time
import json
from urllib.request import urlopen
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


load_dotenv()


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _pose_service_supports_upload() -> bool:
    """Reject a stale pose process that predates uploaded-video analysis."""
    try:
        with urlopen("http://127.0.0.1:8001/openapi.json", timeout=1.0) as response:
            document = json.load(response)
        return "/analyze/video" in document.get("paths", {})
    except Exception:
        return False


def _start_pose_service(project_root: Path):
    enabled = os.getenv("POSE_TRACKING_ENABLED", "true").lower() in {
        "1", "true", "yes", "on"
    }
    auto_start = os.getenv("POSE_AUTO_START", "true").lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled or not auto_start:
        return None
    if _port_is_open("127.0.0.1", 8001):
        if _pose_service_supports_upload():
            print("Using the existing YOLO Pose service on http://127.0.0.1:8001")
        else:
            print(
                "Port 8001 is occupied by an outdated or incompatible service. "
                "Stop that process and restart py run.py."
            )
        return None

    pose_python = project_root / ".venv-pose" / "Scripts" / "python.exe"
    if not pose_python.exists():
        print("YOLO Pose is enabled but .venv-pose is missing. See README setup instructions.")
        return None

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        [str(pose_python), "-m", "pose_service.run"],
        cwd=str(project_root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    for _ in range(120):
        if process.poll() is not None:
            print("YOLO Pose service exited during startup.")
            return None
        if _port_is_open("127.0.0.1", 8001) and _pose_service_supports_upload():
            print(f"YOLO Pose service ready on http://127.0.0.1:8001 (PID {process.pid})")
            return process
        time.sleep(0.25)
    print("YOLO Pose service did not open port 8001 within 30 seconds.")
    process.terminate()
    return None

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    pose_process = _start_pose_service(project_root)
    default_key = project_root / "certs" / "industrial-safety-key.pem"
    default_cert = project_root / "certs" / "industrial-safety-cert.pem"
    ssl_keyfile = os.getenv("SSL_KEYFILE") or (str(default_key) if default_key.exists() else None)
    ssl_certfile = os.getenv("SSL_CERTFILE") or (str(default_cert) if default_cert.exists() else None)
    scheme = "https" if ssl_keyfile and ssl_certfile else "http"
    print(f"Starting Industrial Safety System on {scheme}://0.0.0.0:{os.getenv('PORT', '8000')}")
    try:
        uvicorn.run(
            "app.main:app",
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            reload=os.getenv("RELOAD", "true").lower() == "true",
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )
    finally:
        if pose_process is not None and pose_process.poll() is None:
            pose_process.terminate()
            try:
                pose_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pose_process.kill()
