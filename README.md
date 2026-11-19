# Industrial Safety System

This system uses a pre-trained VideoMAE model to detect industrial hazards from uploaded videos, a computer camera, or a paired phone camera.

## Setup
1. Place your pre-trained model file in `models/videomae_hazard.pth`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python run.py`

## Video sources

- **Upload file** calls `POST /predict/timeline/stream` and consumes an NDJSON stream, so each 5-second VideoMAE window appears as soon as its GPU inference completes. Windows start every 4 seconds, leaving 1 second of overlap to protect chunk boundaries. Each window contains 16 uniformly sampled JPEG frames resized to at most 360p at quality 0.92. At the same time, the local YOLO Pose service tracks people across the uploaded video at 10 sampled FPS and returns timestamped fall events. There is no second detailed scan. The final result combines VideoMAE hazard ranges with YOLO Pose fall ranges. JSON endpoints `POST /predict/timeline/` and `POST /predict/` remain available for backward compatibility.
- **Computer camera** continuously maintains a five-second rolling buffer, sampled as 16 JPEG frames at up to 360p and quality 0.92. The target stride is four seconds; a new window is sent to `POST /predict/live/` only while the GPU is idle, and stale windows are never queued.
- **Phone camera** pairs through a one-time QR code and streams to the dashboard with WebRTC. The dashboard applies the same rolling-buffer pipeline.

The camera preview remains smooth while inference runs. Only one VideoMAE request runs at a time; while the GPU is busy the rolling buffer keeps changing, but no stale window is queued.

## Real-time YOLO Pose tracking

YOLO Pose runs as an isolated Python 3.11 service on port `8001`. The main app proxies `/ws/pose` over its existing HTTP/HTTPS origin, so computer and phone modes work without mixed-content errors.

Create the environment once:

```powershell
py -3.11 -m venv .venv-pose
.\.venv-pose\Scripts\python.exe -m pip install --upgrade pip
.\.venv-pose\Scripts\python.exe -m pip install -r pose_service\requirements.txt
```

Normally only one command is needed; `run.py` starts and stops the pose service automatically:

```powershell
py run.py
```

For troubleshooting, disable `POSE_AUTO_START` and run the pose service manually in a separate window with `.\.venv-pose\Scripts\python.exe -m pose_service.run`.

The first pose-service start downloads `yolo26n-pose.pt`. The browser sends at most one outstanding 640px JPEG at a time, targeting 10 FPS without creating a tracking backlog. Each camera stream gets fresh tracker state. The overlay shows skeletons, bounding boxes, track IDs, and the heuristic states `STANDING`, `FALLING`, or `FALLEN`.

Default pose settings can be overridden in `.env`:

```env
POSE_TRACKING_ENABLED=true
POSE_AUTO_START=true
POSE_SERVICE_WS_URL=ws://127.0.0.1:8001/ws/pose
POSE_MODEL=yolo26n-pose.pt
POSE_TRACKER=bytetrack.yaml
POSE_DEVICE=cpu
POSE_IMAGE_SIZE=640
POSE_CONFIDENCE=0.35
```

The fall state is temporal logic over tracked keypoints, not a separately trained fall classifier. Validate and tune its thresholds using the actual camera angles before treating it as a safety alert.

## Kaggle GPU inference

The browser always calls the local FastAPI server. In `remote` mode, the local
server securely adds the bearer token and proxies uploads to the Kaggle GPU API;
the token is never exposed to frontend JavaScript.

Create `.env` from `.env.example` and set:

```env
INFERENCE_BACKEND=remote
REMOTE_INFERENCE_URL=https://your-current-ngrok-domain
REMOTE_INFERENCE_TOKEN=the-same-token-stored-in-kaggle
REMOTE_INFERENCE_TIMEOUT_SECONDS=180
```

Check the connection at `GET /api/inference/health`. A Kaggle restart creates a
new ngrok URL, so update `REMOTE_INFERENCE_URL` and restart the local server.
Use `INFERENCE_BACKEND=local` to load and run the model on the local machine.

## Phone camera setup

Phone browsers require a secure context for camera access. Configure an HTTPS URL that the phone can reach:

The server now detects the active LAN IPv4 address automatically. When HTTPS is enabled it advertises the stable `industrial-safety.local` hostname over mDNS, so changing Wi-Fi networks does not require editing an IP address. `PUBLIC_BASE_URL` is optional and only acts as an explicit override.

```powershell
$env:PHONE_CAMERA_HOSTNAME="industrial-safety.local"
$env:SSL_CERTFILE="C:\path\to\certificate.pem"
$env:SSL_KEYFILE="C:\path\to\private-key.pem"
python run.py
```

Open the same HTTPS address on the computer, select **Phone camera**, create a connection, and scan the QR code. The phone must trust the development certificate. For connections across different networks, configure a TURN server through `RTC_ICE_SERVERS_JSON`.

For local development, generate the project CA and server certificate once:

```powershell
python scripts\generate_https_cert.py
python run.py
```

`run.py` automatically enables HTTPS when `certs/industrial-safety-cert.pem` and `certs/industrial-safety-key.pem` exist. Install `frontend/certs/industrial-safety-ca.cer` as a trusted CA on each phone once. Private keys and generated certificates are excluded from Git.

## Tests

```powershell
python -m unittest test_live_window.py test_camera_sessions.py test_camera_api.py test_network_address.py test_warmup_and_smoothing.py test_remote_inference.py test_video_timeline.py test_timeline_api.py test_pose_fall_state.py
```
