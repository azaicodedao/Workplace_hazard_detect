# Industrial Safety System

This system uses a pre-trained VideoMAE model to detect industrial hazards from uploaded videos, a computer camera, or a paired phone camera.

## Setup
1. Place your pre-trained model file in `models/videomae_hazard.pth`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python run.py`

## Video sources

- **Upload file** sends the original video to `POST /predict/`. The existing full-video frame extraction and uniform sampling are unchanged.
- **Computer camera** uses the browser camera and sends 16 JPEG frames captured from `0ms` through `5000ms` to `POST /predict/live/`. Live frames preserve up to 480p source resolution and use JPEG quality 0.92 before VideoMAE preprocessing.
- **Phone camera** pairs through a one-time QR code and streams to the dashboard with WebRTC. The dashboard applies the same five-second live capture pipeline.

The camera preview remains smooth while inference runs. Only one inference request runs at a time, and an older waiting window is replaced by the newest complete window to prevent latency buildup.

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
python -m unittest test_live_window.py test_camera_sessions.py test_camera_api.py test_network_address.py test_warmup_and_smoothing.py test_remote_inference.py
```
