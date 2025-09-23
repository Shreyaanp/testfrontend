# mdai controller

FastAPI-based supervisor that coordinates ToF triggers, RealSense capture, and backend pairing for the Jetson kiosk.

## Features (scaffold)

- `/healthz` liveness endpoint and `/preview` MJPEG feed for the kiosk iframe
- Local WebSocket (`/ws/ui`) broadcasting session phase updates to the React frontend
- Session manager that wires pairing HTTP calls, ECDSA nonce signing, backend WebSocket handling, and liveness capture hooks
- ToF polling abstraction with debounce and pluggable distance provider
- RealSense service wrapper that reuses the existing `d435i/mediapipe_liveness.py` pipeline when available and falls back to placeholder frames otherwise

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r controller/requirements.txt
uvicorn controller.app.main:app --reload --host 0.0.0.0 --port 5000
```

Populate the root `.env` with kiosk credentials:

```
DEVICE_ID=alpha
EVM_ADDRESS=0x1531541A8B1C447BC5Bb13DB94F43FE4bA316286
PRIVATE_KEY=0x14b8a7202980cab84d5e03b5f434849000f3f5702b326e6dee40d79a2e6254fd
PUBLIC_KEY=0xf93f9585a80491a36438015197dc05ebadf259620f3973f8a8c4bcb51dd1b6508c2f84880694de3870e4209facb8ab3ad85ce158e7ac934953e7db86dbb5b895
BACKEND_API_URL=https://mdai.mercle.ai
BACKEND_WS_URL=wss://mdai.mercle.ai/ws
```

(Existing `VITE_` entries remain for the React build.)

## Next integration steps

1. Replace `mock_distance_provider` with the actual ToF driver and feed noise-filtered distances.
2. Enable hardware capture in `RealSenseService` by ensuring the Jetson has the `pyrealsense2`, `mediapipe`, and `opencv-python` builds installed.
3. Flesh out backend WebSocket handling (`hardware_register`, richer session messages) once backend contract is final.
4. Add automated tests for session transitions (happy path, cancellation, error handling) using `asyncio` fakes.
