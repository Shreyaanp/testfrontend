# mdai-ui

React + Vite kiosk frontend that runs full-screen in Chromium and reacts to controller WebSocket events.

## Scripts

```bash
npm install
npm run dev    # http://localhost:3000
npm run build  # production bundle
```

Set environment variables when building:

```
VITE_CONTROLLER_WS_URL=ws://127.0.0.1:5000/ws/ui
VITE_PREVIEW_URL=http://127.0.0.1:5000/preview
VITE_CONTROLLER_HTTP_URL=http://127.0.0.1:5000
VITE_BACKEND_URL=https://mdai.mercle.ai
VITE_DEVICE_ID=alpha
```

## Architecture outline

- XState machine (`src/app-state/sessionMachine.ts`) mirrors controller phases and drives the UI stage router.
- `useControllerSocket` listens to the local controller WebSocket, forwards state into XState, and streams auxiliary events into the dashboard.
- `StageRouter` switches between black screen, QR prompt, instruction overlays, and error states over the RealSense preview area.
- The new `ControlPanel` surface (right column) exposes device/back-end info, manual ToF triggers, live metrics, and a timestamped event log.
- RealSense preview is embedded via iframe (`/preview`) and fades in only during capture-related phases, matching the kiosk controller stream.

## TODOs

- Add audible cues / animations for the stabilization countdown.
- Integrate localization and accessibility (screen reader hidden text for instructions).
