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
```

## Architecture outline

- XState machine (`src/app-state/sessionMachine.ts`) mirrors controller phases and drives the UI stage router.
- `useControllerSocket` listens to the local controller WebSocket and forwards events into XState.
- `StageRouter` switches between black screen, QR prompt, instruction overlays, and error states.
- RealSense preview is embedded via iframe (`/preview`) and fades in only during capture-related phases.

## TODOs

- Hook up countdown logic for `expires_in` to display real-time expiry.
- Add audible cues / animations for the stabilization countdown.
- Integrate localization and accessibility (screen reader hidden text for instructions).
