import { useEffect, useMemo } from 'react'
import { useMachine } from '@xstate/react'
import StageRouter from './components/StageRouter'
import { sessionMachine } from './app-state/sessionMachine'
import { useControllerSocket } from './hooks/useControllerSocket'

const previewVisibleStates = new Set([
  'human_detect',
  'stabilizing',
  'uploading',
  'waiting_ack'
])

export default function App() {
  const [state, send] = useMachine(sessionMachine, {
    devTools: true
  })

  useControllerSocket(send)

  const showPreview = useMemo(() => previewVisibleStates.has(state.value as string), [state.value])

  useEffect(() => {
    document.body.style.backgroundColor = '#000'
    return () => {
      document.body.style.backgroundColor = ''
    }
  }, [])

  return (
    <div className="app-shell">
      <StageRouter state={state} />
      <iframe
        title="RealSense preview"
        className={`preview-frame ${showPreview ? 'visible' : 'hidden'}`}
        src={(import.meta.env.VITE_PREVIEW_URL as string | undefined) ?? 'http://127.0.0.1:5000/preview'}
        allow="autoplay"
      />
    </div>
  )
}
