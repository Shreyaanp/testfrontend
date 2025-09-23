import { useEffect, useRef } from 'react'
import type { Sender } from 'xstate'
import type { SessionEvent, SessionPhase } from '../app-state/sessionMachine'

const DEFAULT_WS_URL = 'ws://127.0.0.1:5000/ws/ui'

interface ControllerMessage {
  type: string
  phase?: string
  data?: Record<string, unknown>
  error?: string
}

export function useControllerSocket(send: Sender<SessionEvent>) {
  const retryRef = useRef<number | null>(null)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const wsUrl = (import.meta.env.VITE_CONTROLLER_WS_URL as string | undefined) ?? DEFAULT_WS_URL
    let cancelled = false

    const connect = () => {
      if (cancelled) return
      const socket = new WebSocket(wsUrl)
      socketRef.current = socket

      socket.onopen = () => {
        if (retryRef.current) {
          window.clearTimeout(retryRef.current)
          retryRef.current = null
        }
      }

      socket.onmessage = (event) => {
        try {
          const message: ControllerMessage = JSON.parse(event.data)
          if (message.type === 'heartbeat') {
            send({ type: 'HEARTBEAT' })
            return
          }
          if (message.type === 'state' && typeof message.phase === 'string') {
            send({
              type: 'CONTROLLER_STATE',
              phase: message.phase as SessionPhase,
              data: message.data,
              error: message.error
            })
          }
        } catch (err) {
          console.error('Failed to parse controller message', err)
        }
      }

      socket.onclose = () => {
        if (cancelled) return
        retryRef.current = window.setTimeout(connect, 2000)
      }

      socket.onerror = () => {
        socket.close()
      }
    }

    connect()

    return () => {
      cancelled = true
      if (retryRef.current) {
        window.clearTimeout(retryRef.current)
      }
      socketRef.current?.close()
      socketRef.current = null
    }
  }, [send])
}
