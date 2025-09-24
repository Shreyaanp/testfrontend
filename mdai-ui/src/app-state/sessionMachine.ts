import { assign, createMachine } from 'xstate'

export type SessionPhase =
  | 'idle'
  | 'pairing_request'
  | 'qr_display'
  | 'waiting_activation'
  | 'human_detect'
  | 'stabilizing'
  | 'uploading'
  | 'waiting_ack'
  | 'complete'
  | 'error'

export interface SessionContext {
  pairingToken?: string
  expiresIn?: number
  error?: string
  lastHeartbeatTs?: number
}

export type SessionEvent =
  | {
      type: 'CONTROLLER_STATE'
      phase: SessionPhase
      data?: Record<string, unknown>
      error?: string
    }
  | { type: 'HEARTBEAT' }
  | { type: 'RESET' }

const assignPairingDetails = assign<SessionContext, SessionEvent>({
  pairingToken: (_ctx, event) =>
    event.type === 'CONTROLLER_STATE' && typeof event.data?.pairing_token === 'string'
      ? (event.data.pairing_token as string)
      : undefined,
  expiresIn: (_ctx, event) =>
    event.type === 'CONTROLLER_STATE' && typeof event.data?.expires_in === 'number'
      ? (event.data.expires_in as number)
      : undefined,
  error: (_ctx, event) => (event.type === 'CONTROLLER_STATE' ? event.error : undefined)
})

const assignError = assign<SessionContext, SessionEvent>({
  error: (_ctx, event) => (event.type === 'CONTROLLER_STATE' ? event.error : undefined)
})

const resetContext = assign<SessionContext, SessionEvent>({
  pairingToken: () => undefined,
  expiresIn: () => undefined,
  error: (_ctx, event) => (event.type === 'CONTROLLER_STATE' ? event.error : undefined)
})

const phaseGuard = (phase: SessionPhase) => {
  return (_ctx: SessionContext, event: SessionEvent) =>
    event.type === 'CONTROLLER_STATE' && event.phase === phase
}

export const sessionMachine = createMachine<SessionContext, SessionEvent>(
  {
    id: 'session',
    predictableActionArguments: true,
    initial: 'idle',
    context: {},
    on: {
      CONTROLLER_STATE: [
        { cond: phaseGuard('idle'), target: '.idle', actions: resetContext },
        { cond: phaseGuard('pairing_request'), target: '.pairing_request' },
        { cond: phaseGuard('qr_display'), target: '.qr_display', actions: assignPairingDetails },
        { cond: phaseGuard('waiting_activation'), target: '.waiting_activation' },
        { cond: phaseGuard('human_detect'), target: '.human_detect' },
        { cond: phaseGuard('stabilizing'), target: '.stabilizing' },
        { cond: phaseGuard('uploading'), target: '.uploading' },
        { cond: phaseGuard('waiting_ack'), target: '.waiting_ack' },
        { cond: phaseGuard('complete'), target: '.complete' },
        { cond: phaseGuard('error'), target: '.error', actions: assignError }
      ],
      HEARTBEAT: {
        actions: assign({ lastHeartbeatTs: () => Date.now() })
      },
      RESET: {
        target: '.idle',
        actions: assign({ pairingToken: undefined, expiresIn: undefined, error: undefined })
      }
    },
    states: {
      idle: {},
      pairing_request: {},
      qr_display: {},
      waiting_activation: {},
      human_detect: {},
      stabilizing: {},
      uploading: {},
      waiting_ack: {},
      complete: {},
      error: {}
    }
  }
)
