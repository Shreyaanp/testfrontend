import { assign, createMachine } from 'xstate';
const assignPairingDetails = assign({
    pairingToken: (_ctx, event) => event.type === 'CONTROLLER_STATE' && typeof event.data?.pairing_token === 'string'
        ? event.data.pairing_token
        : undefined,
    expiresIn: (_ctx, event) => event.type === 'CONTROLLER_STATE' && typeof event.data?.expires_in === 'number'
        ? event.data.expires_in
        : undefined,
    error: (_ctx, event) => (event.type === 'CONTROLLER_STATE' ? event.error : undefined)
});
const assignError = assign({
    error: (_ctx, event) => (event.type === 'CONTROLLER_STATE' ? event.error : undefined)
});
const resetContext = assign({
    pairingToken: () => undefined,
    expiresIn: () => undefined,
    error: (_ctx, event) => (event.type === 'CONTROLLER_STATE' ? event.error : undefined)
});
const phaseGuard = (phase) => {
    return (_ctx, event) => event.type === 'CONTROLLER_STATE' && event.phase === phase;
};
export const sessionMachine = createMachine({
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
});
