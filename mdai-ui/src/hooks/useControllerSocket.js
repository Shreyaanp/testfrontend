import { useEffect, useRef } from 'react';
const DEFAULT_WS_URL = 'ws://127.0.0.1:5000/ws/ui';
export function useControllerSocket(send, options) {
    const retryRef = useRef(null);
    const socketRef = useRef(null);
    useEffect(() => {
        const wsUrl = import.meta.env.VITE_CONTROLLER_WS_URL ?? DEFAULT_WS_URL;
        let cancelled = false;
        const connect = () => {
            if (cancelled)
                return;
            options?.onStatusChange?.('connecting');
            const socket = new WebSocket(wsUrl);
            socketRef.current = socket;
            socket.onopen = () => {
                if (retryRef.current) {
                    window.clearTimeout(retryRef.current);
                    retryRef.current = null;
                }
                options?.onStatusChange?.('open');
            };
            socket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    options?.onEvent?.(message);
                    if (message.type === 'heartbeat') {
                        send({ type: 'HEARTBEAT' });
                        return;
                    }
                    if (message.type === 'state' && typeof message.phase === 'string') {
                        send({
                            type: 'CONTROLLER_STATE',
                            phase: message.phase,
                            data: message.data,
                            error: message.error
                        });
                    }
                }
                catch (err) {
                    console.error('Failed to parse controller message', err);
                }
            };
            socket.onclose = () => {
                if (cancelled)
                    return;
                retryRef.current = window.setTimeout(connect, 2000);
                options?.onStatusChange?.('closed');
            };
            socket.onerror = () => {
                socket.close();
            };
        };
        connect();
        return () => {
            cancelled = true;
            if (retryRef.current) {
                window.clearTimeout(retryRef.current);
            }
            socketRef.current?.close();
            socketRef.current = null;
            options?.onStatusChange?.('closed');
        };
    }, [send, options]);
}
