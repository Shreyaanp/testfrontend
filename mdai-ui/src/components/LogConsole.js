import { useEffect, useRef } from 'react';
export default function LogConsole({ entries }) {
    const endRef = useRef(null);
    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [entries]);
    return (<div className="log-console">
      {entries.length === 0 ? (<div className="log-placeholder">Awaiting events…</div>) : (entries.map((entry) => (<div key={entry.id} className={`log-entry ${entry.level}`}>
            <span className="time">{new Date(entry.ts).toLocaleTimeString()}</span>
            <span className="message">{entry.message}</span>
          </div>)))}
      <div ref={endRef}/>
    </div>);
}
