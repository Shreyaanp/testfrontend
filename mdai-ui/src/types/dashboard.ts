export interface MetricsSnapshot {
  stability?: number
  focus?: number
  composite?: number
}

export type LogLevel = 'info' | 'error'

export interface LogEntry {
  id: string
  ts: number
  message: string
  level: LogLevel
}
