export type Level = 'low' | 'medium' | 'high' | 'critical'
export type Context = { category: string; prior_tickets?: number; prior_escalations?: number; tenure_days?: number }
export type Message = { role: 'customer' | 'agent'; text: string; at: string }
export type Prediction = {
  ticket_id: string; position: number; message: Message; risk: number; level: Level
  alert: boolean; alert_reason: string; action: string; delta: number; rapid_increase: boolean
  signals: string[]; contributions: { feature: string; delta: number }[]; version: string
}
export type Ticket = { id: string; context: Context; message_count: number; latest: Prediction | null; alerted: boolean }
export type Detail = { id: string; context: Context; history: Prediction[] }
export type Scenario = { name: string; context: Context; messages: Message[] }
export type Metrics = {
  tickets: number; positives: number; threshold: number; precision: number; recall: number; f1: number
  pr_auc: number; roc_auc: number; prefix_brier: number; alert_rate: number; false_alerts_per_100: number
  confusion_matrix: number[][]; lead_messages: number[]; lead_minutes: number[]; prefix_ece: number; detected_at_least: Record<string, number>
  alert_budget: { budget: number; precision: number; recall: number }[]
}
export type Evaluation = { model: { selected: string; version: string }; holdout: Record<string, Metrics>; inference_latency_ms: { p50: number; p95: number } }

export async function api<T>(path: string, body?: object): Promise<T> {
  const response = await fetch('/api' + path, body && {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status})`)
  }
  return response.json()
}
