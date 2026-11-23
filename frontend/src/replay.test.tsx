// @vitest-environment jsdom
/// <reference types="vite/client" />
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Replay } from './App'
import { type Message, type Prediction, type Scenario } from './types'

vi.mock('recharts', () => {
  const Empty = () => null
  return { Bar: Empty, BarChart: Empty, CartesianGrid: Empty, Legend: Empty, Line: Empty,
    LineChart: Empty, ResponsiveContainer: Empty, Tooltip: Empty, XAxis: Empty, YAxis: Empty }
})

const scenario: Scenario = { name: 'Refund delay', context: { category: 'refund' }, messages: [
  { role: 'customer', text: 'Where is the refund?', at: '2025-02-01T05:00:00Z' },
  { role: 'agent', text: 'I will check.', at: '2025-02-01T05:06:00Z' },
] }
const result = (data: Message, position: number) => ({
  ticket_id: 'demo-ticket', message: data, position, risk: position * 0.3, level: position === 2 ? 'high' : 'low',
  action: position === 2 ? 'Recommend specialist' : 'Continue normal support',
  alert: position === 2, alert_reason: 'Test decision', delta: 0.3,
  signals: [], contributions: [], version: 'v1',
})
const reply = (body: unknown) => new Response(JSON.stringify(body), { status: 200 })
const next = () => screen.getByRole('button', { name: 'Replay next message' }) as HTMLButtonElement

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('Replay', () => {
  it('submits one message per click and renders the returned score and alert', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(reply({ id: 'demo-ticket', context: scenario.context }))
      .mockResolvedValueOnce(reply(result(scenario.messages[0], 1)))
      .mockResolvedValueOnce(reply(result(scenario.messages[1], 2)))
    vi.stubGlobal('fetch', fetcher)
    const refresh = vi.fn()
    render(<Replay scenarios={[scenario]} onChange={refresh} />)
    expect(fetcher).not.toHaveBeenCalled()
    fireEvent.click(next())
    await screen.findByText('Where is the refund?')
    expect(screen.queryByText('I will check.')).toBeNull()
    expect(JSON.parse(fetcher.mock.calls[1][1].body).message).toEqual(scenario.messages[0])
    fireEvent.click(next())
    await screen.findByText('Recommend specialist')
    expect(screen.getByText(/First alert at message 2/)).toBeTruthy()
    expect(next().disabled).toBe(true)
    expect(refresh).toHaveBeenCalledTimes(2)
    fireEvent.click(screen.getByRole('button', { name: 'Start new replay' }))
    expect(screen.queryByText('I will check.')).toBeNull()
    expect(next().disabled).toBe(false)
  })

  it('retries a failed message with the same ticket and idempotency key', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(reply({ id: 'demo-ticket', context: scenario.context }))
      .mockRejectedValueOnce(new Error('Network unavailable'))
      .mockResolvedValueOnce(reply(result(scenario.messages[0], 1)))
    vi.stubGlobal('fetch', fetcher)
    render(<Replay scenarios={[scenario]} onChange={() => {}} />)
    fireEvent.click(next())
    await screen.findByRole('alert')
    expect(screen.getByText('0 / 2 messages')).toBeTruthy()
    fireEvent.click(next())
    await waitFor(() => expect(screen.getByText('1 / 2 messages')).toBeTruthy())
    expect(fetcher.mock.calls[1][0]).toBe(fetcher.mock.calls[2][0])
    expect(fetcher.mock.calls[1][1].body).toBe(fetcher.mock.calls[2][1].body)
  })
})

const integrationOrigin = import.meta.env.VITE_TEST_API_ORIGIN

it.skipIf(!integrationOrigin)('renders real scores and persists replay through the API', async () => {
  const fetchReal = globalThis.fetch.bind(globalThis)
  const predictions: Prediction[] = []
  vi.stubGlobal('fetch', async (path: string, options?: RequestInit) => {
    const response = await fetchReal(integrationOrigin + path, options)
    if (path.endsWith('/messages') && response.ok) predictions.push(await response.clone().json())
    return response
  })
  const scenarios: Scenario[] = await (await fetchReal(integrationOrigin + '/api/demo')).json()
  render(<Replay scenarios={scenarios} onChange={() => {}} />)
  for (let i = 0; i < scenarios[0].messages.length; i++) {
    fireEvent.click(next())
    await screen.findByText(`${i + 1} / ${scenarios[0].messages.length} messages`)
  }
  expect(predictions.some(item => item.alert)).toBe(true)
  expect(new Set(predictions.map(item => item.risk)).size).toBeGreaterThan(1)
  const history: Prediction[] = await (await fetchReal(
    `${integrationOrigin}/api/tickets/${predictions[0].ticket_id}/risk-history`
  )).json()
  expect(history).toEqual(predictions)
})
