import { useEffect, useRef, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type Detail, type Evaluation, type Prediction, type Scenario, type Ticket } from './types'

const percent = (n: number) => `${Math.round(n * 100)}%`
const decimal = (n: number) => n.toFixed(3)
const LEVELS = ['low', 'medium', 'high', 'critical'] as const
const caption = (value: string) => value.replaceAll('_', ' ')

function RiskChart({ history }: { history: Prediction[] }) {
  return <div className="chart" aria-label="Risk by message">
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={history}>
        <CartesianGrid stroke="#e6e8eb" vertical={false} />
        <XAxis dataKey="position" allowDecimals={false} />
        <YAxis domain={[0, 1]} tickFormatter={percent} width={52} />
        <Tooltip formatter={(value) => percent(Number(value))} />
        <Line dataKey="risk" name="Escalation risk" stroke="#166855" strokeWidth={3} dot={{ r: 4 }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  </div>
}

function Conversation({ detail }: { detail: Detail }) {
  const recent = detail.history.at(-1)
  const first = detail.history.find(item => item.alert)
  return <section className="panel">
    <div className="heading"><div><h2>{caption(detail.context.category)} conversation</h2><p className="muted">{detail.id.slice(0, 8)} · {detail.history.length} messages</p></div>
      {recent && <span className={`pill ${recent.level}`}>{recent.level} · {percent(recent.risk)}</span>}</div>
    {!recent ? <p className="muted">No messages yet.</p> : <>
      <p className="advice"><strong>{recent.action}</strong><br />{recent.alert_reason} · {recent.delta >= 0 ? '+' : ''}{(recent.delta * 100).toFixed(1)} pp since last message{recent.rapid_increase && ' · Rapid increase'}{first && ` · First alert at message ${first.position}`}</p>
      <RiskChart history={detail.history} />
      <div className="split">
        <div><h3>Conversation timeline</h3>
          {detail.history.map(item => <article className={`message ${item.message.role}`} key={item.position}>
            <div className="heading"><strong>{item.message.role}</strong><small>#{item.position} · {percent(item.risk)}{item.alert && ' · Alert'}</small></div>
            <p>{item.message.text}</p><time dateTime={item.message.at}>{new Date(item.message.at).toLocaleString()}</time>
          </article>)}
        </div>
        <aside><h3>Observed signals</h3>
          {recent.signals.length ? <ul>{recent.signals.map(signal => <li key={signal}>{signal}</li>)}</ul> : <p className="muted">No supporting signals detected yet.</p>}
          <h3>Model sensitivity</h3><p className="muted">Risk difference from replacing one feature at a time with its training median.</p>
          <ul>{recent.contributions.map(item => <li key={item.feature}>{caption(item.feature)} · {item.delta > 0 ? '+' : ''}{(item.delta * 100).toFixed(2)} pp</li>)}</ul>
          <small>Model {recent.version}</small>
        </aside>
      </div>
    </>}
  </section>
}

export function Replay({ scenarios, onChange }: { scenarios: Scenario[]; onChange: () => void }) {
  const [selection, setSelection] = useState('0')
  const [detail, setDetail] = useState<Detail>()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const pendingId = useRef('')
  const scenario = scenarios[Number(selection)]
  const progress = detail?.history.length ?? 0

  async function next() {
    if (!scenario || busy || progress >= scenario.messages.length) return
    setBusy(true)
    setError('')
    try {
      let active = detail
      if (!active) {
        const ticket = await api<Omit<Detail, 'history'>>('/tickets', scenario.context)
        active = { ...ticket, history: [] }
        setDetail(active)
      }
      pendingId.current ||= crypto.randomUUID()
      const prediction = await api<Prediction>(`/tickets/${active.id}/messages`, {
        message_id: pendingId.current, message: scenario.messages[progress],
      })
      pendingId.current = ''
      setDetail({ ...active, history: [...active.history, prediction] })
      onChange()
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(false)
    }
  }

  function reset() {
    pendingId.current = ''
    setDetail(undefined)
    setError('')
  }

  return <>
    <section className="panel">
      <div className="heading"><div><h2>Live conversation replay</h2><p className="muted">Send one message at a time through the scoring API.</p></div></div>
      <div className="controls">
        <label>Scenario<select aria-label="Scenario" value={selection} disabled={busy} onChange={e => { setSelection(e.target.value); reset() }}>
          {scenarios.map((sample, i) => <option key={sample.name} value={i}>{sample.name}</option>)}
        </select></label>
        <button className="primary" disabled={busy || !scenario || progress === scenario.messages.length} onClick={next}>{busy ? 'Scoring…' : 'Replay next message'}</button>
        <button disabled={busy || !detail} onClick={reset}>Start new replay</button>
        <span className="muted">{progress} / {scenario?.messages.length ?? 0} messages</span>
      </div>
      {error && <p role="alert">{error}</p>}
    </section>
    {detail && <Conversation detail={detail} />}
  </>
}

function Performance({ evaluation }: { evaluation: Evaluation }) {
  const model = evaluation.holdout[evaluation.model.selected]
  const counts = Object.entries(model.detected_at_least).map(([position, tickets]) => ({ position, tickets }))
  const leadCounts = [...new Set(model.lead_messages)].sort((a, b) => a - b)
    .map(position => ({ position, tickets: model.lead_messages.filter(lead => lead === position).length }))
  const sortedMinutes = [...model.lead_minutes].sort((a, b) => a - b)
  const medianLead = sortedMinutes.length ? (sortedMinutes[Math.floor((sortedMinutes.length - 1) / 2)]
    + sortedMinutes[Math.floor(sortedMinutes.length / 2)]) / 2 : null
  return <>
    <section className="panel">
      <div className="heading"><div><h2>Offline holdout evaluation</h2><p className="muted">{model.tickets} synthetic tickets · {model.positives} escalations · Selected on validation</p></div>
      <span className="pill low">{caption(evaluation.model.selected)}</span></div>
      <div className="table-wrap"><table><thead><tr><th>Model / ablation</th><th>Precision</th><th>Recall</th><th>F1</th><th>PR-AUC</th><th>ROC-AUC</th><th>Brier</th><th>False alerts / 100</th></tr></thead><tbody>
        {Object.entries(evaluation.holdout).map(([name, item]) => <tr key={name}>
          <td>{caption(name)}</td><td>{percent(item.precision)}</td><td>{percent(item.recall)}</td><td>{decimal(item.f1)}</td>
          <td>{decimal(item.pr_auc)}</td><td>{decimal(item.roc_auc)}</td><td>{decimal(item.prefix_brier)}</td><td>{item.false_alerts_per_100.toFixed(1)}</td>
        </tr>)}
      </tbody></table></div>
      <p className="muted">Selected model warm CPU scoring p50 {evaluation.inference_latency_ms.p50.toFixed(1)} ms · p95 {evaluation.inference_latency_ms.p95.toFixed(1)} ms</p>
      <p className="muted">Prefix calibration error {decimal(model.prefix_ece)}</p><h3>Confusion matrix</h3>
      <table><thead><tr><th>Observed</th><th>No alert</th><th>Alert</th></tr></thead><tbody>
        {model.confusion_matrix.map((r, i) => <tr key={i}><th>{i ? 'Escalated' : 'No escalation'}</th><td>{r[0]}</td><td>{r[1]}</td></tr>)}
      </tbody></table>
    </section>
    <div className="split">
      <section className="panel"><h2>Early detection</h2><p className="muted">{counts.map(item => `≥${item.position} messages: ${item.tickets}`).join(' · ')}
          {medianLead !== null && ` · Median ${medianLead} min among detections`}</p>
        <p className="muted">Distribution by messages of lead time</p>
        <div className="chart"><ResponsiveContainer><BarChart data={leadCounts}><CartesianGrid stroke="#e6e8eb" vertical={false}/><XAxis dataKey="position" /><YAxis allowDecimals={false}/><Tooltip/><Bar dataKey="tickets" fill="#166855" /></BarChart></ResponsiveContainer></div>
      </section>
      <section className="panel"><h2>Alert budget</h2><p className="muted">Offline ranking by peak pre-event risk</p>
        <div className="chart"><ResponsiveContainer><LineChart data={model.alert_budget}><XAxis dataKey="budget" tickFormatter={percent}/><YAxis domain={[0, 1]} tickFormatter={percent}/><Tooltip formatter={value => percent(Number(value))}/><Legend/><Line dataKey="precision" stroke="#166855"/><Line dataKey="recall" stroke="#ac5827"/></LineChart></ResponsiveContainer></div>
      </section>
    </div>
  </>
}

export default function App() {
  const [tab, setTab] = useState('Monitor')
  const [tickets, setTickets] = useState<Ticket[]>([])
  const [alerts, setAlerts] = useState<Prediction[]>([])
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [evaluation, setEvaluation] = useState<Evaluation>()
  const [detail, setDetail] = useState<Detail>()
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('all')
  const [sort, setSort] = useState<'risk' | 'messages' | 'change'>('risk')
  const [loading, setLoading] = useState(true)

  async function refresh() {
    try {
      const [current, activeAlerts] = await Promise.all([api<Ticket[]>('/tickets?limit=200'), api<Prediction[]>('/alerts')])
      setTickets(current)
      setAlerts(activeAlerts)
    } catch (err) { setError(String(err)) }
  }

  async function load() {
    setError('')
    setLoading(true)
    await Promise.all([refresh(), api<Scenario[]>('/demo').then(setScenarios).catch(err => setError(String(err))),
      api<Evaluation>('/metrics').then(setEvaluation).catch(err => setError(String(err)))])
    setLoading(false)
  }

  useEffect(() => { void load() }, [])

  async function open(id: string) {
    try { setDetail(await api<Detail>(`/tickets/${id}`)) }
    catch (err) { setError(String(err)) }
  }

  function value(ticket: Ticket) {
    if (sort === 'messages') return ticket.message_count
    return sort === 'change' ? ticket.latest?.delta ?? 0 : ticket.latest?.risk ?? 0
  }

  const visible = tickets.filter(t => filter === 'all' || t.latest?.level === filter).sort((a, b) => value(b) - value(a))
  const distribution = LEVELS.map(level => ({ level, tickets: tickets.filter(t => t.latest?.level === level).length }))
  const snapshot = evaluation?.holdout[evaluation.model.selected]
  return <>
    <header><div className="brand"><span className="mark" aria-hidden="true">↗</span>EscalationIQ</div>
      <nav aria-label="Workspace">{['Monitor', 'Live replay', 'Evaluation'].map(name => <button aria-current={tab === name ? 'page' : undefined} key={name} onClick={() => setTab(name)}>{name}</button>)}</nav>
    </header>
    <main>
      <div className="hero"><div><h1>{tab === 'Monitor' ? 'Support operations' : tab === 'Live replay' ? 'Conversation simulator' : 'Model performance'}</h1>
        <p className="muted">{tab === 'Monitor' ? 'Track escalating conversations and specialist review recommendations.' : tab === 'Live replay' ? 'Observe risk and alert decisions as a ticket unfolds.' : 'Review the chronological holdout and alerting tradeoffs.'}</p></div>
        <button disabled={loading} onClick={load}>{loading ? 'Loading…' : 'Refresh'}</button></div>
      {error && <p className="error" role="alert">{error}</p>}
      {tab === 'Monitor' && <>
        <div className="stats">{[
          ['Active tickets', tickets.length], ['High risk', distribution[2].tickets], ['Critical risk', distribution[3].tickets], ['Recent alerts', alerts.length],
        ].map(([name, n]) => <div className="panel stat" key={name}><span>{name}</span><strong>{n}</strong></div>)}</div>
        <div className="split">
          <section className="panel"><h2>Risk distribution</h2><div className="chart compact"><ResponsiveContainer><BarChart data={distribution}><XAxis dataKey="level"/><YAxis allowDecimals={false} width={28}/><Tooltip/><Bar dataKey="tickets" fill="#166855" radius={[4,4,0,0]} /></BarChart></ResponsiveContainer></div></section>
          <section className="panel"><h2>Offline model summary</h2>{snapshot && evaluation ? <><p className="muted">{caption(evaluation.model.selected)} · synthetic holdout</p>
            <dl><dt>PR-AUC</dt><dd>{decimal(snapshot.pr_auc)}</dd><dt>Alert precision</dt><dd>{percent(snapshot.precision)}</dd><dt>Early recall</dt><dd>{percent(snapshot.recall)}</dd></dl></> : <p className="muted">Evaluation loading</p>}</section>
        </div>
        <section className="panel"><div className="heading"><h2>Ticket queue</h2><div className="controls">
          <label>Severity<select value={filter} onChange={e => setFilter(e.target.value)}><option value="all">All levels</option>{LEVELS.map(l => <option key={l}>{l}</option>)}</select></label>
          <label>Sort by<select value={sort} onChange={e => setSort(e.target.value as typeof sort)}><option value="risk">Risk</option><option value="messages">Messages</option><option value="change">Risk change</option></select></label>
        </div></div>
        {visible.length ? <div className="table-wrap"><table><thead><tr><th>Ticket</th><th>Category</th><th>Risk</th><th>Change</th><th>Messages</th><th>Alert sent</th><th>Recommendation</th></tr></thead>
          <tbody>{visible.map(t => <tr key={t.id}><td><button className="link" onClick={() => open(t.id)}>{t.id.slice(0, 8)}</button></td><td>{t.context.category}</td>
            <td>{t.latest ? <span className={`pill ${t.latest.level}`}>{percent(t.latest.risk)}</span> : '—'}</td>
            <td>{t.latest ? `${t.latest.delta >= 0 ? '+' : ''}${(t.latest.delta * 100).toFixed(1)} pp` : '—'}</td>
            <td>{t.message_count}</td><td>{t.alerted ? 'Yes' : '—'}</td><td>{t.latest?.action ?? 'Awaiting message'}</td></tr>)}</tbody></table></div>
          : <p className="muted">No matching tickets. Use Live replay to add a conversation.</p>}
        </section>
        {detail && <Conversation detail={detail}/>}
        <section className="panel"><h2>Recent alerts</h2>{alerts.length ? <ul className="alert-list">{alerts.map(a => <li key={`${a.ticket_id}-${a.position}`}><button className="link" onClick={() => open(a.ticket_id)}>{a.ticket_id.slice(0, 8)}</button> · {a.level} · message {a.position} · {percent(a.risk)}</li>)}</ul> : <p className="muted">No alerts recorded.</p>}</section>
      </>}
      <div hidden={tab !== 'Live replay'}><Replay scenarios={scenarios} onChange={refresh} /></div>
      {tab === 'Evaluation' && evaluation && <Performance evaluation={evaluation} />}
    </main>
  </>
}
