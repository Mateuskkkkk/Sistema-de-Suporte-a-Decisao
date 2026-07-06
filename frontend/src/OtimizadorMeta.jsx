import React, { useEffect, useMemo, useState } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea,
} from 'recharts'
import {
  Activity, CheckCircle2, Database, Play, RefreshCw, Send, SlidersHorizontal,
} from 'lucide-react'

const MESES = ['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN', 'JUL', 'AGO', 'SET', 'OUT', 'NOV', 'DEZ']
const MESES_NOMES = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
const NIVEL_LABELS = ['Normal', 'Alerta', 'Seca', 'Seca Severa']
const CURVE_COLORS = ['#2a9d8f', '#d4a017', '#e07b2a', '#d94040']
const BAND_COLORS = {
  normal: '#2a9d8f',
  alerta: '#d4a017',
  seca: '#e07b2a',
  severa: '#d94040',
}

function makeApi(base) {
  const b = base || import.meta.env?.VITE_API_URL || 'http://127.0.0.1:8000'
  return {
    reservatorios: async () => {
      const r = await fetch(`${b}/api/otimizador/reservatorios`)
      if (!r.ok) throw new Error('Erro ao listar reservatórios')
      return r.json()
    },
    limites: async (reservatorio) => {
      const r = await fetch(`${b}/api/otimizador/limites`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reservatorio }),
      })
      if (!r.ok) throw new Error('Erro ao buscar limites')
      return r.json()
    },
    simular: (payload) => fetch(`${b}/api/otimizador/simular`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  }
}

function Card({ children, style }) {
  return (
    <div style={{
      background: 'var(--card)',
      border: '1.5px solid var(--border)',
      borderRadius: 'var(--radius)',
      boxShadow: 'var(--shadow)',
      ...style,
    }}>
      {children}
    </div>
  )
}

function Field({ label, children }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 5, fontSize: 10, fontWeight: 800, color: 'var(--text-light)', textTransform: 'uppercase' }}>
      {label}
      {children}
    </label>
  )
}

function Input(props) {
  return <input {...props} style={{ width: '100%', padding: '8px 10px', border: '1.5px solid var(--border)', borderRadius: 'var(--radius-xs)', background: '#fff', color: 'var(--text)', fontSize: 12, outline: 'none', ...(props.style || {}) }} />
}

function Select(props) {
  return <select {...props} style={{ width: '100%', padding: '8px 10px', border: '1.5px solid var(--border)', borderRadius: 'var(--radius-xs)', background: '#fff', color: 'var(--text)', fontSize: 12, outline: 'none', ...(props.style || {}) }} />
}

function curvasParaFaixas(result, scenario) {
  if (!result?.matriz_curvas?.length) return []
  const totalNormal = Number(scenario.durb || 0) + Number(scenario.dsupl || 0)
  return result.matriz_curvas.map((curve, idx) => {
    const nivelIdx = idx + 1
    const vazaoNivel = (Number(scenario.durb || 0) * Number(scenario.fracDurb[nivelIdx] || 0))
      + (Number(scenario.dsupl || 0) * Number(scenario.fracDsup[nivelIdx] || 0))
    const racionamento = totalNormal > 0 ? Math.max(0, Math.min(100, (1 - vazaoNivel / totalNormal) * 100)) : 0
    return {
      Faixa: NIVEL_LABELS[nivelIdx],
      Racionamento: Number(racionamento.toFixed(1)),
      ...Object.fromEntries(MESES.map((m, i) => [m, Number((Number(curve[i] || 0) * 100).toFixed(1))])),
    }
  })
}

function buildBandChartData(matrizCurvas) {
  if (!matrizCurvas?.length) return []
  return MESES.map((mes, i) => {
    const alerta = Number((Number(matrizCurvas[0]?.[i] || 0) * 100).toFixed(2))
    const seca = Number((Number(matrizCurvas[1]?.[i] || 0) * 100).toFixed(2))
    const severa = Number((Number(matrizCurvas[2]?.[i] || 0) * 100).toFixed(2))
    return {
      mes,
      severa,
      seca: Math.max(0, seca - severa),
      alerta: Math.max(0, alerta - seca),
      normal: Math.max(0, 100 - alerta),
      limiteAlerta: alerta,
      limiteSeca: seca,
      limiteSevera: severa,
    }
  })
}

function buildHistoricalVolumeData(result, mesIni, anoIni) {
  if (!result?.volumes_historicos?.length) return []
  const simMesIni = result.mes_inicio ?? mesIni
  const simAnoIni = result.ano_inicio ?? anoIni
  const cap = result.capacidade_hm3 || 1

  const data = result.volumes_historicos.map((vol, index) => {
    const mesDoAno = (simMesIni - 1 + index) % 12
    const anoAtual = simAnoIni + Math.floor((simMesIni - 1 + index) / 12)
    const volPerc = (Number(vol || 0) / cap) * 100
    const n0 = result.matriz_curvas?.[0]?.[mesDoAno] ? result.matriz_curvas[0][mesDoAno] * 100 : 0
    const n1 = result.matriz_curvas?.[1]?.[mesDoAno] ? result.matriz_curvas[1][mesDoAno] * 100 : 0
    const n2 = result.matriz_curvas?.[2]?.[mesDoAno] ? result.matriz_curvas[2][mesDoAno] * 100 : 0
    let estado = 0
    if (volPerc < n2) estado = 3
    else if (volPerc < n1) estado = 2
    else if (volPerc < n0) estado = 1

    return {
      data: `${MESES[mesDoAno]}/${anoAtual}`,
      origVol: Number(volPerc.toFixed(2)),
      origEstado: estado,
      vol_0: null,
      vol_1: null,
      vol_2: null,
      vol_3: null,
    }
  })

  for (let i = 0; i < data.length; i += 1) {
    const curr = data[i]
    curr[`vol_${curr.origEstado}`] = curr.origVol
    if (i > 0) {
      const prev = data[i - 1]
      if (prev.origEstado !== curr.origEstado) curr[`vol_${prev.origEstado}`] = curr.origVol
    }
  }

  return data
}

function HistoricalVolumeTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const point = payload.find(p => p?.payload?.origVol !== undefined)?.payload
  if (!point) return null
  const color = CURVE_COLORS[point.origEstado] || CURVE_COLORS[0]
  return (
    <div style={{ background: '#fff', border: '1.5px solid var(--border)', borderRadius: 10, padding: '10px 12px', boxShadow: 'var(--shadow)', fontSize: 11 }}>
      <div style={{ fontWeight: 900, marginBottom: 7, color: 'var(--text)' }}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
        <span style={{ width: 8, height: 8, borderRadius: 999, background: color }} />
        <span style={{ color: 'var(--text-mid)' }}>Volume:</span>
        <strong style={{ color: 'var(--text)' }}>{point.origVol.toFixed(2)}%</strong>
      </div>
      <div style={{ color, fontWeight: 900, textTransform: 'uppercase', fontSize: 9.5 }}>{NIVEL_LABELS[point.origEstado]}</div>
    </div>
  )
}

export default function OtimizadorMeta({ apiUrl, onApplyCurvas }) {
  const api = useMemo(() => makeApi(apiUrl), [apiUrl])
  const [lista, setLista] = useState([])
  const [reservatorio, setReservatorio] = useState('')
  const [bounds, setBounds] = useState({ anoMin: 1911, mesMin: 1, anoMax: 2021, mesMax: 12 })
  const [prob, setProb] = useState(0.25)
  const [iters, setIters] = useState(50)
  const [ninicio, setNinicio] = useState(7)
  const [mesIni, setMesIni] = useState(1)
  const [anoIni, setAnoIni] = useState(1911)
  const [mesFim, setMesFim] = useState(12)
  const [anoFim, setAnoFim] = useState(2021)
  const [scenario, setScenario] = useState({
    durb: 0.5,
    dsupl: 0.3,
    fracDurb: [1, 1, 0.8, 0.5],
    fracDsup: [1, 0.8, 0.5, 0],
    garantiaReq: [0.9, 0.95, 0.98, 1],
  })
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState(null)
  const [msg, setMsg] = useState(null)
  const [refAreaLeft, setRefAreaLeft] = useState(null)
  const [refAreaRight, setRefAreaRight] = useState(null)
  const [zoomDomain, setZoomDomain] = useState(null)

  useEffect(() => {
    api.reservatorios()
      .then(data => {
        const itens = data.lista || []
        setLista(itens)
        setReservatorio(prev => prev || itens[0] || '')
      })
      .catch(e => setMsg({ type: 'error', text: e.message }))
  }, [api])

  useEffect(() => {
    if (!reservatorio) return
    api.limites(reservatorio)
      .then(data => {
        setBounds({
          anoMin: data.ano_min ?? 1911,
          mesMin: data.mes_min ?? 1,
          anoMax: data.ano_max ?? 2021,
          mesMax: data.mes_max ?? 12,
        })
        setAnoIni(data.ano_min ?? 1911)
        setMesIni(data.mes_min ?? 1)
        setAnoFim(data.ano_max ?? 2021)
        setMesFim(data.mes_max ?? 12)
      })
      .catch(() => {})
  }, [api, reservatorio])

  const setArray = (field, idx, value) => {
    setScenario(prev => {
      const arr = [...prev[field]]
      arr[idx] = value
      return { ...prev, [field]: arr }
    })
  }

  const handleRun = async () => {
    if (!reservatorio) return
    setLoading(true)
    setProgress(0)
    setResult(null)
    setMsg(null)
    setZoomDomain(null)

    const payload = {
      cenario_id: Date.now().toString(),
      reservatorio,
      prob,
      iters,
      ninicio,
      mes_inicio: mesIni,
      ano_inicio: anoIni,
      mes_fim: mesFim,
      ano_fim: anoFim,
      durb_m3s: scenario.durb,
      dsupl_m3s: scenario.dsupl,
      frac_durb: scenario.fracDurb,
      frac_dsup: scenario.fracDsup,
      garantia_req: scenario.garantiaReq,
      seed: 42,
    }

    try {
      const response = await api.simular(payload)
      if (!response.ok || !response.body) throw new Error('Falha ao iniciar otimização')
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const chunks = buffer.split('\n\n')
        buffer = chunks.pop() || ''
        for (const chunk of chunks) {
          if (!chunk.startsWith('data: ')) continue
          const data = JSON.parse(chunk.slice(6))
          if (data.status === 'progresso') {
            setProgress(Math.round((data.iteracao / data.total_iteracoes) * 100))
          } else if (data.status === 'sucesso') {
            setProgress(100)
            setResult(data)
          } else if (data.status === 'erro') {
            setMsg({ type: 'error', text: data.mensagem || 'Erro na otimização' })
          }
        }
      }
    } catch (e) {
      setMsg({ type: 'error', text: e.message })
    } finally {
      setLoading(false)
    }
  }

  const apply = () => {
    const faixas = curvasParaFaixas(result, scenario)
    onApplyCurvas?.({
      id: Date.now(),
      reservatorio,
      faixas,
      result,
      scenario,
    })
    setMsg({ type: 'success', text: 'Curvas enviadas para o simulador. Abra o Simulador e selecione o mesmo reservatório.' })
  }

  const chartData = buildBandChartData(result?.matriz_curvas)
  const chartDataVolume = buildHistoricalVolumeData(result, mesIni, anoIni)
  const activeDataVolume = zoomDomain ? chartDataVolume.slice(zoomDomain.start, zoomDomain.end + 1) : chartDataVolume

  const handleVolumeZoom = () => {
    if (!refAreaLeft || !refAreaRight || refAreaLeft === refAreaRight) {
      setRefAreaLeft(null)
      setRefAreaRight(null)
      return
    }
    let start = chartDataVolume.findIndex(d => d.data === refAreaLeft)
    let end = chartDataVolume.findIndex(d => d.data === refAreaRight)
    if (start < 0 || end < 0) {
      setRefAreaLeft(null)
      setRefAreaRight(null)
      return
    }
    if (start > end) [start, end] = [end, start]
    setZoomDomain({ start, end })
    setRefAreaLeft(null)
    setRefAreaRight(null)
  }

  return (
    <div className="sim-root" style={{ minHeight: 600, padding: '18px 26px 48px' }}>
      <style>{`.sim-root{--bg:#fdf6ee;--orange:#e07b2a;--orange-pale:#fdebd3;--orange-deep:#c46318;--teal:#2a9d8f;--teal-pale:#d4f5ef;--blue:#264fa3;--blue-pale:#dde8f8;--red:#d94040;--red-pale:#fde8e8;--yellow:#d4a017;--yellow-pale:#fef3cd;--text:#1e1208;--text-mid:#5a3c24;--text-light:#9a7055;--border:#ecdcc8;--border-light:#f5ebe0;--card:#fff;--shadow:0 2px 16px rgba(150,90,40,.10);--radius:14px;--radius-sm:9px;--radius-xs:6px;font-family:'Sora',sans-serif;background:var(--bg);color:var(--text)}.opt-layout{display:grid;grid-template-columns:320px minmax(0,1fr);gap:16px;align-items:start}.opt-side{position:sticky;top:16px;background:var(--card);border:1.5px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:16px}.opt-side-head{font-size:14px;font-weight:900;margin-bottom:12px;display:flex;gap:8px;align-items:center}.opt-section{border-top:1.5px solid var(--border-light);padding-top:10px}.opt-btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;border:0;border-radius:9px;padding:9px 13px;font-size:12px;font-weight:800;cursor:pointer}.opt-primary{background:linear-gradient(135deg,var(--orange),var(--orange-deep));color:#fff}.opt-ghost{background:#fff;color:var(--text-mid);border:1.5px solid var(--border)}@keyframes opt-spin{to{transform:rotate(360deg)}}.opt-spin{animation:opt-spin 1.1s linear infinite}@media(max-width:920px){.opt-layout{grid-template-columns:1fr}.opt-side{position:relative;top:0}}`}</style>

      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', marginBottom: 14 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 3 }}>
            <Activity size={21} color="var(--orange)" />
            <h2 style={{ fontSize: 19, fontWeight: 800, margin: 0 }}>Otimizador de Níveis Meta</h2>
          </div>
          <p style={{ fontSize: 11.5, color: 'var(--text-light)', margin: 0 }}>Calcule curvas guia por PSO e envie os limites mensais para o simulador.</p>
        </div>
        {result && (
          <button className="opt-btn opt-primary" onClick={apply}>
            <Send size={14} /> Aplicar no Simulador
          </button>
        )}
      </div>

      {msg && (
        <div style={{ marginBottom: 12, padding: '9px 13px', borderRadius: 'var(--radius-sm)', fontSize: 12, fontWeight: 700, background: msg.type === 'success' ? 'var(--teal-pale)' : 'var(--red-pale)', color: msg.type === 'success' ? 'var(--teal)' : 'var(--red)' }}>
          {msg.type === 'success' ? '✓' : '×'} {msg.text}
        </div>
      )}

      <div className="opt-layout">
        <aside className="opt-side">
          <div style={{ fontSize: 14, fontWeight: 900, marginBottom: 12, display: 'flex', gap: 8, alignItems: 'center' }}><SlidersHorizontal size={15} color="var(--orange)" />Parâmetros</div>
          <div style={{ display: 'grid', gap: 10 }}>
            <Field label="Reservatório">
              <Select value={reservatorio} onChange={e => setReservatorio(e.target.value)}>
                {lista.map(r => <option key={r} value={r}>{r}</option>)}
              </Select>
            </Field>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <Field label="Demanda 1 (m³/s)"><Input type="number" step="0.01" value={scenario.durb} onChange={e => setScenario(p => ({ ...p, durb: Number(e.target.value) }))} /></Field>
              <Field label="Demanda 2 (m³/s)"><Input type="number" step="0.01" value={scenario.dsupl} onChange={e => setScenario(p => ({ ...p, dsupl: Number(e.target.value) }))} /></Field>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <Field label="Probabilidade"><Input type="number" min="0" max="1" step="0.01" value={prob} onChange={e => setProb(Number(e.target.value))} /></Field>
              <Field label="Iterações PSO"><Input type="number" min="10" step="10" value={iters} onChange={e => setIters(Number(e.target.value))} /></Field>
            </div>

            <Field label="Início do ano hidrológico">
              <Select value={ninicio} onChange={e => setNinicio(Number(e.target.value))}>
                {MESES_NOMES.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}
              </Select>
            </Field>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <Field label="Mês início"><Select value={mesIni} onChange={e => setMesIni(Number(e.target.value))}>{MESES.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}</Select></Field>
              <Field label="Ano início"><Input type="number" min={bounds.anoMin} max={anoFim} value={anoIni} onChange={e => setAnoIni(Number(e.target.value))} /></Field>
              <Field label="Mês fim"><Select value={mesFim} onChange={e => setMesFim(Number(e.target.value))}>{MESES.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}</Select></Field>
              <Field label="Ano fim"><Input type="number" min={anoIni} max={bounds.anoMax} value={anoFim} onChange={e => setAnoFim(Number(e.target.value))} /></Field>
            </div>

            <div className="opt-section">
              <div style={{ fontSize: 10, fontWeight: 900, color: 'var(--text-light)', textTransform: 'uppercase', marginBottom: 7 }}>Estados operacionais</div>
              {NIVEL_LABELS.map((label, i) => (
                <div key={label} style={{ display: 'grid', gridTemplateColumns: '74px 1fr 1fr 1fr', gap: 5, alignItems: 'center', marginBottom: 5 }}>
                  <span style={{ fontSize: 11, fontWeight: 800, color: CURVE_COLORS[i] }}>{label}</span>
                  <Input type="number" step="1" min="0" max="100" value={Number((scenario.fracDurb[i] * 100).toFixed(1))} onChange={e => setArray('fracDurb', i, Number(e.target.value) / 100)} />
                  <Input type="number" step="1" min="0" max="100" value={Number((scenario.fracDsup[i] * 100).toFixed(1))} onChange={e => setArray('fracDsup', i, Number(e.target.value) / 100)} />
                  <Input type="number" step="1" min="0" max="100" value={Number((scenario.garantiaReq[i] * 100).toFixed(1))} onChange={e => setArray('garantiaReq', i, Number(e.target.value) / 100)} />
                </div>
              ))}
              <div style={{ display: 'grid', gridTemplateColumns: '74px 1fr 1fr 1fr', gap: 5, fontSize: 9.5, color: 'var(--text-light)', fontWeight: 800 }}>
                <span />
                <span>Dem. 1 %</span>
                <span>Dem. 2 %</span>
                <span>Garantia %</span>
              </div>
            </div>

            <button className="opt-btn opt-primary" onClick={handleRun} disabled={loading || !reservatorio} style={{ marginTop: 6, opacity: loading ? 0.7 : 1 }}>
              {loading ? <RefreshCw size={14} className="opt-spin" /> : <Play size={14} />}
              {loading ? `Otimizando ${progress}%` : 'Otimizar Curvas'}
            </button>
          </div>
        </aside>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {!result && !loading && (
            <Card style={{ padding: 48, textAlign: 'center' }}>
              <Database size={32} color="var(--orange)" style={{ opacity: 0.4, marginBottom: 10 }} />
              <div style={{ fontSize: 14, fontWeight: 900, marginBottom: 5 }}>Pronto para otimizar</div>
              <div style={{ fontSize: 12, color: 'var(--text-light)' }}>Escolha o reservatório e os estados operacionais para calcular curvas guia.</div>
            </Card>
          )}

          {loading && (
            <Card style={{ padding: 42, textAlign: 'center' }}>
              <RefreshCw size={32} color="var(--orange)" className="opt-spin" style={{ marginBottom: 12 }} />
              <div style={{ fontSize: 14, fontWeight: 900 }}>Otimizando níveis meta...</div>
              <div style={{ margin: '14px auto 0', width: 'min(360px,100%)', height: 8, borderRadius: 999, background: 'var(--border)', overflow: 'hidden' }}>
                <div style={{ width: `${progress}%`, height: '100%', background: 'var(--orange)' }} />
              </div>
            </Card>
          )}

          {result && !loading && (
            <>
              <Card style={{ padding: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13.5, fontWeight: 900, marginBottom: 12 }}>
                  <CheckCircle2 size={16} color="var(--teal)" /> Resultado para {reservatorio}
                </div>
                <div style={{ height: 330 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ top: 10, right: 12, bottom: 0, left: -18 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#ecdcc8" />
                      <XAxis dataKey="mes" tick={{ fill: '#9a7055', fontSize: 11 }} tickLine={false} />
                      <YAxis domain={[0, 100]} tick={{ fill: '#9a7055', fontSize: 11 }} tickFormatter={v => `${v}%`} tickLine={false} />
                      <Tooltip formatter={v => `${Number(v).toFixed(2)}%`} />
                      <Legend />
                      <Area isAnimationActive={false} type="monotone" stackId="meta" name="Seca Severa" dataKey="severa" stroke={BAND_COLORS.severa} fill={BAND_COLORS.severa} fillOpacity={0.55} />
                      <Area isAnimationActive={false} type="monotone" stackId="meta" name="Seca" dataKey="seca" stroke={BAND_COLORS.seca} fill={BAND_COLORS.seca} fillOpacity={0.5} />
                      <Area isAnimationActive={false} type="monotone" stackId="meta" name="Alerta" dataKey="alerta" stroke={BAND_COLORS.alerta} fill={BAND_COLORS.alerta} fillOpacity={0.48} />
                      <Area isAnimationActive={false} type="monotone" stackId="meta" name="Normal" dataKey="normal" stroke={BAND_COLORS.normal} fill={BAND_COLORS.normal} fillOpacity={0.45} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              {chartDataVolume.length > 0 && (
                <Card style={{ padding: 16 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, marginBottom: 12 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13.5, fontWeight: 900 }}>
                      <Activity size={16} color="var(--orange)" /> Simulacao Historica de Volumes (%)
                    </div>
                    {zoomDomain && (
                      <button className="opt-btn opt-ghost" onClick={() => setZoomDomain(null)} style={{ padding: '6px 10px', fontSize: 10 }}>
                        Resetar Zoom
                      </button>
                    )}
                  </div>
                  <div style={{ height: 300, userSelect: 'none' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={activeDataVolume}
                        margin={{ top: 10, right: 12, bottom: 0, left: -18 }}
                        onMouseDown={e => e && setRefAreaLeft(e.activeLabel ? String(e.activeLabel) : null)}
                        onMouseMove={e => e && refAreaLeft && setRefAreaRight(e.activeLabel ? String(e.activeLabel) : null)}
                        onMouseUp={handleVolumeZoom}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#ecdcc8" />
                        <XAxis dataKey="data" tick={{ fill: '#9a7055', fontSize: 10 }} tickLine={false} minTickGap={36} />
                        <YAxis domain={[0, 100]} tick={{ fill: '#9a7055', fontSize: 11 }} tickFormatter={v => `${v}%`} tickLine={false} />
                        <Tooltip content={<HistoricalVolumeTooltip />} />
                        {refAreaLeft && refAreaRight && <ReferenceArea x1={refAreaLeft} x2={refAreaRight} strokeOpacity={0.3} fill="#2a9d8f" fillOpacity={0.16} />}
                        <Area isAnimationActive={false} type="linear" dataKey="vol_0" stroke={CURVE_COLORS[0]} strokeWidth={2.5} fill={CURVE_COLORS[0]} fillOpacity={0.30} connectNulls={false} />
                        <Area isAnimationActive={false} type="linear" dataKey="vol_1" stroke={CURVE_COLORS[1]} strokeWidth={2.5} fill={CURVE_COLORS[1]} fillOpacity={0.34} connectNulls={false} />
                        <Area isAnimationActive={false} type="linear" dataKey="vol_2" stroke={CURVE_COLORS[2]} strokeWidth={2.5} fill={CURVE_COLORS[2]} fillOpacity={0.36} connectNulls={false} />
                        <Area isAnimationActive={false} type="linear" dataKey="vol_3" stroke={CURVE_COLORS[3]} strokeWidth={2.5} fill={CURVE_COLORS[3]} fillOpacity={0.38} connectNulls={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              )}

              <Card style={{ overflow: 'hidden' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: 'var(--bg)' }}>
                      <th style={{ textAlign: 'left', padding: 9, color: 'var(--text-light)' }}>Nível</th>
                      <th style={{ padding: 9, color: 'var(--text-light)' }}>Rac. aplicado</th>
                      {MESES.map(m => <th key={m} style={{ padding: 9, color: 'var(--text-light)' }}>{m}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {curvasParaFaixas(result, scenario).map((f, i) => (
                      <tr key={f.Faixa}>
                        <td style={{ padding: 9, borderTop: '1px solid var(--border-light)', fontWeight: 900, color: CURVE_COLORS[i + 1] }}>{f.Faixa}</td>
                        <td style={{ padding: 9, borderTop: '1px solid var(--border-light)', textAlign: 'center' }}>{f.Racionamento}%</td>
                        {MESES.map(m => <td key={m} style={{ padding: 9, borderTop: '1px solid var(--border-light)', textAlign: 'center' }}>{f[m]}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
