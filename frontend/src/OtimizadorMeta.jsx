import React, { useEffect, useMemo, useState } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea,
} from 'recharts'
import {
  Activity, CheckCircle2, Database, Play, RefreshCw, Send,
  Download, FileSpreadsheet, Plus, Search, X,
} from 'lucide-react'
import * as XLSX from 'xlsx'

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

const DEFAULT_SCENARIO = {
  durb: 0.5,
  dsupl: 0.3,
  fracDurb: [1, 1, 0.8, 0.5],
  fracDsup: [1, 0.8, 0.5, 0],
  garantiaReq: [0.9, 0.95, 0.98, 1],
}

function createScenario(index) {
  return {
    id: `cenario-${Date.now()}-${index}`,
    name: `Cenário ${index}`,
    ...DEFAULT_SCENARIO,
    fracDurb: [...DEFAULT_SCENARIO.fracDurb],
    fracDsup: [...DEFAULT_SCENARIO.fracDsup],
    garantiaReq: [...DEFAULT_SCENARIO.garantiaReq],
  }
}

function pct(v) {
  return `${(Number(v || 0) * 100).toFixed(1)}%`
}

function downloadText(filename, content, type = 'text/csv;charset=utf-8;') {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

function safeName(value) {
  return String(value || 'otimizacao').replace(/[\\/:*?"<>|]+/g, '').replace(/\s+/g, '_')
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

function ReservatorioSearch({ lista, value, onChange }) {
  const [query, setQuery] = useState(value || '')
  const [open, setOpen] = useState(false)
  const ref = React.useRef(null)

  useEffect(() => { setQuery(value || '') }, [value])

  useEffect(() => {
    const handler = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const filtered = lista
    .filter(nome => String(nome || '').toLowerCase().includes(query.toLowerCase()))
    .slice(0, 60)

  const select = (nome) => {
    setQuery(nome)
    setOpen(false)
    onChange(nome)
  }

  return (
    <div className="opt-search" ref={ref}>
      <Search size={13} className="opt-search-icon" />
      <input
        className="opt-search-input"
        value={query}
        onChange={e => { setQuery(e.target.value); setOpen(true); if (!e.target.value) onChange('') }}
        onFocus={() => setOpen(true)}
        placeholder="Digite para buscar..."
      />
      {open && filtered.length > 0 && (
        <div className="opt-search-menu">
          {filtered.map(nome => (
            <button key={nome} type="button" className="opt-search-item" onMouseDown={() => select(nome)}>
              {nome}
            </button>
          ))}
        </div>
      )}
    </div>
  )
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

export default function OtimizadorMeta({ apiUrl, onApplyCurvas, darkMode = false }) {
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
  const [scenarios, setScenarios] = useState(() => [{ ...createScenario(1), id: 'cenario-1', name: 'Cenário 1' }])
  const [activeScenarioId, setActiveScenarioId] = useState('cenario-1')
  const [loadingId, setLoadingId] = useState(null)
  const [progressByScenario, setProgressByScenario] = useState({})
  const [resultsByScenario, setResultsByScenario] = useState({})
  const [msg, setMsg] = useState(null)
  const [refAreaLeft, setRefAreaLeft] = useState(null)
  const [refAreaRight, setRefAreaRight] = useState(null)
  const [zoomDomain, setZoomDomain] = useState(null)

  const scenario = scenarios.find(s => s.id === activeScenarioId) || scenarios[0]
  const result = resultsByScenario[activeScenarioId] || null
  const loading = loadingId === activeScenarioId
  const progress = progressByScenario[activeScenarioId] || 0

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

  const updateScenario = (updater) => {
    setScenarios(prev => prev.map(item => {
      if (item.id !== activeScenarioId) return item
      const patch = typeof updater === 'function' ? updater(item) : updater
      return { ...item, ...patch }
    }))
  }

  const setArray = (field, idx, value) => {
    updateScenario(prev => {
      const arr = [...prev[field]]
      arr[idx] = value
      return { [field]: arr }
    })
  }

  const addScenario = () => {
    const novo = createScenario(scenarios.length + 1)
    setScenarios(prev => [...prev, novo])
    setActiveScenarioId(novo.id)
    setMsg(null)
    setZoomDomain(null)
  }

  const removeScenario = (id) => {
    if (scenarios.length <= 1) return
    const nextScenarios = scenarios.filter(s => s.id !== id)
    setScenarios(nextScenarios)
    if (activeScenarioId === id) setActiveScenarioId(nextScenarios[0]?.id || 'cenario-1')
    setResultsByScenario(prev => {
      const next = { ...prev }
      delete next[id]
      return next
    })
    setProgressByScenario(prev => {
      const next = { ...prev }
      delete next[id]
      return next
    })
  }

  const handleRun = async () => {
    if (!reservatorio) return
    const scenarioId = activeScenarioId
    const scenarioSnapshot = scenario
    setLoadingId(scenarioId)
    setProgressByScenario(prev => ({ ...prev, [scenarioId]: 0 }))
    setResultsByScenario(prev => ({ ...prev, [scenarioId]: null }))
    setMsg(null)
    setZoomDomain(null)

    const payload = {
      cenario_id: scenarioId,
      reservatorio,
      prob,
      iters,
      ninicio,
      mes_inicio: mesIni,
      ano_inicio: anoIni,
      mes_fim: mesFim,
      ano_fim: anoFim,
      durb_m3s: scenarioSnapshot.durb,
      dsupl_m3s: scenarioSnapshot.dsupl,
      frac_durb: scenarioSnapshot.fracDurb,
      frac_dsup: scenarioSnapshot.fracDsup,
      garantia_req: scenarioSnapshot.garantiaReq,
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
            setProgressByScenario(prev => ({ ...prev, [scenarioId]: Math.round((data.iteracao / data.total_iteracoes) * 100) }))
          } else if (data.status === 'sucesso') {
            setProgressByScenario(prev => ({ ...prev, [scenarioId]: 100 }))
            setResultsByScenario(prev => ({ ...prev, [scenarioId]: data }))
          } else if (data.status === 'erro') {
            setMsg({ type: 'error', text: data.mensagem || 'Erro na otimização' })
          }
        }
      }
    } catch (e) {
      setMsg({ type: 'error', text: e.message })
    } finally {
      setLoadingId(null)
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

  const performanceRows = () => NIVEL_LABELS.map((label, i) => {
    const vazaoTotal = (Number(scenario.durb || 0) * Number(scenario.fracDurb[i] || 0))
      + (Number(scenario.dsupl || 0) * Number(scenario.fracDsup[i] || 0))
    return {
      'Nivel Operacional': label,
      'Vazao Total (L/s)': Number((vazaoTotal * 1000).toFixed(3)),
      'Permanencia Exigida': Number(((Number(scenario.garantiaReq[i] || 0)) * 100).toFixed(2)),
      'Permanencia Obtida': Number(((Number(result?.garantias_obtidas?.[i] || 0)) * 100).toFixed(2)),
    }
  })

  const simulationRows = () => (result?.simulacao_historica || []).map(d => ({
    'Mês/Ano': d.Data,
    'Armazenamento Inicial (hm³)': Number(d['Armazenamento Inicial'] || 0),
    'Armazenamento Final (hm³)': Number(d['Armazenamento Final'] || 0),
    'Afluências (hm³/mês)': Number(d['Afluências (hm³/mês)'] || 0),
    'Evaporação (hm³)': Number(d['Evaporação (hm³)'] || 0),
    'Demanda Solicitada (m³/s)': Number(d['Demanda Solicitada (m³/s)'] || 0),
    'Demanda Atendida (m³/s)': Number(d['Demanda Atendida (m³/s)'] || 0),
    'Demanda Atendida (hm³)': Number(d['Demanda Atendida (m³/s)'] || 0) * 2.592,
    'Racionamento (%)': Number(d['Racionamento (%)'] || 0),
    'Vertimento (hm³)': Number(d['Vertimento (hm³)'] || 0),
    'Falha': d.Falha || 'Não',
    'Modo Operação': d['Modo Operação'] || 'Normal',
  }))

  const exportCurvesCSV = () => {
    if (!result?.matriz_curvas) return
    let csv = 'Mes;Alerta;Seca;Seca Severa\n'
    MESES.forEach((mes, i) => {
      csv += `${mes};${(result.matriz_curvas[0][i] * 100).toFixed(2)};${(result.matriz_curvas[1][i] * 100).toFixed(2)};${(result.matriz_curvas[2][i] * 100).toFixed(2)}\n`
    })
    downloadText(`curvas_${safeName(reservatorio)}.csv`, csv)
  }

  const exportVolumesCSV = () => {
    if (!result?.volumes_historicos?.length) return
    let csv = 'Data;Volume Absoluto (hm3);Volume Percentual (%)\n'
    const simMesIni = result.mes_inicio ?? mesIni
    const simAnoIni = result.ano_inicio ?? anoIni
    const cap = result.capacidade_hm3 || 1
    result.volumes_historicos.forEach((vol, index) => {
      const mesDoAno = (simMesIni - 1 + index) % 12
      const anoAtual = simAnoIni + Math.floor((simMesIni - 1 + index) / 12)
      csv += `${MESES[mesDoAno]}/${anoAtual};${Number(vol).toFixed(2)};${((Number(vol) / cap) * 100).toFixed(2)}\n`
    })
    downloadText(`volumes_${safeName(reservatorio)}.csv`, csv)
  }

  const exportSimulationExcel = () => {
    const rows = simulationRows()
    if (!rows.length) return
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(rows), 'Simulacao')
    XLSX.writeFile(wb, `simulacao_${safeName(reservatorio)}.xlsx`)
  }

  const exportOptimizationExcel = () => {
    if (!result) return
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(performanceRows()), 'Desempenho')
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(curvasParaFaixas(result, scenario)), 'Curvas')
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(buildHistoricalVolumeData(result, mesIni, anoIni).map(d => ({
      Data: d.data,
      'Volume Percentual (%)': d.origVol,
      Estado: NIVEL_LABELS[d.origEstado],
    }))), 'Volumes')
    const rows = simulationRows()
    if (rows.length) XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(rows), 'Simulacao')
    XLSX.writeFile(wb, `otimizacao_${safeName(reservatorio)}.xlsx`)
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
    <div className={`sim-root ${darkMode ? 'opt-dark' : ''}`} style={{ minHeight: 600, padding: '18px 26px 48px' }}>
      <style>{`.sim-root{--bg:#fdf6ee;--orange:#e07b2a;--orange-pale:#fdebd3;--orange-deep:#c46318;--teal:#2a9d8f;--teal-pale:#d4f5ef;--blue:#264fa3;--blue-pale:#dde8f8;--red:#d94040;--red-pale:#fde8e8;--yellow:#d4a017;--yellow-pale:#fef3cd;--text:#1e1208;--text-mid:#5a3c24;--text-light:#9a7055;--border:#ecdcc8;--border-light:#f5ebe0;--card:#fff;--shadow:0 2px 16px rgba(150,90,40,.10);--radius:14px;--radius-sm:9px;--radius-xs:6px;font-family:'Sora',sans-serif;background:var(--bg);color:var(--text)}.opt-layout{display:grid;grid-template-columns:320px minmax(0,1fr);gap:16px;align-items:start}.opt-side{position:sticky;top:16px;background:var(--card);border:1.5px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:16px}.opt-side-head{font-size:14px;font-weight:900;margin-bottom:12px;display:flex;gap:8px;align-items:center}.opt-section{border-top:1.5px solid var(--border-light);padding-top:10px}.opt-btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;border:0;border-radius:9px;padding:9px 13px;font-size:12px;font-weight:800;cursor:pointer}.opt-primary{background:linear-gradient(135deg,var(--orange),var(--orange-deep));color:#fff}.opt-ghost{background:#fff;color:var(--text-mid);border:1.5px solid var(--border)}@keyframes opt-spin{to{transform:rotate(360deg)}}.opt-spin{animation:opt-spin 1.1s linear infinite}@media(max-width:920px){.opt-layout{grid-template-columns:1fr}.opt-side{position:relative;top:0}}`}</style>
      <style>{`.opt-layout{grid-template-columns:340px minmax(0,1fr);gap:0}.opt-side{position:sticky;top:12px;background:#fff;border:1px solid #cbd5e1;border-radius:0;box-shadow:0 10px 24px rgba(15,23,42,.12);padding:0;overflow:hidden;font-family:'JetBrains Mono','Consolas',monospace}.opt-side-top{padding:16px;border-bottom:1px solid #cbd5e1;display:flex;flex-direction:column;gap:14px;background:#fff}.opt-side-body{padding:16px;display:flex;flex-direction:column;gap:22px;background:#fff}.opt-label{display:block;font-size:10px;text-transform:uppercase;color:#475569;font-weight:700;margin-bottom:5px}.opt-label.center{text-align:center}.opt-control-row{display:flex;align-items:center;gap:12px}.opt-control-row input[type=range]{flex:1;accent-color:#0ea5e9}.opt-mini{width:64px;text-align:center}.opt-select-wide{width:80%;margin:0 auto;display:block}.opt-period-row{display:flex;align-items:center;justify-content:center;gap:12px;margin-top:8px}.opt-period-name{width:42px;font-size:9px;text-transform:uppercase;color:#64748b}.opt-period-fields{display:flex;gap:4px}.opt-month{width:86px}.opt-year{width:86px;text-align:center}.opt-tabbar{display:flex;overflow-x:auto;border-bottom:1px solid #cbd5e1;background:#f1f5f9}.opt-tab{border:0;border-right:1px solid #cbd5e1;background:#fff;color:#0284c7;font:700 12px 'JetBrains Mono','Consolas',monospace;padding:10px 16px}.opt-grid2{display:grid;grid-template-columns:1fr 1fr;gap:32px}.opt-matrix-title{text-align:center;font-size:10px;text-transform:uppercase;color:#475569;font-weight:700;margin:0 0 8px}.opt-matrix-labels,.opt-matrix{display:grid;grid-template-columns:repeat(4,1fr)}.opt-matrix-labels span{font-size:9px;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.opt-matrix{border:1px solid #cbd5e1;border-radius:4px;overflow:hidden}.opt-matrix input{border:0;border-right:1px solid #cbd5e1;background:#f1f5f9;text-align:center;font:12px 'JetBrains Mono','Consolas',monospace;padding:7px 4px;min-width:0}.opt-matrix input:last-child{border-right:0}.opt-run{width:100%;padding:12px;border-radius:6px;background:#0284c7;color:#fff;font:800 12px 'JetBrains Mono','Consolas',monospace;text-transform:uppercase;letter-spacing:.08em}.opt-side select,.opt-side input[type=number]{border:1px solid #cbd5e1;background:#f1f5f9;color:#0f172a;border-radius:4px;font:12px 'JetBrains Mono','Consolas',monospace;padding:7px 8px}.opt-perm-table{width:100%;border-collapse:collapse;font:12px 'JetBrains Mono','Consolas',monospace;text-align:center}.opt-perm-table th{color:#64748b;font-size:11px;font-weight:800;padding:8px 6px}.opt-perm-table td{border-top:1px solid #e2e8f0;padding:8px 6px;color:#334155}.opt-perm-table td:first-child{text-align:left;font-weight:800}.opt-section-title{font:800 12px 'JetBrains Mono','Consolas',monospace;text-transform:uppercase;color:#475569;border-bottom:1px solid #e2e8f0;padding-bottom:6px;margin-bottom:8px}@media(max-width:920px){.opt-layout{grid-template-columns:1fr;gap:16px}.opt-side{position:relative;top:0;border-radius:var(--radius)}}`}</style>
      <style>{`.opt-dark{--bg:#050403;--card:#0d0805;--text:#fff7ef;--text-mid:#efd0b8;--text-light:#c0987c;--border:#2a1a10;--border-light:#1f140d;--orange-pale:#3a1d0b;--orange-deep:#ff9b42;--teal-pale:#09231f;--red-pale:#2a0c0c;--yellow-pale:#2a2108;--blue-pale:#071634;--shadow:0 2px 18px rgba(0,0,0,.45)}.opt-side,.opt-side-top,.opt-side-body,.opt-tab{background:var(--card);color:var(--text);font-family:'Sora',sans-serif}.opt-side{border-color:var(--border);border-radius:var(--radius);box-shadow:var(--shadow)}.opt-side-top,.opt-tabbar{border-color:var(--border)}.opt-tabbar{background:var(--bg)}.opt-tab{border-color:var(--border);color:var(--orange-deep)}.opt-label,.opt-period-name,.opt-matrix-title,.opt-perm-table th,.opt-section-title{color:var(--text-light);font-family:'Sora',sans-serif}.opt-matrix,.opt-side select,.opt-side input[type=number]{border-color:var(--border);background:#080503;color:var(--text);font-family:'Sora',sans-serif}.opt-matrix input{border-color:var(--border);background:#080503;color:var(--text);font-family:'Sora',sans-serif}.opt-dark option{background:#080503;color:var(--text)}.opt-control-row input[type=range]{accent-color:var(--orange)}.opt-run{background:linear-gradient(135deg,var(--orange),var(--orange-deep));font-family:'Sora',sans-serif}.opt-perm-table{font-family:'Sora',sans-serif}.opt-perm-table td{border-color:var(--border-light);color:var(--text-mid)}.opt-section-title{border-color:var(--border-light)}.opt-dark .recharts-default-tooltip{background:var(--card)!important;border-color:var(--border)!important;color:var(--text)!important}`}</style>
      <style>{`.opt-search{position:relative;width:88%;margin:0 auto}.opt-search-icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:var(--text-light);pointer-events:none}.opt-search-input{width:100%;border:1.5px solid var(--border);border-radius:8px;background:var(--card);color:var(--text);padding:9px 10px 9px 30px;font:12px 'Sora',sans-serif;outline:none}.opt-search-input:focus{border-color:var(--orange-deep);box-shadow:0 0 0 3px var(--orange-pale)}.opt-search-menu{position:absolute;z-index:40;left:0;right:0;top:calc(100% + 4px);max-height:210px;overflow:auto;border:1.5px solid var(--border);border-radius:8px;background:var(--card);box-shadow:var(--shadow);padding:4px}.opt-search-item{display:block;width:100%;text-align:left;border:0;border-radius:6px;background:transparent;color:var(--text);padding:8px 9px;font:700 11.5px 'Sora',sans-serif;cursor:pointer}.opt-search-item:hover{background:var(--orange-pale);color:var(--orange-deep)}.opt-input-card{border:1.5px solid var(--border);border-radius:10px;background:color-mix(in srgb,var(--card) 82%,var(--bg));padding:12px}.opt-demand-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.opt-demand-input{width:100%;text-align:center;border-radius:8px!important;padding:10px!important;font-size:13px!important;font-weight:800!important}.opt-hydro-select{display:block;margin:0 auto;width:180px;text-align:center}.opt-tab{display:inline-flex;align-items:center;gap:7px}.opt-tab.active{background:var(--orange-pale);color:var(--orange-deep)}.opt-tab-add{border:0;background:transparent;color:var(--text-light);padding:9px 12px;cursor:pointer}.opt-tab-add:hover{color:var(--orange-deep);background:var(--orange-pale)}.opt-tab-close{border:0;background:transparent;color:inherit;padding:0;line-height:0;cursor:pointer;opacity:.7}.opt-tab-close:hover{opacity:1;color:var(--red)}.opt-dark .opt-ghost{background:#0a0604;color:var(--text-light);border-color:var(--border)}.opt-dark .opt-ghost:hover{background:var(--orange-pale);color:var(--orange-deep);border-color:var(--orange-deep)}`}</style>
      <style>{`.opt-side,.opt-side-top,.opt-side-body,.opt-tab,.opt-btn,.opt-label,.opt-period-name,.opt-matrix-title,.opt-perm-table,.opt-section-title,.opt-search-input,.opt-search-item,.opt-side select,.opt-side input[type=number],.opt-matrix input,.opt-run{font-family:'Sora',sans-serif}.opt-input-card{background:transparent!important;border:0!important;border-radius:0!important;padding:0!important}.opt-side-body{gap:18px}.opt-matrix-title{color:var(--text-light);letter-spacing:0}.opt-matrix-labels{gap:4px;margin-bottom:4px}.opt-matrix-labels span{font-family:'Sora',sans-serif;font-weight:800}.opt-demand-input,.opt-side select,.opt-side input[type=number],.opt-matrix input{border-radius:var(--radius-xs)!important}.opt-matrix{gap:4px;border:0!important;border-radius:0!important;background:transparent!important;overflow:visible}.opt-matrix input{background:var(--card);color:var(--text);border:1.5px solid var(--border)!important;box-shadow:none!important;padding:8px 4px}.opt-matrix input:last-child{border-right:1.5px solid var(--border)!important}.sim-root:not(.opt-dark) .opt-side{background:var(--card);border:1.5px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow)}.sim-root:not(.opt-dark) .opt-side-top,.sim-root:not(.opt-dark) .opt-side-body{background:var(--card);border-color:var(--border-light)}.sim-root:not(.opt-dark) .opt-tabbar{background:var(--card);border-color:var(--border-light)}.sim-root:not(.opt-dark) .opt-tab{background:transparent;color:var(--text-light);border-color:var(--border-light)}.sim-root:not(.opt-dark) .opt-tab.active,.sim-root:not(.opt-dark) .opt-tab-add:hover{background:var(--orange-pale);color:var(--orange-deep)}.sim-root:not(.opt-dark) .opt-search-input,.sim-root:not(.opt-dark) .opt-search-menu{background:var(--card);color:var(--text);border-color:var(--border)}.sim-root:not(.opt-dark) .opt-search-item{color:var(--text)}.sim-root:not(.opt-dark) .opt-search-item:hover{background:var(--orange-pale);color:var(--orange-deep)}.sim-root:not(.opt-dark) .opt-side select,.sim-root:not(.opt-dark) .opt-side input[type=number],.sim-root:not(.opt-dark) .opt-matrix input{background:var(--card);color:var(--text);border-color:var(--border)!important}.sim-root:not(.opt-dark) .opt-ghost{background:var(--card);color:var(--text-mid);border-color:var(--border)}.opt-dark .opt-side select,.opt-dark .opt-side input[type=number],.opt-dark .opt-matrix input{background:#080503;color:var(--text);border-color:var(--border)!important}`}</style>

      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', marginBottom: 14 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 3 }}>
            <Activity size={21} color="var(--orange)" />
            <h2 style={{ fontSize: 19, fontWeight: 800, margin: 0 }}>Otimizador de Níveis Meta</h2>
          </div>
          <p style={{ fontSize: 11.5, color: 'var(--text-light)', margin: 0 }}>Calcule curvas guia por PSO e envie os limites mensais para o simulador.</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          {result && (
            <>
              <button className="opt-btn opt-ghost" onClick={exportCurvesCSV}><Download size={14} /> CSV Curvas</button>
              <button className="opt-btn opt-ghost" onClick={exportVolumesCSV}><Download size={14} /> CSV Volumes</button>
              <button className="opt-btn opt-ghost" onClick={exportSimulationExcel}><FileSpreadsheet size={14} /> Planilha Simulação</button>
              <button className="opt-btn opt-ghost" onClick={exportOptimizationExcel}><FileSpreadsheet size={14} /> Salvar Resultados</button>
              <button className="opt-btn opt-primary" onClick={apply}>
                <Send size={14} /> Aplicar no Simulador
              </button>
            </>
          )}
        </div>
      </div>

      {msg && (
        <div style={{ marginBottom: 12, padding: '9px 13px', borderRadius: 'var(--radius-sm)', fontSize: 12, fontWeight: 700, background: msg.type === 'success' ? 'var(--teal-pale)' : 'var(--red-pale)', color: msg.type === 'success' ? 'var(--teal)' : 'var(--red)' }}>
          {msg.type === 'success' ? '✓' : '×'} {msg.text}
        </div>
      )}

      <div className="opt-layout">
        <aside className="opt-side">
          <div className="opt-side-top">
            <div>
              <label className="opt-label center">Reservatório</label>
              <ReservatorioSearch lista={lista} value={reservatorio} onChange={setReservatorio} />
            </div>

            <div>
              <label className="opt-label">Prob. Afluência</label>
              <div className="opt-control-row">
                <input type="range" min="0.05" max="0.95" step="0.05" value={prob} onChange={e => setProb(Number(e.target.value))} />
                <input className="opt-mini" type="number" min="0" max="1" step="0.05" value={prob} onChange={e => setProb(Number(e.target.value))} />
              </div>
            </div>

            <div>
              <label className="opt-label">Iterações PSO</label>
              <div className="opt-control-row">
                <input type="range" min="10" max="500" step="10" value={iters} onChange={e => setIters(Number(e.target.value))} />
                <input className="opt-mini" type="number" min="10" step="10" value={iters} onChange={e => setIters(Number(e.target.value))} />
              </div>
            </div>

            <div>
              <label className="opt-label">Período de Simulação</label>
              <div className="opt-period-row">
                <span className="opt-period-name">Inicio</span>
                <div className="opt-period-fields">
                  <select className="opt-month" value={mesIni} onChange={e => setMesIni(Number(e.target.value))}>{MESES.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}</select>
                  <input className="opt-year" type="number" min={bounds.anoMin} max={anoFim} value={anoIni} onChange={e => setAnoIni(Number(e.target.value))} />
                </div>
              </div>
              <div className="opt-period-row">
                <span className="opt-period-name">Fim</span>
                <div className="opt-period-fields">
                  <select className="opt-month" value={mesFim} onChange={e => setMesFim(Number(e.target.value))}>{MESES.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}</select>
                  <input className="opt-year" type="number" min={anoIni} max={bounds.anoMax} value={anoFim} onChange={e => setAnoFim(Number(e.target.value))} />
                </div>
              </div>
            </div>

            <div>
              <label className="opt-label center">Mês Início Ano Hidrológico</label>
              <select className="opt-month opt-hydro-select" value={ninicio} onChange={e => setNinicio(Number(e.target.value))}>
                {MESES_NOMES.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}
              </select>
            </div>
          </div>

          <div className="opt-tabbar">
            {scenarios.map(s => (
              <button
                key={s.id}
                className={`opt-tab ${s.id === activeScenarioId ? 'active' : ''}`}
                type="button"
                onClick={() => { setActiveScenarioId(s.id); setZoomDomain(null) }}
              >
                {s.name}
                {scenarios.length > 1 && (
                  <span
                    className="opt-tab-close"
                    role="button"
                    tabIndex={0}
                    onClick={e => { e.stopPropagation(); removeScenario(s.id) }}
                  >
                    <X size={12} />
                  </span>
                )}
              </button>
            ))}
            <button className="opt-tab-add" type="button" onClick={addScenario} title="Adicionar cenário">
              <Plus size={15} />
            </button>
          </div>

          <div className="opt-side-body">
            <div className="opt-input-card opt-demand-grid">
              <label>
                <span className="opt-label center">Demanda 1 (m³/s)</span>
                <input className="opt-demand-input" type="number" step="0.01" value={scenario.durb} onChange={e => updateScenario({ durb: Number(e.target.value) })} />
              </label>
              <label>
                <span className="opt-label center">Demanda 2 (m³/s)</span>
                <input className="opt-demand-input" type="number" step="0.01" value={scenario.dsupl} onChange={e => updateScenario({ dsupl: Number(e.target.value) })} />
              </label>
            </div>

            <div className="opt-input-card">
              <p className="opt-matrix-title">Permanências Requeridas (%)</p>
              <div className="opt-matrix-labels">{NIVEL_LABELS.map((label, i) => <span key={label} style={{ color: CURVE_COLORS[i] }}>{label}</span>)}</div>
              <div className="opt-matrix">
                {scenario.garantiaReq.map((v, i) => (
                  <input key={i} type="number" step="1" min="0" max="100" value={Number((v * 100).toFixed(1))} onChange={e => setArray('garantiaReq', i, Number(e.target.value) / 100)} />
                ))}
              </div>
            </div>

            <div className="opt-input-card">
              <p className="opt-matrix-title">Atendimento da Demanda 1</p>
              <div className="opt-matrix">
                {scenario.fracDurb.map((v, i) => (
                  <input key={i} type="number" step="1" min="0" max="100" value={Number((v * 100).toFixed(1))} onChange={e => setArray('fracDurb', i, Number(e.target.value) / 100)} />
                ))}
              </div>
            </div>

            <div className="opt-input-card">
              <p className="opt-matrix-title">Atendimento da Demanda 2 (%)</p>
              <div className="opt-matrix">
                {scenario.fracDsup.map((v, i) => (
                  <input key={i} type="number" step="1" min="0" max="100" value={Number((v * 100).toFixed(1))} onChange={e => setArray('fracDsup', i, Number(e.target.value) / 100)} />
                ))}
              </div>
            </div>

            <button className="opt-btn opt-run" onClick={handleRun} disabled={loading || !reservatorio} style={{ opacity: loading ? 0.7 : 1 }}>
              {loading ? <RefreshCw size={14} className="opt-spin" /> : <Play size={14} />}
              {loading ? `Otimizando ${progress}%` : 'Simular Cenário'}
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

              <Card style={{ padding: 16 }}>
                <div className="opt-section-title">Desempenho e Vazões</div>
                <table className="opt-perm-table">
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'left' }}>Nivel Operacional</th>
                      <th>Vazao Total (L/s)</th>
                      <th>Permanencia Exigida</th>
                      <th>Permanencia Obtida</th>
                    </tr>
                  </thead>
                  <tbody>
                    {NIVEL_LABELS.map((label, i) => {
                      const vazaoTotal = (Number(scenario.durb || 0) * Number(scenario.fracDurb[i] || 0))
                        + (Number(scenario.dsupl || 0) * Number(scenario.fracDsup[i] || 0))
                      const exigida = Number(scenario.garantiaReq[i] || 0)
                      const obtida = Number(result.garantias_obtidas?.[i] || 0)
                      const ok = obtida >= exigida - 0.01
                      return (
                        <tr key={label}>
                          <td style={{ color: CURVE_COLORS[i], display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ width: 8, height: 8, borderRadius: 999, background: CURVE_COLORS[i], display: 'inline-block' }} />
                            {label}
                          </td>
                          <td>{(vazaoTotal * 1000).toFixed(1)}</td>
                          <td>{pct(exigida)}</td>
                          <td style={{ color: ok ? '#10b981' : '#ef4444', fontWeight: 900 }}>{pct(obtida)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
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

              <Card style={{ padding: 16, overflow: 'hidden' }}>
                <div className="opt-section-title">Valores das Curvas (% Volume)</div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: 'var(--bg)' }}>
                      <th style={{ textAlign: 'left', padding: 9, color: 'var(--text-light)' }}>Nível</th>
                      {MESES.map(m => <th key={m} style={{ padding: 9, color: 'var(--text-light)' }}>{m}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {curvasParaFaixas(result, scenario).map((f, i) => (
                      <tr key={f.Faixa}>
                        <td style={{ padding: 9, borderTop: '1px solid var(--border-light)', fontWeight: 900, color: CURVE_COLORS[i + 1] }}>{f.Faixa}</td>
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
