import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea,
} from 'recharts'
import {
  Activity, CheckCircle2, Database, Play, RefreshCw, Send,
  Download, FileSpreadsheet, FileText, Plus, Search, X,
} from 'lucide-react'
import * as XLSX from './utils/planilha'
import { elementToPngDataUrl } from './components/ChartExportMenu'

const MESES = ['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN', 'JUL', 'AGO', 'SET', 'OUT', 'NOV', 'DEZ']
const MESES_NOMES = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
const NIVEL_LABELS = ['Normal', 'Alerta', 'Seca', 'Seca Severa']
const CURVE_COLORS = ['#2a9d8f', '#d4a017', '#e07b2a', '#d94040']
const MAX_FAIXAS = 4

function labelsForBands(count) {
  return NIVEL_LABELS.slice(0, Math.min(MAX_FAIXAS, Math.max(2, Number(count) || MAX_FAIXAS)))
}

function defaultsForBands(count) {
  const labels = labelsForBands(count)
  const templates = {
    2: { fracDurb: [1, 0.5], fracDsup: [1, 0], garantiaReq: [0.9, 1] },
    3: { fracDurb: [1, 0.8, 0.5], fracDsup: [1, 0.5, 0], garantiaReq: [0.9, 0.98, 1] },
    4: { fracDurb: [1, 1, 0.8, 0.5], fracDsup: [1, 0.8, 0.5, 0], garantiaReq: [0.9, 0.95, 0.98, 1] },
  }
  return { labels, ...templates[labels.length] }
}

const m3sToLps = value => Number(((Number(value) || 0) * 1000).toFixed(3))
const lpsToM3s = value => Math.max(0, (Number(value) || 0) / 1000)

const DEFAULT_SCENARIO = {
  durb: 0.5,
  dsupl: 0,
  quantidadeFaixas: 4,
  ...defaultsForBands(4),
}

function createScenario(index) {
  return {
    id: `cenario-${Date.now()}-${index}`,
    name: `Cenário ${index}`,
    ...DEFAULT_SCENARIO,
    labels: [...DEFAULT_SCENARIO.labels],
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

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
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

function useBoxZoom(data, key = 'data') {
  const [left, setLeft] = useState(null)
  const [right, setRight] = useState(null)
  const [domain, setDomain] = useState(null)
  const activeData = useMemo(() => {
    if (!domain || !data?.length) return data
    const start = data.findIndex(d => String(d[key]) === String(domain.start))
    const end = data.findIndex(d => String(d[key]) === String(domain.end))
    if (start < 0 || end < 0) return data
    return data.slice(Math.min(start, end), Math.max(start, end) + 1)
  }, [data, domain, key])
  return {
    data: activeData,
    isZoomed: Boolean(domain),
    reset: () => setDomain(null),
    props: {
      onMouseDown: e => e?.activeLabel !== undefined && setLeft(e.activeLabel),
      onMouseMove: e => left !== null && e?.activeLabel !== undefined && setRight(e.activeLabel),
      onMouseUp: () => {
        if (left !== null && right !== null && String(left) !== String(right)) setDomain({ start: left, end: right })
        setLeft(null)
        setRight(null)
      },
    },
    area: left !== null && right !== null
      ? <ReferenceArea x1={left} x2={right} strokeOpacity={0.3} fill="#2a9d8f" fillOpacity={0.16} />
      : null,
  }
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
  const labels = scenario.labels || result.faixas_nomes || labelsForBands(scenario.quantidadeFaixas)
  const totalNormal = Number(scenario.durb || 0) + Number(scenario.dsupl || 0)
  return result.matriz_curvas.map((curve, idx) => {
    const nivelIdx = idx + 1
    const vazaoNivel = (Number(scenario.durb || 0) * Number(scenario.fracDurb[nivelIdx] || 0))
      + (Number(scenario.dsupl || 0) * Number(scenario.fracDsup[nivelIdx] || 0))
    const racionamento = totalNormal > 0 ? Math.max(0, Math.min(100, (1 - vazaoNivel / totalNormal) * 100)) : 0
    return {
      Faixa: labels[nivelIdx] || `Faixa ${nivelIdx + 1}`,
      Racionamento: Number(racionamento.toFixed(1)),
      NomeFaixaNormal: labels[0] || 'Normal',
      _tipoFaixa: 'restrita',
      _estadoIndice: nivelIdx,
      _cor: CURVE_COLORS[nivelIdx],
      ...Object.fromEntries(MESES.map((m, i) => [m, Number((Number(curve[i] || 0) * 100).toFixed(1))])),
    }
  })
}

function buildBandChartData(matrizCurvas, labels) {
  if (!matrizCurvas?.length) return { data: [], bands: [] }
  const restricted = labels.slice(1).map((label, index) => ({
    label,
    stateIndex: index + 1,
    thresholdIndex: index,
    key: `faixa_${index + 1}`,
  })).reverse()
  const bands = [
    ...restricted,
    { label: labels[0], stateIndex: 0, key: 'faixa_0' },
  ]
  const data = MESES.map((mes, monthIndex) => {
    const row = { mes }
    let previous = 0
    restricted.forEach(item => {
      const threshold = Number((Number(matrizCurvas[item.thresholdIndex]?.[monthIndex] || 0) * 100).toFixed(2))
      row[item.key] = Math.max(0, threshold - previous)
      row[`limite_${item.stateIndex}`] = threshold
      previous = threshold
    })
    row.faixa_0 = Math.max(0, 100 - previous)
    return row
  })
  return { data, bands }
}

function buildHistoricalVolumeData(result, mesIni, anoIni, labels) {
  if (!result?.volumes_historicos?.length) return []
  const simMesIni = result.mes_inicio ?? mesIni
  const simAnoIni = result.ano_inicio ?? anoIni
  const cap = result.capacidade_hm3 || 1

  const data = result.volumes_historicos.map((vol, index) => {
    const mesDoAno = (simMesIni - 1 + index) % 12
    const anoAtual = simAnoIni + Math.floor((simMesIni - 1 + index) / 12)
    const volPerc = (Number(vol || 0) / cap) * 100
    let estado = 0
    for (let thresholdIndex = 0; thresholdIndex < (result.matriz_curvas?.length || 0); thresholdIndex += 1) {
      const threshold = Number(result.matriz_curvas[thresholdIndex]?.[mesDoAno] || 0) * 100
      if (volPerc < threshold) estado = thresholdIndex + 1
    }

    const row = {
      data: `${MESES[mesDoAno]}/${anoAtual}`,
      origVol: Number(volPerc.toFixed(2)),
      origEstado: estado,
    }
    labels.forEach((_, stateIndex) => { row[`vol_${stateIndex}`] = null })
    return row
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

function HistoricalVolumeTooltip({ active, payload, label, labels }) {
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
      <div style={{ color, fontWeight: 900, textTransform: 'uppercase', fontSize: 9.5 }}>{labels[point.origEstado]}</div>
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
  const levelsChartRef = useRef(null)
  const volumeChartRef = useRef(null)
  const sideRef = useRef(null)

  const scenario = scenarios.find(s => s.id === activeScenarioId) || scenarios[0]
  const result = resultsByScenario[activeScenarioId] || null
  const loading = loadingId === activeScenarioId
  const progress = progressByScenario[activeScenarioId] || 0

  useEffect(() => {
    const updateSideHeight = () => {
      const element = sideRef.current
      if (!element || window.innerWidth <= 720) {
        element?.style.removeProperty('--opt-side-height')
        return
      }
      const visibleTop = Math.max(74, element.getBoundingClientRect().top)
      element.style.setProperty('--opt-side-height', `${Math.max(240, window.innerHeight - visibleTop - 12)}px`)
    }
    const frame = window.requestAnimationFrame(updateSideHeight)
    window.addEventListener('resize', updateSideHeight)
    window.addEventListener('scroll', updateSideHeight, { passive: true })
    return () => {
      window.cancelAnimationFrame(frame)
      window.removeEventListener('resize', updateSideHeight)
      window.removeEventListener('scroll', updateSideHeight)
    }
  }, [msg, result, loading, scenario?.quantidadeFaixas])

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

  const setFaixaNome = (idx, value) => {
    updateScenario(prev => {
      const labels = [...(prev.labels || labelsForBands(prev.quantidadeFaixas))]
      labels[idx] = value
      return { labels }
    })
  }

  const restoreFaixaNome = (idx) => {
    if (String(scenario.labels?.[idx] || '').trim()) return
    setFaixaNome(idx, labelsForBands(scenario.quantidadeFaixas)[idx])
  }

  const setQuantidadeFaixas = (value) => {
    const quantidadeFaixas = Math.min(MAX_FAIXAS, Math.max(2, Number(value) || 2))
    updateScenario(prev => {
      const defaults = defaultsForBands(quantidadeFaixas)
      const resize = (values, fallback) => Array.from(
        { length: quantidadeFaixas },
        (_, index) => Number(values?.[index] ?? fallback[index])
      )
      return {
        quantidadeFaixas,
        labels: Array.from(
          { length: quantidadeFaixas },
          (_, index) => prev.labels?.[index] ?? defaults.labels[index]
        ),
        fracDurb: resize(prev.fracDurb, defaults.fracDurb),
        fracDsup: resize(prev.fracDsup, defaults.fracDsup),
        garantiaReq: resize(prev.garantiaReq, defaults.garantiaReq),
      }
    })
    setResultsByScenario(prev => ({ ...prev, [activeScenarioId]: null }))
    setZoomDomain(null)
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
    const nomesConfigurados = (scenario.labels || []).map(label => String(label).trim())
    if (nomesConfigurados.length !== scenario.quantidadeFaixas || nomesConfigurados.some(nome => !nome)) {
      setMsg({ type: 'error', text: 'Preencha o nome de todas as faixas de operação.' })
      return
    }
    if (new Set(nomesConfigurados.map(nome => nome.toLocaleLowerCase('pt-BR'))).size !== nomesConfigurados.length) {
      setMsg({ type: 'error', text: 'Use um nome diferente para cada faixa de operação.' })
      return
    }
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
      quantidade_faixas: scenarioSnapshot.quantidadeFaixas,
      faixas_nomes: nomesConfigurados,
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
    setMsg({ type: 'success', text: 'Curvas enviadas para o simulador com reservatório e demanda preenchidos automaticamente.' })
  }

  const labelsAtivos = scenario?.labels || result?.faixas_nomes || labelsForBands(scenario?.quantidadeFaixas)

  const performanceRows = () => labelsAtivos.map((label, i) => {
    const vazaoTotal = (Number(scenario.durb || 0) * Number(scenario.fracDurb[i] || 0))
      + (Number(scenario.dsupl || 0) * Number(scenario.fracDsup[i] || 0))
    return {
      'Nível Operacional': label,
      'Vazão Total (L/s)': Number((vazaoTotal * 1000).toFixed(3)),
      'Garantia Requerida': Number(((Number(scenario.garantiaReq[i] || 0)) * 100).toFixed(2)),
      'Garantia Obtida': Number(((Number(result?.garantias_obtidas?.[i] || 0)) * 100).toFixed(2)),
    }
  })

  const nomesResultado = result?.faixas_nomes || labelsForBands(scenario?.quantidadeFaixas)
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
    'Modo Operação': labelsAtivos[nomesResultado.indexOf(d['Modo Operação'])] || d['Modo Operação'] || labelsAtivos[0],
  }))

  const exportCurvesCSV = () => {
    if (!result?.matriz_curvas) return
    let csv = `Mês;${labelsAtivos.slice(1).join(';')}\n`
    MESES.forEach((mes, i) => {
      csv += `${mes};${result.matriz_curvas.map(curve => (Number(curve[i] || 0) * 100).toFixed(2)).join(';')}\n`
    })
    downloadText(`curvas_${safeName(reservatorio)}.csv`, csv)
  }

  const exportVolumesCSV = () => {
    if (!result?.volumes_historicos?.length) return
    let csv = 'Data;Volume Absoluto (hm³);Volume Percentual (%)\n'
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
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(rows), 'Simulação')
    XLSX.writeFile(wb, `simulacao_${safeName(reservatorio)}.xlsx`)
  }

  const exportOptimizationExcel = () => {
    if (!result) return
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(performanceRows()), 'Desempenho')
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(curvasParaFaixas(result, scenario)), 'Curvas')
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(buildHistoricalVolumeData(result, mesIni, anoIni, labelsAtivos).map(d => ({
      Data: d.data,
      'Volume Percentual (%)': d.origVol,
      Estado: labelsAtivos[d.origEstado],
    }))), 'Volumes')
    const rows = simulationRows()
    if (rows.length) XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(rows), 'Simulação')
    XLSX.writeFile(wb, `otimizacao_${safeName(reservatorio)}.xlsx`)
  }

  const exportOptimizationReport = async () => {
    if (!result) return
    setMsg({ type: 'success', text: 'Gerando relat\u00f3rio visual...' })
    try {
      const [levelsImage, volumeImage] = await Promise.all([
        elementToPngDataUrl(levelsChartRef.current),
        elementToPngDataUrl(volumeChartRef.current),
      ])
      const performance = performanceRows()
      const curves = curvasParaFaixas(result, scenario)
      const performanceHtml = performance.map(row => `
        <tr>
          <td>${escapeHtml(row['Nível Operacional'])}</td>
          <td>${escapeHtml(row['Vazão Total (L/s)'])}</td>
          <td>${escapeHtml(row['Garantia Requerida'])}%</td>
          <td>${escapeHtml(row['Garantia Obtida'])}%</td>
        </tr>`).join('')
      const curvesHtml = curves.map(row => `
        <tr>
          <td>${escapeHtml(row.Faixa)}</td>
          ${MESES.map(month => `<td>${escapeHtml(row[month])}</td>`).join('')}
        </tr>`).join('')
      const html = `<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Relat&oacute;rio da otimiza&ccedil;&atilde;o - ${escapeHtml(reservatorio)}</title>
  <style>
    body{font-family:Arial,sans-serif;margin:0;color:#1e1208;background:#fff}main{max-width:1120px;margin:0 auto;padding:32px}
    h1{font-size:24px;margin:0 0 5px}h2{font-size:16px;margin:28px 0 10px;color:#c46318}p{margin:4px 0;color:#6f4b32}
    .meta{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:22px 0}.meta div{border:1px solid #ecdcc8;padding:10px}.meta b{display:block;font-size:11px;color:#9a7055;margin-bottom:4px}
    figure{margin:12px 0 24px;border:1px solid #ecdcc8;padding:12px;break-inside:avoid}figure img{display:block;width:100%;height:auto}figcaption{font-size:12px;font-weight:700;margin-bottom:9px}
    table{width:100%;border-collapse:collapse;font-size:11px}th,td{border:1px solid #ecdcc8;padding:7px;text-align:center}th{background:#fdf6ee;color:#6f4b32}td:first-child,th:first-child{text-align:left;font-weight:700}
    footer{margin-top:28px;padding-top:12px;border-top:1px solid #ecdcc8;font-size:10px;color:#9a7055}@media print{main{max-width:none;padding:12mm}figure{page-break-inside:avoid}}
  </style>
</head>
<body><main>
  <h1>Relat&oacute;rio da otimiza&ccedil;&atilde;o de n&iacute;veis meta</h1>
  <p>${escapeHtml(reservatorio)} &middot; ${escapeHtml(scenario.name)}</p>
  <div class="meta">
    <div><b>Demanda</b>${escapeHtml(m3sToLps(scenario.durb))} L/s</div>
    <div><b>Per&iacute;odo</b>${MESES[mesIni - 1]}/${anoIni} a ${MESES[mesFim - 1]}/${anoFim}</div>
    <div><b>Probabilidade de aflu&ecirc;ncia</b>${escapeHtml((prob * 100).toFixed(0))}%</div>
    <div><b>Itera&ccedil;&otilde;es PSO</b>${escapeHtml(iters)}</div>
  </div>
  ${levelsImage ? `<figure><figcaption>Curvas dos n&iacute;veis meta</figcaption><img src="${levelsImage}" alt="Curvas dos n&iacute;veis meta"></figure>` : ''}
  <h2>Desempenho por n&iacute;vel operacional</h2>
  <table><thead><tr><th>N&iacute;vel</th><th>Vaz&atilde;o total (L/s)</th><th>Garantia requerida</th><th>Garantia obtida</th></tr></thead><tbody>${performanceHtml}</tbody></table>
  ${volumeImage ? `<figure><figcaption>S&eacute;rie hist&oacute;rica de volumes</figcaption><img src="${volumeImage}" alt="S&eacute;rie hist&oacute;rica de volumes"></figure>` : ''}
  <h2>Valores mensais das curvas (% do volume)</h2>
  <table><thead><tr><th>N&iacute;vel</th>${MESES.map(month => `<th>${month}</th>`).join('')}</tr></thead><tbody>${curvesHtml}</tbody></table>
  <footer>Gerado pelo Sistema de Suporte &agrave; Decis&atilde;o em ${new Date().toLocaleString('pt-BR')}.</footer>
</main></body></html>`
      downloadText(`relatorio_otimizacao_${safeName(reservatorio)}.html`, html, 'text/html;charset=utf-8;')
      setMsg({ type: 'success', text: 'Relat\u00f3rio visual exportado com gr\u00e1ficos e valores dos n\u00edveis.' })
    } catch (error) {
      setMsg({ type: 'error', text: `Falha ao gerar relat\u00f3rio: ${error.message}` })
    }
  }
  const chartModel = buildBandChartData(result?.matriz_curvas, labelsAtivos)
  const chartDataVolume = buildHistoricalVolumeData(result, mesIni, anoIni, labelsAtivos)
  const bandZoom = useBoxZoom(chartModel.data, 'mes')
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
      <style>{`.opt-search{position:relative;width:88%;margin:0 auto}.opt-search-icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:var(--text-light);pointer-events:none}.opt-search-input{width:100%;border:1.5px solid var(--border);border-radius:8px;background:var(--card);color:var(--text);padding:9px 10px 9px 30px;font:12px 'Sora',sans-serif;outline:none}.opt-search-input:focus{border-color:var(--orange-deep);box-shadow:0 0 0 3px var(--orange-pale)}.opt-search-menu{position:absolute;z-index:40;left:0;right:0;top:calc(100% + 4px);max-height:210px;overflow:auto;border:1.5px solid var(--border);border-radius:8px;background:var(--card);box-shadow:var(--shadow);padding:4px}.opt-search-item{display:block;width:100%;text-align:left;border:0;border-radius:6px;background:transparent;color:var(--text);padding:8px 9px;font:700 11.5px 'Sora',sans-serif;cursor:pointer}.opt-search-item:hover{background:var(--orange-pale);color:var(--orange-deep)}.opt-input-card{border:1.5px solid var(--border);border-radius:10px;background:color-mix(in srgb,var(--card) 82%,var(--bg));padding:12px}.opt-demand-grid{display:grid;grid-template-columns:1fr;gap:12px}.opt-demand-input{width:100%;text-align:center;border-radius:8px!important;padding:10px!important;font-size:13px!important;font-weight:800!important}.opt-hydro-select{display:block;margin:0 auto;width:180px;text-align:center}.opt-tab{display:inline-flex;align-items:center;gap:7px}.opt-tab.active{background:var(--orange-pale);color:var(--orange-deep)}.opt-tab-add{border:0;background:transparent;color:var(--text-light);padding:9px 12px;cursor:pointer}.opt-tab-add:hover{color:var(--orange-deep);background:var(--orange-pale)}.opt-tab-close{border:0;background:transparent;color:inherit;padding:0;line-height:0;cursor:pointer;opacity:.7}.opt-tab-close:hover{opacity:1;color:var(--red)}.opt-dark .opt-ghost{background:#0a0604;color:var(--text-light);border-color:var(--border)}.opt-dark .opt-ghost:hover{background:var(--orange-pale);color:var(--orange-deep);border-color:var(--orange-deep)}`}</style>
      <style>{`.opt-side,.opt-side-top,.opt-side-body,.opt-tab,.opt-btn,.opt-label,.opt-period-name,.opt-matrix-title,.opt-perm-table,.opt-section-title,.opt-search-input,.opt-search-item,.opt-side select,.opt-side input[type=number],.opt-matrix input,.opt-run{font-family:'Sora',sans-serif}.opt-input-card{background:transparent!important;border:0!important;border-radius:0!important;padding:0!important}.opt-side-body{gap:18px}.opt-matrix-title{color:var(--text-light);letter-spacing:0}.opt-matrix-labels{gap:4px;margin-bottom:4px}.opt-matrix-labels span{font-family:'Sora',sans-serif;font-weight:800}.opt-demand-input,.opt-side select,.opt-side input[type=number],.opt-matrix input{border-radius:var(--radius-xs)!important}.opt-matrix{gap:4px;border:0!important;border-radius:0!important;background:transparent!important;overflow:visible}.opt-matrix input{background:var(--card);color:var(--text);border:1.5px solid var(--border)!important;box-shadow:none!important;padding:8px 4px}.opt-matrix input:last-child{border-right:1.5px solid var(--border)!important}.sim-root:not(.opt-dark) .opt-side{background:var(--card);border:1.5px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow)}.sim-root:not(.opt-dark) .opt-side-top,.sim-root:not(.opt-dark) .opt-side-body{background:var(--card);border-color:var(--border-light)}.sim-root:not(.opt-dark) .opt-tabbar{background:var(--card);border-color:var(--border-light)}.sim-root:not(.opt-dark) .opt-tab{background:transparent;color:var(--text-light);border-color:var(--border-light)}.sim-root:not(.opt-dark) .opt-tab.active,.sim-root:not(.opt-dark) .opt-tab-add:hover{background:var(--orange-pale);color:var(--orange-deep)}.sim-root:not(.opt-dark) .opt-search-input,.sim-root:not(.opt-dark) .opt-search-menu{background:var(--card);color:var(--text);border-color:var(--border)}.sim-root:not(.opt-dark) .opt-search-item{color:var(--text)}.sim-root:not(.opt-dark) .opt-search-item:hover{background:var(--orange-pale);color:var(--orange-deep)}.sim-root:not(.opt-dark) .opt-side select,.sim-root:not(.opt-dark) .opt-side input[type=number],.sim-root:not(.opt-dark) .opt-matrix input{background:var(--card);color:var(--text);border-color:var(--border)!important}.sim-root:not(.opt-dark) .opt-ghost{background:var(--card);color:var(--text-mid);border-color:var(--border)}.opt-dark .opt-side select,.opt-dark .opt-side input[type=number],.opt-dark .opt-matrix input{background:#080503;color:var(--text);border-color:var(--border)!important}`}</style>

      <style>{`.opt-side{position:sticky!important;top:74px!important;height:var(--opt-side-height,calc(100vh - 190px));max-height:var(--opt-side-height,calc(100vh - 190px));display:flex;flex-direction:column;overflow:hidden!important}.opt-side-scroll{min-height:0;flex:1;overflow-y:auto;overscroll-behavior:contain;scrollbar-gutter:stable}.opt-side-scroll::-webkit-scrollbar{width:8px}.opt-side-scroll::-webkit-scrollbar-thumb{background:var(--border);border:2px solid var(--card);border-radius:8px}.opt-run-dock{flex:none;padding:12px 16px 16px;border-top:1.5px solid var(--border-light);background:var(--card);position:relative;z-index:2}.opt-run-dock .opt-run{margin:0}.opt-band-names{display:flex;flex-direction:column;gap:6px;margin:10px auto 0;width:88%}.opt-band-name-row{display:grid;grid-template-columns:10px minmax(0,1fr);align-items:center;gap:7px}.opt-band-dot{width:8px;height:8px;border-radius:50%;display:block}.opt-band-name-row input{width:100%;min-width:0;border:1.5px solid var(--border);border-radius:var(--radius-xs);background:var(--card);color:var(--text);font:700 11px 'Sora',sans-serif;padding:7px 8px;outline:none}.opt-band-name-row input:focus{border-color:var(--orange-deep);box-shadow:0 0 0 2px var(--orange-pale)}.opt-dark .opt-band-name-row input{background:#080503;color:var(--text);border-color:var(--border)}@media(min-width:721px){.opt-layout{grid-template-columns:340px minmax(0,1fr)!important}}@media(max-width:720px){.opt-side{position:relative!important;top:0!important;height:auto;max-height:none;overflow:visible!important}.opt-side-scroll{overflow:visible;scrollbar-gutter:auto}}`}</style>

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
              <button className="opt-btn opt-ghost" onClick={exportOptimizationExcel}><FileSpreadsheet size={14} /> Excel Resultados</button>
              <button className="opt-btn opt-ghost" onClick={exportOptimizationReport}><FileText size={14} /> Salvar relat&oacute;rio</button>
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
        <aside className="opt-side" ref={sideRef}>
          <div className="opt-side-scroll">
          <div className="opt-side-top">
            <div>
              <label className="opt-label center">Reservatório</label>
              <ReservatorioSearch lista={lista} value={reservatorio} onChange={setReservatorio} />
            </div>

            <div>
              <label className="opt-label">Prob. de Afluência</label>
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
                <span className="opt-period-name">Início</span>
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
              <label className="opt-label center">Mês Inicial do Ano Hidrológico</label>
              <select className="opt-month opt-hydro-select" value={ninicio} onChange={e => setNinicio(Number(e.target.value))}>
                {MESES_NOMES.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}
              </select>
            </div>

            <div>
              <label className="opt-label center">Faixas de Operação</label>
              <select className="opt-month opt-hydro-select" value={scenario.quantidadeFaixas} onChange={e => setQuantidadeFaixas(e.target.value)}>
                {[2, 3, 4].map(quantidade => <option key={quantidade} value={quantidade}>{quantidade} faixas</option>)}
              </select>
              <div className="opt-band-names" aria-label="Nomes das faixas de operação">
                {scenario.labels.map((label, index) => (
                  <label className="opt-band-name-row" key={index}>
                    <span className="opt-band-dot" style={{ background: CURVE_COLORS[index] }} />
                    <input
                      type="text"
                      maxLength={32}
                      aria-label={`Nome da faixa ${index + 1}`}
                      value={label}
                      onChange={e => setFaixaNome(index, e.target.value)}
                      onBlur={() => restoreFaixaNome(index)}
                    />
                  </label>
                ))}
              </div>
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
                <span className="opt-label center">Demanda (L/s)</span>
                <input className="opt-demand-input" type="number" min="0" step="10" value={m3sToLps(scenario.durb)} onChange={e => updateScenario({ durb: lpsToM3s(e.target.value) })} />
              </label>
            </div>

            <div className="opt-input-card">
              <p className="opt-matrix-title">Garantias Requeridas (%)</p>
              <div className="opt-matrix-labels" style={{ gridTemplateColumns: `repeat(${labelsAtivos.length}, minmax(0, 1fr))` }}>{labelsAtivos.map((label, i) => <span key={label} style={{ color: CURVE_COLORS[i] }}>{label}</span>)}</div>
              <div className="opt-matrix" style={{ gridTemplateColumns: `repeat(${labelsAtivos.length}, minmax(0, 1fr))` }}>
                {scenario.garantiaReq.map((v, i) => (
                  <input key={i} type="number" step="1" min="0" max="100" value={Number((v * 100).toFixed(1))} onChange={e => setArray('garantiaReq', i, Number(e.target.value) / 100)} />
                ))}
              </div>
            </div>

            <div className="opt-input-card">
              <p className="opt-matrix-title">Atendimento da Demanda (%)</p>
              <div className="opt-matrix" style={{ gridTemplateColumns: `repeat(${labelsAtivos.length}, minmax(0, 1fr))` }}>
                {scenario.fracDurb.map((v, i) => (
                  <input key={i} type="number" step="1" min="0" max="100" value={Number((v * 100).toFixed(1))} onChange={e => setArray('fracDurb', i, Number(e.target.value) / 100)} />
                ))}
              </div>
            </div>

          </div>
          </div>

          <div className="opt-run-dock">
            <button className="opt-btn opt-run" onClick={handleRun} disabled={loading || !reservatorio} style={{ opacity: loading ? 0.7 : 1 }}>
              {loading ? <RefreshCw size={14} className="opt-spin" /> : <Play size={14} />}
              {loading ? `Otimizando ${progress}%` : 'Otimizar Cenário'}
            </button>
          </div>
        </aside>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {!result && !loading && (
            <Card style={{ padding: 48, textAlign: 'center' }}>
              <Database size={32} color="var(--orange)" style={{ opacity: 0.4, marginBottom: 10 }} />
              <div style={{ fontSize: 14, fontWeight: 900, marginBottom: 5 }}>Pronto para otimizar</div>
              <div style={{ fontSize: 12, color: 'var(--text-light)' }}>Escolha o reservatório e as faixas de operação para calcular as curvas-guia.</div>
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
                  {bandZoom.isZoomed && (
                    <button className="opt-btn opt-ghost" onClick={bandZoom.reset} style={{ marginLeft: 'auto', padding: '6px 10px', fontSize: 10 }}>
                      Resetar Zoom
                    </button>
                  )}
                </div>
                <div ref={levelsChartRef} data-chart-name="curvas_niveis_meta" style={{ height: 330 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={bandZoom.data} margin={{ top: 10, right: 12, bottom: 0, left: -18 }} {...bandZoom.props}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#ecdcc8" />
                      <XAxis dataKey="mes" tick={{ fill: '#9a7055', fontSize: 11 }} tickLine={false} />
                      <YAxis domain={[0, 100]} tick={{ fill: '#9a7055', fontSize: 11 }} tickFormatter={v => `${v}%`} tickLine={false} />
                      <Tooltip formatter={v => `${Number(v).toFixed(2)}%`} />
                      <Legend />
                      {chartModel.bands.map(band => (
                        <Area key={band.key} isAnimationActive={false} type="linear" stackId="meta" name={band.label} dataKey={band.key} stroke={CURVE_COLORS[band.stateIndex]} fill={CURVE_COLORS[band.stateIndex]} fillOpacity={0.48} />
                      ))}
                      {bandZoom.area}
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              <Card style={{ padding: 16 }}>
                <div className="opt-section-title">Desempenho e Vazões</div>
                <table className="opt-perm-table">
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'left' }}>Nível Operacional</th>
                      <th>Vazão Total (L/s)</th>
                      <th>Garantia Requerida</th>
                      <th>Garantia Obtida</th>
                    </tr>
                  </thead>
                  <tbody>
                    {labelsAtivos.map((label, i) => {
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
                      <Activity size={16} color="var(--orange)" /> Simulação Histórica de Volumes (%)
                    </div>
                    {zoomDomain && (
                      <button className="opt-btn opt-ghost" onClick={() => setZoomDomain(null)} style={{ padding: '6px 10px', fontSize: 10 }}>
                        Resetar Zoom
                      </button>
                    )}
                  </div>
                  <div ref={volumeChartRef} data-chart-name="serie_historica_volumes" style={{ height: 300, userSelect: 'none' }}>
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
                        <Tooltip content={<HistoricalVolumeTooltip labels={labelsAtivos} />} />
                        {refAreaLeft && refAreaRight && <ReferenceArea x1={refAreaLeft} x2={refAreaRight} strokeOpacity={0.3} fill="#2a9d8f" fillOpacity={0.16} />}
                        {labelsAtivos.map((label, stateIndex) => (
                          <Area key={label} isAnimationActive={false} type="linear" dataKey={`vol_${stateIndex}`} name={label} stroke={CURVE_COLORS[stateIndex]} strokeWidth={2.5} fill={CURVE_COLORS[stateIndex]} fillOpacity={0.30 + stateIndex * 0.025} connectNulls={false} />
                        ))}
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              )}

              <Card style={{ padding: 16, overflow: 'hidden' }}>
                <div className="opt-section-title">Valores das Curvas (% do Volume)</div>
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
                        <td style={{ padding: 9, borderTop: '1px solid var(--border-light)', fontWeight: 900, color: CURVE_COLORS[i + 1] || CURVE_COLORS[0] }}>{f.Faixa}</td>
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
