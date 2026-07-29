import { useEffect, useMemo, useState } from 'react'
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, ReferenceArea,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import {
  Activity, BarChart3, BrainCircuit, Check, Download, RefreshCw, Search,
  SlidersHorizontal, TrendingUp, Waves,
} from 'lucide-react'
import * as XLSX from 'xlsx'

const MESES = ['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN', 'JUL', 'AGO', 'SET', 'OUT', 'NOV', 'DEZ']

function apiBase(apiUrl) {
  return apiUrl || import.meta.env?.VITE_API_URL || 'http://127.0.0.1:8000'
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(typeof err.detail === 'string' ? err.detail : 'Falha ao processar.')
  }
  return res.json()
}

function Card({ children, style }) {
  return <div style={{ background: 'var(--card)', border: '1.5px solid var(--border)', borderRadius: 'var(--radius)', boxShadow: 'var(--shadow)', ...style }}>{children}</div>
}

function Field({ label, children }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <span style={{ fontSize: 10.5, fontWeight: 800, color: 'var(--text-light)', textTransform: 'uppercase' }}>{label}</span>
      {children}
    </label>
  )
}

function Control({ as, children, style, ...props }) {
  const base = {
    width: '100%',
    border: '1.5px solid var(--border)',
    borderRadius: 'var(--radius-xs)',
    background: 'var(--card)',
    color: 'var(--text)',
    padding: '8px 10px',
    font: "700 12px 'Sora', sans-serif",
    outline: 'none',
  }
  if (as === 'select') return <select {...props} style={{ ...base, cursor: 'pointer', ...style }}>{children}</select>
  return <input {...props} style={{ ...base, ...style }} />
}

function Metric({ label, value, sub, icon: Icon }) {
  return (
    <Card style={{ padding: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
        <div>
          <div style={{ fontSize: 10.5, fontWeight: 900, color: 'var(--text-light)', textTransform: 'uppercase' }}>{label}</div>
          <div style={{ fontSize: 22, fontWeight: 900, color: 'var(--orange-deep)', marginTop: 5 }}>{value}</div>
          {sub && <div style={{ fontSize: 11, color: 'var(--text-light)' }}>{sub}</div>}
        </div>
        {Icon && <Icon size={18} color="var(--orange)" />}
      </div>
    </Card>
  )
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: 'var(--card)', border: '1.5px solid var(--border)', borderRadius: 8, padding: '8px 10px', boxShadow: 'var(--shadow)', fontSize: 11 }}>
      <div style={{ fontWeight: 900, color: 'var(--text)', marginBottom: 4 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, fontWeight: 800 }}>
          {p.name}: {Number(p.value || 0).toFixed(3)}
        </div>
      ))}
    </div>
  )
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

function ZoomReset({ zoom }) {
  if (!zoom?.isZoomed) return null
  return <button className="pv-btn pv-ghost" onClick={zoom.reset} style={{ padding: '6px 10px', fontSize: 10 }}>Resetar zoom</button>
}

function safeSheetName(name) {
  return String(name).replace(/[\\/?*[\]:]/g, '').slice(0, 31) || 'Planilha'
}

function exportVazoesGarantia(permanencia, reservatorio) {
  const wb = XLSX.utils.book_new()
  if (permanencia?.resultados?.length) {
    const qxxRows = permanencia.resultados.map(row => ({
      'Vazão de garantia': row.referencia,
      'Garantia requerida (%)': row.garantia_requerida,
      'Garantia obtida (%)': row.garantia_obtida,
      'Vazão (L/s)': Number(row.vazao_m3s || 0) * 1000,
    }))
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(qxxRows), safeSheetName('Vazoes de garantia'))
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(permanencia.curva || []), safeSheetName('Curva garantia'))
  }
  XLSX.writeFile(wb, `vazoes_garantia_${String(reservatorio || 'reservatorio').replace(/\s+/g, '_')}.xlsx`)
}

function exportPrevisao(previsao, reservatorio, importancia) {
  const wb = XLSX.utils.book_new()
  if (previsao?.previsao?.length) {
    const validacaoPorData = new Map((previsao.validacao || []).map(row => [row.data, row]))
    const historicoRows = (previsao.historico || []).map(row => {
      const validacao = validacaoPorData.get(row.data)
      return {
        Data: row.data,
        Tipo: validacao ? 'Histórico usado na validação' : 'Histórico usado no treino',
        'Vazão observada (m³/s)': row.vazao_m3s,
        'Afluência observada (hm³/mês)': row.afluencia_hm3_mes,
        'Previsão retrospectiva (m³/s)': validacao?.previsto_m3s ?? null,
        'Erro retrospectivo (m³/s)': validacao?.erro_m3s ?? null,
        'Previsão futura (m³/s)': null,
        'Afluência futura (hm³/mês)': null,
      }
    })
    const previsaoRows = (previsao.previsao || []).map(row => ({
      Data: row.data,
      Tipo: 'Previsão futura',
      'Vazão observada (m³/s)': null,
      'Afluência observada (hm³/mês)': null,
      'Previsão retrospectiva (m³/s)': null,
      'Erro retrospectivo (m³/s)': null,
      'Previsão futura (m³/s)': row.vazao_m3s,
      'Afluência futura (hm³/mês)': row.afluencia_hm3_mes,
    }))
    const metricas = [{
      Reservatório: previsao.reservatorio || reservatorio,
      Método: previsao.metodo,
      Modelo: String(previsao.modelo || '').toUpperCase(),
      K: previsao.parametros?.k,
      'Defasagens de afluência': previsao.parametros?.lags,
      'Horizonte (meses)': previsao.parametros?.horizonte,
      'Defasagem climática (meses)': previsao.parametros?.lag_climatico,
      'Meses de teste': previsao.parametros?.teste_meses,
      'Indicadores selecionados': (previsao.indicadores || []).map(item => item.label).join(', ') || 'Nenhum (modelo hidrológico e sazonal)',
      'Período inicial': previsao.periodo?.inicio,
      'Período final': previsao.periodo?.fim,
      'Meses usados': previsao.periodo?.meses,
      'MAE (m³/s)': previsao.metricas?.mae_m3s,
      'RMSE (m³/s)': previsao.metricas?.rmse_m3s,
      NSE: previsao.metricas?.nse,
      Correlação: previsao.metricas?.correlacao,
      'Viés (m³/s)': previsao.metricas?.vies_m3s,
    }]
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet([...historicoRows, ...previsaoRows]), safeSheetName('Serie mensal e previsao'))
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(previsao.validacao || []), safeSheetName('Validacao'))
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(previsao.previsao || []), safeSheetName('Previsao futura'))
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(metricas), safeSheetName('Parametros e metricas'))
    if (importancia?.indicadores?.length) {
      const rows = importancia.indicadores.map(item => ({
        Posição: item.posicao,
        Indicador: item.label,
        'Contribuição relativa (%)': item.contribuicao_relativa_percent,
        'Variabilidade individual R² (%)': item.variabilidade_individual_r2_percent,
        'Aumento do MSE por permutação (%)': item.aumento_mse_percent,
        Selecionado: (previsao.indicadores || []).some(selected => selected.id === item.id) ? 'Sim' : 'Não',
      }))
      XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(rows), safeSheetName('Importancia indicadores'))
    }
  }
  XLSX.writeFile(wb, `previsao_afluencia_${String(reservatorio || 'reservatorio').replace(/\s+/g, '_')}.xlsx`)
}

export default function PrevisaoVazoes({ apiUrl, darkMode = false, mode = 'qxx' }) {
  const base = apiBase(apiUrl)
  const [reservatorios, setReservatorios] = useState([])
  const [query, setQuery] = useState('')
  const [reservatorio, setReservatorio] = useState('')
  const [activeTab, setActiveTab] = useState(mode)
  const [periodo, setPeriodo] = useState({ mesInicial: 1, anoInicial: 1911, mesFinal: 12, anoFinal: 2021 })
  const [qxx, setQxx] = useState({ volInicial: 100 })
  const [knn, setKnn] = useState({
    modelo: 'knn',
    k: 5,
    lags: 12,
    horizonte: 3,
    lagClimatico: 3,
    testeMeses: 120,
    metodoImportancia: 'permutacao',
  })
  const [loading, setLoading] = useState(false)
  const [loadingImportance, setLoadingImportance] = useState(false)
  const [msg, setMsg] = useState(null)
  const [permanencia, setPermanencia] = useState(null)
  const [previsao, setPrevisao] = useState(null)
  const [importancia, setImportancia] = useState(null)
  const [selectedIndicators, setSelectedIndicators] = useState([])

  useEffect(() => {
    fetch(`${base}/api/reservatorios`)
      .then(r => r.json())
      .then(data => {
        const lista = (Array.isArray(data) ? data : []).map(r => r.CORPO).filter(Boolean)
        setReservatorios(lista)
        if (lista[0]) {
          setReservatorio(lista[0])
          setQuery(lista[0])
        }
      })
      .catch(e => setMsg({ type: 'error', text: e.message }))
  }, [base])

  const filtrados = useMemo(() => reservatorios.filter(r => r.toLowerCase().includes(query.toLowerCase())).slice(0, 80), [reservatorios, query])

  const payloadBase = () => ({
    reservatorio,
    mes_inicial: Number(periodo.mesInicial),
    ano_inicial: Number(periodo.anoInicial),
    mes_final: Number(periodo.mesFinal),
    ano_final: Number(periodo.anoFinal),
  })

  const runQxx = async () => {
    if (!reservatorio) return
    setLoading(true)
    setMsg(null)
    try {
      const data = await postJson(`${base}/api/vazoes/permanencia`, {
        ...payloadBase(),
        vol_inicial_percent: Number(qxx.volInicial),
      })
      setPermanencia(data)
      setMsg({ type: 'success', text: 'Vazões de garantia calculadas.' })
    } catch (e) {
      setMsg({ type: 'error', text: e.message })
    } finally {
      setLoading(false)
    }
  }

  const forecastPayload = () => ({
    ...payloadBase(),
    modelo: knn.modelo,
    k: Number(knn.k),
    lags: Number(knn.lags),
    horizonte: Number(knn.horizonte),
    lag_climatico: Number(knn.lagClimatico),
    teste_meses: Number(knn.testeMeses),
  })

  const updateForecastConfig = changes => {
    setKnn(current => ({ ...current, ...changes }))
    setImportancia(null)
    setSelectedIndicators([])
    setPrevisao(null)
  }

  const recommendedIndicators = items => {
    const selected = []
    for (const item of items) {
      const conflictsWithDipole = item.id === 'dipolo'
        ? selected.includes('tna') || selected.includes('tsa')
        : ['tna', 'tsa'].includes(item.id) && selected.includes('dipolo')
      if (!conflictsWithDipole) selected.push(item.id)
      if (selected.length === 3) break
    }
    return selected
  }

  const analyzeIndicators = async () => {
    if (!reservatorio) return
    setLoadingImportance(true)
    setMsg(null)
    try {
      const data = await postJson(`${base}/api/previsao/importancia`, {
        ...forecastPayload(),
        metodo_importancia: knn.metodoImportancia,
      })
      setImportancia(data)
      setSelectedIndicators(recommendedIndicators(data.indicadores))
      setPrevisao(null)
      setMsg({ type: 'success', text: 'Indicadores classificados. Três variáveis não redundantes foram pré-selecionadas.' })
    } catch (e) {
      setMsg({ type: 'error', text: e.message })
    } finally {
      setLoadingImportance(false)
    }
  }

  const toggleIndicator = indicator => {
    setSelectedIndicators(current => current.includes(indicator)
      ? current.filter(item => item !== indicator)
      : [...current, indicator])
    setPrevisao(null)
  }

  const runKnn = async () => {
    if (!reservatorio) return
    setLoading(true)
    setMsg(null)
    try {
      const data = await postJson(`${base}/api/previsao/executar`, {
        ...forecastPayload(),
        indicadores: selectedIndicators,
      })
      setPrevisao(data)
      setMsg({ type: 'success', text: `Previsão ${knn.modelo.toUpperCase()} concluída.` })
    } catch (e) {
      setMsg({ type: 'error', text: e.message })
    } finally {
      setLoading(false)
    }
  }

  const forecastChart = useMemo(() => {
    const hist = (previsao?.historico || []).slice(-48).map(d => ({ data: d.data, Historico: d.vazao_m3s }))
    const fut = (previsao?.previsao || []).map(d => ({ data: d.data, Previsao: d.vazao_m3s }))
    return [...hist, ...fut]
  }, [previsao])

  const validationChart = useMemo(() => (previsao?.validacao || []).map(d => ({ data: d.data, Observado: d.observado_m3s, Previsto: d.previsto_m3s })), [previsao])
  const qxxCurve = useMemo(() => (permanencia?.curva || []).map(row => ({
    ...row,
    vazao_ls: Number(row.vazao_m3s || 0) * 1000,
  })), [permanencia])
  const qxxZoom = useBoxZoom(qxxCurve, 'garantia')
  const forecastZoom = useBoxZoom(forecastChart)
  const validationZoom = useBoxZoom(validationChart)
  const q100 = permanencia?.destaques?.Q100 || permanencia?.vazao_plena
  const q95 = permanencia?.destaques?.Q95
  const q90 = permanencia?.destaques?.Q90
  const isFixedMode = mode === 'qxx' || mode === 'knn'
  const pageTitle = activeTab === 'qxx' ? 'Vazões de Garantia' : 'Previsão de Afluência'
  const pageSubtitle = activeTab === 'qxx'
    ? 'Cálculo de Q1 a Q100 por garantia mensal da demanda.'
    : 'Compare KNN e XGBoost, classifique indicadores climáticos e escolha as variáveis do modelo.'

  useEffect(() => {
    if (mode === 'qxx' || mode === 'knn') setActiveTab(mode)
  }, [mode])

  return (
    <div className={`pv-root ${darkMode ? 'pv-dark' : ''}`} style={{ minHeight: 600, padding: '18px 26px 48px' }}>
      <style>{`.pv-root{--bg:#fdf6ee;--orange:#e07b2a;--orange-pale:#fdebd3;--orange-deep:#c46318;--teal:#2a9d8f;--teal-pale:#d4f5ef;--blue:#264fa3;--blue-pale:#dde8f8;--red:#d94040;--red-pale:#fde8e8;--text:#1e1208;--text-mid:#5a3c24;--text-light:#9a7055;--border:#ecdcc8;--border-light:#f5ebe0;--card:#fff;--shadow:0 2px 16px rgba(150,90,40,.10);--radius:14px;--radius-sm:9px;--radius-xs:6px;font-family:'Sora',sans-serif;background:var(--bg);color:var(--text)}.pv-dark{--bg:#050403;--card:#0d0805;--text:#fff7ef;--text-mid:#efd0b8;--text-light:#c0987c;--border:#2a1a10;--border-light:#1f140d;--orange-pale:#3a1d0b;--orange-deep:#ff9b42;--teal-pale:#09231f;--blue-pale:#071634;--red-pale:#2a0c0c;--shadow:0 2px 18px rgba(0,0,0,.45)}.pv-layout{display:grid;grid-template-columns:360px minmax(0,1fr);gap:16px;align-items:start}.pv-btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;border:0;border-radius:9px;padding:9px 13px;font-size:12px;font-weight:900;cursor:pointer;font-family:'Sora',sans-serif}.pv-btn:disabled{cursor:not-allowed;opacity:.55}.pv-primary{background:linear-gradient(135deg,var(--orange),var(--orange-deep));color:#fff}.pv-ghost{background:var(--card);color:var(--text-mid);border:1.5px solid var(--border)}.pv-tabbar{display:flex;gap:4px;background:var(--card);border:1.5px solid var(--border);border-radius:9px;padding:3px;width:fit-content}.pv-tab{border:0;border-radius:7px;background:transparent;color:var(--text-light);font:900 12px 'Sora',sans-serif;padding:8px 13px;cursor:pointer}.pv-tab.on{background:var(--orange-pale);color:var(--orange-deep)}.pv-section{border-top:1.5px solid var(--border-light);padding-top:13px;margin-top:13px}.pv-model-switch{display:grid;grid-template-columns:1fr 1fr;gap:4px;background:var(--border-light);padding:4px;border-radius:9px}.pv-model-option{border:0;border-radius:7px;padding:8px 6px;background:transparent;color:var(--text-light);font:900 11px 'Sora',sans-serif;cursor:pointer}.pv-model-option.on{background:var(--card);color:var(--orange-deep);box-shadow:0 1px 5px rgba(80,40,10,.12)}.pv-indicator-list{display:flex;flex-direction:column;gap:6px;margin-top:8px}.pv-indicator{display:grid;grid-template-columns:18px minmax(0,1fr) 45px;gap:8px;align-items:center;width:100%;padding:8px;border:1.5px solid var(--border-light);border-radius:8px;background:var(--card);color:var(--text);text-align:left;cursor:pointer;font-family:'Sora',sans-serif}.pv-indicator.on{border-color:var(--teal);background:var(--teal-pale)}.pv-check{width:16px;height:16px;border:1.5px solid var(--border);border-radius:4px;display:flex;align-items:center;justify-content:center;background:var(--card)}.pv-indicator.on .pv-check{background:var(--teal);border-color:var(--teal);color:#fff}.pv-importance-track{height:4px;border-radius:3px;background:var(--border-light);overflow:hidden;margin-top:4px}.pv-importance-fill{height:100%;background:var(--orange);border-radius:3px}.pv-table-wrap{max-height:430px;overflow:auto;border:1.5px solid var(--border);border-radius:var(--radius-sm)}.pv-table{width:100%;border-collapse:collapse;font-size:12px}.pv-table th{position:sticky;top:0;background:var(--card);color:var(--text-light);text-align:left;padding:8px;border-bottom:1.5px solid var(--border-light)}.pv-table td{padding:8px;border-bottom:1px solid var(--border-light)}@keyframes pv-spin{to{transform:rotate(360deg)}}.pv-spin{animation:pv-spin 1s linear infinite}@media(max-width:920px){.pv-layout{grid-template-columns:1fr}.pv-tabbar{width:100%}.pv-tab{flex:1}}`}</style>

      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', marginBottom: 14 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
            <Waves size={22} color="var(--orange)" />
            <h2 style={{ margin: 0, fontSize: 19, fontWeight: 900 }}>{pageTitle}</h2>
          </div>
          <p style={{ margin: '4px 0 0', color: 'var(--text-light)', fontSize: 11.5 }}>{pageSubtitle}</p>
        </div>
        {((activeTab === 'qxx' && permanencia) || (activeTab === 'knn' && previsao)) && (
          <button
            className="pv-btn pv-ghost"
            onClick={() => activeTab === 'qxx'
              ? exportVazoesGarantia(permanencia, reservatorio)
              : exportPrevisao(previsao, reservatorio, importancia)}
          >
            <Download size={14} /> Excel
          </button>
        )}
      </div>

      {!isFixedMode && (
        <div className="pv-tabbar" style={{ marginBottom: 14 }}>
          <button className={`pv-tab ${activeTab === 'qxx' ? 'on' : ''}`} onClick={() => setActiveTab('qxx')}>Vazões de Garantia</button>
          <button className={`pv-tab ${activeTab === 'knn' ? 'on' : ''}`} onClick={() => setActiveTab('knn')}>Previsão de Afluência</button>
        </div>
      )}

      {msg && (
        <div style={{ marginBottom: 12, padding: '9px 13px', borderRadius: 'var(--radius-sm)', fontSize: 12, fontWeight: 800, background: msg.type === 'success' ? 'var(--teal-pale)' : 'var(--red-pale)', color: msg.type === 'success' ? 'var(--teal)' : 'var(--red)' }}>
          {msg.text}
        </div>
      )}

      <div className="pv-layout">
        <Card style={{ padding: 16, position: 'sticky', top: 14, maxHeight: 'calc(100vh - 28px)', overflowY: 'auto' }}>
          <Field label="Reservatório">
            <div style={{ position: 'relative' }}>
              <Search size={13} style={{ position: 'absolute', left: 10, top: 11, color: 'var(--text-light)' }} />
              <Control value={query} onChange={e => setQuery(e.target.value)} placeholder="Digite para buscar..." style={{ paddingLeft: 30 }} />
              {query && filtrados.length > 0 && query !== reservatorio && (
                <div style={{ position: 'absolute', zIndex: 20, left: 0, right: 0, top: 'calc(100% + 4px)', maxHeight: 220, overflow: 'auto', background: 'var(--card)', border: '1.5px solid var(--border)', borderRadius: 'var(--radius-xs)', boxShadow: 'var(--shadow)', padding: 4 }}>
                  {filtrados.map(nome => (
                    <button key={nome} type="button" onClick={() => {
                      setReservatorio(nome)
                      setQuery(nome)
                      setImportancia(null)
                      setSelectedIndicators([])
                      setPrevisao(null)
                    }} style={{ display: 'block', width: '100%', textAlign: 'left', border: 0, background: 'transparent', color: 'var(--text)', padding: '7px 9px', borderRadius: 6, cursor: 'pointer', fontWeight: 800 }}>
                      {nome}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </Field>

          <div className="pv-section">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <Field label="Início">
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
                  <Control as="select" value={periodo.mesInicial} onChange={e => setPeriodo(p => ({ ...p, mesInicial: e.target.value }))}>{MESES.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}</Control>
                  <Control type="number" value={periodo.anoInicial} onChange={e => setPeriodo(p => ({ ...p, anoInicial: e.target.value }))} />
                </div>
              </Field>
              <Field label="Fim">
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
                  <Control as="select" value={periodo.mesFinal} onChange={e => setPeriodo(p => ({ ...p, mesFinal: e.target.value }))}>{MESES.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}</Control>
                  <Control type="number" value={periodo.anoFinal} onChange={e => setPeriodo(p => ({ ...p, anoFinal: e.target.value }))} />
                </div>
              </Field>
            </div>
          </div>

          {activeTab === 'qxx' ? (
            <>
              <div className="pv-section">
                <Field label="Volume inicial (%)">
                  <Control type="number" min="0" max="100" step="1" value={qxx.volInicial} onChange={e => setQxx({ volInicial: e.target.value })} />
                </Field>
                <div style={{ marginTop: 9, fontSize: 11.5, lineHeight: 1.45, color: 'var(--text-light)' }}>
                  Vazão de garantia = maior demanda constante atendida com a garantia requerida.
                </div>
              </div>
              <button className="pv-btn pv-primary" onClick={runQxx} disabled={loading || !reservatorio} style={{ width: '100%', marginTop: 16, opacity: loading ? 0.75 : 1 }}>
                {loading ? <RefreshCw size={14} className="pv-spin" /> : <BarChart3 size={14} />}
                {loading ? 'Calculando...' : 'Calcular Vazões de Garantia'}
              </button>
            </>
          ) : (
            <>
              <div className="pv-section">
                <Field label="Modelo">
                  <div className="pv-model-switch">
                    <button type="button" className={`pv-model-option ${knn.modelo === 'knn' ? 'on' : ''}`} onClick={() => updateForecastConfig({ modelo: 'knn' })}>KNN</button>
                    <button type="button" className={`pv-model-option ${knn.modelo === 'xgboost' ? 'on' : ''}`} onClick={() => updateForecastConfig({ modelo: 'xgboost' })}>XGBoost</button>
                  </div>
                </Field>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {knn.modelo === 'knn' && (
                    <Field label="Vizinhos (K)">
                      <Control type="number" min="1" max="50" value={knn.k} onChange={e => updateForecastConfig({ k: e.target.value })} />
                    </Field>
                  )}
                  <Field label="Lags de afluência">
                    <Control type="number" min="1" max="24" value={knn.lags} onChange={e => updateForecastConfig({ lags: e.target.value })} />
                  </Field>
                  <Field label="Horizonte (meses)">
                    <Control type="number" min="1" max="12" value={knn.horizonte} onChange={e => updateForecastConfig({ horizonte: e.target.value })} />
                  </Field>
                  <Field label="Lag climático">
                    <Control type="number" min="1" max="12" value={knn.lagClimatico} onChange={e => updateForecastConfig({ lagClimatico: e.target.value })} />
                  </Field>
                  <Field label="Validação (meses)">
                    <Control type="number" min="12" max="240" value={knn.testeMeses} onChange={e => updateForecastConfig({ testeMeses: e.target.value })} />
                  </Field>
                </div>
              </div>

              <div className="pv-section">
                <Field label="Método de importância">
                  <Control as="select" value={knn.metodoImportancia} onChange={e => updateForecastConfig({ metodoImportancia: e.target.value })}>
                    <option value="permutacao">Permutação do modelo</option>
                    <option value="select_k_best">Select K Best</option>
                    <option value="ganho_xgboost">Ganho do XGBoost</option>
                    <option value="copeland">Copeland unificado</option>
                  </Control>
                </Field>
                <button className="pv-btn pv-ghost" onClick={analyzeIndicators} disabled={loadingImportance || loading || !reservatorio} style={{ width: '100%', marginTop: 9 }}>
                  {loadingImportance ? <RefreshCw size={14} className="pv-spin" /> : <SlidersHorizontal size={14} />}
                  {loadingImportance ? 'Analisando...' : 'Analisar Indicadores'}
                </button>

                {importancia?.indicadores?.length ? (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 12 }}>
                      <div style={{ fontSize: 10.5, fontWeight: 900, color: 'var(--text-light)', textTransform: 'uppercase' }}>Variáveis do modelo</div>
                      <button type="button" onClick={() => {
                        setSelectedIndicators(importancia.indicadores.map(item => item.id))
                        setPrevisao(null)
                      }} style={{ border: 0, background: 'transparent', color: 'var(--orange-deep)', font: "800 10px 'Sora', sans-serif", cursor: 'pointer' }}>Selecionar todas</button>
                    </div>
                    <div className="pv-indicator-list">
                      {importancia.indicadores.map(item => {
                        const checked = selectedIndicators.includes(item.id)
                        return (
                          <button key={item.id} type="button" className={`pv-indicator ${checked ? 'on' : ''}`} onClick={() => toggleIndicator(item.id)}>
                            <span className="pv-check">{checked && <Check size={11} />}</span>
                            <span style={{ minWidth: 0 }}>
                              <span style={{ display: 'block', fontSize: 11, fontWeight: 900 }}>{item.label}</span>
                              <span style={{ display: 'block', fontSize: 9.5, color: 'var(--text-light)', marginTop: 2 }}>R² individual {Number(item.variabilidade_individual_r2_percent).toFixed(1)}%</span>
                              <span className="pv-importance-track"><span className="pv-importance-fill" style={{ width: `${Math.min(100, item.contribuicao_relativa_percent)}%`, display: 'block' }} /></span>
                            </span>
                            <span style={{ fontSize: 10.5, fontWeight: 900, color: 'var(--orange-deep)', textAlign: 'right' }}>{Number(item.contribuicao_relativa_percent).toFixed(1)}%</span>
                          </button>
                        )
                      })}
                    </div>
                    {selectedIndicators.includes('dipolo') && selectedIndicators.includes('tna') && selectedIndicators.includes('tsa') && (
                      <div style={{ marginTop: 8, padding: 8, borderRadius: 7, background: 'var(--orange-pale)', color: 'var(--text-mid)', fontSize: 10.5, lineHeight: 1.4 }}>
                        O dipolo é calculado por TNA − TSA. Usar os três juntos adiciona variáveis redundantes.
                      </div>
                    )}
                    <div style={{ marginTop: 8, color: 'var(--text-light)', fontSize: 9.5, lineHeight: 1.4 }}>
                      A contribuição relativa soma 100%. O R² individual é apenas a associação isolada de cada índice.
                    </div>
                  </>
                ) : (
                  <div style={{ marginTop: 9, color: 'var(--text-light)', fontSize: 10.5, lineHeight: 1.45 }}>
                    Analise os índices para obter o ranking e selecionar as variáveis climáticas.
                  </div>
                )}
              </div>

              <button className="pv-btn pv-primary" onClick={runKnn} disabled={loading || loadingImportance || !reservatorio || !importancia} style={{ width: '100%', marginTop: 16 }}>
                {loading ? <RefreshCw size={14} className="pv-spin" /> : <BrainCircuit size={14} />}
                {loading ? 'Calculando...' : `Gerar Previsão ${knn.modelo.toUpperCase()}`}
              </button>
            </>
          )}
        </Card>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {activeTab === 'qxx' && (
            permanencia ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))', gap: 10 }}>
                  <Metric label="Meses usados" value={permanencia.periodo?.meses || 0} sub={`${permanencia.periodo?.inicio} a ${permanencia.periodo?.fim}`} icon={BarChart3} />
                  <Metric label="Q100 - Vazão plena" value={`${(q100?.vazao_m3s ?? 0).toFixed(3)} m³/s`} sub={`${((q100?.vazao_m3s ?? 0) * 1000).toFixed(1)} L/s · garantia 100%`} icon={Waves} />
                  <Metric label="Q95" value={`${(q95?.vazao_m3s ?? 0).toFixed(3)} m³/s`} sub={`${((q95?.vazao_m3s ?? 0) * 1000).toFixed(1)} L/s · garantia ${Number(q95?.garantia_obtida ?? 0).toFixed(2)}%`} icon={TrendingUp} />
                  <Metric label="Q90" value={`${(q90?.vazao_m3s ?? 0).toFixed(3)} m³/s`} sub={`${((q90?.vazao_m3s ?? 0) * 1000).toFixed(1)} L/s · garantia ${Number(q90?.garantia_obtida ?? 0).toFixed(2)}%`} icon={TrendingUp} />
                </div>

                <Card style={{ padding: 14 }}>
                  <div style={{ fontSize: 14, fontWeight: 900, marginBottom: 3 }}>Curva de Vazões de Garantia</div>
                  <div style={{ fontSize: 11.5, color: 'var(--text-light)', marginBottom: 8 }}>Cada ponto representa a maior vazão constante que atende à garantia requerida.</div>
                  <ZoomReset zoom={qxxZoom}/>
                  <div style={{ height: 285 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={qxxZoom.data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }} {...qxxZoom.props}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="garantia" tick={{ fontSize: 10, fill: 'var(--text-light)' }} unit="%" />
                        <YAxis yAxisId="m3s" tick={{ fontSize: 10, fill: 'var(--text-light)' }} />
                        <YAxis yAxisId="ls" orientation="right" tick={{ fontSize: 10, fill: 'var(--orange-deep)' }} />
                        <Tooltip content={<ChartTooltip />} />
                        <Area yAxisId="m3s" dataKey="vazao_m3s" name="Vazão de garantia (m³/s)" stroke="#264fa3" fill="#264fa3" fillOpacity={0.14} strokeWidth={2} dot={false} />
                        <Line yAxisId="ls" type="monotone" dataKey="vazao_ls" name="Vazão de garantia (L/s)" stroke="#e07b2a" strokeWidth={2} dot={false} />
                        {qxxZoom.area}
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </Card>

                <Card style={{ padding: 14 }}>
                  <div style={{ fontSize: 14, fontWeight: 900, marginBottom: 8 }}>Tabela de Vazões de Garantia</div>
                  <div className="pv-table-wrap">
                    <table className="pv-table">
                      <thead><tr><th>Vazão de garantia</th><th>Garantia requerida</th><th>Garantia obtida</th><th>Vazão (L/s)</th></tr></thead>
                      <tbody>
                        {permanencia.resultados.map(row => (
                          <tr key={row.referencia}>
                            <td style={{ fontWeight: 900, color: row.garantia_requerida === 100 ? 'var(--orange-deep)' : 'var(--text)' }}>{row.referencia}</td>
                            <td>{row.garantia_requerida}%</td>
                            <td>{Number(row.garantia_obtida).toFixed(2)}%</td>
                            <td>{(Number(row.vazao_m3s) * 1000).toFixed(1)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </>
            ) : (
              <Card style={{ padding: 46, textAlign: 'center' }}>
                <BarChart3 size={34} color="var(--orange)" style={{ opacity: 0.55, marginBottom: 10 }} />
                <div style={{ fontSize: 15, fontWeight: 900 }}>Calcule as vazões de garantia</div>
                <div style={{ fontSize: 12, color: 'var(--text-light)', marginTop: 4 }}>O cálculo simula demandas constantes e calcula a garantia obtida.</div>
              </Card>
            )
          )}

          {activeTab === 'knn' && (
            previsao ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))', gap: 10 }}>
                  <Metric label="Meses usados" value={previsao.periodo?.meses || 0} sub={`${previsao.periodo?.inicio} a ${previsao.periodo?.fim}`} icon={BarChart3} />
                  <Metric label="Modelo" value={String(previsao.modelo || '').toUpperCase()} sub={`${previsao.indicadores?.length || 0} indicador(es) climático(s)`} icon={BrainCircuit} />
                  {previsao?.metricas?.rmse_m3s !== undefined && <Metric label="RMSE validação" value={previsao.metricas.rmse_m3s.toFixed(3)} sub="m³/s" icon={TrendingUp} />}
                  {previsao?.metricas?.mae_m3s !== undefined && <Metric label="MAE validação" value={previsao.metricas.mae_m3s.toFixed(3)} sub="m³/s" icon={TrendingUp} />}
                  {previsao?.metricas?.nse !== null && previsao?.metricas?.nse !== undefined && <Metric label="NSE validação" value={previsao.metricas.nse.toFixed(3)} sub="1,0 representa ajuste perfeito" icon={Activity} />}
                </div>

                {importancia?.indicadores?.length > 0 && (
                  <Card style={{ padding: 14 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, marginBottom: 9 }}>
                      <div>
                        <div style={{ fontSize: 14, fontWeight: 900 }}>Indicadores Climáticos</div>
                        <div style={{ fontSize: 11, color: 'var(--text-light)', marginTop: 2 }}>{importancia.metodo_label} · contribuição relativa normalizada</div>
                      </div>
                      <div style={{ padding: '5px 8px', borderRadius: 7, background: 'var(--teal-pale)', color: 'var(--teal)', fontSize: 10, fontWeight: 900 }}>
                        {previsao.indicadores?.length || 0} selecionado(s)
                      </div>
                    </div>
                    <div className="pv-table-wrap" style={{ maxHeight: 300 }}>
                      <table className="pv-table">
                        <thead>
                          <tr><th>Posição</th><th>Indicador</th><th>Usado</th><th>Contribuição relativa</th><th>R² individual</th><th>ΔMSE permutação</th></tr>
                        </thead>
                        <tbody>
                          {importancia.indicadores.map(item => {
                            const used = (previsao.indicadores || []).some(selected => selected.id === item.id)
                            return (
                              <tr key={item.id}>
                                <td>{item.posicao}º</td>
                                <td style={{ fontWeight: 900 }}>{item.label}</td>
                                <td style={{ color: used ? 'var(--teal)' : 'var(--text-light)', fontWeight: 900 }}>{used ? 'Sim' : 'Não'}</td>
                                <td>{Number(item.contribuicao_relativa_percent).toFixed(2)}%</td>
                                <td>{Number(item.variabilidade_individual_r2_percent).toFixed(2)}%</td>
                                <td>{Number(item.aumento_mse_percent).toFixed(2)}%</td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                )}

                <Card style={{ padding: 14 }}>
                  <div style={{ fontSize: 14, fontWeight: 900, marginBottom: 3 }}>Previsão {String(previsao.modelo || '').toUpperCase()}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-light)', marginBottom: 8 }}>Modelos diretos e independentes para cada horizonte futuro.</div>
                  <ZoomReset zoom={forecastZoom}/>
                  <div style={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={forecastZoom.data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }} {...forecastZoom.props}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="data" tick={{ fontSize: 10, fill: 'var(--text-light)' }} />
                        <YAxis tick={{ fontSize: 10, fill: 'var(--text-light)' }} />
                        <Tooltip content={<ChartTooltip />} />
                        <Line type="monotone" dataKey="Historico" name="Histórico" stroke="#264fa3" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="Previsao" name="Previsão" stroke="#e07b2a" strokeWidth={2.5} dot={{ r: 3 }} />
                        {forecastZoom.area}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Card>

                {validationChart.length > 0 && (
                  <Card style={{ padding: 14 }}>
                    <div style={{ fontSize: 14, fontWeight: 900, marginBottom: 8 }}>Validação Retrospectiva</div>
                    <ZoomReset zoom={validationZoom}/>
                    <div style={{ height: 260 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={validationZoom.data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }} {...validationZoom.props}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                          <XAxis dataKey="data" tick={{ fontSize: 10, fill: 'var(--text-light)' }} />
                          <YAxis tick={{ fontSize: 10, fill: 'var(--text-light)' }} />
                          <Tooltip content={<ChartTooltip />} />
                          <Line type="monotone" dataKey="Observado" stroke="#2a9d8f" strokeWidth={2} dot={false} />
                          <Line type="monotone" dataKey="Previsto" stroke="#e07b2a" strokeWidth={2} dot={false} />
                          {validationZoom.area}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </Card>
                )}
              </>
            ) : (
              <Card style={{ padding: 46, textAlign: 'center' }}>
                <BrainCircuit size={34} color="var(--orange)" style={{ opacity: 0.55, marginBottom: 10 }} />
                <div style={{ fontSize: 15, fontWeight: 900 }}>Configure sua previsão</div>
                <div style={{ fontSize: 12, color: 'var(--text-light)', marginTop: 4 }}>Classifique os indicadores, escolha as variáveis e execute KNN ou XGBoost.</div>
              </Card>
            )
          )}
        </div>
      </div>
    </div>
  )
}
