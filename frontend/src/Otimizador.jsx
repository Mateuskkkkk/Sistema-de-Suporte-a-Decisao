/**
 * Otimizador.jsx
 * Página autónoma do Optimizador de Curvas Guia (NSGA-II).
 * Design idêntico ao Simulador.jsx.
 *
 * Props:
 *   apiUrl  — URL base do backend (ex: "http://localhost:8000")
 *
 * Uso:
 *   import Otimizador from './Otimizador'
 *   <Otimizador apiUrl={import.meta.env.VITE_API_URL} />
 *
 * Dependências: react, recharts, lucide-react, xlsx
 */

import React, { useState, useEffect, useMemo, useRef } from 'react'
import {
  AreaChart, Area, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ZAxis,
} from 'recharts'
import {
  Cpu, RefreshCw, AlertTriangle, CheckCircle2,
  Download, FileSpreadsheet, Waves, Info,
  Shield, Database, Calendar, ChevronDown,
  Zap, ArrowRight, BarChart3, Target,
} from 'lucide-react'
import * as XLSX from 'xlsx'

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const MESES     = ['JAN','FEV','MAR','ABR','MAI','JUN','JUL','AGO','SET','OUT','NOV','DEZ']
const CURVAS    = ['normal','atencao','seca','seca_severa']
const CURVA_LABELS = { normal:'Normal', atencao:'Atenção', seca:'Seca', seca_severa:'Seca Severa' }

const CSS = `
  .ot-root *, .ot-root *::before, .ot-root *::after { box-sizing:border-box; }
  .ot-root {
    --bg:#fdf6ee; --orange:#e07b2a; --orange-light:#f5a654;
    --orange-pale:#fdebd3; --orange-deep:#c46318; --orange-glow:rgba(224,123,42,0.18);
    --teal:#2a9d8f; --teal-pale:#d4f5ef;
    --blue:#264fa3; --blue-light:#4a7cc7; --blue-pale:#dde8f8;
    --red:#d94040; --red-pale:#fde8e8;
    --yellow:#d4a017; --yellow-pale:#fef3cd;
    --text:#1e1208; --text-mid:#5a3c24; --text-light:#9a7055;
    --border:#ecdcc8; --border-light:#f5ebe0; --card:#ffffff;
    --shadow-sm:0 1px 4px rgba(150,90,40,0.08);
    --shadow:0 2px 16px rgba(150,90,40,0.10);
    --radius:14px; --radius-sm:9px; --radius-xs:6px;
    font-family:'Sora',sans-serif;
    background:var(--bg); color:var(--text);
    -webkit-font-smoothing:antialiased;
  }
  .ot-root ::-webkit-scrollbar { width:5px; height:5px; }
  .ot-root ::-webkit-scrollbar-track { background:var(--bg); }
  .ot-root ::-webkit-scrollbar-thumb { background:var(--border); border-radius:10px; }
  .ot-root input,.ot-root select,.ot-root button { font-family:'Sora',sans-serif; }
  @keyframes ot-fade { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  @keyframes ot-spin { to{transform:rotate(360deg)} }
  @keyframes ot-pulse { 0%,100%{opacity:1} 50%{opacity:.45} }
  .ot-fade { animation:ot-fade .35s ease both; }
  .ot-spin { animation:ot-spin 1.4s linear infinite; }
  .ot-pulse { animation:ot-pulse 1.6s ease-in-out infinite; }
  .ot-ghost { display:flex;align-items:center;gap:6px;background:none;border:1.5px solid var(--border);border-radius:var(--radius-xs);padding:6px 13px;font-size:11.5px;font-weight:600;color:var(--text-mid);cursor:pointer;transition:all .15s; }
  .ot-ghost:hover { border-color:var(--orange);color:var(--orange); }
  .ot-tab { padding:8px 16px;border-radius:9px;border:none;font-size:12px;font-weight:700;cursor:pointer;transition:all .15s;white-space:nowrap; }
  .ot-tab.on  { background:var(--orange-pale);color:var(--orange-deep); }
  .ot-tab.off { background:none;color:var(--text-light); }
  .ot-tab.off:hover { background:var(--orange-pale);color:var(--orange); }
  .ot-tr:hover td { background:var(--orange-pale)!important; }
`

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────

function Card({ children, style, className='' }) {
  return (
    <div className={className} style={{
      background:'var(--card)', border:'1.5px solid var(--border)',
      borderRadius:'var(--radius)', boxShadow:'var(--shadow)', ...style
    }}>{children}</div>
  )
}

function Label({ icon: Icon, children }) {
  return (
    <div style={{ display:'flex', alignItems:'center', gap:7, fontSize:10.5,
      fontWeight:700, color:'var(--text-light)', textTransform:'uppercase',
      letterSpacing:'0.06em', marginBottom:7, marginTop:15 }}>
      {Icon && <Icon size={12} strokeWidth={2.5}/>}{children}
    </div>
  )
}

function FC({ as='input', children, style, ...props }) {
  const base = { width:'100%', padding:'7px 10px', border:'1.5px solid var(--border)',
    borderRadius:'var(--radius-xs)', background:'#fff', color:'var(--text)',
    fontSize:12.5, outline:'none', transition:'border-color .15s', ...style }
  const onF = e => e.target.style.borderColor='var(--orange)'
  const onB = e => e.target.style.borderColor='var(--border)'
  if (as==='select') return <select style={{...base,appearance:'none',cursor:'pointer'}} onFocus={onF} onBlur={onB} {...props}>{children}</select>
  return <input style={base} onFocus={onF} onBlur={onB} {...props}/>
}

function MCard({ label, value, sub, variant='default', icon: Icon }) {
  const C = {
    default:{a:'var(--orange)',b:'var(--orange-pale)',t:'var(--orange)'},
    success:{a:'var(--teal)',  b:'var(--teal-pale)',  t:'var(--teal)'},
    danger: {a:'var(--red)',   b:'var(--red-pale)',   t:'var(--red)'},
    info:   {a:'var(--blue)',  b:'var(--blue-pale)',  t:'var(--blue)'},
    yellow: {a:'var(--yellow)',b:'var(--yellow-pale)',t:'var(--yellow)'},
  }[variant]
  return (
    <div className="ot-fade" style={{ background:'var(--card)', border:'1.5px solid var(--border)',
      borderRadius:'var(--radius)', padding:'14px 16px', boxShadow:'var(--shadow)',
      position:'relative', overflow:'hidden' }}>
      <div style={{ position:'absolute',top:0,left:0,right:0,height:3,
        background:C.a,borderRadius:'4px 4px 0 0' }}/>
      <div style={{ display:'flex',alignItems:'flex-start',justifyContent:'space-between',marginBottom:7 }}>
        <div style={{ fontSize:9.5,fontWeight:700,color:'var(--text-light)',
          textTransform:'uppercase',letterSpacing:'0.06em' }}>{label}</div>
        {Icon && <div style={{ width:24,height:24,borderRadius:6,background:C.b,
          display:'flex',alignItems:'center',justifyContent:'center' }}>
          <Icon size={12} color={C.a} strokeWidth={2.5}/>
        </div>}
      </div>
      <div style={{ fontSize:22,fontWeight:800,color:C.t,fontFamily:'JetBrains Mono',
        lineHeight:1,marginBottom:4 }}>{value}</div>
      <div style={{ fontSize:10.5,color:'var(--text-light)',lineHeight:1.4 }}>{sub}</div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// COR POR NÍVEL (verde → vermelho, igual ao NiveisMeta do Simulador)
// ─────────────────────────────────────────────────────────────────────────────

function nivelColor(idx, total) {
  if (total <= 1) return '#2a9d8f'
  const t     = idx / (total - 1)
  const stops = [[42,157,143],[212,160,23],[224,123,42],[217,64,64]]
  const seg   = (stops.length - 1) * t
  const lo    = Math.floor(seg)
  const hi    = Math.min(lo + 1, stops.length - 1)
  const frac  = seg - lo
  const r = Math.round(stops[lo][0] + (stops[hi][0]-stops[lo][0])*frac)
  const g = Math.round(stops[lo][1] + (stops[hi][1]-stops[lo][1])*frac)
  const b = Math.round(stops[lo][2] + (stops[hi][2]-stops[lo][2])*frac)
  return `rgb(${r},${g},${b})`
}

// ─────────────────────────────────────────────────────────────────────────────
// GRÁFICO DE BANDAS (igual ao NiveisMeta, reutilizável)
// ─────────────────────────────────────────────────────────────────────────────

function GraficoCurvas({ curvas }) {
  if (!curvas) return null

  // Curvas ordenadas do limite mais alto (verde) para o mais baixo (vermelho)
  // Normal → Atenção → Seca → Seca Severa  (já vêm ordenadas do backend)
  const ordered = [
    { key:'normal',      label:'Normal',      vals: curvas.normal },
    { key:'atencao',     label:'Atenção',     vals: curvas.atencao },
    { key:'seca',        label:'Seca',        vals: curvas.seca },
    { key:'seca_severa', label:'Seca Severa', vals: curvas.seca_severa },
  ]
  const n    = ordered.length
  const cores = ordered.map((_,i) => nivelColor(i, n))

  // Dataset empilhado (diferenças) — mesmo padrão do NiveisMeta
  const ascOrder = [...ordered].reverse()   // Seca Severa primeiro (base)

  const data = MESES.map((mes, mi) => {
    const limites = ascOrder.map(c => c.vals[mi])
    const p = { mes }
    p['_b0'] = limites[0]
    for (let i = 1; i < ascOrder.length; i++)
      p[`_b${i}`] = Math.max(0, limites[i] - limites[i-1])
    p[`_b${ascOrder.length}`] = Math.max(0, 100 - limites[ascOrder.length-1])
    return p
  })

  const coresBandas = [
    ...ascOrder.map((_,i) => nivelColor(n-1-i, n)),
    '#e8e0d4',
  ]

  const Tip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    const mi = MESES.indexOf(label)
    return (
      <div style={{ background:'#fff', border:'1.5px solid var(--border)',
        borderRadius:10, padding:'9px 13px', boxShadow:'var(--shadow)', fontSize:11 }}>
        <div style={{ fontWeight:700, marginBottom:5, color:'var(--text)' }}>{label}</div>
        {ordered.map((c,i) => (
          <div key={i} style={{ display:'flex',gap:7,alignItems:'center',marginBottom:2 }}>
            <div style={{ width:7,height:7,borderRadius:'50%',background:cores[i] }}/>
            <span style={{ color:'var(--text-mid)' }}>{c.label}:</span>
            <span style={{ fontWeight:600,fontFamily:'JetBrains Mono',color:'var(--text)' }}>
              ≤ {c.vals[mi].toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div>
      <div style={{ height:260 }}>
        <ResponsiveContainer>
          <AreaChart data={data} margin={{top:4,right:20,left:0,bottom:4}}>
            <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
            <XAxis dataKey="mes" tick={{fontSize:10,fill:'var(--text-light)'}}/>
            <YAxis domain={[0,100]} tick={{fontSize:10,fill:'var(--text-light)'}}
              label={{value:'% Cap.',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
            <Tooltip content={<Tip/>}/>
            {coresBandas.map((cor,i) => (
              <Area key={i} type="linear" dataKey={`_b${i}`} stackId="s"
                stroke={i===coresBandas.length-1?'none':cor}
                strokeWidth={i===coresBandas.length-1?0:2}
                fill={cor}
                fillOpacity={i===coresBandas.length-1?0.12:0.55}
                dot={false} activeDot={false} legendType="none"/>
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
      {/* Legenda */}
      <div style={{ display:'flex',gap:8,flexWrap:'wrap',marginTop:12 }}>
        {ordered.map((c,i) => (
          <span key={i} style={{ display:'inline-flex',alignItems:'center',gap:6,
            fontSize:10.5,borderRadius:20,padding:'3px 11px',fontWeight:600,
            background:`${cores[i]}22`,color:cores[i],border:`1.5px solid ${cores[i]}66` }}>
            <span style={{ width:8,height:8,borderRadius:'50%',background:cores[i],
              display:'inline-block',flexShrink:0 }}/>
            {c.label}
          </span>
        ))}
        <span style={{ display:'inline-flex',alignItems:'center',gap:6,fontSize:10.5,
          borderRadius:20,padding:'3px 11px',fontWeight:600,
          background:'#e8e0d422',color:'var(--text-light)',border:'1.5px solid #e8e0d466' }}>
          <span style={{ width:8,height:8,borderRadius:'50%',background:'#c8b8a0',
            display:'inline-block',flexShrink:0 }}/>
          Sem restrição
        </span>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// GRÁFICO FRONTEIRA DE PARETO
// ─────────────────────────────────────────────────────────────────────────────

function GraficoPareto({ resultado }) {
  if (!resultado) return null
  // O backend devolve apenas o vencedor — simulamos a fronteira com os dados disponíveis
  const { fitness_vencedor, n_solucoes_pareto } = resultado
  const ponto = {
    f1: parseFloat(fitness_vencedor.f1_deficit_quadratico.toFixed(2)),
    f2: parseFloat(fitness_vencedor.f2_vertimento_hm3.toFixed(2)),
  }

  const Tip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    return (
      <div style={{ background:'#fff',border:'1.5px solid var(--border)',
        borderRadius:10,padding:'9px 13px',boxShadow:'var(--shadow)',fontSize:11 }}>
        <div style={{ fontWeight:700,marginBottom:5,color:'var(--text)' }}>
          Solução vencedora
        </div>
        <div>F1 (défice²): <strong style={{ fontFamily:'JetBrains Mono' }}>{d.f1}</strong></div>
        <div>F2 (vertimento hm³): <strong style={{ fontFamily:'JetBrains Mono' }}>{d.f2}</strong></div>
      </div>
    )
  }

  return (
    <div>
      <div style={{ height:220 }}>
        <ResponsiveContainer>
          <ScatterChart margin={{top:8,right:20,bottom:20,left:0}}>
            <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
            <XAxis dataKey="f1" name="F1 Défice²" type="number"
              tick={{fontSize:10,fill:'var(--text-light)'}}
              label={{value:'F1 — Défice Quadrático',position:'insideBottom',
                offset:-12,fill:'var(--text-light)',fontSize:10}}/>
            <YAxis dataKey="f2" name="F2 Vertimento (hm³)" type="number"
              tick={{fontSize:10,fill:'var(--text-light)'}}
              label={{value:'F2 — Vertimento (hm³)',angle:-90,
                position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
            <ZAxis range={[120,120]}/>
            <Tooltip content={<Tip/>}/>
            <Scatter data={[ponto]} fill="var(--orange)" shape="star"/>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div style={{ marginTop:8,padding:'8px 13px',background:'var(--blue-pale)',
        borderRadius:'var(--radius-sm)',fontSize:11,color:'var(--blue)',lineHeight:1.6 }}>
        <Info size={12} style={{ marginRight:5,verticalAlign:'middle' }}/>
        O backend devolve a <strong>solução vencedora</strong> da fronteira de Pareto
        (menor F1). A fronteira completa continha <strong>{n_solucoes_pareto}</strong> soluções
        não-dominadas. Quanto mais à esquerda e abaixo, melhor.
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// TABELA DAS CURVAS
// ─────────────────────────────────────────────────────────────────────────────

function TabelaCurvas({ curvas }) {
  if (!curvas) return null
  const n    = CURVAS.length
  const cores = CURVAS.map((_,i) => nivelColor(i, n))

  return (
    <div style={{ overflowX:'auto' }}>
      <table style={{ width:'100%',borderCollapse:'collapse',fontSize:11.5 }}>
        <thead>
          <tr style={{ background:'var(--bg)' }}>
            <th style={{ padding:'7px 12px',textAlign:'left',fontSize:10,fontWeight:700,
              textTransform:'uppercase',letterSpacing:'0.05em',color:'var(--text-light)',
              borderBottom:'1.5px solid var(--border)',whiteSpace:'nowrap' }}>Nível</th>
            {MESES.map(m => (
              <th key={m} style={{ padding:'7px 8px',textAlign:'right',fontSize:10,
                fontWeight:700,textTransform:'uppercase',letterSpacing:'0.05em',
                color:'var(--text-light)',borderBottom:'1.5px solid var(--border)',
                whiteSpace:'nowrap' }}>{m}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {CURVAS.map((c,ci) => (
            <tr key={c} className="ot-tr" style={{ background:ci%2===0?'transparent':'rgba(236,220,200,0.15)' }}>
              <td style={{ padding:'6px 12px',borderBottom:'1px solid var(--border-light)' }}>
                <span style={{ display:'inline-flex',alignItems:'center',gap:6,
                  fontSize:11,borderRadius:20,padding:'2px 10px',fontWeight:700,
                  background:`${cores[ci]}22`,color:cores[ci],
                  border:`1.5px solid ${cores[ci]}55` }}>
                  <span style={{ width:7,height:7,borderRadius:'50%',
                    background:cores[ci],display:'inline-block' }}/>
                  {CURVA_LABELS[c]}
                </span>
              </td>
              {curvas[c].map((v,mi) => (
                <td key={mi} style={{ padding:'6px 8px',textAlign:'right',
                  borderBottom:'1px solid var(--border-light)',
                  fontFamily:'JetBrains Mono',fontSize:11,color:'var(--text-mid)' }}>
                  {v.toFixed(1)}%
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// BARRA DE PROGRESSO ANIMADA (simulada — o backend não faz streaming)
// ─────────────────────────────────────────────────────────────────────────────

function ProgressBar({ running, nGen }) {
  const [pct, setPct] = useState(0)
  const ref = useRef(null)

  useEffect(() => {
    if (!running) { setPct(0); return }
    setPct(0)
    // Simula progresso ao longo do tempo esperado (≈ 2s/geração como estimativa)
    const total   = nGen * 2000   // ms estimados
    const step    = 200           // actualiza a cada 200ms
    const inc     = (step / total) * 92  // chega até 92% — o resto quando terminar
    ref.current   = setInterval(() => {
      setPct(p => Math.min(p + inc, 92))
    }, step)
    return () => clearInterval(ref.current)
  }, [running, nGen])

  useEffect(() => {
    if (!running && pct > 0) {
      setPct(100)
      setTimeout(() => setPct(0), 800)
    }
  }, [running])

  if (pct === 0) return null

  return (
    <div style={{ marginTop:10 }}>
      <div style={{ display:'flex',justifyContent:'space-between',
        fontSize:10.5,color:'var(--text-light)',marginBottom:4 }}>
        <span className="ot-pulse">A optimizar curvas guia…</span>
        <span style={{ fontFamily:'JetBrains Mono' }}>{Math.round(pct)}%</span>
      </div>
      <div style={{ height:6,background:'var(--border)',borderRadius:99,overflow:'hidden' }}>
        <div style={{ width:`${pct}%`,height:'100%',
          background:'linear-gradient(90deg,var(--orange),var(--orange-deep))',
          borderRadius:99,transition:'width .25s ease' }}/>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// EXPORTAR EXCEL
// ─────────────────────────────────────────────────────────────────────────────

function exportarCurvas(resultado) {
  if (!resultado) return
  const wb = XLSX.utils.book_new()
  const { curvas_guia_percentual: c, reservatorio_nome, demanda_alvo } = resultado

  // Aba 1: Curvas guia
  const rows = MESES.map((mes, mi) => ({
    'Mês':           mes,
    'Normal (%)':       c.normal[mi],
    'Atenção (%)':      c.atencao[mi],
    'Seca (%)':         c.seca[mi],
    'Seca Severa (%)':  c.seca_severa[mi],
  }))
  const ws1 = XLSX.utils.json_to_sheet(rows)
  XLSX.utils.book_append_sheet(wb, ws1, 'Curvas Guia')

  // Aba 2: Resumo
  const resumo = [
    { 'Parâmetro': 'Reservatório',             'Valor': reservatorio_nome },
    { 'Parâmetro': 'Código',                   'Valor': resultado.reservatorio_cod },
    { 'Parâmetro': 'Demanda Alvo (m³/s)',      'Valor': demanda_alvo },
    { 'Parâmetro': 'Período',                  'Valor': `${resultado.periodo.ano_inicial}–${resultado.periodo.ano_final}` },
    { 'Parâmetro': 'Meses simulados',          'Valor': resultado.periodo.n_meses },
    { 'Parâmetro': 'F1 — Défice quadrático',   'Valor': resultado.fitness_vencedor.f1_deficit_quadratico },
    { 'Parâmetro': 'F2 — Vertimento (hm³)',    'Valor': resultado.fitness_vencedor.f2_vertimento_hm3 },
    { 'Parâmetro': 'Soluções Pareto',          'Valor': resultado.n_solucoes_pareto },
  ]
  const ws2 = XLSX.utils.json_to_sheet(resumo)
  XLSX.utils.book_append_sheet(wb, ws2, 'Resumo')

  XLSX.writeFile(wb, `curvas_guia_${resultado.reservatorio_cod}.xlsx`)
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENTE PRINCIPAL
// ─────────────────────────────────────────────────────────────────────────────

export default function Otimizador({ apiUrl, onAplicarCurvas }) {
  const baseUrl = apiUrl || import.meta.env?.VITE_API_URL || 'http://localhost:8000'

  // Estado da lista de reservatórios
  const [resList, setResList]   = useState([])
  const [apiError, setApiError] = useState(null)

  // Formulário
  const [cod,          setCod]          = useState('')
  const [anoIni,       setAnoIni]       = useState(1990)
  const [anoFim,       setAnoFim]       = useState(2020)
  const [demanda,      setDemanda]      = useState(5.0)
  const [volIniPct,    setVolIniPct]    = useState(50)
  const [nGen,         setNGen]         = useState(80)
  const [popSize,      setPopSize]      = useState(200)
  const [searchQuery,  setSearchQuery]  = useState('')
  const [dropOpen,     setDropOpen]     = useState(false)
  const dropRef = useRef(null)

  // Resultado
  const [resultado,  setResultado]  = useState(null)
  const [running,    setRunning]    = useState(false)
  const [error,      setError]      = useState(null)
  const [activeTab,  setActiveTab]  = useState('grafico')

  // Fechar dropdown ao clicar fora
  useEffect(() => {
    const h = e => { if (dropRef.current && !dropRef.current.contains(e.target)) setDropOpen(false) }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [])

  // Carregar reservatórios
  useEffect(() => {
    fetch(`${baseUrl}/api/reservatorios`)
      .then(r => r.json())
      .then(data => setResList(data))
      .catch(e => setApiError(e.message))
  }, [baseUrl])

  const resNome = useMemo(() =>
    resList.find(r => r.COD === cod)?.CORPO || '', [cod, resList])

  const filtered = useMemo(() =>
    resList.filter(r => r.CORPO.toLowerCase().includes(searchQuery.toLowerCase())).slice(0, 50),
    [resList, searchQuery])

  // Submeter optimização
  const handleRun = async () => {
    if (!cod) return
    setRunning(true); setError(null); setResultado(null)
    try {
      const res = await fetch(`${baseUrl}/api/otimizar-curvas`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cod, ano_inicial: parseInt(anoIni), ano_final: parseInt(anoFim),
          demanda_alvo: parseFloat(demanda),
          vol_inicial_pct: parseFloat(volIniPct),
          n_gen: parseInt(nGen), pop_size: parseInt(popSize),
        })
      })
      if (!res.ok) {
        const e = await res.json().catch(() => ({}))
        throw new Error(typeof e.detail === 'string' ? e.detail : JSON.stringify(e.detail) || 'Erro')
      }
      const data = await res.json()
      setResultado(data)
      setActiveTab('grafico')
    } catch (e) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  // Aplicar nos Níveis Meta (callback para o componente pai)
  const handleAplicar = () => {
    if (!resultado || !onAplicarCurvas) return
    // Converte as curvas no formato esperado pelo PlanoSecasPanel
    const { curvas_guia_percentual: c } = resultado
    const faixas = [
      { Faixa:'Normal',      Racionamento:0,  ...Object.fromEntries(MESES.map((m,i) => [m, c.normal[i]])) },
      { Faixa:'Atenção',     Racionamento:20, ...Object.fromEntries(MESES.map((m,i) => [m, c.atencao[i]])) },
      { Faixa:'Seca',        Racionamento:40, ...Object.fromEntries(MESES.map((m,i) => [m, c.seca[i]])) },
      { Faixa:'Seca Severa', Racionamento:70, ...Object.fromEntries(MESES.map((m,i) => [m, c.seca_severa[i]])) },
    ]
    onAplicarCurvas({ cod, nome: resNome, faixas })
  }

  const RES_TABS = [
    { id:'grafico',  label:'📈 Curvas Guia' },
    { id:'pareto',   label:'🎯 Fronteira de Pareto' },
    { id:'tabela',   label:'📋 Tabela' },
  ]

  return (
    <div className="ot-root" style={{ minHeight:600, paddingBottom:48 }}>
      <style>{CSS}</style>

      {/* ── Top bar ── */}
      <div style={{ padding:'18px 26px 0', display:'flex', alignItems:'flex-start',
        justifyContent:'space-between', gap:12, flexWrap:'wrap' }}>
        <div>
          <div style={{ display:'flex', alignItems:'center', gap:9, marginBottom:3 }}>
            <Cpu size={21} color="var(--orange)" strokeWidth={2}/>
            <h2 style={{ fontSize:19, fontWeight:800, color:'var(--text)',
              letterSpacing:'-0.01em', margin:0 }}>
              Optimizador de Curvas Guia
            </h2>
          </div>
          <p style={{ fontSize:11.5, color:'var(--text-light)', margin:0 }}>
            NSGA-II · Minimiza falhas de abastecimento e vertimento simultaneamente
          </p>
        </div>
        {resultado && (
          <div style={{ display:'flex', gap:7, flexWrap:'wrap' }}>
            <button className="ot-ghost" onClick={() => exportarCurvas(resultado)}>
              <FileSpreadsheet size={12}/> Excel
            </button>
            {onAplicarCurvas && (
              <button onClick={handleAplicar} style={{
                display:'flex', alignItems:'center', gap:6,
                background:'linear-gradient(135deg,var(--teal),#1e7d6f)',
                border:'none', borderRadius:'var(--radius-xs)', padding:'6px 14px',
                fontSize:12, fontWeight:700, color:'#fff', cursor:'pointer',
                boxShadow:'0 3px 12px rgba(42,157,143,0.3)', transition:'all .15s',
              }}>
                <ArrowRight size={12}/> Aplicar nos Níveis Meta
              </button>
            )}
          </div>
        )}
      </div>

      {/* ── API error ── */}
      {apiError && (
        <div style={{ margin:'12px 26px 0', padding:'10px 14px', background:'#fffbea',
          border:'1.5px solid #f5c842', borderRadius:'var(--radius-sm)',
          display:'flex', gap:9, alignItems:'flex-start' }}>
          <AlertTriangle size={13} color="#b48a0c" style={{ marginTop:1 }}/>
          <div style={{ fontSize:11, color:'#7a5c00' }}>
            <strong>API não encontrada.</strong> Verifique a URL do backend.
            <br/><span style={{ fontSize:10, opacity:.75 }}>{apiError}</span>
          </div>
        </div>
      )}

      {/* ── Main grid ── */}
      <div style={{ padding:'14px 26px 0', display:'grid',
        gridTemplateColumns:'295px 1fr', gap:16, alignItems:'start' }}>

        {/* ── Painel de configuração ── */}
        <Card style={{ padding:'18px 14px', position:'sticky', top:16 }}>
          <div style={{ fontSize:14.5, fontWeight:800, color:'var(--text)', marginBottom:2 }}>
            Configuração
          </div>
          <div style={{ fontSize:11, color:'var(--text-light)', marginBottom:14 }}>
            Reservatório: <strong style={{ color:'var(--orange-deep)' }}>
              {resNome || 'Nenhum selecionado'}
            </strong>
          </div>

          {/* Busca de reservatório */}
          <Label icon={Database}>Reservatório</Label>
          <div ref={dropRef} style={{ position:'relative' }}>
            <input
              value={searchQuery}
              onChange={e => { setSearchQuery(e.target.value); setDropOpen(true); if (!e.target.value) setCod('') }}
              onFocus={() => setDropOpen(true)}
              placeholder="Digite para buscar…"
              style={{ width:'100%', padding:'7px 10px', border:'1.5px solid var(--border)',
                borderRadius:'var(--radius-xs)', background:'#fff', color:'var(--text)',
                fontSize:12.5, outline:'none', transition:'border-color .15s' }}
              onFocusCapture={e => e.target.style.borderColor='var(--orange)'}
              onBlurCapture={e => e.target.style.borderColor='var(--border)'}
            />
            {dropOpen && filtered.length > 0 && (
              <div style={{ position:'absolute', top:'100%', left:0, right:0,
                background:'#fff', border:'1.5px solid var(--border)',
                borderRadius:'var(--radius-xs)', boxShadow:'var(--shadow)',
                zIndex:999, maxHeight:180, overflowY:'auto', marginTop:2 }}>
                {filtered.map(r => (
                  <div key={r.COD}
                    onMouseDown={() => { setCod(r.COD); setSearchQuery(r.CORPO); setDropOpen(false) }}
                    style={{ padding:'7px 11px', fontSize:12, cursor:'pointer',
                      borderBottom:'1px solid var(--border-light)', transition:'background .1s' }}
                    onMouseEnter={e => e.currentTarget.style.background='var(--orange-pale)'}
                    onMouseLeave={e => e.currentTarget.style.background='#fff'}>
                    <span style={{ fontWeight:600, color:'var(--text)' }}>{r.CORPO}</span>
                    <span style={{ fontSize:10, color:'var(--text-light)', marginLeft:8,
                      fontFamily:'JetBrains Mono' }}>{r.COD}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          {cod && (
            <div style={{ fontSize:9.5, color:'var(--text-light)', marginTop:3,
              fontFamily:'JetBrains Mono' }}>
              COD: {cod} · Cap: {resList.find(r=>r.COD===cod)?.['CAPAC (m³)']?.toFixed(2)} hm³
            </div>
          )}

          {/* Período */}
          <Label icon={Calendar}>Período</Label>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:6 }}>
            <div>
              <div style={{ fontSize:10, color:'var(--text-light)', marginBottom:3, fontWeight:600 }}>
                Ano Inicial
              </div>
              <FC type="number" value={anoIni} onChange={e=>setAnoIni(e.target.value)}
                min="1900" max="2100" placeholder="Ano"/>
            </div>
            <div>
              <div style={{ fontSize:10, color:'var(--text-light)', marginBottom:3, fontWeight:600 }}>
                Ano Final
              </div>
              <FC type="number" value={anoFim} onChange={e=>setAnoFim(e.target.value)}
                min="1900" max="2100" placeholder="Ano"/>
            </div>
          </div>

          {/* Demanda */}
          <Label icon={Waves}>Demanda Alvo (m³/s)</Label>
          <FC type="number" value={demanda} onChange={e=>setDemanda(e.target.value)}
            min="0" step="0.1"/>
          <div style={{ fontSize:9.5, color:'var(--text-light)', marginTop:3,
            fontFamily:'JetBrains Mono' }}>
            = {(demanda * 2.592e6 / 1e6).toFixed(3)} hm³/mês
          </div>

          {/* Vol inicial */}
          <Label icon={Database}>Vol. Inicial (%)</Label>
          <FC type="number" value={volIniPct} onChange={e=>setVolIniPct(e.target.value)}
            min="0" max="100" step="1"/>

          {/* Parâmetros do NSGA-II */}
          <Label icon={Cpu}>Parâmetros NSGA-II</Label>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:6 }}>
            <div>
              <div style={{ fontSize:10, color:'var(--text-light)', marginBottom:3, fontWeight:600 }}>
                Gerações
              </div>
              <FC type="number" value={nGen} onChange={e=>setNGen(e.target.value)}
                min="20" max="500" step="10"/>
            </div>
            <div>
              <div style={{ fontSize:10, color:'var(--text-light)', marginBottom:3, fontWeight:600 }}>
                População
              </div>
              <FC type="number" value={popSize} onChange={e=>setPopSize(e.target.value)}
                min="40" max="1000" step="20"/>
            </div>
          </div>

          {/* Info tempo estimado */}
          <div style={{ marginTop:10, padding:'8px 11px', background:'var(--blue-pale)',
            borderRadius:'var(--radius-xs)', fontSize:10.5, color:'var(--blue)',
            display:'flex', gap:6, alignItems:'flex-start' }}>
            <Info size={12} style={{ flexShrink:0, marginTop:1 }}/>
            <span>
              Tempo estimado: <strong>~{Math.round(nGen * popSize / 1000 * 0.8)} – {Math.round(nGen * popSize / 1000 * 1.5)} s</strong>.
              Aumente as gerações para curvas mais refinadas.
            </span>
          </div>

          <ProgressBar running={running} nGen={nGen}/>

          {/* Botão */}
          <button onClick={handleRun} disabled={running || !cod}
            style={{ width:'100%', marginTop:14, padding:12,
              background: running || !cod ? 'var(--border)'
                : 'linear-gradient(135deg,var(--orange),var(--orange-deep))',
              border:'none', borderRadius:'var(--radius-sm)',
              color: running || !cod ? 'var(--text-light)' : '#fff',
              fontSize:13, fontWeight:800,
              cursor: running || !cod ? 'not-allowed' : 'pointer',
              boxShadow: running ? 'none' : '0 4px 18px var(--orange-glow)',
              transition:'all .2s', letterSpacing:'0.02em' }}
            onMouseEnter={e=>{ if(!running&&cod) e.currentTarget.style.transform='translateY(-1px)' }}
            onMouseLeave={e=>e.currentTarget.style.transform='none'}>
            {running
              ? <span style={{ display:'flex', alignItems:'center', justifyContent:'center', gap:8 }}>
                  <RefreshCw size={14} className="ot-spin"/> A optimizar…
                </span>
              : '▶ Iniciar Optimização'}
          </button>
        </Card>

        {/* ── Área de resultados ── */}
        <div style={{ display:'flex', flexDirection:'column', gap:12 }}>

          {/* Erro */}
          {error && (
            <div style={{ background:'var(--red-pale)', border:'1.5px solid var(--red)',
              borderRadius:'var(--radius-sm)', padding:'10px 14px',
              display:'flex', alignItems:'center', gap:9 }}>
              <AlertTriangle size={13} color="var(--red)"/>
              <span style={{ flex:1, fontSize:11.5, color:'var(--red)', fontWeight:500 }}>{error}</span>
              <button onClick={()=>setError(null)} style={{ background:'none', border:'none',
                cursor:'pointer', color:'var(--red)', fontSize:16, lineHeight:1 }}>×</button>
            </div>
          )}

          {/* A carregar */}
          {running && !resultado && (
            <Card style={{ padding:'52px 20px', display:'flex', flexDirection:'column',
              alignItems:'center', gap:14 }}>
              <div style={{ position:'relative' }}>
                <div style={{ width:60, height:60, borderRadius:'50%',
                  background:'var(--orange-pale)', border:'2px solid var(--orange-light)',
                  display:'flex', alignItems:'center', justifyContent:'center' }}>
                  <Cpu size={26} color="var(--orange)" strokeWidth={1.5}/>
                </div>
                <RefreshCw size={18} color="var(--orange)" className="ot-spin"
                  style={{ position:'absolute', bottom:-2, right:-2,
                    background:'var(--card)', borderRadius:'50%', padding:2 }}/>
              </div>
              <div style={{ textAlign:'center' }}>
                <div style={{ fontSize:14, fontWeight:800, color:'var(--text)', marginBottom:5 }}>
                  NSGA-II em execução
                </div>
                <div style={{ fontSize:11.5, color:'var(--text-light)', maxWidth:320 }}>
                  Avaliando {popSize} indivíduos ao longo de {nGen} gerações.
                  Aguarde — isto pode demorar alguns minutos.
                </div>
              </div>
            </Card>
          )}

          {/* Estado inicial */}
          {!running && !resultado && !error && (
            <Card style={{ display:'flex', flexDirection:'column', alignItems:'center',
              justifyContent:'center', padding:'56px 20px', gap:14 }}>
              <div style={{ width:64, height:64, borderRadius:'50%',
                background:'var(--orange-pale)', border:'2px solid var(--orange-light)',
                display:'flex', alignItems:'center', justifyContent:'center' }}>
                <Target size={28} color="var(--orange)" strokeWidth={1.5}/>
              </div>
              <div style={{ textAlign:'center' }}>
                <div style={{ fontSize:14.5, fontWeight:800, color:'var(--text)', marginBottom:5 }}>
                  Pronto para optimizar
                </div>
                <div style={{ fontSize:11.5, color:'var(--text-light)', maxWidth:320 }}>
                  Selecione um reservatório, defina o período e a demanda alvo,
                  e clique em <strong>Iniciar Optimização</strong>.
                </div>
              </div>
            </Card>
          )}

          {/* Resultados */}
          {resultado && (
            <>
              {/* KPIs */}
              <div style={{ display:'grid',
                gridTemplateColumns:'repeat(auto-fill,minmax(150px,1fr))', gap:10 }}
                className="ot-fade">
                <MCard label="Garantia (F1)" icon={CheckCircle2}
                  value={resultado.fitness_vencedor.f1_deficit_quadratico.toFixed(1)}
                  sub="défice quadrático total"
                  variant={resultado.fitness_vencedor.f1_deficit_quadratico < 1 ? 'success' : 'yellow'}/>
                <MCard label="Vertimento (F2)" icon={Droplets}
                  value={`${resultado.fitness_vencedor.f2_vertimento_hm3.toFixed(1)} hm³`}
                  sub="total no período"
                  variant={resultado.fitness_vencedor.f2_vertimento_hm3 < 10 ? 'success' : 'info'}/>
                <MCard label="Soluções Pareto" icon={BarChart3}
                  value={resultado.n_solucoes_pareto}
                  sub="soluções não dominadas"
                  variant="info"/>
                <MCard label="Meses" icon={Calendar}
                  value={resultado.periodo.n_meses}
                  sub={`${resultado.periodo.ano_inicial}–${resultado.periodo.ano_final}`}
                  variant="default"/>
              </div>

              {/* Tabs de resultado */}
              <div style={{ display:'flex', gap:3, background:'var(--card)',
                border:'1.5px solid var(--border)', borderRadius:'var(--radius-sm)',
                padding:3, width:'fit-content', boxShadow:'var(--shadow-sm)',
                flexWrap:'wrap' }}>
                {RES_TABS.map(t => (
                  <button key={t.id}
                    className={`ot-tab ${activeTab===t.id?'on':'off'}`}
                    onClick={() => setActiveTab(t.id)}>{t.label}</button>
                ))}
              </div>

              {/* Gráfico de bandas */}
              {activeTab === 'grafico' && (
                <Card className="ot-fade" style={{ padding:'16px 18px' }}>
                  <div style={{ marginBottom:14 }}>
                    <div style={{ fontSize:13, fontWeight:800, color:'var(--text)' }}>
                      Curvas Guia Optimizadas — {resultado.reservatorio_nome}
                    </div>
                    <div style={{ fontSize:11, color:'var(--text-light)', marginTop:2 }}>
                      Bandas mensais de volume: verde = seguro · vermelho = crítico
                    </div>
                  </div>
                  <GraficoCurvas curvas={resultado.curvas_guia_percentual}/>
                </Card>
              )}

              {/* Fronteira de Pareto */}
              {activeTab === 'pareto' && (
                <Card className="ot-fade" style={{ padding:'16px 18px' }}>
                  <div style={{ marginBottom:12 }}>
                    <div style={{ fontSize:13, fontWeight:800, color:'var(--text)' }}>
                      Fronteira de Pareto
                    </div>
                    <div style={{ fontSize:11, color:'var(--text-light)', marginTop:2 }}>
                      Solução vencedora destacada — menor F1 (prioridade de abastecimento)
                    </div>
                  </div>
                  <GraficoPareto resultado={resultado}/>
                </Card>
              )}

              {/* Tabela */}
              {activeTab === 'tabela' && (
                <Card className="ot-fade" style={{ overflow:'hidden' }}>
                  <div style={{ padding:'11px 16px', borderBottom:'1.5px solid var(--border)',
                    background:'var(--bg)', display:'flex', alignItems:'center',
                    justifyContent:'space-between' }}>
                    <div>
                      <span style={{ fontSize:13, fontWeight:800, color:'var(--text)' }}>
                        Curvas Guia — valores mensais (% da capacidade)
                      </span>
                    </div>
                  </div>
                  <TabelaCurvas curvas={resultado.curvas_guia_percentual}/>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
