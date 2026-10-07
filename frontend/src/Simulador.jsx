import React, { useState, useEffect, useMemo } from 'react'
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar, ComposedChart,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea,
  ScatterChart, Scatter,
} from 'recharts'
import {
  Waves, Download, RefreshCw, AlertTriangle, CheckCircle2, AlertCircle,
  Plus, Trash2, Database, Calendar, Settings2, Zap, ChevronDown,
  ChevronLeft, ChevronRight, BarChart3, TrendingDown, Droplets,
  Save, Info, Shield, ArrowLeftRight, FileSpreadsheet, BarChart2,
  Activity,
} from 'lucide-react'
import * as XLSX from './utils/planilha'
import { downloadElementAsPng } from './components/ChartExportMenu'

// nomes dos meses abreviados, usados em vários lugares do app
const MESES = ['JAN','FEV','MAR','ABR','MAI','JUN','JUL','AGO','SET','OUT','NOV','DEZ']
const CENARIOS_HIDROLOGICOS = [
  { id: 'historico', label: 'Histórico' },
  { id: 'afluencia_zero', label: 'Afluência zero' },
  { id: 'seco_50', label: 'Seco -50%' },
  { id: 'umido_120', label: 'Úmido +20%' },
  { id: 'fator_personalizado', label: 'Percentual personalizado' },
  { id: 'seca_repetida', label: 'Repetir seca histórica' },
  { id: 'reamostragem_anual', label: 'Reamostragem de anos' },
]
const ANO_MIN_SERIE = 1911
const ANO_MAX_SERIE = 2021

const m3sToLps = value => Number(((parseFloat(value) || 0) * 1000).toFixed(3))
const lpsToM3s = value => Math.max(0, (parseFloat(value) || 0) / 1000)

function getCapacidadeHm3(reservatorio) {
  if (!reservatorio) return 0
  const chaveCapacidade = Object.keys(reservatorio).find(k => k.startsWith('CAPAC'))
  return parseFloat(reservatorio.capacidade_hm3 ?? reservatorio[chaveCapacidade] ?? 0) || 0
}

// lista de anos disponíveis pra seleção: de 1911 até 2020
const ANOS  = Array.from({ length: 110 }, (_, i) => 1911 + i)

// paleta de cores usada nos gráficos quando tem mais de um reservatório
const COLORS = [
  { stroke: '#264fa3', fill: '#264fa3', fillOp: 0.15 },
  { stroke: '#e07b2a', fill: '#e07b2a', fillOp: 0.15 },
  { stroke: '#2a9d8f', fill: '#2a9d8f', fillOp: 0.15 },
  { stroke: '#9b2dca', fill: '#9b2dca', fillOp: 0.15 },
]

// CSS global injetado no componente — define as variáveis de cor, fontes,
// animações e os estilos das classes reutilizáveis (.sim-tab, .sim-ghost, etc.)
const CSS = `
  .sim-root *, .sim-root *::before, .sim-root *::after { box-sizing: border-box; }
  .sim-root {
    --bg: #fdf6ee; --orange: #e07b2a; --orange-light: #f5a654;
    --orange-pale: #fdebd3; --orange-deep: #c46318; --orange-glow: rgba(224,123,42,0.18);
    --teal: #2a9d8f; --teal-pale: #d4f5ef;
    --blue: #264fa3; --blue-light: #4a7cc7; --blue-pale: #dde8f8;
    --red: #d94040; --red-pale: #fde8e8;
    --yellow: #d4a017; --yellow-pale: #fef3cd;
    --text: #1e1208; --text-mid: #5a3c24; --text-light: #855f40;
    --border: #ecdcc8; --border-light: #f5ebe0; --card: #ffffff;
    --shadow-sm: 0 1px 4px rgba(150,90,40,0.08);
    --shadow: 0 2px 16px rgba(150,90,40,0.10);
    --radius: 14px; --radius-sm: 9px; --radius-xs: 6px;
    font-family: 'Sora', sans-serif;
    background: var(--bg); color: var(--text);
    -webkit-font-smoothing: antialiased;
  }
  .sim-root ::-webkit-scrollbar { width: 5px; height: 5px; }
  .sim-root ::-webkit-scrollbar-track { background: var(--bg); }
  .sim-root ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
  .sim-root input, .sim-root select, .sim-root textarea, .sim-root button { font-family: 'Sora', sans-serif; }
  @keyframes sim-fade { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
  @keyframes sim-spin  { to { transform: rotate(360deg); } }
  .sim-fade { animation: sim-fade 0.35s ease both; }
  .sim-spin { animation: sim-spin 1.2s linear infinite; }
  .sim-tab { padding:8px 16px; border-radius:9px; border:none; font-size:12px; font-weight:700; cursor:pointer; transition:all 0.15s; white-space:nowrap; }
  .sim-tab.on  { background:var(--orange-pale); color:var(--orange-deep); }
  .sim-tab.off { background:none; color:var(--text-light); }
  .sim-tab.off:hover { background:var(--orange-pale); color:var(--orange); }
  .sim-ghost { display:flex; align-items:center; gap:6px; background:none; border:1.5px solid var(--border); border-radius:var(--radius-xs); padding:6px 12px; font-size:11.5px; font-weight:600; color:var(--text-mid); cursor:pointer; transition:all 0.15s; }
  .sim-ghost:hover { border-color:var(--orange); color:var(--orange); }
  .sim-plano-inp { width:100%; border:1.5px solid transparent; border-radius:4px; background:transparent; text-align:center; font-size:11px; font-family:'Sora',sans-serif; color:var(--text); padding:3px 2px; transition:all 0.15s; outline:none; }
  .sim-plano-inp:focus { border-color:var(--orange); background:var(--orange-pale); }
  .sim-tr:hover td { background: var(--orange-pale) !important; }
  .sim-root button:focus-visible, .sim-root input:focus-visible, .sim-root select:focus-visible {
    outline: 2px solid var(--orange-deep); outline-offset: 2px;
  }
  .sim-root .sim-invalid { border-color: var(--red) !important; background: var(--red-pale) !important; }
  .sim-erro-campo { font-size: 10px; color: var(--red); margin-top: 3px; font-weight: 600; line-height: 1.3; }
  .sim-layout { padding: 14px 26px 0; display: grid; grid-template-columns: 295px minmax(0, 1fr); gap: 16px; align-items: start; }
  .sim-config-card { position: sticky; top: 16px; }
  .sim-header { padding: 18px 26px 0; }
  @media (max-width: 900px) {
    .sim-layout { grid-template-columns: minmax(0, 1fr); padding: 12px 12px 0; }
    .sim-config-card { position: static; }
    .sim-header { padding: 14px 12px 0; }
    .sim-root .recharts-legend-wrapper { font-size: 9px !important; }
  }
  @media (prefers-reduced-motion: reduce) {
    .sim-fade, .sim-spin { animation: none !important; }
  }
`

// cria o objeto de API com os métodos de busca e simulação
// usa a URL passada como prop ou a variável de ambiente VITE_API_URL
function makeApi(base) {
  const b = base || import.meta.env?.VITE_API_URL || 'http://127.0.0.1:8000'

  // faz um GET simples e retorna o JSON; lança erro se der ruim
  const get  = async (p) => {
    const r = await fetch(`${b}${p}`)
    if (!r.ok) {
      const e = await r.json().catch(() => ({}))
      throw new Error(typeof e.detail === 'string' ? e.detail : `Erro ${r.status} ao consultar ${p}`)
    }
    return r.json()
  }

  // faz um POST com body JSON; trata os erros de validação do backend
  const post = async (p, body) => {
    const r = await fetch(`${b}${p}`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) })
    if (!r.ok) {
      const e = await r.json().catch(() => ({}))
      if (Array.isArray(e.detail)) throw new Error(e.detail.map(x => String(x.msg || '').replace(/^Value error,\s*/, '')).join(' '))
      throw new Error(typeof e.detail==='string' ? e.detail : JSON.stringify(e.detail) || 'Erro')
    }
    return r.json()
  }

  return {
    fetchReservatorios: () => get('/api/reservatorios'),
    fetchPresets:       () => get('/api/presets'),
    fetchPlanoSecas:    (cod) => get(`/api/plano-secas/${encodeURIComponent(cod)}`),
    runSimulacao:       (payload) => post('/api/simular', payload),
  }
}

// exporta os resultados da simulação pra um arquivo Excel (.xlsx)
function exportExcel(resultados, modo) {
  const wb = XLSX.utils.book_new()
  const segundos = 2.592e6
  const isSerie = modo === 'Série'

  resultados.forEach(r => {
    const rows = r.dados.map(d => {
      const row = {
        'Mês/Ano':                     d.Data,
        'Armazenamento Inicial (hm³)': parseFloat(d['Armazenamento Inicial'] ?? 0),
        'Armazenamento Final (hm³)':   parseFloat(d['Armazenamento Final'] ?? 0),
        'Afluências (hm³/mês)':        parseFloat(d['Afluências (hm³/mês)'] ?? 0),
        'Evaporação (hm³)':            parseFloat(d['Evaporação (hm³)'] ?? 0),
        'Demanda Solicitada (m³/s)':   parseFloat(d['Demanda Solicitada (m³/s)'] ?? 0),
        'Demanda Atendida (m³/s)':     parseFloat(d['Demanda Atendida (m³/s)'] ?? 0),
        'Retirada Total (m³/s)':       parseFloat(d['Retirada Total (m³/s)'] ?? 0),
        'Demanda Atendida (hm³)':      parseFloat(d['Demanda Atendida (m³/s)'] ?? 0) * (segundos / 1e6),
        'Racionamento (%)':            parseFloat(d['Racionamento (%)'] ?? 0),
        'Vertimento (hm³)':            parseFloat(d['Vertimento (hm³)'] ?? 0),
        'Falha':                       d['Falha'] ?? 'Não',
        'Modo Operação':               d['Modo Operação'] ?? 'Normal',
      }
      if (isSerie) {
        row['Transferência Recebida (m³/s)'] = parseFloat(d['Transferência Recebida (m³/s)'] ?? 0)
        row['Transferência Enviada (m³/s)']  = parseFloat(d['Transferência Enviada (m³/s)'] ?? 0)
      }
      return row
    })
    const ws = XLSX.utils.json_to_sheet(rows)
    XLSX.utils.book_append_sheet(wb, ws, r.reservatorio.slice(0, 31))
  })
  const nomeAcude = resultados[0]?.reservatorio 
    ? resultados[0].reservatorio.replace(/\s+/g, '_') 
    : 'hidrica'

  XLSX.writeFile(wb, `simulacao_${nomeAcude}.xlsx`)
}

function Card({ children, style, className = '' }) {
  return <div className={className} style={{ background:'var(--card)', border:'1.5px solid var(--border)', borderRadius:'var(--radius)', boxShadow:'var(--shadow)', ...style }}>{children}</div>
}

function Label({ icon: Icon, children }) {
  return (
    <div style={{ display:'flex', alignItems:'center', gap:7, fontSize:10.5, fontWeight:700, color:'var(--text-light)', textTransform:'uppercase', letterSpacing:'0.06em', marginBottom:7, marginTop:15 }}>
      {Icon && <Icon size={12} strokeWidth={2.5} />}{children}
    </div>
  )
}

function FC({ as='input', children, style, invalid=false, className='', ...props }) {
  const base = { width:'100%', padding:'7px 10px', border:'1.5px solid var(--border)', borderRadius:'var(--radius-xs)', background:'#fff', color:'var(--text)', fontSize:12.5, fontFamily:'Sora, sans-serif', outline:'none', transition:'border-color 0.15s', ...style }
  const onF = e => e.target.style.borderColor = 'var(--orange)'
  const onB = e => e.target.style.borderColor = 'var(--border)'
  const extra = { className: `${className} ${invalid ? 'sim-invalid' : ''}`.trim(), 'aria-invalid': invalid || undefined }
  if (as === 'select') return <select style={{ ...base, appearance:'none', cursor:'pointer' }} onFocus={onF} onBlur={onB} {...extra} {...props}>{children}</select>
  return <input style={base} onFocus={onF} onBlur={onB} {...extra} {...props} />
}

// mensagem de erro exibida logo abaixo de um campo
function ErroCampo({ id, children }) {
  if (!children) return null
  return <div id={id} className="sim-erro-campo" role="alert">{children}</div>
}

// valida a configuração antes do envio; devolve { geral: [...], itens: [{campo: msg}] }
function validarConfiguracao({ items, modo, mesIni, anoIni, mesFim, anoFim, cenarioHidrologico, fatorAfluencia, secaAnoIni, secaAnoFim, histerese }) {
  const itens = items.map(it => {
    const e = {}
    if (!String(it.nome || '').trim()) e.nome = 'Selecione o reservatório.'
    const vol = parseFloat(it.volPct)
    if (!Number.isFinite(vol) || vol < 0 || vol > 100) e.volPct = 'Use um valor entre 0 e 100%.'
    const dem = parseFloat(it.demanda1 ?? it.demanda)
    if (!Number.isFinite(dem) || dem < 0) e.demanda = 'A demanda não pode ser negativa.'
    const gat = parseFloat(it.gatilho)
    if (modo !== 'Individual' && (!Number.isFinite(gat) || gat < 0 || gat > 100)) e.gatilho = 'Use um valor entre 0 e 100%.'
    return e
  })
  const geral = {}
  const ai = parseInt(anoIni), af = parseInt(anoFim)
  if (!Number.isFinite(ai) || ai < ANO_MIN_SERIE || ai > ANO_MAX_SERIE) geral.anoIni = `Ano entre ${ANO_MIN_SERIE} e ${ANO_MAX_SERIE}.`
  if (!Number.isFinite(af) || af < ANO_MIN_SERIE || af > ANO_MAX_SERIE) geral.anoFim = `Ano entre ${ANO_MIN_SERIE} e ${ANO_MAX_SERIE}.`
  if (!geral.anoIni && !geral.anoFim && ai * 12 + MESES.indexOf(mesIni) > af * 12 + MESES.indexOf(mesFim)) geral.periodo = 'O início deve ser anterior ao fim.'
  if (cenarioHidrologico === 'fator_personalizado') {
    const fp = parseFloat(fatorAfluencia)
    if (!Number.isFinite(fp) || fp < 0 || fp > 500) geral.fator = 'Use um percentual entre 0 e 500%.'
  }
  if (cenarioHidrologico === 'seca_repetida') {
    const a = parseInt(secaAnoIni), b = parseInt(secaAnoFim)
    if (!Number.isFinite(a) || !Number.isFinite(b) || a < ANO_MIN_SERIE || b > ANO_MAX_SERIE) geral.seca = `Informe anos entre ${ANO_MIN_SERIE} e ${ANO_MAX_SERIE}.`
    else if (a > b) geral.seca = 'O ano inicial da seca deve ser anterior ao final.'
  }
  const h = parseFloat(histerese)
  if (modo !== 'Individual' && (!Number.isFinite(h) || h < 0 || h > 100)) geral.histerese = 'Use um valor entre 0 e 100 pontos percentuais.'
  const temErro = Object.keys(geral).length > 0 || itens.some(e => Object.keys(e).length > 0)
  return { geral, itens, temErro }
}

const CTip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background:'#fff', border:'1.5px solid var(--border)', borderRadius:10, padding:'9px 13px', boxShadow:'var(--shadow)', fontSize:11 }}>
      <div style={{ fontWeight:700, marginBottom:5, color:'var(--text)' }}>{label}</div>
      {payload.map((p,i) => (
        <div key={i} style={{ display:'flex', gap:7, alignItems:'center', marginBottom:2 }}>
          <div style={{ width:7, height:7, borderRadius:'50%', background:p.color }} />
          <span style={{ color:'var(--text-mid)' }}>{p.name}:</span>
          <span style={{ fontWeight:600, fontFamily:'JetBrains Mono', color:'var(--text)' }}>{typeof p.value==='number'?p.value.toFixed(2):p.value}</span>
        </div>
      ))}
    </div>
  )
}

function tickFmt(v) { if (!v) return ''; const p=v.split('-'); return `${p[1]}/${p[0]?.slice(2)}` }

function MCard({ label, value, sub, variant='default', icon:Icon }) {
  const C = { default:{a:'var(--orange)',b:'var(--orange-pale)',t:'var(--orange)'}, success:{a:'var(--teal)',b:'var(--teal-pale)',t:'var(--teal)'}, danger:{a:'var(--red)',b:'var(--red-pale)',t:'var(--red)'}, info:{a:'var(--blue)',b:'var(--blue-pale)',t:'var(--blue)'}, yellow:{a:'var(--yellow)',b:'var(--yellow-pale)',t:'var(--yellow)'} }[variant]
  return (
    <div className="sim-fade" style={{ background:'var(--card)', border:'1.5px solid var(--border)', borderRadius:'var(--radius)', padding:'14px 16px', boxShadow:'var(--shadow)', position:'relative', overflow:'hidden' }}>
      <div style={{ position:'absolute', top:0, left:0, right:0, height:3, background:C.a, borderRadius:'4px 4px 0 0' }} />
      <div style={{ display:'flex', alignItems:'flex-start', justifyContent:'space-between', marginBottom:7 }}>
        <div style={{ fontSize:9.5, fontWeight:700, color:'var(--text-light)', textTransform:'uppercase', letterSpacing:'0.06em' }}>{label}</div>
        {Icon && <div style={{ width:24, height:24, borderRadius:6, background:C.b, display:'flex', alignItems:'center', justifyContent:'center' }}><Icon size={12} color={C.a} strokeWidth={2.5} /></div>}
      </div>
      <div style={{ fontSize:24, fontWeight:800, color:C.t, fontFamily:'JetBrains Mono', lineHeight:1, marginBottom:4 }}>{value}</div>
      <div style={{ fontSize:10.5, color:'var(--text-light)', lineHeight:1.4 }}>{sub}</div>
    </div>
  )
}

function calcFalhasConjuntas(resultados, modo) {
  if (!resultados?.length) return []
  const n = resultados[0].dados.length
  return Array.from({length:n}, (_,t) =>
    resultados.some(r=>r.dados[t]?.['Falha']==='Sim')
  )
}

function MetricsRow({ resultados, modo }) {
  if (!resultados?.length) return null
  const totalMeses = resultados[0].dados.length
  const falhasConj = calcFalhasConjuntas(resultados, modo)
  const falhasSist = falhasConj.filter(Boolean).length
  
  let rac=0, racM=0, atend=0, solic=0, vert=0, evap=0, transf=0

  // Agrupa os dados por mês para não somar as métricas do sistema em dobro
  for (let t = 0; t < totalMeses; t++) {
    let teveRacNoMes = false
    let racMaxNoMes = 0
    
    resultados.forEach(r => {
      const d = r.dados[t]
      if (!d) return
      
      const rc = parseFloat(d['Racionamento (%)'])||0
      if (rc > 0) {
        teveRacNoMes = true
        racMaxNoMes = Math.max(racMaxNoMes, rc) // Pega o racionamento mais severo do sistema
      }
      
      atend += parseFloat(d['Demanda Atendida (m³/s)'])||0
      solic += parseFloat(d['Demanda Solicitada (m³/s)'])||0
      vert  += parseFloat(d['Vertimento (hm³)'])||0
      evap  += parseFloat(d['Evaporação (hm³)'])||0
      transf += parseFloat(d['Transferência Recebida (m³/s)'])||0
    })
    
    // Contabiliza o mês de racionamento apenas 1 vez para o sistema
    if (teveRacNoMes) {
      racM++
      rac += racMaxNoMes
    }
  }

  const freq  = totalMeses>0?((falhasSist/totalMeses)*100).toFixed(1):'0.0'
  const at    = solic>0?((atend/solic)*100).toFixed(1):'100.0'
  const rm    = racM>0?(rac/racM).toFixed(1):'0.0'
  const ok    = parseFloat(freq)===0

  return (
    <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(160px,1fr))', gap:10 }}>
      <div className="sim-fade" style={{ background:'var(--card)', border:`1.5px solid ${ok?'var(--teal-pale)':'var(--red-pale)'}`, borderRadius:'var(--radius)', padding:'14px 16px', boxShadow:'var(--shadow)', position:'relative', overflow:'hidden' }}>
        <div style={{ position:'absolute', top:0, left:0, right:0, height:3, background:ok?'var(--teal)':'var(--red)', borderRadius:'4px 4px 0 0' }} />
        <div style={{ display:'flex', justifyContent:'space-between', marginBottom:7 }}>
          <div style={{ fontSize:9.5, fontWeight:700, color:'var(--text-light)', textTransform:'uppercase', letterSpacing:'0.06em' }}>Freq. Falha Sist.</div>
          <div style={{ width:24, height:24, borderRadius:6, background:ok?'var(--teal-pale)':'var(--red-pale)', display:'flex', alignItems:'center', justifyContent:'center' }}>
            {ok?<CheckCircle2 size={12} color="var(--teal)"/>:<AlertCircle size={12} color="var(--red)"/>}
          </div>
        </div>
        <div style={{ fontSize:24, fontWeight:800, color:ok?'var(--teal)':'var(--red)', fontFamily:'JetBrains Mono', lineHeight:1, marginBottom:4 }}>{freq}%</div>
        <div style={{ fontSize:10.5, color:'var(--text-light)' }}>{ok?'✓ Sistema 100% abastecido':`${falhasSist} mês(es) em ${totalMeses}`}</div>
      </div>
      <MCard label="Atendimento Médio" value={`${at}%`} sub="Demanda atendida vs solicitada" variant={parseFloat(at)>=95?'success':parseFloat(at)>=80?'yellow':'danger'} icon={BarChart3}/>
      <MCard label="Meses Simulados" value={totalMeses} sub={`${resultados.length} reservatório(s)`} variant="info" icon={TrendingDown}/>
      <MCard label="Rac. Médio" value={`${rm}%`} sub={`em ${racM} mês(es) c/ restrição`} variant={parseFloat(rm)>20?'danger':parseFloat(rm)>0?'yellow':'success'} icon={AlertCircle}/>
      <MCard label="Evaporação Total" value={evap.toFixed(1)} sub="hm³ no período" variant="default" icon={Droplets}/>
      <MCard label="Vertimento Total" value={vert.toFixed(1)} sub="hm³ excedente" variant={vert>0?'info':'success'} icon={Droplets}/>
      {transf>0 && <MCard label="Transferências" value={transf.toFixed(1)} sub="m³/s acumulados" variant="yellow" icon={ArrowLeftRight}/>}
    </div>
  )
}
function MesesAbastecidos({ resultados, modo, params }) {
  if (!resultados?.length) return null
  const totalSist = resultados[0].dados.length
  return (
    <Card className="sim-fade" style={{ padding:'16px 20px' }}>
      <div style={{ fontSize:13, fontWeight:800, color:'var(--text)', marginBottom:12, display:'flex', alignItems:'center', gap:8 }}>
        <Activity size={15} color="var(--orange)"/>
        Meses Abastecidos por Reservatório
      </div>
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(200px,1fr))', gap:10 }}>
        {resultados.map((r,i) => {
          const isParalelo = modo === 'Paralelo'
          const mesesComResp = isParalelo
            ? r.dados.filter(d => (parseFloat(d['Demanda Solicitada (m³/s)'])||0) > 0)
            : r.dados
          const atend = mesesComResp.filter(d => d['Falha']==='Não').length
          const base  = mesesComResp.length
          const pct   = base > 0 ? ((atend / base) * 100) : 0
          const cor     = pct>=95?'var(--teal)':pct>=80?'var(--yellow)':'var(--red)'
          const corPale = pct>=95?'var(--teal-pale)':pct>=80?'var(--yellow-pale)':'var(--red-pale)'
          return (
            <div key={i} style={{ background:'var(--bg)', border:'1.5px solid var(--border)', borderRadius:'var(--radius-sm)', padding:'12px 14px' }}>
              <div style={{ fontSize:11, fontWeight:700, color:'var(--text-mid)', marginBottom:6, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{r.reservatorio}</div>
              <div style={{ display:'flex', alignItems:'baseline', gap:5, marginBottom:6 }}>
                <span style={{ fontSize:22, fontWeight:800, color:cor, fontFamily:'JetBrains Mono' }}>{atend}</span>
                <span style={{ fontSize:11, color:'var(--text-light)' }}>
                  {isParalelo ? `/ ${base} meses c/ responsabilidade` : `/ ${totalSist} meses`}
                </span>
              </div>
              <div style={{ height:6, background:'var(--border)', borderRadius:99, overflow:'hidden' }}>
                <div style={{ width:`${pct}%`, height:'100%', background:cor, borderRadius:99, transition:'width 0.6s ease' }}/>
              </div>
              <div style={{ marginTop:5, display:'flex', justifyContent:'space-between' }}>
                <span style={{ fontSize:10, background:corPale, color:cor, borderRadius:20, padding:'1px 8px', fontWeight:700 }}>{pct.toFixed(1)}% atendido</span>
                {base-atend>0 && <span style={{ fontSize:10, color:'var(--red)' }}>{base-atend} falha(s)</span>}
              </div>
              {isParalelo && base < totalSist && (
                <div style={{ marginTop:4, fontSize:10, color:'var(--text-light)' }}>
                  {totalSist-base} mês(es) sob responsabilidade do outro reservatório
                </div>
              )}
            </div>
          )
        })}
      </div>
    </Card>
  )
}

function FailureDetail({ resultados }) {
  if (!resultados?.length) return null
  const falhas = []
  
  resultados.forEach(r => r.dados.forEach(d => {
    if (d['Falha']==='Sim') {
      falhas.push({ 
        reservatorio:r.reservatorio, 
        data:d.Data, 
        volIni:parseFloat(d['Armazenamento Inicial']||0).toFixed(2), 
        demSol:parseFloat(d['Demanda Solicitada (m³/s)']||0).toFixed(3), 
        demAt:parseFloat(d['Demanda Atendida (m³/s)']||0).toFixed(3), 
        rac:parseFloat(d['Racionamento (%)']||0).toFixed(1), 
        modo:d['Modo Operação'] 
      })
    }
  }))

  // Ordena cronologicamente para a tabela ficar bonita
  falhas.sort((a,b) => a.data.localeCompare(b.data))

  return (
    <Card style={{ padding:'16px 20px', borderColor:falhas.length>0?'var(--red-pale)':'var(--teal-pale)' }}>
      <div style={{ display:'flex', alignItems:'center', gap:9, marginBottom:falhas.length?12:0 }}>
        {falhas.length>0?<AlertCircle size={16} color="var(--red)"/>:<CheckCircle2 size={16} color="var(--teal)"/>}
        <div>
          <div style={{ fontSize:13, fontWeight:800, color:'var(--text)' }}>Falha de Atendimento da Demanda</div>
          {!falhas.length && <div style={{ fontSize:11.5, color:'var(--teal)', marginTop:2, fontWeight:600 }}>✓ Nenhuma falha no período.</div>}
        </div>
      </div>
      {falhas.length>0 && (
        <div style={{ maxHeight:250, overflowY:'auto', display:'flex', flexDirection:'column', gap:4 }}>
          {falhas.map((f,i) => (
            <div key={i} style={{ background:'var(--red-pale)', borderRadius:'var(--radius-xs)', padding:'8px 12px', display:'flex', alignItems:'center', justifyContent:'space-between', gap:8, flexWrap:'wrap' }}>
              <div style={{ display:'flex', alignItems:'center', gap:7 }}>
                <div style={{ background:'var(--red)', color:'#fff', borderRadius:5, padding:'1px 7px', fontSize:10, fontWeight:700, fontFamily:'JetBrains Mono' }}>{f.data}</div>
                <span style={{ fontSize:11.5, fontWeight:700, color:'var(--red)' }}>{f.reservatorio}</span>
              </div>
              <div style={{ display:'flex', gap:10, fontSize:10.5, color:'var(--text-mid)', flexWrap:'wrap' }}>
                <span>Vol: <strong style={{ fontFamily:'JetBrains Mono' }}>{f.volIni} hm³</strong></span>
                <span>Sol.: <strong style={{ fontFamily:'JetBrains Mono' }}>{f.demSol}</strong></span>
                <span>At.: <strong style={{ fontFamily:'JetBrains Mono', color:'var(--red)' }}>{f.demAt}</strong></span>
                {parseFloat(f.rac)>0 && <span>Rac: <strong>{f.rac}%</strong></span>}
                <span style={{ background:'rgba(217,64,64,0.1)', borderRadius:4, padding:'1px 5px', fontSize:10, fontWeight:600, color:'var(--red)' }}>{f.modo}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}
function ChartCard({ title, subtitle, children, action }) {
  return (
    <Card className="sim-fade" style={{ padding:'16px 18px' }}>
      <div style={{ marginBottom:12, display:'flex', alignItems:'flex-start', justifyContent:'space-between', gap:10 }}>
        <div>
          <div style={{ fontSize:13, fontWeight:800, color:'var(--text)' }}>{title}</div>
          {subtitle && <div style={{ fontSize:11, color:'var(--text-light)', marginTop:2 }}>{subtitle}</div>}
        </div>
        {action}
      </div>
      {children}
    </Card>
  )
}

function ExportTableImageButton({ targetId, filename }) {
  return (
    <button
      type="button"
      className="sim-ghost"
      title="Exportar tabela como imagem PNG"
      onClick={() => downloadElementAsPng(document.getElementById(targetId), filename)}
      style={{ flexShrink:0 }}
    >
      <Download size={12}/> Imagem PNG
    </button>
  )
}

function ResSel({ resultados, sel, onChange }) {
  if (resultados.length <= 1) return null
  return (
    <div style={{display:'flex',gap:4,flexWrap:'wrap',marginBottom:8}}>
      <button onClick={()=>onChange('todos')} style={{padding:'4px 11px',borderRadius:20,border:`1.5px solid ${sel==='todos'?'var(--orange)':'var(--border)'}`,background:sel==='todos'?'var(--orange-pale)':'none',color:sel==='todos'?'var(--orange-deep)':'var(--text-light)',fontSize:10.5,fontWeight:700,cursor:'pointer',transition:'all 0.15s'}}>Sobrepostos</button>
      {resultados.map((r,i)=>(
        <button key={i} onClick={()=>onChange(i)} style={{padding:'4px 11px',borderRadius:20,border:`1.5px solid ${sel===i?COLORS[i%4].stroke:'var(--border)'}`,background:sel===i?'rgba('+hexToRgb(COLORS[i%4].stroke)+',0.1)':'none',color:sel===i?COLORS[i%4].stroke:'var(--text-light)',fontSize:10.5,fontWeight:700,cursor:'pointer',transition:'all 0.15s'}}>{r.reservatorio}</button>
      ))}
    </div>
  )
}

function hexToRgb(hex){const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return `${r},${g},${b}`}

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

  const zoomProps = {
    onMouseDown: e => e?.activeLabel !== undefined && setLeft(e.activeLabel),
    onMouseMove: e => left !== null && e?.activeLabel !== undefined && setRight(e.activeLabel),
    onMouseUp: () => {
      if (left !== null && right !== null && String(left) !== String(right)) setDomain({ start: left, end: right })
      setLeft(null)
      setRight(null)
    },
  }

  return {
    data: activeData,
    isZoomed: Boolean(domain),
    reset: () => setDomain(null),
    props: zoomProps,
    area: left !== null && right !== null
      ? <ReferenceArea x1={left} x2={right} strokeOpacity={0.3} fill="#2a9d8f" fillOpacity={0.16} />
      : null,
  }
}

function ZoomReset({ zoom }) {
  if (!zoom?.isZoomed) return null
  return (
    <button className="sim-ghost" onClick={zoom.reset} style={{ marginBottom: 8, padding: '5px 9px', fontSize: 10 }}>
      Resetar zoom
    </button>
  )
}

function normalizarNomeFaixa(nome) {
  return String(nome || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .trim()
    .toLowerCase()
}

function nomeFaixaGrafico(d) {
  if (d?.Falha === 'Sim') return 'Colapso'
  return String(d?.['Modo Operação'] || '').trim() || 'Normal'
}

function corFaixaGrafico(nome, racionamento = 0) {
  const faixa = normalizarNomeFaixa(nome)
  if (faixa.includes('colapso') || faixa.includes('falha')) return '#7f1d1d'
  if (faixa.includes('normal') || faixa.includes('acima do teto')) return '#2a9d8f'
  if (faixa.includes('alerta') || faixa.includes('atencao')) return '#d4a017'
  if (faixa.includes('severa') || faixa.includes('emerg') || faixa.includes('critic')) return '#d94040'
  if (faixa.includes('seca')) return '#e07b2a'
  if (racionamento >= 70) return '#d94040'
  if (racionamento >= 35) return '#e07b2a'
  if (racionamento > 0) return '#d4a017'
  return '#264fa3'
}

function Charts({ resultados, params, modo, usarNiveisMeta = false }) {
  const [selVol,setSelVol]=useState('todos')
  const [selDem,setSelDem]=useState('todos')
  const [selRac,setSelRac]=useState('todos')
  const [selBal,setSelBal]=useState('todos')
  const [selTr,setSelTr]=useState('todos')
  const [marcarFalhas,setMarcarFalhas]=useState(true)
  const [destacarRac,setDestacarRac]=useState(false)

  if (!resultados?.length) return null

  const allD = [...new Set(resultados.flatMap(r => r.dados.map(d => d.Data)))].sort()
  const iv   = Math.max(0, Math.floor(allD.length/12)-1)
  const resSel = (sel) => sel==='todos' ? resultados : [resultados[sel]]
  const volSeriesMeta = {}

  const mkVolData = (sel) => {
    const res = resSel(sel)
    const prevByRes = {}
    return allD.map(data => {
      const p = {data, __rac: 0}
      res.forEach((r,i) => {
        const d = r.dados.find(x=>x.Data===data)
        const ri = resultados.indexOf(r)
        const cap = params?.[ri]?.capacidade || 1
        if(d){
          const vf = parseFloat(d['Armazenamento Final'])||0
          const volPct = cap>0 ? parseFloat(((vf/cap)*100).toFixed(1)) : 0
          p.__rac = Math.max(p.__rac, parseFloat(d['Racionamento (%)']) || 0)
          if (d['Falha'] === 'Sim') p[`Falha (${r.reservatorio})`] = volPct
          if (usarNiveisMeta) {
            const faixa = nomeFaixaGrafico(d)
            const chave = `Vol.${faixa} (${r.reservatorio})`
            const racionamento = parseFloat(d['Racionamento (%)']) || 0
            p[chave] = volPct
            volSeriesMeta[chave] = {
              nome: `${faixa} (${r.reservatorio})`,
              cor: corFaixaGrafico(faixa, racionamento),
            }
            const anterior = prevByRes[r.reservatorio]
            if (anterior && anterior !== chave) p[anterior] = volPct
            prevByRes[r.reservatorio] = chave
          } else {
            p[`Vol.% (${r.reservatorio})`] = volPct
          }
        }
      })
      return p
    })
  }

  const mkDemData = (sel) => {
    const res = resSel(sel)
    return allD.map(data => { const p={data}; res.forEach(r=>{ const d=r.dados.find(x=>x.Data===data); if(d){p[`Sol.(${r.reservatorio})`]=parseFloat(d['Demanda Solicitada (m³/s)'])||0; p[`At.(${r.reservatorio})`]=parseFloat(d['Demanda Atendida (m³/s)'])||0} }); return p })
  }

  const mkRacData = (sel) => {
    const res = resSel(sel)
    return allD.map(data => { const p={data}; res.forEach(r=>{ const d=r.dados.find(x=>x.Data===data); if(d) p[r.reservatorio]=parseFloat(d['Racionamento (%)'])||0 }); return p })
  }

  const mkBalData = (sel) => {
    const res = resSel(sel)
    return allD.map(data => { const p={data}; res.forEach(r=>{ const d=r.dados.find(x=>x.Data===data); if(d){p[`Afluência(${r.reservatorio})`]=parseFloat(d['Afluências (hm³/mês)'])||0; p[`Evap.(${r.reservatorio})`]=parseFloat(d['Evaporação (hm³)'])||0; p[`Demanda Atendida(${r.reservatorio})`]=(parseFloat(d['Demanda Atendida (m³/s)'])||0)*2.592; p[`Vertimento(${r.reservatorio})`]=parseFloat(d['Vertimento (hm³)'])||0} }); return p })
  }

  const mkSerieVazoesData = (sel) => {
    const res = resSel(sel)
    return allD.map(data => { const p={data}; res.forEach(r=>{ const d=r.dados.find(x=>x.Data===data); if(d) p[r.reservatorio]=parseFloat(d['Vazão (m³/s)'])||0 }); return p })
  }

  const mkTrData = (sel) => {
    const res = resSel(sel)
    return allD.map(data => { const p={data}; res.forEach(r=>{ const d=r.dados.find(x=>x.Data===data); if(d){const ev=parseFloat(d['Transferência Enviada (m³/s)'])||0; if(ev>0)p[`Env.(${r.reservatorio})`]=ev} }); return p })
  }

  const volData = mkVolData(selVol)
  const demData = mkDemData(selDem)
  const racData = mkRacData(selRac)
  const serieVazoesData = mkSerieVazoesData(selBal)
  const trData  = mkTrData(selTr)
  const volZoom = useBoxZoom(volData)
  const demZoom = useBoxZoom(demData)
  const racZoom = useBoxZoom(racData)
  const serieVazoesZoom = useBoxZoom(serieVazoesData)
  const trZoom = useBoxZoom(trData)

  const keysFromAllRows = (data, predicate=()=>true) => [
    ...new Set(data.flatMap(row=>Object.keys(row).filter(k=>k!=='data'&&predicate(k))))
  ]
  const volKeys = keysFromAllRows(volData,k=>k.startsWith('Vol.'))
  const falhaKeys = keysFromAllRows(volData,k=>k.startsWith('Falha ('))
  const temRacionamento = volData.some(p => p.__rac > 0)
  // períodos contínuos com racionamento, calculados sobre o trecho visível (zoom)
  const periodosRac = []
  if (destacarRac) {
    let inicio = null
    volZoom.data.forEach((p, idx) => {
      const ativo = p.__rac > 0
      if (ativo && inicio === null) inicio = p.data
      const proximoAtivo = volZoom.data[idx + 1]?.__rac > 0
      if (ativo && !proximoAtivo) { periodosRac.push([inicio, p.data]); inicio = null }
    })
  }
  const racKeys = keysFromAllRows(racData)
  const serieKeys = keysFromAllRows(serieVazoesData)
  const trKeys = keysFromAllRows(trData)
  const hasTransf = modo==='Série' && trKeys.length>0
  const activeRes = (sel) => sel==='todos'?resultados:[resultados[sel]]

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
      <ChartCard title="Volume Armazenado (%)" subtitle={marcarFalhas && falhaKeys.length ? 'Pontos vermelhos: meses com falha de atendimento' : undefined}>
        <ResSel resultados={resultados} sel={selVol} onChange={setSelVol}/>
        <div style={{display:'flex',gap:14,flexWrap:'wrap',marginBottom:6,fontSize:10.5,color:'var(--text-mid)'}}>
          <label style={{display:'inline-flex',alignItems:'center',gap:5,cursor:'pointer'}}>
            <input type="checkbox" checked={marcarFalhas} onChange={e=>setMarcarFalhas(e.target.checked)}/> Marcar meses com falha
          </label>
          {temRacionamento&&(
            <label style={{display:'inline-flex',alignItems:'center',gap:5,cursor:'pointer'}}>
              <input type="checkbox" checked={destacarRac} onChange={e=>setDestacarRac(e.target.checked)}/> Destacar períodos com racionamento
            </label>
          )}
        </div>
        <ZoomReset zoom={volZoom}/>
        <div style={{ height:250 }}>
          <ResponsiveContainer>
            <ComposedChart data={volZoom.data} margin={{top:4,right:28,left:0,bottom:0}} {...volZoom.props}>
              <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
              <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
              <YAxis yAxisId="vol" domain={[0,100]} tick={{fontSize:10,fill:'var(--blue)'}} label={{value:'%',angle:-90,position:'insideLeft',fill:'var(--blue)',fontSize:10}}/>
              <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
              {usarNiveisMeta
                ? volKeys.map((k)=>{
                    const meta = volSeriesMeta[k] || { nome:k.replace(/^Vol\./, ''), cor:COLORS[0].stroke }
                    return <Line key={k} yAxisId="vol" type="linear" dataKey={k} name={meta.nome} stroke={meta.cor} strokeWidth={2.2} dot={false} connectNulls={false}/>
                  })
                : volKeys.map((k,i)=><Area key={k} yAxisId="vol" type="monotone" dataKey={k} stroke={COLORS[i%4].stroke} fill={COLORS[i%4].fill} fillOpacity={COLORS[i%4].fillOp} strokeWidth={2} dot={false}/>)}
              {periodosRac.map(([x1,x2],idx)=>(
                <ReferenceArea key={`rac${idx}`} yAxisId="vol" x1={x1} x2={x2} fill="#d4a017" fillOpacity={0.14} strokeOpacity={0} ifOverflow="hidden"/>
              ))}
              {marcarFalhas&&falhaKeys.map(k=>(
                <Line key={k} yAxisId="vol" dataKey={k} name={k} stroke="none" legendType="circle" isAnimationActive={false}
                  dot={{r:2.6,fill:'#d94040',stroke:'#fff',strokeWidth:0.6}} activeDot={{r:4,fill:'#d94040'}}/>
              ))}
              {volZoom.area}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      <ChartCard title="Demanda: Solicitada vs Atendida" subtitle="m³/s mensal">
        <ResSel resultados={resultados} sel={selDem} onChange={setSelDem}/>
        <ZoomReset zoom={demZoom}/>
        <div style={{ height:200 }}>
          <ResponsiveContainer>
            <LineChart data={demZoom.data} margin={{top:4,right:20,left:0,bottom:0}} {...demZoom.props}>
              <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
              <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
              <YAxis tick={{fontSize:10,fill:'var(--text-light)'}} label={{value:'m³/s',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
              <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
              {activeRes(selDem).map((r,i)=>{const gi=resultados.indexOf(r);return[
                <Line key={`s${gi}`} type="monotone" dataKey={`Sol.(${r.reservatorio})`} stroke={COLORS[gi%4].stroke} strokeWidth={2} strokeDasharray={"5 3"} dot={false}/>,
                <Line key={`a${gi}`} type="monotone" dataKey={`At.(${r.reservatorio})`}  stroke={COLORS[gi%4].stroke} strokeWidth={2} dot={false}/>,
              ]})}
              {demZoom.area}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {racKeys.length>0 && (
        <ChartCard title="Racionamento Mensal" subtitle="Níveis Meta — Racionamento aplicado (%)">
          <ResSel resultados={resultados} sel={selRac} onChange={setSelRac}/>
          <ZoomReset zoom={racZoom}/>
          <div style={{ height:180 }}>
            <ResponsiveContainer>
              <BarChart data={racZoom.data} margin={{top:4,right:20,left:0,bottom:0}} {...racZoom.props}>
                <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
                <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
                <YAxis domain={[0,100]} tick={{fontSize:10,fill:'var(--text-light)'}} label={{value:'%',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
                <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
                {racKeys.map((k,i)=><Bar key={k} dataKey={k} fill={COLORS[resultados.findIndex(r=>r.reservatorio===k)%4]?.stroke||COLORS[0].stroke} fillOpacity={0.75} radius={[3,3,0,0]}/>)}
                {racZoom.area}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      )}

      <ChartCard title="Série de Vazões Afluentes" subtitle="Vazão mensal afluente a cada reservatório (m³/s)">
        <ResSel resultados={resultados} sel={selBal} onChange={setSelBal}/>
        <ZoomReset zoom={serieVazoesZoom}/>
        <div style={{ height:185 }}>
          <ResponsiveContainer>
            <LineChart data={serieVazoesZoom.data} margin={{top:4,right:20,left:0,bottom:0}} {...serieVazoesZoom.props}>
              <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
              <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
              <YAxis tick={{fontSize:10,fill:'var(--text-light)'}} label={{value:'m³/s',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
              <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
              {serieKeys.map((k,i)=><Line key={k} type="monotone" dataKey={k} stroke={COLORS[i%4].stroke} strokeWidth={1.8} dot={false}/>)}
              {serieVazoesZoom.area}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {hasTransf && (
        <ChartCard title="Transferências entre Reservatórios" subtitle="m³/s mensal">
          <ResSel resultados={resultados} sel={selTr} onChange={setSelTr}/>
          <ZoomReset zoom={trZoom}/>
          <div style={{ height:180 }}>
            <ResponsiveContainer>
              <BarChart data={trZoom.data} margin={{top:4,right:20,left:0,bottom:0}} {...trZoom.props}>
                <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
                <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
                <YAxis tick={{fontSize:10,fill:'var(--text-light)'}} label={{value:'m³/s',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
                <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
                {trKeys.map((k)=><Bar key={k} dataKey={k} fill="#9b2dca" fillOpacity={0.7} radius={[3,3,0,0]}/>)}
                {trZoom.area}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      )}
    </div>
  )
}

function VazoesDetail({ resultados, modo }) {
  const [sel, setSel] = useState(0)
  const [page, setPage] = useState(0)
  const PAGE = 18

  if (!resultados?.length) return null

  const r = resultados[sel]
  const dados = r.dados
  const allD  = [...new Set(resultados.flatMap(x => x.dados.map(d => d.Data)))].sort()
  const iv    = Math.max(0, Math.floor(allD.length/12)-1)

  const serieData = allD.map(data => {
    const p = { data }
    resultados.forEach(res => {
      const d = res.dados.find(x => x.Data === data)
      if (d) {
        p[`Afluência(${res.reservatorio})`] = parseFloat(d['Afluências (hm³/mês)']) || 0
        p[`Demanda Atendida(${res.reservatorio})`] = (parseFloat(d['Demanda Atendida (m³/s)']) || 0) * 2.592
        p[`Evap.(${res.reservatorio})`] = parseFloat(d['Evaporação (hm³)']) || 0
      }
    })
    return p
  })

  const totalPg = Math.ceil(dados.length / PAGE)
  const pagDados = dados.slice(page * PAGE, (page+1)*PAGE)
  const balZoom = useBoxZoom(serieData)

  const VCOLS_ALL = [
    { key:'Data',                           label:'Mês/Ano',     mono:true  },
    { key:'Vazão (m³/s)',                   label:'Vazão (m³/s)',mono:true  },
    { key:'Afluências (hm³/mês)',           label:'Afluência (hm³)',mono:true},
    { key:'Evaporação (hm³)',               label:'Evap. (hm³)', mono:true  },
    { key:'Armazenamento Inicial',          label:'Vol. Ini.',   mono:true  },
    { key:'Armazenamento Final',            label:'Vol. Fin.',   mono:true  },
    { key:'Demanda Solicitada (m³/s)',      label:'Dem. Sol.',   mono:true  },
    { key:'Demanda Atendida (m³/s)',        label:'Dem. At.',    mono:true  },
    { key:'Retirada Total (m³/s)',          label:'Ret. Total',  mono:true  },
    { key:'Transferência Recebida (m³/s)',  label:'Tr. Rec.',    mono:true  },
    { key:'Transferência Enviada (m³/s)',   label:'Tr. Env.',    mono:true  },
    { key:'Racionamento (%)',               label:'Rac.(%)',     mono:true  },
    { key:'Vertimento (hm³)',               label:'Vertimento',  mono:true  },
    { key:'Falha',                          label:'Falha',       align:'center'},
    { key:'Modo Operação',                  label:'Modo',        align:'center'},
  ]

  const VCOLS = modo==='Série' ? VCOLS_ALL : VCOLS_ALL.filter(c=>!c.key.includes('Transferência'))

  function fv(val,key){
    if(val===null||val===undefined||val==='') return '—'
    if(key==='Falha'||key==='Modo Operação'||key==='Data') return val
    const n=parseFloat(val); return isNaN(n)?val:n.toFixed(3)
  }

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
      {resultados.length>1 && (
        <div style={{ display:'flex', gap:5, flexWrap:'wrap' }}>
          {resultados.map((res,i)=>(
            <button key={i} onClick={()=>{setSel(i);setPage(0)}}
              style={{ padding:'5px 13px', borderRadius:20, border:`1.5px solid ${sel===i?'var(--orange)':'var(--border)'}`, background:sel===i?'var(--orange-pale)':'var(--card)', color:sel===i?'var(--orange-deep)':'var(--text-light)', fontSize:11.5, fontWeight:700, cursor:'pointer', transition:'all 0.15s' }}>
              {res.reservatorio}
            </button>
          ))}
        </div>
      )}

      <ChartCard title="Balanço Hídrico Mensal" subtitle="Afluência, evaporação e demanda atendida (hm³/mês)">
        <ZoomReset zoom={balZoom}/>
        <div style={{ height:220 }}>
          <ResponsiveContainer>
            <BarChart data={balZoom.data} margin={{top:4,right:20,left:0,bottom:0}} {...balZoom.props}>
              <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
              <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
              <YAxis tick={{fontSize:10,fill:'var(--text-light)'}} label={{value:'hm³',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
              <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
              {resultados.map((res,i)=>[
                <Bar key={`af${i}`} dataKey={`Afluência(${res.reservatorio})`} fill="#2a9d8f" fillOpacity={0.65} radius={[3,3,0,0]}/>,
                <Bar key={`da${i}`} dataKey={`Demanda Atendida(${res.reservatorio})`} fill="#d4a017" fillOpacity={0.7} radius={[3,3,0,0]}/>,
                <Bar key={`ev${i}`} dataKey={`Evap.(${res.reservatorio})`} fill="#e07b2a" fillOpacity={0.65} radius={[3,3,0,0]}/>,
              ])}
              {balZoom.area}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      <Card style={{ overflow:'hidden' }}>
        <div style={{ padding:'11px 16px', borderBottom:'1.5px solid var(--border)', display:'flex', alignItems:'center', justifyContent:'space-between', background:'var(--bg)' }}>
          <div>
            <span style={{ fontSize:13, fontWeight:800, color:'var(--text)' }}>{r.reservatorio}</span>
            <span style={{ fontSize:10.5, color:'var(--text-light)', marginLeft:10 }}>{dados.length} registros mensais</span>
          </div>
        </div>
        <div style={{ overflowX:'auto' }}>
          <table style={{ width:'100%', borderCollapse:'collapse', fontSize:11 }}>
            <thead>
              <tr style={{ background:'var(--bg)' }}>
                {VCOLS.map(c=>(
                  <th key={c.key} style={{ padding:'7px 10px', textAlign:c.align||'right', fontSize:9.5, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.05em', color:'var(--text-light)', borderBottom:'1.5px solid var(--border)', whiteSpace:'nowrap' }}>{c.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pagDados.map((d,ri)=>{
                const isFail = d['Falha']==='Sim'
                const hasRac = parseFloat(d['Racionamento (%)'])>0
                return (
                  <tr key={ri} className="sim-tr" style={{ background:isFail?'rgba(217,64,64,0.04)':'transparent' }}>
                    {VCOLS.map(c=>{
                      const val = fv(d[c.key],c.key)
                      let col='var(--text-mid)', fw=400
                      if(c.key==='Armazenamento Inicial'){col='var(--blue-light)';fw=600}
                      if(c.key==='Armazenamento Final'){col='var(--blue)';fw=700}
                      if(c.key==='Falha'){col=val==='Sim'?'var(--red)':'var(--teal)';fw=700}
                      if(c.key==='Racionamento (%)'&&hasRac){col='var(--yellow)';fw=600}
                      if(c.key==='Data'){col='var(--text)';fw=600}
                      if(c.key==='Vazão (m³/s)'){col='var(--blue-light)';fw=600}
                      return (
                        <td key={c.key} style={{ padding:'6px 10px', textAlign:c.align||'right', borderBottom:'1px solid var(--border-light)', fontFamily:c.mono?'JetBrains Mono':'Sora', color:col, fontWeight:fw, whiteSpace:'nowrap' }}>
                          {c.key==='Falha'
                            ? <span style={{ display:'inline-flex', padding:'1px 6px', borderRadius:20, background:val==='Sim'?'var(--red-pale)':'var(--teal-pale)', fontSize:10 }}>{val==='Sim'?'✗ Sim':'✓ Não'}</span>
                            : c.key==='Modo Operação'
                              ? <span style={{ display:'inline-flex', padding:'1px 6px', borderRadius:20, background:val==='Normal'?'var(--blue-pale)':val?.includes('FALHA')?'var(--red-pale)':'var(--yellow-pale)', color:val==='Normal'?'var(--blue)':val?.includes('FALHA')?'var(--red)':'var(--yellow)', fontSize:10, fontWeight:600 }}>{val}</span>
                              : val}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {totalPg>1 && (
          <div style={{ padding:'9px 16px', borderTop:'1.5px solid var(--border)', display:'flex', alignItems:'center', justifyContent:'space-between' }}>
            <span style={{ fontSize:10.5, color:'var(--text-light)' }}>Pág. {page+1}/{totalPg} · {dados.length} registros</span>
            <div style={{ display:'flex', gap:5 }}>
              <button onClick={()=>setPage(p=>Math.max(0,p-1))} disabled={page===0} className="sim-ghost" style={{ opacity:page===0?0.4:1 }}><ChevronLeft size={10}/> Ant.</button>
              <button onClick={()=>setPage(p=>Math.min(totalPg-1,p+1))} disabled={page>=totalPg-1} className="sim-ghost" style={{ opacity:page>=totalPg-1?0.4:1 }}>Próx. <ChevronRight size={10}/></button>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}

function GarantiaAnalise({ resultados, modo, vazaoConjunta, params }) {
  if (!resultados?.length) return null

  const nomeFaixaGarantia = valor => valor === 'Acima do Teto' ? 'Normal' : (valor || 'Normal')
  const ordemFaixaGarantia = valor => ({
    normal: 0,
    alerta: 1,
    seca: 2,
    'seca severa': 3,
  })[nomeFaixaGarantia(valor).trim().toLocaleLowerCase('pt-BR')] ?? 99

  const totalMeses = resultados[0].dados.length
  const dfs = resultados.map(r => r.dados)

  const vazoesSystem = new Array(totalMeses).fill(0)
  dfs.forEach(df => df.forEach((d,t) => { vazoesSystem[t] += parseFloat(d['Demanda Atendida (m³/s)'])||0 }))

  // Correção: para o cálculo de garantia, a falha acontece se QUALQUER
  // reservatório não atingiu sua meta individual ou conjunta
  const falhasGarantia = Array.from({length:totalMeses}, (_,t) =>
    resultados.some(r=>r.dados[t]?.['Falha']==='Sim')
  )

  const numFalhas  = falhasGarantia.filter(Boolean).length
  const garantiaSistema = ((totalMeses - numFalhas) / totalMeses * 100)

  const vazoesSemFalha  = vazoesSystem.filter((_,t) => !falhasGarantia[t])
  const vazaoMedia   = vazoesSemFalha.length ? vazoesSemFalha.reduce((a,b)=>a+b,0)/vazoesSemFalha.length : 0
  const vazaoMaxima  = vazoesSemFalha.length ? Math.max(...vazoesSemFalha) : 0
  const vazaoMinima  = vazoesSemFalha.length ? Math.min(...vazoesSemFalha) : 0
  const demNominal = (params||[]).reduce((s,p)=>s+(p?.demanda_nominal||0),0) + (modo==='Paralelo'?(vazaoConjunta||0):0)

  const grouped = {}
  vazoesSystem.forEach((v,t) => {
    if (!falhasGarantia[t]) {
      const k = parseFloat(v.toFixed(3))
      grouped[k] = (grouped[k]||0)+1
    }
  })
  const sortedKeys = Object.keys(grouped).map(Number).sort((a,b)=>b-a)
  let cumFreq = 0
  const resumo = sortedKeys.map(k => {
    const perm = grouped[k]
    const freq = (perm/totalMeses*100)
    cumFreq += freq
    const atendimento = demNominal > 0 ? Math.max(0, Math.min(100, (k / demNominal) * 100)) : 0
    return { vazao:k, perm, freq:freq.toFixed(2), garantia:cumFreq.toFixed(2), atendimento: atendimento.toFixed(2) }
  })

  if (numFalhas>0) resumo.push({ vazao:'FALHA', perm:numFalhas, freq:(numFalhas/totalMeses*100).toFixed(2), garantia:'-', atendimento:'0.00' })

  const curvData = sortedKeys.map((k,i) => ({
    vazao: k,
    garantia: parseFloat(resumo[i].garantia),
    permanencia: parseFloat(resumo[i].freq),
  }))

  const garantiaZoom = useBoxZoom(curvData, 'garantia')

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:14 }}>
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(150px,1fr))', gap:10 }}>
        <MCard label="Garantia Sistema"  value={`${garantiaSistema.toFixed(2)}%`} sub="Meses sem falha / total" variant={garantiaSistema>=95?'success':garantiaSistema>=80?'yellow':'danger'} icon={Shield}/>
        <MCard label="Meses Simulados"   value={totalMeses}  sub={`${numFalhas} com falha`}     variant="info"    icon={TrendingDown}/>
        <MCard label="Dem. Nominal Total" value={`${demNominal.toFixed(3)}`} sub="m³/s"        variant="default" icon={BarChart3}/>
        <MCard label="Vazão Média"        value={vazaoMedia.toFixed(3)}  sub="m³/s" variant="default" icon={Waves}/>
        <MCard label="Vazão Máxima"       value={vazaoMaxima.toFixed(3)} sub="m³/s" variant="success" icon={Waves}/>
        <MCard label="Vazão Mínima"       value={vazaoMinima.toFixed(3)} sub="m³/s" variant={vazaoMinima>0?'info':'danger'} icon={Waves}/>
      </div>

      {curvData.length>1 && (
        <ChartCard title="Curva de Permanência e Garantia" subtitle="Garantia acumulada (%) × Vazão total do sistema (m³/s)">
          <ZoomReset zoom={garantiaZoom}/>
          <div style={{ height:230 }}>
            <ResponsiveContainer>
              <AreaChart data={garantiaZoom.data} margin={{top:4,right:20,left:0,bottom:0}} {...garantiaZoom.props}>
                <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
                <XAxis dataKey="garantia" type="number" domain={[0,100]} tick={{fontSize:10,fill:'var(--text-light)'}} label={{value:'Garantia Acumulada (%)',position:'insideBottom',offset:-2,fill:'var(--text-light)',fontSize:10}}/>
                <YAxis tick={{fontSize:10,fill:'var(--blue)'}} label={{value:'Vazão (m³/s)',angle:-90,position:'insideLeft',fill:'var(--blue)',fontSize:10}}/>
                <Tooltip content={<CTip/>}/>
                <Area type="monotone" dataKey="vazao" name="Vazão (m³/s)" stroke="#264fa3" fill="#264fa3" fillOpacity={0.15} strokeWidth={2} dot={false}/>
                {garantiaZoom.area}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      )}

      <ChartCard
        title="Análise de Vazões Totais do Sistema"
        subtitle="Permanência, frequência e garantia acumulada"
        action={<ExportTableImageButton targetId="garantia-tabela-sistema" filename="analise_vazoes_totais_sistema"/>}
      >
        <div style={{ overflowX:'auto' }}>
          <table id="garantia-tabela-sistema" style={{ width:'100%', borderCollapse:'collapse', fontSize:11.5 }}>
            <thead>
              <tr style={{ background:'var(--bg)' }}>
                {['Vazão Total Sistema (m³/s)','Atendimento (%)','Permanência (meses)','Frequência (%)','Garantia Acumulada (%)'].map(h=>(
                  <th key={h} style={{ padding:'7px 12px', textAlign:'right', fontSize:10, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.05em', color:'var(--text-light)', borderBottom:'1.5px solid var(--border)', whiteSpace:'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {resumo.map((row,i)=>(
                <tr key={i} className="sim-tr" style={{ background:row.vazao==='FALHA'?'var(--red-pale)':i%2===0?'transparent':'rgba(236,220,200,0.15)' }}>
                  <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono', fontWeight:row.vazao==='FALHA'?700:400, color:row.vazao==='FALHA'?'var(--red)':'var(--text)' }}>{row.vazao==='FALHA'?'FALHA':parseFloat(row.vazao).toFixed(3)}</td>
                  <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono', color:row.vazao==='FALHA'?'var(--red)':'var(--teal)', fontWeight:600 }}>{row.vazao==='FALHA'?'0.00%':`${row.atendimento}%`}</td>
                  <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono' }}>{row.perm}</td>
                  <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono' }}>{row.freq}%</td>
                  <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono', color:row.garantia==='-'?'var(--text-light)':'var(--blue)', fontWeight:row.garantia==='-'?400:600 }}>{row.garantia === '-' ? '—' : `${row.garantia}%`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ChartCard>

      <div style={{ fontSize:13, fontWeight:800, color:'var(--text)', marginTop:4, display:'flex', alignItems:'center', gap:8 }}>
        <Database size={14} color="var(--orange)"/>
        Detalhamento por Reservatório
      </div>

      {resultados.map((r, idx) => {
        const df    = r.dados
        const p     = params?.[idx]
        const demNom = p?.demanda_nominal || 0
        const demConj = modo==='Paralelo'&&idx===0 ? (vazaoConjunta||0) : 0
        const demTot = demNom + demConj

        const gruposVistos = new Set()
        const grupos = []
        df.forEach(d => {
          if (d['Falha'] === 'Sim') return
          const rac      = parseFloat(d['Racionamento (%)']) || 0
          const nome     = nomeFaixaGarantia(d['Modo Operação'])
          const chave    = `${nome}__${rac}`
          if (!gruposVistos.has(chave)) {
            gruposVistos.add(chave)
            grupos.push({ nome, rac, chave })
          }
        })

        grupos.sort((a,b) => ordemFaixaGarantia(a.nome) - ordemFaixaGarantia(b.nome) || a.rac - b.rac)

        let cumG = 0
        const tabelaRes = []

        grupos.forEach(({ nome, rac, chave }) => {
          const filtro = df.filter(d =>
            d['Falha'] === 'Não' &&
            (parseFloat(d['Racionamento (%)']) || 0) === rac &&
            nomeFaixaGarantia(d['Modo Operação']) === nome
          )
          if (!filtro.length) return
          const vazAlvo = demTot * (1 - rac / 100)
          const freq    = (filtro.length / totalMeses) * 100
          cumG += freq
          tabelaRes.push({ faixa:nome, rac:rac.toFixed(1), vazAlvo:vazAlvo.toFixed(3), count:filtro.length, freq:freq.toFixed(2), garantia:cumG.toFixed(2) })
        })

        const cntFalha = df.filter(d=>d['Falha']==='Sim').length
        if (cntFalha>0) tabelaRes.push({ faixa:'FALHA', rac:'FALHA', vazAlvo:'0.000', count:cntFalha, freq:(cntFalha/totalMeses*100).toFixed(2), garantia:'-' })

        return (
          <Card key={idx} style={{ overflow:'hidden' }}>
            <div style={{ padding:'11px 16px', background:'var(--bg)', borderBottom:'1.5px solid var(--border)', display:'flex', alignItems:'center', justifyContent:'space-between', flexWrap:'wrap', gap:6 }}>
              <div>
                <span style={{ fontSize:13, fontWeight:800, color:'var(--text)' }}>{r.reservatorio}</span>
                <span style={{ fontSize:11, color:'var(--text-light)', marginLeft:10 }}>Demanda Total: <strong style={{ fontFamily:'JetBrains Mono' }}>{demTot.toFixed(3)} m³/s</strong></span>
              </div>
              <div style={{ display:'flex', alignItems:'center', gap:7, flexWrap:'wrap' }}>
                {demConj>0 && (
                  <span style={{ fontSize:10.5, background:'var(--blue-pale)', color:'var(--blue)', borderRadius:6, padding:'2px 9px', fontWeight:600 }}>
                    {demNom.toFixed(3)} (espec.) + {demConj.toFixed(3)} (conjunta) = {demTot.toFixed(3)} m³/s
                  </span>
                )}
                <ExportTableImageButton targetId={`garantia-tabela-res-${idx}`} filename={`garantia_${r.reservatorio}`}/>
              </div>
            </div>
            <div style={{ overflowX:'auto' }}>
              <table id={`garantia-tabela-res-${idx}`} style={{ width:'100%', borderCollapse:'collapse', fontSize:11.5 }}>
                <thead>
                  <tr style={{ background:'var(--bg)' }}>
                    {['Nível Meta','Racionamento (%)','Vazão Total (m³/s)','Meses Responsável','Frequência (%)','Garantia (%)'].map(h=>(
                      <th key={h} style={{ padding:'7px 12px', textAlign:'right', fontSize:10, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.05em', color:'var(--text-light)', borderBottom:'1.5px solid var(--border)', whiteSpace:'nowrap' }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tabelaRes.map((row,i)=>(
                    <tr key={i} className="sim-tr" style={{ background:row.faixa.includes('FALHA')?'var(--red-pale)':i%2===0?'transparent':'rgba(236,220,200,0.15)' }}>
                      <td style={{ padding:'6px 12px', textAlign:'left', borderBottom:'1px solid var(--border-light)', fontWeight:600, color:row.faixa.includes('FALHA')?'var(--red)':'var(--text-mid)' }}>{row.faixa}</td>
                      <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono', color:row.rac==='FALHA'?'var(--red)':parseFloat(row.rac)>0?'var(--yellow)':'var(--teal)' }}>{row.rac==='FALHA'?'—':`${row.rac}%`}</td>
                      <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono' }}>{row.vazAlvo}</td>
                      <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono', fontWeight:600 }}>{row.count}</td>
                      <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono' }}>{row.freq}%</td>
                      <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono', color:row.garantia==='-'?'var(--text-light)':'var(--blue)', fontWeight:row.garantia==='-'?400:700 }}>{row.garantia==='-'?'—':`${row.garantia}%`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )
      })}
    </div>
  )
}

const RCOLS_BASE = [
  {key:'Data',label:'Data',align:'left'},{key:'Armazenamento Inicial',label:'Vol. Ini (hm³)',mono:true},{key:'Afluências (hm³/mês)',label:'Afluência (hm³)',mono:true},{key:'Evaporação (hm³)',label:'Evap. (hm³)',mono:true},{key:'Demanda Solicitada (m³/s)',label:'Dem. Sol.',mono:true},{key:'Demanda Atendida (m³/s)',label:'Dem. At.',mono:true},{key:'Retirada Total (m³/s)',label:'Ret. Total',mono:true},{key:'Transferência Recebida (m³/s)',label:'Tr. Rec.',mono:true},{key:'Transferência Enviada (m³/s)',label:'Tr. Env.',mono:true},{key:'Racionamento (%)',label:'Rac.(%)',mono:true},{key:'Vertimento (hm³)',label:'Vertimento',mono:true},{key:'Armazenamento Final',label:'Vol. Fin.',mono:true},{key:'Falha',label:'Falha',align:'center'},{key:'Modo Operação',label:'Modo',align:'center'},
]

function getRCols(modo) {
  if (modo === 'Série') return RCOLS_BASE
  return RCOLS_BASE.filter(c => !c.key.includes('Transferência'))
}

const PG=15

function fmtR(v,k){if(v===null||v===undefined||v==='')return'—';if(k==='Falha'||k==='Modo Operação'||k==='Data')return v;const n=parseFloat(v);return isNaN(n)?v:n.toFixed(2)}

function ResultsTable({ resultados, modo }) {
  const [sel,setSel]=useState(0)
  const [pg,setPg]=useState(0)
  if(!resultados?.length) return null
  const RCOLS = getRCols(modo)
  const r=resultados[sel]
  const totalPg=Math.ceil(r.dados.length/PG)
  const pagD=r.dados.slice(pg*PG,(pg+1)*PG)
  return (
    <div>
      {resultados.length>1&&(
        <div style={{display:'flex',gap:5,marginBottom:10,flexWrap:'wrap'}}>
          {resultados.map((x,i)=>(
            <button key={i} onClick={()=>{setSel(i);setPg(0)}} style={{padding:'5px 12px',borderRadius:20,border:`1.5px solid ${sel===i?'var(--orange)':'var(--border)'}`,background:sel===i?'var(--orange-pale)':'var(--card)',color:sel===i?'var(--orange-deep)':'var(--text-light)',fontSize:11.5,fontWeight:700,cursor:'pointer',transition:'all 0.15s'}}>{x.reservatorio}</button>
          ))}
        </div>
      )}
      <Card style={{overflow:'hidden'}}>
        <div style={{padding:'11px 16px',borderBottom:'1.5px solid var(--border)',display:'flex',alignItems:'center',justifyContent:'space-between',background:'var(--bg)'}}>
          <div><span style={{fontSize:13,fontWeight:800,color:'var(--text)'}}>{r.reservatorio}</span><span style={{fontSize:10.5,color:'var(--text-light)',marginLeft:10}}>{r.dados.length} registros</span></div>
        </div>
        <div style={{overflowX:'auto'}}>
          <table style={{width:'100%',borderCollapse:'collapse',fontSize:11}}>
            <thead><tr style={{background:'var(--bg)'}}>
              {RCOLS.map(c=><th key={c.key} style={{padding:'7px 10px',textAlign:c.align||'right',fontSize:9.5,fontWeight:700,textTransform:'uppercase',letterSpacing:'0.05em',color:'var(--text-light)',borderBottom:'1.5px solid var(--border)',whiteSpace:'nowrap'}}>{c.label}</th>)}
            </tr></thead>
            <tbody>
              {pagD.map((d,ri)=>{
                const fail=d['Falha']==='Sim',rac=parseFloat(d['Racionamento (%)'])>0
                return(
                  <tr key={ri} className="sim-tr" style={{background:fail?'rgba(217,64,64,0.04)':'transparent'}}>
                    {RCOLS.map(c=>{
                      const v=fmtR(d[c.key],c.key)
                      let col='var(--text-mid)',fw=400
                      if(c.key==='Armazenamento Inicial'){col='var(--blue-light)';fw=600}
                      if(c.key==='Armazenamento Final'){col='var(--blue)';fw=700}
                      if(c.key==='Falha'){col=v==='Sim'?'var(--red)':'var(--teal)';fw=700}
                      if(c.key==='Racionamento (%)'&&rac){col='var(--yellow)';fw=600}
                      if(c.key==='Data'){col='var(--text)';fw=600}
                      return(
                        <td key={c.key} style={{padding:'6px 10px',textAlign:c.align||'right',borderBottom:'1px solid var(--border-light)',fontFamily:c.mono?'JetBrains Mono':'Sora',color:col,fontWeight:fw,whiteSpace:'nowrap'}}>
                          {c.key==='Falha'?<span style={{display:'inline-flex',padding:'1px 6px',borderRadius:20,background:v==='Sim'?'var(--red-pale)':'var(--teal-pale)',fontSize:10}}>{v==='Sim'?'✗ Sim':'✓ Não'}</span>
                          :c.key==='Modo Operação'?<span style={{display:'inline-flex',padding:'1px 6px',borderRadius:20,background:v==='Normal'?'var(--blue-pale)':v?.includes('FALHA')?'var(--red-pale)':'var(--yellow-pale)',color:v==='Normal'?'var(--blue)':v?.includes('FALHA')?'var(--red)':'var(--yellow)',fontSize:10,fontWeight:600}}>{v}</span>
                          :v}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {totalPg>1&&(
          <div style={{padding:'9px 16px',borderTop:'1.5px solid var(--border)',display:'flex',alignItems:'center',justifyContent:'space-between'}}>
            <span style={{fontSize:10.5,color:'var(--text-light)'}}>Pág. {pg+1}/{totalPg}</span>
            <div style={{display:'flex',gap:5}}>
              <button onClick={()=>setPg(p=>Math.max(0,p-1))} disabled={pg===0} className="sim-ghost" style={{opacity:pg===0?0.4:1}}><ChevronLeft size={10}/> Ant.</button>
              <button onClick={()=>setPg(p=>Math.min(totalPg-1,p+1))} disabled={pg>=totalPg-1} className="sim-ghost" style={{opacity:pg>=totalPg-1?0.4:1}}>Próx. <ChevronRight size={10}/></button>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}

const PGPS_PERMANENCIAS = [
  {estado:'Normal',meta:90,color:'var(--teal)'},
  {estado:'Alerta',meta:5,color:'var(--yellow)'},
  {estado:'Seca',meta:3,color:'var(--orange)'},
  {estado:'Seca Severa',meta:2,color:'var(--red)'},
]

function PgpsValidation({ resultados }) {
  const controlador = resultados.find(r=>String(r.reservatorio).toLowerCase()==='fogareiro')
  const dados = controlador?.dados||[]
  if(!dados.length) return null
  const contagens = dados.reduce((acc,row)=>{
    const estado=row['Modo Operação']||'Normal'
    acc[estado]=(acc[estado]||0)+1
    return acc
  },{})

  return (
    <Card style={{padding:0,overflow:'hidden'}}>
      <div style={{padding:'13px 16px 10px',borderBottom:'1px solid var(--border-light)'}}>
        <div style={{fontSize:13.5,fontWeight:800,color:'var(--text)'}}>Validação do Cenário 1 (PGPS)</div>
        <div style={{fontSize:10.5,color:'var(--text-light)',marginTop:2}}>Permanência do hidrossistema definida pelo volume mensal do Fogareiro.</div>
      </div>
      <div style={{overflowX:'auto'}}>
        <table style={{width:'100%',borderCollapse:'collapse',fontSize:11.5}}>
          <thead><tr style={{background:'var(--bg)',color:'var(--text-light)'}}>
            {['Estado','Meta do plano','Obtida','Meses','Diferença'].map(h=><th key={h} style={{padding:'8px 12px',textAlign:h==='Estado'?'left':'right',fontWeight:700,borderBottom:'1px solid var(--border)'}}>{h}</th>)}
          </tr></thead>
          <tbody>{PGPS_PERMANENCIAS.map(item=>{
            const meses=contagens[item.estado]||0
            const obtida=meses/dados.length*100
            const diferenca=obtida-item.meta
            return <tr key={item.estado} style={{borderBottom:'1px solid var(--border-light)'}}>
              <td style={{padding:'8px 12px',fontWeight:700,color:item.color}}>{item.estado}</td>
              <td style={{padding:'8px 12px',textAlign:'right'}}>{item.meta.toFixed(1)}%</td>
              <td style={{padding:'8px 12px',textAlign:'right',fontWeight:700}}>{obtida.toFixed(2)}%</td>
              <td style={{padding:'8px 12px',textAlign:'right',fontFamily:'JetBrains Mono, monospace'}}>{meses}</td>
              <td style={{padding:'8px 12px',textAlign:'right',color:Math.abs(diferenca)<=0.5?'var(--teal)':'var(--text-mid)'}}>{diferenca>=0?'+':''}{diferenca.toFixed(2)} p.p.</td>
            </tr>
          })}</tbody>
        </table>
      </div>
    </Card>
  )
}

const FAIXAS_COR = {
  'Acima do Teto':{bg:'var(--teal-pale)',t:'var(--teal)'},
  'Normal':{bg:'var(--blue-pale)',t:'var(--blue)'},
  'Atenção':{bg:'var(--yellow-pale)',t:'var(--yellow)'},
  'Alerta':{bg:'var(--orange-pale)',t:'var(--orange-deep)'},
  'Emergência':{bg:'var(--red-pale)',t:'var(--red)'},
}

// =============================================================================
// CORRIGIDO: PlanoSecasPanel agora recebe onFaixasChange e chama ao salvar sessão
// =============================================================================
function PlanoSecasPanel({ api, reservatorios, onFaixasChange, faixasSessao, onOpenOtimizador }) {
  const [faixas,setFaixas]=useState(null)
  const [faixasOriginal,setFaixasOriginal]=useState(null)
  const [loading,setLoading]=useState(false)
  const [msg,setMsg]=useState(null)

  const [selRes, setSelRes] = useState(0)
  const reservatorio = reservatorios?.[selRes] || null

  const resKey = useMemo(
    ()=>(reservatorios||[]).map(r=>r.cod).join(','),
    [reservatorios]
  )

  useEffect(()=>{
    setSelRes(0)
    setFaixas(null)
    setFaixasOriginal(null)
    setMsg(null)
  },[resKey])

  useEffect(()=>{
    if(!reservatorio?.cod) return
    setFaixas(null); setFaixasOriginal(null)
    setLoading(true); setMsg(null)
    api.fetchPlanoSecas(reservatorio.cod)
      .then(d=>{
        const faixasAtivas = faixasSessao?.[reservatorio.cod] || faixasSessao?.[reservatorio.nome]
        setFaixas(JSON.parse(JSON.stringify(faixasAtivas || d)))
        setFaixasOriginal(JSON.parse(JSON.stringify(d)))
        if (faixasAtivas) setMsg({type:'session',text:'Curvas carregadas na sessão para este reservatório.'})
      })
      .catch(e=>{
        setFaixas([]); setFaixasOriginal([])
        setMsg({type:'error',text:`Não foi possível carregar os níveis meta da base de dados: ${e.message}`})
      })
      .finally(()=>setLoading(false))
  },[reservatorio?.cod, reservatorio?.nome, faixasSessao])

  const set=(idx,f,v)=>setFaixas(p=>p.map((x,i)=>i===idx?{...x,[f]:v}:x))
  const add=()=>setFaixas(p=>[...(p||[]),{Faixa:'Novo Nível',Racionamento:0,...Object.fromEntries(MESES.map(m=>[m,100]))}])
  const del=(idx)=>setFaixas(p=>p.filter((_,i)=>i!==idx))

  const revert=()=>{
    if(!faixasOriginal) return
    setFaixas(JSON.parse(JSON.stringify(faixasOriginal)))
    // ao reverter, notifica o pai para remover as faixas customizadas desse reservatório
    onFaixasChange && onFaixasChange(reservatorio.cod, null)
    setMsg({type:'info',text:'Revertido para o estado salvo na base de dados.'})
  }

  // CORRIGIDO: saveSession agora propaga as faixas editadas para o componente pai
  const saveSession=()=>{
    // converte os valores de string para número antes de mandar pro pai
    const faixasNormalizadas = (faixas || []).map(f => ({
      ...f,
      Racionamento: parseFloat(f.Racionamento) || 0,
      ...Object.fromEntries(MESES.map(m => [m, parseFloat(f[m]) || 0]))
    }))
    onFaixasChange && onFaixasChange(reservatorio.cod, faixasNormalizadas)
    setMsg({type:'session',text:'Aplicado na sessão. As alterações serão usadas na próxima simulação.'})
  }

  const hasChanges = faixasOriginal !== null && JSON.stringify(faixas) !== JSON.stringify(faixasOriginal)
  const isSessionOnly = hasChanges

  if(!reservatorios?.length || !reservatorio?.cod) return (
    <Card style={{padding:'32px 24px',textAlign:'center'}}>
      <Shield size={28} color="var(--text-light)" style={{marginBottom:10,opacity:0.35}}/>
      <div style={{fontSize:12.5,color:'var(--text-light)'}}>Selecione um reservatório na aba <strong>Configuração</strong>.</div>
    </Card>
  )

  return (
    <div style={{display:'flex',flexDirection:'column',gap:12}}>
      <Card style={{padding:'14px 18px'}}>
        <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',flexWrap:'wrap',gap:10}}>
          <div>
            <div style={{fontSize:13.5,fontWeight:800,color:'var(--text)',display:'flex',alignItems:'center',gap:8}}><Shield size={15} color="var(--orange)"/>Níveis Meta</div>
            {reservatorios?.length>1&&(
              <div style={{display:'flex',gap:4,flexWrap:'wrap',marginTop:6}}>
                {reservatorios.map((r,i)=>(
                  <button key={i} onClick={()=>setSelRes(i)}
                    style={{padding:'3px 10px',borderRadius:20,border:`1.5px solid ${selRes===i?'var(--orange)':'var(--border)'}`,background:selRes===i?'var(--orange-pale)':'none',color:selRes===i?'var(--orange-deep)':'var(--text-light)',fontSize:11,fontWeight:700,cursor:'pointer',transition:'all 0.15s'}}>
                    {r.nome||`Res. ${i+1}`}
                  </button>
                ))}
              </div>
            )}
            <div style={{fontSize:11,color:'var(--text-light)',marginTop:4}}>
              <strong>{reservatorio.nome}</strong> · Cod: <span style={{fontFamily:'JetBrains Mono'}}>{reservatorio.cod}</span>
            </div>
          </div>

          <div style={{display:'flex',gap:6,flexWrap:'wrap',alignItems:'center'}}>
            {isSessionOnly && (
              <span style={{fontSize:10,background:'var(--yellow-pale)',color:'var(--yellow)',borderRadius:20,padding:'2px 8px',fontWeight:700,border:'1px solid var(--yellow)'}}>
                Não salvo na base
              </span>
            )}
            <button className="sim-ghost" onClick={add}><Plus size={11}/> Faixa</button>
            {hasChanges && (
              <button className="sim-ghost" onClick={revert}
                style={{borderColor:'var(--red)',color:'var(--red)'}}>
                <RefreshCw size={11}/> Reverter
              </button>
            )}
            {hasChanges && (
              <button className="sim-ghost" onClick={saveSession}
                style={{borderColor:'var(--teal)',color:'var(--teal)'}}>
                <Save size={11}/> Aplicar na Sessão
              </button>
            )}
          </div>
        </div>
      </Card>

      <div style={{display:'flex',gap:8,padding:'9px 13px',background:'var(--blue-pale)',borderRadius:'var(--radius-sm)',alignItems:'flex-start'}}>
        <Info size={13} color="var(--blue)" style={{flexShrink:0,marginTop:1}}/>
        <div style={{fontSize:11,color:'var(--blue)',lineHeight:1.6}}>Os valores <strong>JAN…DEZ</strong> são o limite máximo de volume (% da capacidade) que ativa o nível nesse mês. <strong>Racionamento</strong> = (%) de redução na demanda. Clique em <strong>Aplicar na Sessão</strong> para usar nas simulações. Para persistir permanentemente edite o arquivo <strong>banco_site.db</strong>.</div>
      </div>

      {faixas && faixas.length > 0 && <NiveisMeta faixas={faixas}/>}

      {loading?<Card style={{padding:'36px',textAlign:'center'}}><RefreshCw size={26} color="var(--orange)" className="sim-spin" style={{marginBottom:9}}/><div style={{fontSize:11.5,color:'var(--text-light)'}}>Carregando…</div></Card>
      :faixas&&faixas.length>0?(
        <Card style={{overflow:'hidden'}}>
          <div style={{overflowX:'auto'}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:11}}>
              <thead><tr style={{background:'var(--bg)'}}>
                <th style={{padding:'7px 10px',textAlign:'left',fontSize:9.5,fontWeight:700,textTransform:'uppercase',letterSpacing:'0.05em',color:'var(--text-light)',borderBottom:'1.5px solid var(--border)',whiteSpace:'nowrap'}}>Nível</th>
                <th style={{padding:'7px 8px',textAlign:'center',fontSize:9.5,fontWeight:700,textTransform:'uppercase',letterSpacing:'0.05em',color:'var(--text-light)',borderBottom:'1.5px solid var(--border)',whiteSpace:'nowrap'}}>Rac.(%)</th>
                {MESES.map(m=><th key={m} style={{padding:'7px 5px',textAlign:'center',fontSize:9.5,fontWeight:700,textTransform:'uppercase',letterSpacing:'0.05em',color:'var(--text-light)',borderBottom:'1.5px solid var(--border)',whiteSpace:'nowrap'}}>{m}</th>)}
                <th style={{padding:'7px 8px',borderBottom:'1.5px solid var(--border)'}}/>
              </tr></thead>
              <tbody>
                {faixas.map((f,i)=>{
                  const c=FAIXAS_COR[f.Faixa]||{bg:'var(--border-light)',t:'var(--text-mid)'}
                  return(
                    <tr key={i} style={{background:i%2===0?'transparent':'rgba(236,220,200,0.12)'}}>
                      <td style={{padding:'6px 8px',borderBottom:'1px solid var(--border-light)'}}>
                        <input className="sim-plano-inp" value={f.Faixa} onChange={e=>set(i,'Faixa',e.target.value)} style={{textAlign:'left',fontFamily:'Sora',fontWeight:700,color:c.t,background:c.bg,borderRadius:5,padding:'3px 8px',width:'100%',border:'1.5px solid transparent'}} onFocus={e=>e.target.style.borderColor='var(--orange)'} onBlur={e=>e.target.style.borderColor='transparent'}/>
                      </td>
                      <td style={{padding:'6px 5px',borderBottom:'1px solid var(--border-light)'}}>
                        <input type="number" className="sim-plano-inp" min="0" max="100" step="1" value={f.Racionamento} onChange={e=>set(i,'Racionamento',e.target.value)}/>
                      </td>
                      {MESES.map(m=>(
                        <td key={m} style={{padding:'6px 3px',borderBottom:'1px solid var(--border-light)'}}>
                          <input type="number" className="sim-plano-inp" min="0" max="100" step="0.1" value={f[m]} onChange={e=>set(i,m,e.target.value)}/>
                        </td>
                      ))}
                      <td style={{padding:'6px 8px',borderBottom:'1px solid var(--border-light)',textAlign:'center'}}>
                        <button onClick={()=>del(i)} style={{background:'none',border:'none',cursor:'pointer',color:'var(--text-light)',padding:3,borderRadius:4,transition:'color 0.15s'}} onMouseEnter={e=>e.currentTarget.style.color='var(--red)'} onMouseLeave={e=>e.currentTarget.style.color='var(--text-light)'}><Trash2 size={12}/></button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
      ):(
        <Card style={{padding:'28px',textAlign:'center'}}>
          <Shield size={24} color="var(--text-light)" style={{opacity:0.35,marginBottom:10}}/>
          <div style={{fontSize:12,color:'var(--text-light)',lineHeight:1.6,marginBottom:12}}>
            Este reservatório ainda não possui níveis meta no banco de dados.
          </div>
          <div style={{display:'flex',gap:8,justifyContent:'center',flexWrap:'wrap'}}>
            {onOpenOtimizador && (
              <button className="sim-ghost" onClick={onOpenOtimizador} style={{borderColor:'var(--orange)',color:'var(--orange-deep)'}}>
                <Activity size={12}/> Abrir Otimizador
              </button>
            )}
            <button className="sim-ghost" onClick={add}>
              <Plus size={11}/> Criar Faixa Manual
            </button>
          </div>
        </Card>
      )}

      {msg&&<div style={{marginTop:9,padding:'7px 11px',borderRadius:'var(--radius-xs)',
        background:msg.type==='success'?'var(--teal-pale)':msg.type==='info'?'var(--blue-pale)':msg.type==='session'?'var(--yellow-pale)':'var(--red-pale)',
        color:msg.type==='success'?'var(--teal)':msg.type==='info'?'var(--blue)':msg.type==='session'?'var(--yellow)':'var(--red)',
        fontSize:11.5,fontWeight:600,lineHeight:1.5}}>
        {msg.type==='success'?'✓':msg.type==='session'?'⚡':msg.type==='info'?'ℹ':'✗'} {msg.text}
      </div>}
    </div>
  )
}

function nivelColor(idx, total) {
  if (total <= 1) return '#2a9d8f'
  const t = idx / (total - 1)
  const stops = [
    [42,157,143],
    [212,160,23],
    [224,123,42],
    [217,64,64],
  ]
  const seg  = (stops.length - 1) * t
  const lo   = Math.floor(seg)
  const hi   = Math.min(lo + 1, stops.length - 1)
  const frac = seg - lo
  const r = Math.round(stops[lo][0] + (stops[hi][0]-stops[lo][0]) * frac)
  const g = Math.round(stops[lo][1] + (stops[hi][1]-stops[lo][1]) * frac)
  const b = Math.round(stops[lo][2] + (stops[hi][2]-stops[lo][2]) * frac)
  return `rgb(${r},${g},${b})`
}

function NiveisMeta({ faixas }) {
  const nomeFaixaNormal = faixas.find(f => f.NomeFaixaNormal)?.NomeFaixaNormal || 'Normal'
  const faixasOrdenadas = [...faixas].sort((a, b) => {
    const ma = MESES.reduce((s,m) => s+(parseFloat(a[m])||0), 0)
    const mb = MESES.reduce((s,m) => s+(parseFloat(b[m])||0), 0)
    return mb - ma
  })

  // A faixa Normal ocupa sempre o restante entre o maior limite e 100%.
  const faixasRestritas = faixasOrdenadas.filter(f => (
    f._tipoFaixa === 'restrita' || String(f.Faixa || '').trim().toLowerCase() !== 'normal'
  ))
  const n = faixasRestritas.length

  const cores = faixasRestritas.map((f, i) => {
    if (f._cor) return f._cor
    const nome = String(f.Faixa || '').toLowerCase()
    if (nome.includes('alerta') || nome.includes('atenção') || nome.includes('atencao')) return '#d4a017'
    if (nome.includes('severa') || nome.includes('emergência') || nome.includes('emergencia') || nome.includes('crítico') || nome.includes('critico')) return '#d94040'
    if (nome.includes('seca')) return '#e07b2a'
    return nivelColor(i, n)
  })

  const bandasGrafico = [
    ...faixasRestritas.map((faixa, i) => ({
      faixa,
      chave: `faixa_${i + 1}`,
      nome: faixa.Faixa || `Faixa ${i + 1}`,
      cor: cores[i],
    })).reverse(),
    { chave: 'faixa_0', nome: nomeFaixaNormal, cor: '#2a9d8f' },
  ]

  const data = MESES.map(mes => {
    const linha = { mes }
    let limiteAnterior = 0
    bandasGrafico.slice(0, -1).forEach(banda => {
      const limite = parseFloat(banda.faixa[mes]) || 0
      linha[banda.chave] = Math.max(0, limite - limiteAnterior)
      limiteAnterior = limite
    })
    linha.faixa_0 = Math.max(0, 100 - limiteAnterior)
    return linha
  })
  const metaZoom = useBoxZoom(data, 'mes')

  const Tip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div style={{ background:'#fff', border:'1.5px solid var(--border)', borderRadius:10, padding:'9px 13px', boxShadow:'var(--shadow)', fontSize:11 }}>
        <div style={{ fontWeight:700, marginBottom:6, color:'var(--text)' }}>{label}</div>
        {faixasRestritas.map((f, i) => (
          <div key={i} style={{ display:'flex', gap:7, alignItems:'center', marginBottom:2 }}>
            <div style={{ width:7, height:7, borderRadius:'50%', background:cores[i] }}/>
            <span style={{ color:'var(--text-mid)' }}>{f.Faixa}:</span>
            <span style={{ fontWeight:600, fontFamily:'JetBrains Mono', color:'var(--text)' }}>≤ {parseFloat(f[label])||0}%</span>
          </div>
        ))}
      </div>
    )
  }

  return (
    <Card className="sim-fade" style={{ padding:'16px 18px' }}>
      <div style={{ marginBottom:14 }}>
        <div style={{ fontSize:13, fontWeight:800, color:'var(--text)' }}>Limites de Ativação por Mês</div>
        <div style={{ fontSize:11, color:'var(--text-light)', marginTop:2 }}>
          Bandas de volume: verde = zona segura · vermelho = nível crítico ativo
        </div>
      </div>
      <ZoomReset zoom={metaZoom}/>
      <div style={{ height:260 }}>
        <ResponsiveContainer>
          <AreaChart data={metaZoom.data} margin={{top:4,right:20,left:0,bottom:4}} {...metaZoom.props}>
            <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
            <XAxis dataKey="mes" tick={{fontSize:10,fill:'var(--text-light)'}}/>
            <YAxis domain={[0,100]} tick={{fontSize:10,fill:'var(--text-light)'}}
              label={{value:'% Cap.',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
            <Tooltip content={<Tip/>}/>
            {bandasGrafico.map(banda => (
              <Area key={banda.chave} type="linear" dataKey={banda.chave} stackId="meta" name={banda.nome} stroke={banda.cor} fill={banda.cor} fillOpacity={banda.chave === 'faixa_0' ? 0.45 : 0.52} dot={false} activeDot={false}/>
            ))}
            {metaZoom.area}
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div style={{ display:'flex', gap:8, flexWrap:'wrap', marginTop:12 }}>
        {faixasRestritas.map((f, i) => {
          const cor = cores[i]
          const rac = parseFloat(f.Racionamento) || 0
          const racTexto = Number.isInteger(rac) ? String(rac) : rac.toFixed(2)
          return (
            <span key={i} style={{ display:'inline-flex', alignItems:'center', gap:6, fontSize:10.5, borderRadius:20, padding:'3px 11px', fontWeight:600, background:`${cor}22`, color:cor, border:`1.5px solid ${cor}66` }}>
              <span style={{ width:8, height:8, borderRadius:'50%', background:cor, display:'inline-block', flexShrink:0 }}/>
              {f.Faixa}{rac > 0 ? ` — ${racTexto}% de Racionamento` : ' — Sem Racionamento'}
            </span>
          )
        })}
        <span style={{ display:'inline-flex', alignItems:'center', gap:6, fontSize:10.5, borderRadius:20, padding:'3px 11px', fontWeight:600, background:'#e8e0d422', color:'var(--text-light)', border:'1.5px solid #e8e0d466' }}>
          <span style={{ width:8, height:8, borderRadius:'50%', background:'#c8b8a0', display:'inline-block', flexShrink:0 }}/>
          {nomeFaixaNormal} — Sem restrição
        </span>
      </div>
    </Card>
  )
}

function ResSearch({ resList, value, onChange }) {
  const [query, setQuery] = useState(value || '')
  const [open,  setOpen]  = useState(false)
  const [ativo, setAtivo] = useState(-1)
  const ref = React.useRef(null)
  const listaId = React.useId()

  useEffect(() => { setQuery(value || '') }, [value])

  useEffect(() => {
    const handler = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const filtered = resList.filter(r =>
    r.CORPO.toLowerCase().includes(query.toLowerCase())
  ).slice(0, 50)

  const select = (nome) => {
    setQuery(nome)
    setOpen(false)
    onChange(nome)
  }

  const onKeyDown = e => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault()
      if (!open) { setOpen(true); return }
      const passo = e.key === 'ArrowDown' ? 1 : -1
      setAtivo(i => Math.max(0, Math.min(filtered.length - 1, i + passo)))
    } else if (e.key === 'Enter' && open && filtered[ativo]) {
      e.preventDefault()
      select(filtered[ativo].CORPO)
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  return (
    <div ref={ref} style={{position:'relative'}}>
      <input
        value={query}
        onChange={e=>{ setQuery(e.target.value); setOpen(true); setAtivo(-1); if(!e.target.value) onChange('') }}
        onFocus={()=>setOpen(true)}
        onKeyDown={onKeyDown}
        placeholder="Digite para buscar…"
        aria-label="Buscar reservatório"
        role="combobox" aria-expanded={open && filtered.length > 0} aria-autocomplete="list"
        aria-controls={listaId}
        aria-activedescendant={open && filtered[ativo] ? `${listaId}-${ativo}` : undefined}
        style={{width:'100%',padding:'7px 10px',border:'1.5px solid var(--border)',borderRadius:'var(--radius-xs)',background:'var(--card)',color:'var(--text)',fontSize:12.5,outline:'none',transition:'border-color 0.15s'}}
        onMouseEnter={e=>e.target.style.borderColor='var(--orange)'}
        onMouseLeave={e=>{ if(document.activeElement!==e.target) e.target.style.borderColor='var(--border)' }}
        onFocusCapture={e=>e.target.style.borderColor='var(--orange)'}
        onBlurCapture={e=>e.target.style.borderColor='var(--border)'}
      />
      {open && filtered.length > 0 && (
        <div id={listaId} role="listbox" aria-label="Reservatórios encontrados" style={{position:'absolute',top:'100%',left:0,right:0,background:'var(--card)',border:'1.5px solid var(--border)',borderRadius:'var(--radius-xs)',boxShadow:'var(--shadow)',zIndex:999,maxHeight:200,overflowY:'auto',marginTop:2}}>
          {filtered.map((r,i)=>(
            <div key={r.COD} id={`${listaId}-${i}`} role="option" aria-selected={i===ativo}
              onMouseDown={()=>select(r.CORPO)}
              style={{padding:'7px 11px',fontSize:12,cursor:'pointer',borderBottom:'1px solid var(--border-light)',transition:'background 0.1s',background:i===ativo?'var(--orange-pale)':'var(--card)'}}
              onMouseEnter={()=>setAtivo(i)}>
              <span style={{fontWeight:600,color:'var(--text)'}}>{r.CORPO}</span>
              <span style={{fontSize:10,color:'var(--text-light)',marginLeft:8,fontFamily:'Sora, sans-serif'}}>{r.COD}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ResCard({ res, index, resList, onChange, onRemove, modoLocked, modo, cenarioHidrossistema, erros = {} }) {
  const [open,setOpen]=useState(true)
  const isPgpsFq = cenarioHidrossistema==='pgps_fogareiro_quixeramobim_cenario_1'
  const showGatilho = modo !== 'Individual' && (isPgpsFq ? String(res.cod)==='16' : index===0)
  return (
    <div style={{background:'var(--bg)',border:'1.5px solid var(--border)',borderRadius:'var(--radius-sm)',marginBottom:6,overflow:'hidden'}}>
      <div role="button" tabIndex={0} aria-expanded={open} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setOpen(!open)}}} style={{display:'flex',alignItems:'center',justifyContent:'space-between',padding:'8px 10px',cursor:'pointer',borderBottom:open?'1.5px solid var(--border-light)':'none'}} onClick={()=>setOpen(!open)}>
        <div style={{display:'flex',alignItems:'center',gap:7}}>
          <div style={{width:19,height:19,borderRadius:'50%',background:'var(--orange-pale)',border:'1.5px solid var(--orange-light)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:9,fontWeight:800,color:'var(--orange-deep)',flexShrink:0}}>{index+1}</div>
          <span style={{fontSize:12,fontWeight:700,color:'var(--orange-deep)'}}>{res.nome||`Reservatório ${index+1}`}</span>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:4}}>
          {index>0&&<button aria-label={`Remover ${res.nome||`reservatório ${index+1}`}`} title="Remover reservatório" onClick={e=>{e.stopPropagation();onRemove(index)}} style={{background:'none',border:'none',cursor:'pointer',color:'var(--text-light)',padding:3,borderRadius:4}} onMouseEnter={e=>e.currentTarget.style.color='var(--red)'} onMouseLeave={e=>e.currentTarget.style.color='var(--text-light)'}><Trash2 size={11}/></button>}
          <ChevronDown size={12} color="var(--text-light)" style={{transform:open?'rotate(180deg)':'none',transition:'transform 0.2s'}}/>
        </div>
      </div>
      {open&&(
        <div style={{padding:'10px 10px 12px'}}>
          <div style={{marginBottom:8}}>
            <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Reservatório</div>
            <ResSearch resList={resList} value={res.nome} onChange={val=>{
              const s=resList.find(r=>r.CORPO===val)
              onChange(index,{nome:val,cod:s?.COD||'',capacidade:getCapacidadeHm3(s),est_evap:s?.['Est. Evap.']||'',volPct:50,vol_inicial:getCapacidadeHm3(s)*0.5})
            }}/>
            {res.capacidade>0&&<div style={{fontSize:9.5,color:'var(--text-light)',marginTop:2,fontFamily:'Sora, sans-serif'}}>Cap: {res.capacidade.toFixed(2)} hm³ · COD: {res.cod}</div>}
            <ErroCampo>{erros.nome}</ErroCampo>
          </div>
          <div style={{display:'grid',gridTemplateColumns:showGatilho?'1fr 1fr':'1fr 1fr',gap:6}}>
            <div>
              <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Vol. Inicial (%)</div>
              <FC type="number" min="0" max="100" step="1" aria-label={`Volume inicial de ${res.nome||'reservatório'} (%)`} invalid={Boolean(erros.volPct)} value={res.volPct??50} onChange={e=>{const p=parseFloat(e.target.value);const v=Number.isFinite(p)?p:0;onChange(index,{volPct:v,vol_inicial:(res.capacidade*Math.max(0,Math.min(100,v)))/100})}}/>
              <ErroCampo>{erros.volPct}</ErroCampo>
              {res.capacidade>0&&<div style={{fontSize:9,color:'var(--text-light)',marginTop:2,fontFamily:'Sora, sans-serif'}}>= {((res.capacidade*(res.volPct??50))/100).toFixed(2)} hm³</div>}
            </div>
            <div>
              <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Demanda (L/s)</div>
              <FC type="number" min="0" step="10" aria-label={`Demanda de ${res.nome||'reservatório'} (L/s)`} invalid={Boolean(erros.demanda)} value={m3sToLps(res.demanda1 ?? res.demanda ?? 0)} onChange={e=>{
                const bruto = parseFloat(e.target.value)
                const demanda1 = Number.isFinite(bruto) ? bruto / 1000 : 0
                onChange(index,{demanda1,demanda:demanda1})
              }}/>
              <ErroCampo>{erros.demanda}</ErroCampo>
            </div>
            {showGatilho&&(
              <div>
                <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Gatilho Transf. (%)</div>
                <FC type="number" min="0" max="100" step="1" aria-label="Gatilho de transferência (% da capacidade)" invalid={Boolean(erros.gatilho)} value={isPgpsFq?30:res.gatilho} disabled={isPgpsFq} onChange={e=>{const g=parseFloat(e.target.value);onChange(index,{gatilho:Number.isFinite(g)?g:0})}}/>
                <ErroCampo>{erros.gatilho}</ErroCampo>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// -----------------------------------------------------------------------------
// Indicadores de desempenho (Hashimoto et al., 1982) calculados pela API
// -----------------------------------------------------------------------------
const fmtNum = (v, casas = 1) => (v === null || v === undefined || Number.isNaN(Number(v))) ? '—' : Number(v).toLocaleString('pt-BR', { minimumFractionDigits: casas, maximumFractionDigits: casas })

const LINHAS_INDICADORES = [
  { chave: 'confiabilidade_percent', rotulo: 'Confiabilidade (%)', casas: 1, dica: 'Fração dos meses sem falha.' },
  { chave: 'resiliencia_percent', rotulo: 'Resiliência (%)', casas: 1, dica: 'Probabilidade de sair da falha no mês seguinte.' },
  { chave: 'vulnerabilidade_percent', rotulo: 'Vulnerabilidade (%)', casas: 1, dica: 'Média, entre os eventos de falha, do maior déficit relativo do evento.' },
  { chave: 'meses_falha', rotulo: 'Meses com falha', casas: 0 },
  { chave: 'eventos_falha', rotulo: 'Eventos de falha', casas: 0 },
  { chave: 'duracao_maxima_falha_meses', rotulo: 'Maior evento (meses)', casas: 0 },
  { chave: 'deficit_acumulado_hm3', rotulo: 'Déficit acumulado (hm³)', casas: 2 },
  { chave: 'atendimento_demanda_aplicada_percent', rotulo: 'Atendimento da demanda aplicada (%)', casas: 2 },
  { chave: 'atendimento_demanda_solicitada_percent', rotulo: 'Atendimento da demanda solicitada (%)', casas: 2 },
  { chave: 'meses_racionamento', rotulo: 'Meses com racionamento', casas: 0 },
]

function linhasSistema(sistema) {
  if (!sistema) return []
  const linhas = []
  if (sistema.modo === 'paralelo') {
    linhas.push(['Meses com falha da demanda conjunta', fmtNum(sistema.meses_falha_demanda_conjunta, 0)])
    linhas.push(['Atendimento da demanda conjunta (%)', fmtNum(sistema.atendimento_demanda_conjunta_percent, 2)])
    linhas.push(['Mudanças de unidade responsável', fmtNum(sistema.mudancas_de_responsavel, 0)])
  }
  if (sistema.modo === 'serie') {
    linhas.push(['Meses com transferência', fmtNum(sistema.meses_com_transferencia, 0)])
    linhas.push(['Acionamentos da transferência', fmtNum(sistema.acionamentos_transferencia, 0)])
    linhas.push(['Volume transferido (hm³)', fmtNum(sistema.volume_transferido_hm3, 2)])
  }
  if (sistema.modo !== 'individual') linhas.push(['Falhas sistêmicas (todas as unidades)', fmtNum(sistema.falhas_sistemicas, 0)])
  return linhas
}

function IndicadoresDesempenho({ resultados, sistema }) {
  if (!resultados?.length || !resultados[0].indicadores) return null
  const cel = { padding:'6px 10px', borderBottom:'1px solid var(--border-light)', textAlign:'right', fontFamily:'JetBrains Mono', fontSize:11.5 }
  const sis = linhasSistema(sistema)
  return (
    <Card className="sim-fade" style={{ padding:'16px 20px' }}>
      <div style={{ fontSize:13, fontWeight:800, color:'var(--text)', marginBottom:4, display:'flex', alignItems:'center', gap:8 }}>
        <Shield size={15} color="var(--orange)"/> Indicadores de Desempenho
      </div>
      <div style={{ fontSize:10.5, color:'var(--text-light)', marginBottom:10 }}>Confiabilidade, resiliência e vulnerabilidade segundo Hashimoto, Stedinger e Loucks (1982).</div>
      <div style={{ overflowX:'auto' }}>
        <table style={{ width:'100%', borderCollapse:'collapse', fontSize:11.5 }}>
          <thead>
            <tr>
              <th scope="col" style={{ ...cel, textAlign:'left', fontFamily:'Sora', color:'var(--text-light)', fontSize:10.5 }}>Indicador</th>
              {resultados.map((r,i)=><th scope="col" key={i} style={{ ...cel, fontFamily:'Sora', color:COLORS[i%4].stroke, fontSize:10.5 }}>{r.reservatorio}</th>)}
            </tr>
          </thead>
          <tbody>
            {LINHAS_INDICADORES.map(l=>(
              <tr key={l.chave} className="sim-tr">
                <th scope="row" title={l.dica} style={{ ...cel, textAlign:'left', fontFamily:'Sora', fontWeight:600, color:'var(--text-mid)' }}>{l.rotulo}</th>
                {resultados.map((r,i)=><td key={i} style={cel}>{fmtNum(r.indicadores?.[l.chave], l.casas)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sis.length>0&&(
        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(190px,1fr))', gap:8, marginTop:12 }}>
          {sis.map(([rotulo, valor])=>(
            <div key={rotulo} style={{ background:'var(--bg)', border:'1.5px solid var(--border)', borderRadius:'var(--radius-sm)', padding:'9px 12px' }}>
              <div style={{ fontSize:9.5, fontWeight:700, color:'var(--text-light)', textTransform:'uppercase', letterSpacing:'0.05em' }}>{rotulo}</div>
              <div style={{ fontSize:18, fontWeight:800, color:'var(--text)', fontFamily:'JetBrains Mono', marginTop:3 }}>{valor}</div>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

// -----------------------------------------------------------------------------
// Comparação de cenários: cenário fixado (A) × simulação atual (B)
// -----------------------------------------------------------------------------
function descreverCenario(meta) {
  if (!meta) return ''
  const partes = [meta.modo]
  partes.push(meta.usarNiveisMeta ? 'com níveis meta' : 'sem níveis meta')
  if (meta.cenarioHidrologicoNome) partes.push(meta.cenarioHidrologicoNome)
  if (meta.periodo) partes.push(meta.periodo)
  if (meta.histerese > 0) partes.push(`histerese ${meta.histerese} p.p.`)
  if (meta.demandas) partes.push(`demandas ${meta.demandas}`)
  return partes.join(' · ')
}

function ComparacaoCenarios({ cenarioA, resultados, simMeta }) {
  const nomesA = cenarioA.resultados.map(r => r.reservatorio)
  const comuns = resultados.map(r => r.reservatorio).filter(n => nomesA.includes(n))
  const [sel, setSel] = useState(comuns[0] || null)
  if (!comuns.length) {
    return (
      <Card style={{ padding:'18px 20px', fontSize:12, color:'var(--text-mid)' }}>
        Os dois cenários não têm reservatórios em comum. Simule os mesmos reservatórios para comparar.
      </Card>
    )
  }
  const nomeSel = comuns.includes(sel) ? sel : comuns[0]
  const rA = cenarioA.resultados.find(r => r.reservatorio === nomeSel)
  const rB = resultados.find(r => r.reservatorio === nomeSel)
  const capA = cenarioA.simMeta?.params?.[cenarioA.resultados.indexOf(rA)]?.capacidade || 1
  const capB = simMeta?.params?.[resultados.indexOf(rB)]?.capacidade || 1
  const porData = new Map()
  rA.dados.forEach(d => porData.set(d.Data, { data: d.Data, 'Cenário A': +((parseFloat(d['Armazenamento Final'])||0) / capA * 100).toFixed(1) }))
  rB.dados.forEach(d => {
    const p = porData.get(d.Data) || { data: d.Data }
    p['Cenário B (atual)'] = +((parseFloat(d['Armazenamento Final'])||0) / capB * 100).toFixed(1)
    porData.set(d.Data, p)
  })
  const serie = [...porData.values()].sort((a,b)=>a.data.localeCompare(b.data))
  const iv = Math.max(0, Math.floor(serie.length/12)-1)

  const volMin = (r, cap) => Math.min(...r.dados.map(d => parseFloat(d['Armazenamento Final'])||0)) / cap * 100
  const soma = (r, chave) => r.dados.reduce((acc, d) => acc + (parseFloat(d[chave])||0), 0)
  const linhas = [
    ...LINHAS_INDICADORES.map(l => ({ rotulo: l.rotulo, a: rA.indicadores?.[l.chave], b: rB.indicadores?.[l.chave], casas: l.casas, melhorMaior: !['vulnerabilidade_percent','meses_falha','eventos_falha','duracao_maxima_falha_meses','deficit_acumulado_hm3','meses_racionamento'].includes(l.chave) })),
    { rotulo: 'Volume mínimo (% cap.)', a: volMin(rA, capA), b: volMin(rB, capB), casas: 1, melhorMaior: true },
    { rotulo: 'Vertimento total (hm³)', a: soma(rA, 'Vertimento (hm³)'), b: soma(rB, 'Vertimento (hm³)'), casas: 1 },
    { rotulo: 'Evaporação total (hm³)', a: soma(rA, 'Evaporação (hm³)'), b: soma(rB, 'Evaporação (hm³)'), casas: 1 },
  ]
  const cel = { padding:'6px 10px', borderBottom:'1px solid var(--border-light)', textAlign:'right', fontFamily:'JetBrains Mono', fontSize:11.5 }
  const corDif = (l) => {
    const d = (Number(l.b) || 0) - (Number(l.a) || 0)
    if (Math.abs(d) < 1e-9 || l.melhorMaior === undefined) return 'var(--text-mid)'
    return (d > 0) === l.melhorMaior ? 'var(--teal)' : 'var(--red)'
  }
  const sisA = linhasSistema(cenarioA.simMeta?.indicadoresSistema)
  const sisB = linhasSistema(simMeta?.indicadoresSistema)

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
      <Card className="sim-fade" style={{ padding:'16px 20px' }}>
        <div style={{ fontSize:13, fontWeight:800, color:'var(--text)', marginBottom:8, display:'flex', alignItems:'center', gap:8 }}>
          <ArrowLeftRight size={15} color="var(--orange)"/> Comparação de Cenários
        </div>
        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(240px,1fr))', gap:8, marginBottom:12 }}>
          <div style={{ background:'var(--blue-pale)', borderRadius:'var(--radius-sm)', padding:'8px 12px', fontSize:11, color:'var(--text-mid)' }}>
            <strong style={{ color:'var(--blue)' }}>Cenário A (fixado)</strong><br/>{descreverCenario(cenarioA.simMeta)}
          </div>
          <div style={{ background:'var(--orange-pale)', borderRadius:'var(--radius-sm)', padding:'8px 12px', fontSize:11, color:'var(--text-mid)' }}>
            <strong style={{ color:'var(--orange-deep)' }}>Cenário B (atual)</strong><br/>{descreverCenario(simMeta)}
          </div>
        </div>
        {comuns.length>1&&(
          <div style={{ display:'flex', gap:4, flexWrap:'wrap', marginBottom:10 }}>
            {comuns.map(n=>(
              <button key={n} className={`sim-tab ${n===nomeSel?'on':'off'}`} onClick={()=>setSel(n)}>{n}</button>
            ))}
          </div>
        )}
        <div style={{ overflowX:'auto' }}>
          <table style={{ width:'100%', borderCollapse:'collapse', fontSize:11.5 }}>
            <thead>
              <tr>
                <th scope="col" style={{ ...cel, textAlign:'left', fontFamily:'Sora', color:'var(--text-light)', fontSize:10.5 }}>{nomeSel}</th>
                <th scope="col" style={{ ...cel, fontFamily:'Sora', color:'var(--blue)', fontSize:10.5 }}>Cenário A</th>
                <th scope="col" style={{ ...cel, fontFamily:'Sora', color:'var(--orange-deep)', fontSize:10.5 }}>Cenário B</th>
                <th scope="col" style={{ ...cel, fontFamily:'Sora', color:'var(--text-light)', fontSize:10.5 }}>Diferença (B − A)</th>
              </tr>
            </thead>
            <tbody>
              {linhas.map(l=>(
                <tr key={l.rotulo} className="sim-tr">
                  <th scope="row" style={{ ...cel, textAlign:'left', fontFamily:'Sora', fontWeight:600, color:'var(--text-mid)' }}>{l.rotulo}</th>
                  <td style={cel}>{fmtNum(l.a, l.casas)}</td>
                  <td style={cel}>{fmtNum(l.b, l.casas)}</td>
                  <td style={{ ...cel, color:corDif(l), fontWeight:700 }}>{(Number(l.b)||0)-(Number(l.a)||0) > 0 ? '+' : ''}{fmtNum((Number(l.b)||0)-(Number(l.a)||0), l.casas)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {(sisA.length>0||sisB.length>0)&&(
          <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(240px,1fr))', gap:8, marginTop:12, fontSize:11 }}>
            {[['Sistema – cenário A', sisA, 'var(--blue)'], ['Sistema – cenário B', sisB, 'var(--orange-deep)']].map(([titulo, itens, cor])=>(
              <div key={titulo} style={{ border:'1.5px solid var(--border)', borderRadius:'var(--radius-sm)', padding:'8px 12px' }}>
                <div style={{ fontWeight:800, color:cor, marginBottom:4 }}>{titulo}</div>
                {itens.length ? itens.map(([r,v])=><div key={r} style={{ display:'flex', justifyContent:'space-between', gap:8, color:'var(--text-mid)' }}><span>{r}</span><strong style={{ fontFamily:'JetBrains Mono' }}>{v}</strong></div>) : <span style={{ color:'var(--text-light)' }}>Modo individual.</span>}
              </div>
            ))}
          </div>
        )}
      </Card>
      <ChartCard title={`Volume armazenado – ${nomeSel}`} subtitle="% da capacidade nos dois cenários">
        <div style={{ height:250 }}>
          <ResponsiveContainer>
            <LineChart data={serie} margin={{top:4,right:20,left:0,bottom:0}}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)"/>
              <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
              <YAxis domain={[0,100]} tick={{fontSize:10,fill:'var(--text-light)'}}/>
              <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
              <Line type="monotone" dataKey="Cenário A" stroke="#264fa3" strokeWidth={1.8} dot={false}/>
              <Line type="monotone" dataKey="Cenário B (atual)" stroke="#e07b2a" strokeWidth={1.8} dot={false}/>
            </LineChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>
    </div>
  )
}

const ITEM_VAZIO = {nome:'',cod:'',capacidade:0,est_evap:'',volPct:50,vol_inicial:0,demanda1:0.5,demanda:0.5,gatilho:30}
const TIPO_ARQUIVO_CONFIG = 'ssd-reservatorios-configuracao'

function ConfigPanel({ resList, presets, onSimulate, loading, onResChange, onReset, onPresetApply, appliedCurvas, obterExtrasConfig, onAbrirConfig }) {
  const [items,setItems]=useState([{nome:'',cod:'',capacidade:0,est_evap:'',volPct:50,vol_inicial:0,demanda1:0,demanda:0,gatilho:10}])
  const [modo,setModo]=useState('Individual')
  const [modoLocked,setModoLocked]=useState(false)
  const [vazaoConj,setVazaoConj]=useState(0)
  const [atendimentoTransferencia,setAtendimentoTransferencia]=useState(100)
  const [histerese,setHisterese]=useState(0)
  const [cenarioHidrologico,setCenarioHidrologico]=useState('historico')
  const [fatorAfluencia,setFatorAfluencia]=useState(100)
  const [secaAnoIni,setSecaAnoIni]=useState(2012)
  const [secaAnoFim,setSecaAnoFim]=useState(2017)
  const [semente,setSemente]=useState(1)
  const [mesIni,setMesIni]=useState('JAN'),[anoIni,setAnoIni]=useState(1911)
  const [mesFim,setMesFim]=useState('DEZ'),[anoFim,setAnoFim]=useState(2017)
  const [presetSel,setPresetSel]=useState('')
  const [cenarioHidrossistema,setCenarioHidrossistema]=useState(null)
  const [tentouEnviar,setTentouEnviar]=useState(false)
  const [avisoArquivo,setAvisoArquivo]=useState(null)

  useEffect(() => {
    if (!appliedCurvas?.reservatorio || !resList.length) return
    const f = resList.find(r => r.CORPO === appliedCurvas.reservatorio || String(r.COD) === String(appliedCurvas.reservatorio))
    const demanda1 = Math.max(0, Number(appliedCurvas.scenario?.durb || 0) + Number(appliedCurvas.scenario?.dsupl || 0))
    const item = {
      nome: f?.CORPO || appliedCurvas.reservatorio,
      cod: f?.COD || appliedCurvas.reservatorio,
      capacidade: getCapacidadeHm3(f),
      est_evap: f?.['Est. Evap.'] || '',
      volPct: 50,
      vol_inicial: getCapacidadeHm3(f) * 0.5,
      demanda1,
      demanda: demanda1,
      gatilho: 10,
    }
    setPresetSel('')
    setCenarioHidrossistema(null)
    setModo('Individual')
    setModoLocked(false)
    setItems([item])
    onResChange&&onResChange([item])
    onReset&&onReset({ keepCurvas: true })
  }, [appliedCurvas?.id, resList])

  const validacao = useMemo(() => validarConfiguracao({
    items, modo, mesIni, anoIni, mesFim, anoFim, cenarioHidrologico, fatorAfluencia, secaAnoIni, secaAnoFim, histerese,
  }), [items, modo, mesIni, anoIni, mesFim, anoFim, cenarioHidrologico, fatorAfluencia, secaAnoIni, secaAnoFim, histerese])
  // os erros só aparecem depois da primeira tentativa de envio ou quando o campo já foi preenchido
  const eg = validacao.geral

  const atualizarItems=n=>{setItems(n);onResChange&&onResChange(n)}
  const change=(idx,patch)=>atualizarItems(items.map((it,i)=>i===idx?{...it,...patch}:it))

  const applyPreset=(nome)=>{
    const p=presets.find(x=>x.nome===nome)
    if(!p) return
    setModo(p.modo); setModoLocked(true)
    const ni=p.reservatorios.map(cod=>{
      const f=resList.find(r=>r.COD===cod||r.CORPO===cod)
      const defaults=p.defaults?.[String(cod)]||{}
      const capacidade=getCapacidadeHm3(f)
      const volPct=defaults.vol_inicial_percent??50
      const demanda1=lpsToM3s(defaults.demanda_lps??0)
      return {nome:f?.CORPO||cod,cod:f?.COD||cod,capacidade,est_evap:f?.['Est. Evap.']||'',volPct,vol_inicial:capacidade*volPct/100,demanda1,demanda:demanda1,gatilho:defaults.gatilho_percent??10}
    })
    setCenarioHidrossistema(p.cenario_hidrossistema||null)
    setVazaoConj(p.vazao_transferencia_lps??0)
    setAtendimentoTransferencia(p.atendimento_transferencia_percent??100)
    if(p.periodo){
      setMesIni(p.periodo.mes_inicial);setAnoIni(p.periodo.ano_inicial)
      setMesFim(p.periodo.mes_final);setAnoFim(p.periodo.ano_final)
    }
    setItems(ni)
    onResChange&&onResChange(ni)
    onReset&&onReset()
    onPresetApply&&onPresetApply(p)
  }

  const clearPreset=()=>{
    setPresetSel('')
    setCenarioHidrossistema(null)
    setVazaoConj(0)
    setAtendimentoTransferencia(100)
    setModoLocked(false)
    const empty = [{...ITEM_VAZIO}]
    setItems(empty)
    onResChange&&onResChange(empty)
    onReset&&onReset()
  }

  const submit=()=>{
    setTentouEnviar(true)
    if (validacao.temErro) return
    onSimulate({
      reservatorios:items.map(it=>({nome:String(it.nome||''),cod:String(it.cod||''),capacidade:parseFloat(it.capacidade)||0,est_evap:String(it.est_evap??''),vol_inicial:parseFloat(it.vol_inicial)||0,demanda:parseFloat(it.demanda1 ?? it.demanda)||0,demanda1:parseFloat(it.demanda1 ?? it.demanda)||0,gatilho:parseFloat(it.gatilho)||0})),
      modo:String(modo),vazao_conjunta:modo==='Individual'?0:lpsToM3s(vazaoConj),
      atendimento_transferencia:modo==='Série'?Math.max(0,Math.min(100,parseFloat(atendimentoTransferencia)||0)):100,
      histerese_transferencia:modo==='Individual'?0:Math.max(0,parseFloat(histerese)||0),
      cenario_hidrologico:String(cenarioHidrologico),
      fator_afluencia_percent:parseFloat(fatorAfluencia)||0,
      seca_ano_inicial:cenarioHidrologico==='seca_repetida'?parseInt(secaAnoIni):null,
      seca_ano_final:cenarioHidrologico==='seca_repetida'?parseInt(secaAnoFim):null,
      semente:parseInt(semente)||1,
      cenario_hidrossistema:cenarioHidrossistema,
      mes_inicial:String(mesIni),ano_inicial:parseInt(anoIni),
      mes_final:String(mesFim),ano_final:parseInt(anoFim),
    })
  }

  // ---- salvar e abrir configuração (arquivo JSON) ----
  const salvarConfiguracao=()=>{
    const config = {
      tipo: TIPO_ARQUIVO_CONFIG, versao: 1, salvo_em: new Date().toISOString(),
      items, modo, modoLocked, presetSel, cenarioHidrossistema, vazaoConj, atendimentoTransferencia, histerese,
      cenarioHidrologico, fatorAfluencia, secaAnoIni, secaAnoFim, semente, mesIni, anoIni, mesFim, anoFim,
      ...(obterExtrasConfig ? obterExtrasConfig() : {}),
    }
    const nome = String(items[0]?.nome || 'simulacao').normalize('NFD').replace(/[̀-ͯ]/g,'').replace(/[^\w-]+/g,'_')
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' })
    const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: `configuracao_${nome}.json` })
    document.body.appendChild(a); a.click(); a.remove()
    setAvisoArquivo({ tipo:'ok', texto:'Configuração salva.' })
  }

  const abrirConfiguracao=async(evento)=>{
    const arquivo = evento.target.files?.[0]
    evento.target.value = ''
    if (!arquivo) return
    try {
      const c = JSON.parse(await arquivo.text())
      if (c.tipo !== TIPO_ARQUIVO_CONFIG || !Array.isArray(c.items) || !c.items.length) throw new Error('o arquivo não é uma configuração do simulador.')
      const itensValidos = c.items.map(it => ({ ...ITEM_VAZIO, ...it }))
      setItems(itensValidos); onResChange&&onResChange(itensValidos)
      setModo(c.modo || 'Individual'); setModoLocked(Boolean(c.modoLocked))
      setPresetSel(c.presetSel || ''); setCenarioHidrossistema(c.cenarioHidrossistema || null)
      setVazaoConj(c.vazaoConj ?? 0); setAtendimentoTransferencia(c.atendimentoTransferencia ?? 100); setHisterese(c.histerese ?? 0)
      setCenarioHidrologico(c.cenarioHidrologico || 'historico'); setFatorAfluencia(c.fatorAfluencia ?? 100)
      setSecaAnoIni(c.secaAnoIni ?? 2012); setSecaAnoFim(c.secaAnoFim ?? 2017); setSemente(c.semente ?? 1)
      setMesIni(c.mesIni || 'JAN'); setAnoIni(c.anoIni ?? 1911); setMesFim(c.mesFim || 'DEZ'); setAnoFim(c.anoFim ?? 2017)
      onAbrirConfig && onAbrirConfig(c)
      setAvisoArquivo({ tipo:'ok', texto:`Configuração aberta: ${arquivo.name}` })
    } catch (erro) {
      setAvisoArquivo({ tipo:'erro', texto:`Não foi possível abrir a configuração: ${erro.message}` })
    }
  }

  const rotuloCampo = {fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600,textTransform:'uppercase',letterSpacing:'0.05em'}
  const bloqueado = loading || validacao.temErro && tentouEnviar

  return (
    <Card className="sim-config-card" style={{padding:'18px 14px'}}>
      <div style={{display:'flex',alignItems:'flex-start',justifyContent:'space-between',gap:8}}>
        <div>
          <div style={{fontSize:14.5,fontWeight:800,color:'var(--text)',marginBottom:2}}>Configuração</div>
          <div style={{fontSize:11,color:'var(--text-light)',marginBottom:10}}>Cenário: <strong style={{color:'var(--orange-deep)'}}>{items[0]?.nome||'Nenhum selecionado'}</strong></div>
        </div>
      </div>
      <div style={{display:'flex',gap:5,flexWrap:'wrap',marginBottom:4}}>
        <button type="button" className="sim-ghost" onClick={salvarConfiguracao} title="Salvar a configuração atual em um arquivo JSON"><Save size={12}/> Salvar</button>
        <label className="sim-ghost" style={{cursor:'pointer'}} title="Abrir uma configuração salva anteriormente">
          <FileSpreadsheet size={12}/> Abrir
          <input type="file" accept="application/json,.json" onChange={abrirConfiguracao} style={{display:'none'}} aria-label="Abrir arquivo de configuração"/>
        </label>
      </div>
      {avisoArquivo&&<div role="status" style={{fontSize:10.5,marginBottom:4,color:avisoArquivo.tipo==='erro'?'var(--red)':'var(--teal)',fontWeight:600}}>{avisoArquivo.texto}</div>}

      {presets.length>0&&(
        <>
          <Label icon={Zap}>Hidrossistema</Label>
          <div style={{display:'flex',gap:5}}>
            <FC as="select" aria-label="Hidrossistema pré-configurado" style={{flex:1}} value={presetSel} onChange={e=>{setPresetSel(e.target.value);applyPreset(e.target.value)}}>
              <option value="">Configuração manual…</option>
              {presets.map(p=><option key={p.nome} value={p.nome}>{p.nome}</option>)}
            </FC>
            {presetSel&&<button onClick={clearPreset} aria-label="Limpar hidrossistema" style={{background:'none',border:'1.5px solid var(--border)',borderRadius:'var(--radius-xs)',padding:'0 8px',cursor:'pointer',color:'var(--text-light)',fontSize:14,transition:'all 0.15s'}} title="Limpar preset" onMouseEnter={e=>e.currentTarget.style.color='var(--red)'} onMouseLeave={e=>e.currentTarget.style.color='var(--text-light)'}><X size={13}/></button>}
          </div>
          {presetSel&&<div style={{marginTop:5,fontSize:10.5,color:'var(--blue)',background:'var(--blue-pale)',borderRadius:5,padding:'3px 9px',display:'inline-flex',alignItems:'center',gap:5}}><Info size={11}/> Modo de operação: <strong>{modo}</strong></div>}
        </>
      )}

      <Label icon={Database}>Reservatórios</Label>
      {items.map((res,i)=>(
        <ResCard key={i} res={res} index={i} resList={resList} onChange={change} onRemove={idx=>atualizarItems(items.filter((_,j)=>j!==idx))} modoLocked={modoLocked} modo={modo} cenarioHidrossistema={cenarioHidrossistema}
          erros={Object.fromEntries(Object.entries(validacao.itens[i]||{}).filter(([k])=>tentouEnviar || k!=='nome'))}/>
      ))}

      <button onClick={()=>atualizarItems([...items,{...ITEM_VAZIO}])}
        style={{width:'100%',padding:'6px',background:'none',border:'1.5px dashed var(--border)',borderRadius:'var(--radius-sm)',color:'var(--text-light)',fontSize:11,cursor:'pointer',marginBottom:2,transition:'all 0.15s'}}
        onMouseEnter={e=>{e.currentTarget.style.borderColor='var(--orange)';e.currentTarget.style.color='var(--orange)';e.currentTarget.style.background='var(--orange-pale)'}}
        onMouseLeave={e=>{e.currentTarget.style.borderColor='var(--border)';e.currentTarget.style.color='var(--text-light)';e.currentTarget.style.background='none'}}>
        <Plus size={10} style={{marginRight:4}}/> Adicionar Reservatório
      </button>

      <Label icon={Settings2}>Modo de Operação</Label>
      <div role="radiogroup" aria-label="Modo de operação" style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:5,marginBottom:4}}>
        {['Individual','Série','Paralelo'].map(m=>(
          <button key={m} role="radio" aria-checked={modo===m} aria-disabled={modoLocked&&modo!==m} onClick={()=>!modoLocked&&setModo(m)}
            style={{padding:'7px 4px',border:`1.5px solid ${modo===m?'var(--orange)':'var(--border)'}`,borderRadius:'var(--radius-xs)',background:modo===m?'var(--orange-pale)':'none',color:modo===m?'var(--orange-deep)':'var(--text-light)',fontSize:11,fontWeight:700,cursor:modoLocked?'not-allowed':'pointer',transition:'all 0.15s',opacity:modoLocked&&modo!==m?0.4:1}}>
            {m}
          </button>
        ))}
      </div>

      {modo!=='Individual'&&(
        <div style={{marginTop:9}}>
          <div style={rotuloCampo}>{modo==='Série'?'Vazão de Transferência (L/s)':'Vazão Conjunta (L/s)'}</div>
          <FC type="number" min="0" step="10" aria-label={modo==='Série'?'Vazão de transferência (L/s)':'Vazão conjunta (L/s)'} value={vazaoConj} onChange={e=>setVazaoConj(Math.max(0, parseFloat(e.target.value)||0))}/>
          {modo==='Série'&&<div style={{marginTop:7}}>
            <div style={rotuloCampo}>Atendimento da Transferência (%)</div>
            <FC type="number" min="0" max="100" step="1" aria-label="Atendimento da transferência (%)" value={atendimentoTransferencia} onChange={e=>setAtendimentoTransferencia(Math.max(0,Math.min(100,parseFloat(e.target.value)||0)))}/>
          </div>}
          <div style={{marginTop:7}}>
            <div style={rotuloCampo} title="Pontos percentuais da capacidade acima do gatilho necessários para encerrar a ação">Histerese do gatilho (p.p.)</div>
            <FC type="number" min="0" max="100" step="1" aria-label="Histerese do gatilho em pontos percentuais" invalid={Boolean(eg.histerese)} value={histerese} onChange={e=>setHisterese(e.target.value)}/>
            <div style={{fontSize:9.5,color:'var(--text-light)',marginTop:2,lineHeight:1.35}}>
              {parseFloat(histerese)>0
                ? `${modo==='Série'?'A transferência':'A demanda conjunta'} só é encerrada quando o volume passa de gatilho + ${parseFloat(histerese)} p.p.`
                : 'Zero: a regra liga e desliga no próprio gatilho.'}
            </div>
            <ErroCampo>{eg.histerese}</ErroCampo>
          </div>
        </div>
      )}

      <Label icon={Activity}>Cenário Hidrológico</Label>
      <FC as="select" aria-label="Cenário hidrológico" value={cenarioHidrologico} onChange={e=>setCenarioHidrologico(e.target.value)}>
        {CENARIOS_HIDROLOGICOS.map(c=><option key={c.id} value={c.id}>{c.label}</option>)}
      </FC>
      {cenarioHidrologico==='fator_personalizado'&&(
        <div style={{marginTop:7}}>
          <div style={rotuloCampo}>Afluência (% da histórica)</div>
          <FC type="number" min="0" max="500" step="5" aria-label="Afluência em percentual da histórica" invalid={Boolean(eg.fator)} value={fatorAfluencia} onChange={e=>setFatorAfluencia(e.target.value)}/>
          <ErroCampo>{eg.fator}</ErroCampo>
        </div>
      )}
      {cenarioHidrologico==='seca_repetida'&&(
        <div style={{marginTop:7}}>
          <div style={rotuloCampo}>Anos da seca a repetir</div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:4}}>
            <FC type="number" min={ANO_MIN_SERIE} max={ANO_MAX_SERIE} aria-label="Ano inicial da seca" invalid={Boolean(eg.seca)} value={secaAnoIni} onChange={e=>setSecaAnoIni(e.target.value)}/>
            <FC type="number" min={ANO_MIN_SERIE} max={ANO_MAX_SERIE} aria-label="Ano final da seca" invalid={Boolean(eg.seca)} value={secaAnoFim} onChange={e=>setSecaAnoFim(e.target.value)}/>
          </div>
          <div style={{fontSize:9.5,color:'var(--text-light)',marginTop:2,lineHeight:1.35}}>Os anos escolhidos são repetidos em sequência ao longo de todo o período simulado.</div>
          <ErroCampo>{eg.seca}</ErroCampo>
        </div>
      )}
      {cenarioHidrologico==='reamostragem_anual'&&(
        <div style={{marginTop:7}}>
          <div style={rotuloCampo}>Semente do sorteio</div>
          <FC type="number" min="0" step="1" aria-label="Semente do sorteio" value={semente} onChange={e=>setSemente(e.target.value)}/>
          <div style={{fontSize:9.5,color:'var(--text-light)',marginTop:2,lineHeight:1.35}}>Cada ano simulado recebe um ano histórico sorteado. A mesma semente reproduz o mesmo sorteio.</div>
        </div>
      )}

      <Label icon={Calendar}>Período</Label>
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:6}}>
        <div>
          <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Início</div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:4}}>
            <FC as="select" aria-label="Mês inicial" value={mesIni} onChange={e=>setMesIni(e.target.value)} style={{fontSize:11}}>{MESES.map(m=><option key={m}>{m}</option>)}</FC>
            <FC type="number" aria-label="Ano inicial" invalid={Boolean(eg.anoIni||eg.periodo)} value={anoIni} onChange={e=>setAnoIni(e.target.value)} min={ANO_MIN_SERIE} max={ANO_MAX_SERIE} style={{fontSize:11}} placeholder="Ano"/>
          </div>
          <ErroCampo>{eg.anoIni}</ErroCampo>
        </div>
        <div>
          <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Fim</div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:4}}>
            <FC as="select" aria-label="Mês final" value={mesFim} onChange={e=>setMesFim(e.target.value)} style={{fontSize:11}}>{MESES.map(m=><option key={m}>{m}</option>)}</FC>
            <FC type="number" aria-label="Ano final" invalid={Boolean(eg.anoFim||eg.periodo)} value={anoFim} onChange={e=>setAnoFim(e.target.value)} min={ANO_MIN_SERIE} max={ANO_MAX_SERIE} style={{fontSize:11}} placeholder="Ano"/>
          </div>
          <ErroCampo>{eg.anoFim}</ErroCampo>
        </div>
      </div>
      <ErroCampo>{eg.periodo}</ErroCampo>

      {tentouEnviar&&validacao.temErro&&(
        <div role="alert" style={{marginTop:12,fontSize:11,color:'var(--red)',background:'var(--red-pale)',borderRadius:'var(--radius-xs)',padding:'7px 10px',fontWeight:600}}>
          Corrija os campos destacados antes de simular.
        </div>
      )}

      <button onClick={submit} disabled={bloqueado} aria-busy={loading}
        style={{width:'100%',marginTop:16,padding:12,background:bloqueado?'var(--border)':'linear-gradient(135deg,var(--orange),var(--orange-deep))',border:'none',borderRadius:'var(--radius-sm)',color:bloqueado?'var(--text-light)':'#fff',fontSize:13,fontWeight:800,cursor:bloqueado?'not-allowed':'pointer',boxShadow:loading?'none':'0 4px 18px var(--orange-glow)',transition:'all 0.2s',letterSpacing:'0.02em'}}
        onMouseEnter={e=>{if(!loading)e.currentTarget.style.transform='translateY(-1px)'}}
        onMouseLeave={e=>e.currentTarget.style.transform='none'}>
        {loading?'Simulando…':'▶ Gerar Simulação'}
      </button>
    </Card>
  )
}

function X({ size=14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
    </svg>
  )
}

// =============================================================================
// CORRIGIDO: componente raiz agora gerencia planoSecasSession e injeta no payload
// =============================================================================
export default function SimuladorHidrico({ apiUrl, curvasOtimizadas, darkMode = false, onOpenOtimizador }) {
  const api = useMemo(() => makeApi(apiUrl), [apiUrl])
  const [resList,setResList]=useState([])
  const [presets,setPresets]=useState([])
  const [resultados,setResultados]=useState(null)
  const [simMeta,setSimMeta]=useState(null)
  const [loading,setLoading]=useState(false)
  const [error,setError]=useState(null)
  const [apiError,setApiError]=useState(null)
  const [activeTab,setActiveTab]=useState('padrao')
  const [resultTab,setResultTab]=useState('graficos')
  const [activeRes,setActiveRes]=useState([])
  const [cenarioA,setCenarioA]=useState(null)        // cenário fixado para comparação
  const [segundosSimulando,setSegundosSimulando]=useState(0)

  useEffect(() => {
    if (!loading) { setSegundosSimulando(0); return }
    const inicio = Date.now()
    const id = setInterval(() => setSegundosSimulando(Math.floor((Date.now() - inicio) / 1000)), 500)
    return () => clearInterval(id)
  }, [loading])

  // NOVO: armazena as faixas customizadas por código de reservatório
  // estrutura: { [cod]: FaixaCustom[] | null }
  // null = usar o banco; array = usar as faixas editadas na sessão
  const [planoSecasSession, setPlanoSecasSession] = useState({})

  useEffect(() => {
    if (!curvasOtimizadas?.reservatorio || !curvasOtimizadas?.faixas?.length) return
    setPlanoSecasSession(prev => ({
      ...prev,
      [curvasOtimizadas.reservatorio]: curvasOtimizadas.faixas,
    }))
    setActiveTab('meta')
  }, [curvasOtimizadas?.id])

  useEffect(()=>{
    Promise.all([api.fetchReservatorios(),api.fetchPresets()])
      .then(([r,p])=>{setResList(r);setPresets(p)})
      .catch(e=>setApiError(e.message))
  },[api])

  const handleReset=(options={})=>{
    setResultados(null)
    setSimMeta(null)
    setError(null)
    setActiveTab(options.keepCurvas ? 'meta' : 'padrao')
    setResultTab('graficos')
    // NOVO: limpa as faixas de sessão ao resetar o cenário
    if (!options.keepCurvas) setPlanoSecasSession({})
  }

  // NOVO: recebe a notificação do PlanoSecasPanel quando o usuário clica "Aplicar na Sessão"
  // faixas === null significa que foi revertido → volta a usar o banco
  const handleFaixasChange = (cod, faixas) => {
    setPlanoSecasSession(prev => ({ ...prev, [cod]: faixas }))
  }

  const handlePresetApply = (preset) => {
    if (!preset?.cenario_hidrossistema) return
    setActiveTab('meta')
    const niveis = preset.niveis_meta
    if (niveis?.reservatorio_cod && niveis?.faixas?.length) {
      setPlanoSecasSession(prev => ({
        ...prev,
        [String(niveis.reservatorio_cod)]: niveis.faixas,
      }))
    }
  }

  const handleSimulate=async(payload)=>{
    setLoading(true);setError(null)
    try{
      // CORRIGIDO: injeta plano_secas_custom em cada reservatório que tiver faixas na sessão
      const payloadComPlano = {
        ...payload,
        usar_niveis_meta: activeTab === 'meta',
        reservatorios: payload.reservatorios.map(r => ({
          ...r,
          plano_secas_custom: planoSecasSession[r.cod] || planoSecasSession[r.nome] || null,
        }))
      }

      const data=await api.runSimulacao(payloadComPlano)
      setResultados(data.resultados)
      setSimMeta({
        modo:payload.modo,
        vazaoConjunta:payload.vazao_conjunta,
        params:payload.reservatorios.map(r=>({demanda_nominal:r.demanda,capacidade:r.capacidade})),
        usarNiveisMeta: activeTab === 'meta',
        cenarioHidrossistema: payload.cenario_hidrossistema,
        indicadoresSistema: data.indicadores_sistema,
        cenarioHidrologicoNome: data.cenario_hidrologico?.nome,
        periodo: `${payload.mes_inicial}/${payload.ano_inicial}–${payload.mes_final}/${payload.ano_final}`,
        histerese: payload.histerese_transferencia || 0,
        demandas: payload.reservatorios.map(r=>`${m3sToLps(r.demanda)} L/s`).join(', '),
        geradoEm: new Date(),
      })
      setResultTab(t => t === 'comparar' && cenarioA ? 'comparar' : 'graficos')
      setTimeout(()=>document.getElementById('sim-anchor')?.scrollIntoView({behavior:'smooth',block:'start'}),200)
    }catch(e){setError(e.message)}
    finally{setLoading(false)}
  }

  const MAIN_TABS=[
    {id:'padrao', label:'Simulação Padrão'},
    {id:'meta',   label:'Simulação com Níveis Meta'},
  ]
  const RES_TABS=[
    {id:'graficos',  label:'Gráficos'},
    {id:'vazoes',    label:'Balanço Hídrico'},
    {id:'garantia',  label:'Garantia'},
    ...(cenarioA ? [{id:'comparar', label:'Comparar cenários'}] : []),
  ]

  const fixarCenario = () => {
    if (!resultados) return
    setCenarioA({ resultados, simMeta })
  }

  // dados extras guardados no arquivo de configuração e restaurados ao abri-lo
  const obterExtrasConfig = () => ({ usarNiveisMeta: activeTab === 'meta', planoSecasSessao: planoSecasSession })
  const handleAbrirConfig = (config) => {
    setResultados(null); setSimMeta(null); setError(null); setResultTab('graficos')
    setPlanoSecasSession(config.planoSecasSessao || {})
    setActiveTab(config.usarNiveisMeta ? 'meta' : 'padrao')
  }

  // indica visualmente se há faixas customizadas ativas na sessão
  const temPlanoCustom = Object.values(planoSecasSession).some(v => v !== null && v !== undefined)
  const temCurvaOtimizada = Boolean(curvasOtimizadas?.reservatorio)

  return (
    <div className={`sim-root ${darkMode ? 'app-dark' : ''}`} style={{minHeight:600,paddingBottom:48}}>
      <style>{CSS}</style>
      <style>{`.sim-root.app-dark{--bg:#050403;--card:#0d0805;--text:#fff7ef;--text-mid:#efd0b8;--text-light:#c0987c;--border:#2a1a10;--border-light:#1f140d;--orange-pale:#3a1d0b;--orange-deep:#ff9b42;--teal-pale:#09231f;--red-pale:#2a0c0c;--yellow-pale:#2a2108;--blue-pale:#071634;--shadow-sm:0 1px 6px rgba(0,0,0,.35);--shadow:0 2px 18px rgba(0,0,0,.45)}.sim-root.app-dark input,.sim-root.app-dark select,.sim-root.app-dark textarea{background:#080503!important;color:var(--text)!important;border-color:var(--border)!important}.sim-root.app-dark option{background:#080503;color:var(--text)}.sim-root.app-dark .sim-ghost{background:#0a0604;color:var(--text-light);border-color:var(--border)}.sim-root.app-dark .sim-ghost:hover{background:var(--orange-pale);color:var(--orange-deep);border-color:var(--orange-deep)}.sim-root.app-dark .recharts-default-tooltip{background:var(--card)!important;border-color:var(--border)!important;color:var(--text)!important}`}</style>

      <div className="sim-header" style={{display:'flex',alignItems:'flex-start',justifyContent:'space-between',gap:12,flexWrap:'wrap'}}>
        <div>
          <div style={{display:'flex',alignItems:'center',gap:9,marginBottom:3}}>
            <Waves size={21} color="var(--orange)" strokeWidth={2}/>
            <h2 style={{fontSize:19,fontWeight:800,color:'var(--text)',letterSpacing:'-0.01em',margin:0}}>Simulador de Balanço Hídrico</h2>
          </div>
          <p style={{fontSize:11.5,color:'var(--text-light)',margin:0}}>
            {resultados?.[0]?.reservatorio
              ? <>Cenário: <strong style={{color:'var(--orange-deep)'}}>{resultados[0].reservatorio}</strong> · Série histórica processada.</>
              : 'Configure os reservatórios e clique em Gerar Simulação.'}
            {/* NOVO: badge indicando que há níveis meta customizados ativos */}
            {temPlanoCustom && (
              <span style={{marginLeft:8,fontSize:10,background:'var(--yellow-pale)',color:'var(--yellow)',borderRadius:20,padding:'1px 8px',fontWeight:700,border:'1px solid var(--yellow)'}}>
                ⚡ Níveis meta customizados ativos
              </span>
            )}
            {temCurvaOtimizada && (
              <span style={{marginLeft:8,fontSize:10,background:'var(--teal-pale)',color:'var(--teal)',borderRadius:20,padding:'1px 8px',fontWeight:700,border:'1px solid var(--teal)'}}>
                Curvas otimizadas: {curvasOtimizadas.reservatorio}
              </span>
            )}
          </p>
        </div>
        <div style={{display:'flex',gap:7,alignItems:'center',flexWrap:'wrap'}}>
          <div style={{display:'flex',gap:3,background:'var(--card)',border:'1.5px solid var(--border)',borderRadius:'var(--radius-sm)',padding:3,boxShadow:'var(--shadow-sm)'}}>
            {MAIN_TABS.map(t=><button key={t.id} className={`sim-tab ${activeTab===t.id?'on':'off'}`} onClick={()=>setActiveTab(t.id)}>{t.label}</button>)}
          </div>
          {resultados&&(
            <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
              <button className="sim-ghost" onClick={()=>{fixarCenario(); setResultTab('graficos')}} title="Guarda esta simulação como cenário A para comparar com as próximas">
                <ArrowLeftRight size={12}/> {cenarioA ? 'Substituir cenário A' : 'Fixar para comparar'}
              </button>
              {cenarioA&&<button className="sim-ghost" onClick={()=>{setCenarioA(null); if(resultTab==='comparar') setResultTab('graficos')}} title="Remove o cenário fixado">
                <Trash2 size={12}/> Remover cenário A
              </button>}
              <button className="sim-ghost" onClick={()=>exportExcel(resultados, simMeta?.modo||'Individual')}>
                <FileSpreadsheet size={12}/> Excel
              </button>
              <button className="sim-ghost" onClick={()=>{const b=new Blob([JSON.stringify(resultados,null,2)],{type:'application/json'});const a=Object.assign(document.createElement('a'),{href:URL.createObjectURL(b),download:'simulacao.json'});a.click()}}>
                <Download size={12}/> JSON
              </button>
            </div>
          )}
        </div>
      </div>

      {apiError&&(
        <div style={{margin:'12px 26px 0',padding:'10px 14px',background:'#fffbea',border:'1.5px solid #f5c842',borderRadius:'var(--radius-sm)',display:'flex',gap:9,alignItems:'flex-start'}}>
          <AlertTriangle size={13} color="#b48a0c" style={{marginTop:1}}/>
          <div style={{fontSize:11,color:'#7a5c00',lineHeight:1.6}}>
            <strong>API não encontrada.</strong> Configure <code style={{fontFamily:'JetBrains Mono',background:'#fef3cd',padding:'1px 4px',borderRadius:3}}>VITE_API_URL</code> ou passe a prop <code style={{fontFamily:'JetBrains Mono',background:'#fef3cd',padding:'1px 4px',borderRadius:3}}>apiUrl</code>.<br/>
            <span style={{fontSize:10,opacity:0.75}}>{apiError}</span>
          </div>
        </div>
      )}

      <div className="sim-layout">

        <ConfigPanel resList={resList} presets={presets} onSimulate={handleSimulate} loading={loading} onResChange={setActiveRes} onReset={handleReset} onPresetApply={handlePresetApply} appliedCurvas={curvasOtimizadas} obterExtrasConfig={obterExtrasConfig} onAbrirConfig={handleAbrirConfig}/>

        <div style={{display:'flex',flexDirection:'column',gap:12,minWidth:0}}>

          {(activeTab==='padrao' || activeTab==='meta')&&(
            <>
              {activeTab==='meta'&&(
                <PlanoSecasPanel
                  api={api}
                  reservatorios={activeRes}
                  onFaixasChange={handleFaixasChange}
                  faixasSessao={planoSecasSession}
                  onOpenOtimizador={onOpenOtimizador}
                />
              )}

              {error&&(
                <div style={{background:'var(--red-pale)',border:'1.5px solid var(--red)',borderRadius:'var(--radius-sm)',padding:'10px 14px',display:'flex',alignItems:'center',gap:9}}>
                  <AlertTriangle size={13} color="var(--red)"/>
                  <span style={{flex:1,fontSize:11.5,color:'var(--red)',fontWeight:500}}>{error}</span>
                  <button onClick={()=>setError(null)} aria-label="Fechar mensagem de erro" style={{background:'none',border:'none',cursor:'pointer',color:'var(--red)',fontSize:16,lineHeight:1}}>×</button>
                </div>
              )}

              {loading&&(
                <Card style={{padding:'46px 20px',display:'flex',flexDirection:'column',alignItems:'center',gap:12}}>
                  <div role="status" aria-live="polite" style={{display:'flex',flexDirection:'column',alignItems:'center',gap:12}}>
                    <RefreshCw size={32} color="var(--orange)" className="sim-spin" aria-hidden="true"/>
                    <div style={{fontSize:13,fontWeight:700,color:'var(--text)'}}>Simulando… {segundosSimulando > 0 ? `${segundosSimulando} s` : ''}</div>
                    <div style={{fontSize:11,color:'var(--text-light)',textAlign:'center'}}>
                      Processando a série de vazões e calculando o balanço hídrico mês a mês.
                      {segundosSimulando >= 15 && <><br/>Períodos longos e vários reservatórios podem levar mais tempo. Se o servidor estiver em plano gratuito, a primeira requisição também pode demorar enquanto ele inicia.</>}
                    </div>
                  </div>
                </Card>
              )}

              {!loading&&!resultados&&!error&&(
                <Card style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',padding:'56px 20px',gap:13}}>
                  <div style={{width:64,height:64,borderRadius:'50%',background:'var(--orange-pale)',border:'2px solid var(--orange-light)',display:'flex',alignItems:'center',justifyContent:'center'}}>
                    <Waves size={28} color="var(--orange)" strokeWidth={1.5}/>
                  </div>
                  <div style={{textAlign:'center'}}>
                    <div style={{fontSize:14.5,fontWeight:800,color:'var(--text)',marginBottom:5}}>Pronto para simular</div>
                    <div style={{fontSize:11.5,color:'var(--text-light)',maxWidth:310}}>Configure os reservatórios, período e demandas no painel e clique em <strong>Gerar Simulação</strong>.</div>
                  </div>
                </Card>
              )}

              {!loading&&resultados&&(
                <>
                  <div id="sim-anchor"/>
                  <div role="tablist" aria-label="Resultados" style={{display:'flex',gap:3,background:'var(--card)',border:'1.5px solid var(--border)',borderRadius:'var(--radius-sm)',padding:3,width:'fit-content',maxWidth:'100%',boxShadow:'var(--shadow-sm)',flexWrap:'wrap'}}>
                    {RES_TABS.map(t=><button key={t.id} role="tab" aria-selected={resultTab===t.id} className={`sim-tab ${resultTab===t.id?'on':'off'}`} onClick={()=>setResultTab(t.id)}>{t.label}</button>)}
                  </div>

                  {resultTab==='comparar' && cenarioA ? (
                    <ComparacaoCenarios cenarioA={cenarioA} resultados={resultados} simMeta={simMeta}/>
                  ) : (<>
                  <MetricsRow resultados={resultados} modo={simMeta?.modo||'Individual'}/>
                  <IndicadoresDesempenho resultados={resultados} sistema={simMeta?.indicadoresSistema}/>
                  <MesesAbastecidos resultados={resultados} modo={simMeta?.modo||'Individual'} params={simMeta?.params}/>
                  <FailureDetail resultados={resultados} modo={simMeta?.modo||'Individual'}/>
                  {simMeta?.cenarioHidrossistema==='pgps_fogareiro_quixeramobim_cenario_1'&&<PgpsValidation resultados={resultados}/>}

                  {resultTab==='graficos'  && <Charts resultados={resultados} params={simMeta?.params} modo={simMeta?.modo||'Individual'} usarNiveisMeta={simMeta?.usarNiveisMeta}/>}
                  {resultTab==='vazoes'    && <VazoesDetail resultados={resultados} modo={simMeta?.modo||'Individual'}/>}
                  {resultTab==='garantia'  && simMeta && <GarantiaAnalise resultados={resultados} modo={simMeta.modo} vazaoConjunta={simMeta.vazaoConjunta} params={simMeta.params}/>}
                  </>)}
                </>
              )}
            </>
          )}

        </div>
      </div>
    </div>
  )
}
