
import React, { useState, useEffect, useMemo } from 'react'
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter,
} from 'recharts'
import {
  Waves, Download, RefreshCw, AlertTriangle, CheckCircle2, AlertCircle,
  Plus, Trash2, Database, Calendar, Settings2, Zap, ChevronDown,
  ChevronLeft, ChevronRight, BarChart3, TrendingDown, Droplets,
  Save, Info, Shield, ArrowLeftRight, FileSpreadsheet, BarChart2,
  Activity,
} from 'lucide-react'
import * as XLSX from 'xlsx'

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const MESES = ['JAN','FEV','MAR','ABR','MAI','JUN','JUL','AGO','SET','OUT','NOV','DEZ']
const ANOS  = Array.from({ length: 110 }, (_, i) => 1911 + i)
const COLORS = [
  { stroke: '#264fa3', fill: '#264fa3', fillOp: 0.15 },
  { stroke: '#e07b2a', fill: '#e07b2a', fillOp: 0.15 },
  { stroke: '#2a9d8f', fill: '#2a9d8f', fillOp: 0.15 },
  { stroke: '#9b2dca', fill: '#9b2dca', fillOp: 0.15 },
]

const CSS = `
  .sim-root *, .sim-root *::before, .sim-root *::after { box-sizing: border-box; }
  .sim-root {
    --bg: #fdf6ee; --orange: #e07b2a; --orange-light: #f5a654;
    --orange-pale: #fdebd3; --orange-deep: #c46318; --orange-glow: rgba(224,123,42,0.18);
    --teal: #2a9d8f; --teal-pale: #d4f5ef;
    --blue: #264fa3; --blue-light: #4a7cc7; --blue-pale: #dde8f8;
    --red: #d94040; --red-pale: #fde8e8;
    --yellow: #d4a017; --yellow-pale: #fef3cd;
    --text: #1e1208; --text-mid: #5a3c24; --text-light: #9a7055;
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
  .sim-plano-inp { width:100%; border:1.5px solid transparent; border-radius:4px; background:transparent; text-align:center; font-size:11px; font-family:'JetBrains Mono',monospace; color:var(--text); padding:3px 2px; transition:all 0.15s; outline:none; }
  .sim-plano-inp:focus { border-color:var(--orange); background:var(--orange-pale); }
  .sim-tr:hover td { background: var(--orange-pale) !important; }
`

// ─────────────────────────────────────────────────────────────────────────────
// API
// ─────────────────────────────────────────────────────────────────────────────

function makeApi(base) {
  const b = base || import.meta.env?.VITE_API_URL || 'http://localhost:8000'
  const get  = async (p) => { const r = await fetch(`${b}${p}`); if (!r.ok) throw new Error(`Erro ${r.status}: ${p}`); return r.json() }
  const post = async (p, body) => {
    const r = await fetch(`${b}${p}`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) })
    if (!r.ok) {
      const e = await r.json().catch(() => ({}))
      if (Array.isArray(e.detail)) throw new Error(e.detail.map(x => (x.loc?x.loc.join('→')+': ':'')+x.msg).join(' | '))
      throw new Error(typeof e.detail==='string' ? e.detail : JSON.stringify(e.detail) || 'Erro')
    }
    return r.json()
  }
  return {
    fetchReservatorios: () => get('/api/reservatorios'),
    fetchPresets:       () => get('/api/presets'),
    fetchPlanoSecas:    (cod) => get(`/api/plano-secas/${cod}`).catch(() => []),
    // savePlanoSecas removido — o plano é gerido apenas na sessão
    runSimulacao:       (payload) => post('/api/simular', payload),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// EXCEL EXPORT (SheetJS)
// ─────────────────────────────────────────────────────────────────────────────

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
        'Demanda Atendida (hm³)':      parseFloat(d['Demanda Atendida (m³/s)'] ?? 0) * (segundos / 1e6),
        'Racionamento (%)':            parseFloat(d['Racionamento (%)'] ?? 0),
        'Vertimento (hm³)':            parseFloat(d['Vertimento (hm³)'] ?? 0),
        'Falha':                       d['Falha'] ?? 'Não',
        'Modo Operação':               d['Modo Operação'] ?? 'Normal',
      }
      // Transferência física apenas no modo Série
      if (isSerie) {
        row['Transferência Recebida (m³/s)'] = parseFloat(d['Transferência Recebida (m³/s)'] ?? 0)
        row['Transferência Enviada (m³/s)']  = parseFloat(d['Transferência Enviada (m³/s)'] ?? 0)
      }
      return row
    })
    const ws = XLSX.utils.json_to_sheet(rows)
    XLSX.utils.book_append_sheet(wb, ws, r.reservatorio.slice(0, 31))
  })

  XLSX.writeFile(wb, 'simulacao_hidrica.xlsx')
}

// ─────────────────────────────────────────────────────────────────────────────
// SMALL ATOMS
// ─────────────────────────────────────────────────────────────────────────────

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

function FC({ as='input', children, style, ...props }) {
  const base = { width:'100%', padding:'7px 10px', border:'1.5px solid var(--border)', borderRadius:'var(--radius-xs)', background:'#fff', color:'var(--text)', fontSize:12.5, outline:'none', transition:'border-color 0.15s', ...style }
  const onF = e => e.target.style.borderColor = 'var(--orange)'
  const onB = e => e.target.style.borderColor = 'var(--border)'
  if (as === 'select') return <select style={{ ...base, appearance:'none', cursor:'pointer' }} onFocus={onF} onBlur={onB} {...props}>{children}</select>
  return <input style={base} onFocus={onF} onBlur={onB} {...props} />
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

// ─────────────────────────────────────────────────────────────────────────────
// METRICS ROW
// ─────────────────────────────────────────────────────────────────────────────

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

// Falha sistémica: Paralelo=qualquer falha, Série/Individual=TODOS falham
function calcFalhasConjuntas(resultados, modo) {
  if (!resultados?.length) return []
  const n = resultados[0].dados.length
  // Falha sistémica em todos os modos: apenas quando TODOS os reservatórios falham
  return Array.from({length:n}, (_,t) =>
    resultados.every(r=>r.dados[t]?.['Falha']==='Sim')
  )
}

function MetricsRow({ resultados, modo }) {
  if (!resultados?.length) return null
  const totalMeses = resultados[0].dados.length  // período do sistema, não soma
  const falhasConj = calcFalhasConjuntas(resultados, modo)
  const falhasSist = falhasConj.filter(Boolean).length
  let rac=0, racM=0, atend=0, solic=0, vert=0, evap=0, transf=0
  resultados.forEach(r => r.dados.forEach(d => {
    const rc = parseFloat(d['Racionamento (%)'])||0
    if (rc>0){rac+=rc;racM++}
    atend += parseFloat(d['Demanda Atendida (m³/s)'])||0
    solic  += parseFloat(d['Demanda Solicitada (m³/s)'])||0
    vert   += parseFloat(d['Vertimento (hm³)'])||0
    evap   += parseFloat(d['Evaporação (hm³)'])||0
    transf += parseFloat(d['Transferência Recebida (m³/s)'])||0
  }))
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

// ─────────────────────────────────────────────────────────────────────────────
// MESES ABASTECIDOS (por reservatório)
// ─────────────────────────────────────────────────────────────────────────────

function MesesAbastecidos({ resultados, modo, params }) {
  if (!resultados?.length) return null
  const totalSist = resultados[0].dados.length  // período do sistema
  return (
    <Card className="sim-fade" style={{ padding:'16px 20px' }}>
      <div style={{ fontSize:13, fontWeight:800, color:'var(--text)', marginBottom:12, display:'flex', alignItems:'center', gap:8 }}>
        <Activity size={15} color="var(--orange)"/>
        Meses Abastecidos por Reservatório
      </div>
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(200px,1fr))', gap:10 }}>
        {resultados.map((r,i) => {
          const isParalelo = modo === 'Paralelo'

          // No Paralelo: cada reservatório só é responsável nos meses em que Dem.Sol. > 0
          // (quando Dem.Sol.=0 a responsabilidade estava no outro reservatório)
          // Base = meses com responsabilidade; Atendidos = base com Falha=Não
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

// ─────────────────────────────────────────────────────────────────────────────
// FAILURE DETAIL
// ─────────────────────────────────────────────────────────────────────────────

function FailureDetail({ resultados }) {
  if (!resultados?.length) return null
  const falhas = []
  resultados.forEach(r => r.dados.forEach(d => {
    if (d['Falha']==='Sim') falhas.push({ reservatorio:r.reservatorio, data:d.Data, volIni:parseFloat(d['Armazenamento Inicial']||0).toFixed(2), demSol:parseFloat(d['Demanda Solicitada (m³/s)']||0).toFixed(3), demAt:parseFloat(d['Demanda Atendida (m³/s)']||0).toFixed(3), rac:parseFloat(d['Racionamento (%)']||0).toFixed(1), modo:d['Modo Operação'] })
  }))
  return (
    <Card style={{ padding:'16px 20px', borderColor:falhas.length>0?'var(--red-pale)':'var(--teal-pale)' }}>
      <div style={{ display:'flex', alignItems:'center', gap:9, marginBottom:falhas.length?12:0 }}>
        {falhas.length>0?<AlertCircle size={16} color="var(--red)"/>:<CheckCircle2 size={16} color="var(--teal)"/>}
        <div>
          <div style={{ fontSize:13, fontWeight:800, color:'var(--text)' }}>Detalhamento das Falhas</div>
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

// ─────────────────────────────────────────────────────────────────────────────
// CHARTS (simulação)
// ─────────────────────────────────────────────────────────────────────────────

function ChartCard({ title, subtitle, children }) {
  return (
    <Card className="sim-fade" style={{ padding:'16px 18px' }}>
      <div style={{ marginBottom:12 }}>
        <div style={{ fontSize:13, fontWeight:800, color:'var(--text)' }}>{title}</div>
        {subtitle && <div style={{ fontSize:11, color:'var(--text-light)', marginTop:2 }}>{subtitle}</div>}
      </div>
      {children}
    </Card>
  )
}

// Seletor de reservatório para gráficos
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

function Charts({ resultados, params, modo }) {
  const [selVol,setSelVol]=useState('todos')
  const [selDem,setSelDem]=useState('todos')
  const [selRac,setSelRac]=useState('todos')
  const [selBal,setSelBal]=useState('todos')
  const [selTr,setSelTr]=useState('todos')

  if (!resultados?.length) return null
  const allD = [...new Set(resultados.flatMap(r => r.dados.map(d => d.Data)))].sort()
  const iv   = Math.max(0, Math.floor(allD.length/12)-1)

  // filtrar reservatórios conforme seletor
  const resSel = (sel) => sel==='todos' ? resultados : [resultados[sel]]

  // Volume em %
  const mkVolData = (sel) => {
    const res = resSel(sel)
    return allD.map(data => {
      const p = {data}
      res.forEach((r,i) => {
        const d = r.dados.find(x=>x.Data===data)
        const ri = resultados.indexOf(r)
        const cap = params?.[ri]?.capacidade || 1
        if(d){
          const vf = parseFloat(d['Armazenamento Final'])||0
          p[`Vol.% (${r.reservatorio})`] = cap>0 ? parseFloat(((vf/cap)*100).toFixed(1)) : 0
          p[`Afluência (${r.reservatorio})`] = parseFloat(d['Afluências (hm³/mês)'])||0
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
    return allD.map(data => { const p={data}; res.forEach(r=>{ const d=r.dados.find(x=>x.Data===data); if(d){p[`Afluência(${r.reservatorio})`]=parseFloat(d['Afluências (hm³/mês)'])||0; p[`Evap.(${r.reservatorio})`]=parseFloat(d['Evaporação (hm³)'])||0; p[`Vertimento(${r.reservatorio})`]=parseFloat(d['Vertimento (hm³)'])||0} }); return p })
  }
  const mkTrData = (sel) => {
    const res = resSel(sel)
    return allD.map(data => { const p={data}; res.forEach(r=>{ const d=r.dados.find(x=>x.Data===data); if(d){const rc=parseFloat(d['Transferência Recebida (m³/s)'])||0; const ev=parseFloat(d['Transferência Enviada (m³/s)'])||0; if(rc>0||ev>0){p[`Rec.(${r.reservatorio})`]=rc; p[`Env.(${r.reservatorio})`]=ev}} }); return p })
  }

  const volData = mkVolData(selVol)
  const demData = mkDemData(selDem)
  const racData = mkRacData(selRac)
  const balData = mkBalData(selBal)
  const trData  = mkTrData(selTr)
  // Transferência física só existe no modo Série
  const hasTransf = modo==='Série' && allD.some(data=>{ const p=mkTrData('todos').find(x=>x.data===data); return p&&Object.keys(p).length>1 })

  const volKeys = volData[0] ? Object.keys(volData[0]).filter(k=>k!=='data'&&k.startsWith('Vol.')) : []
  const aflKeys = volData[0] ? Object.keys(volData[0]).filter(k=>k!=='data'&&k.startsWith('Afluência')) : []
  const racKeys = racData[0] ? Object.keys(racData[0]).filter(k=>k!=='data') : []
  const activeRes = (sel) => sel==='todos'?resultados:[resultados[sel]]

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
      {/* Volume em % */}
      <ChartCard title="Volume Armazenado (%)">
        <ResSel resultados={resultados} sel={selVol} onChange={setSelVol}/>
        <div style={{ height:250 }}>
          <ResponsiveContainer>
            <AreaChart data={volData} margin={{top:4,right:28,left:0,bottom:0}}>
              <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
              <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
              <YAxis yAxisId="vol" domain={[0,100]} tick={{fontSize:10,fill:'var(--blue)'}} label={{value:'%',angle:-90,position:'insideLeft',fill:'var(--blue)',fontSize:10}}/>
              <YAxis yAxisId="afl" orientation="right" tick={{fontSize:10,fill:'var(--teal)'}} label={{value:'Afluência(hm³)',angle:90,position:'insideRight',fill:'var(--teal)',fontSize:10}}/>
              <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
              {volKeys.map((k,i)=><Area key={k} yAxisId="vol" type="monotone" dataKey={k} stroke={COLORS[i%4].stroke} fill={COLORS[i%4].fill} fillOpacity={COLORS[i%4].fillOp} strokeWidth={2} dot={false}/>)}
              {aflKeys.map((k,i)=><Line key={k} yAxisId="afl" type="monotone" dataKey={k} stroke={COLORS[(i+2)%4].stroke} strokeWidth={1.5} dot={false} strokeDasharray={"4 2"}/>)}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {/* Demanda */}
      <ChartCard title="Demanda: Solicitada vs Atendida" subtitle="m³/s mensal">
        <ResSel resultados={resultados} sel={selDem} onChange={setSelDem}/>
        <div style={{ height:200 }}>
          <ResponsiveContainer>
            <LineChart data={demData} margin={{top:4,right:20,left:0,bottom:0}}>
              <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
              <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
              <YAxis tick={{fontSize:10,fill:'var(--text-light)'}} label={{value:'m³/s',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
              <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
              {activeRes(selDem).map((r,i)=>{const gi=resultados.indexOf(r);return[
                <Line key={`s${gi}`} type="monotone" dataKey={`Sol.(${r.reservatorio})`} stroke={COLORS[gi%4].stroke} strokeWidth={2} strokeDasharray={"5 3"} dot={false}/>,
                <Line key={`a${gi}`} type="monotone" dataKey={`At.(${r.reservatorio})`}  stroke={COLORS[gi%4].stroke} strokeWidth={2} dot={false}/>,
              ]})}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {/* Racionamento */}
      {racKeys.length>0 && (
        <ChartCard title="Racionamento Mensal" subtitle="Níveis Meta — Racionamento aplicado (%)">
          <ResSel resultados={resultados} sel={selRac} onChange={setSelRac}/>
          <div style={{ height:180 }}>
            <ResponsiveContainer>
              <BarChart data={racData} margin={{top:4,right:20,left:0,bottom:0}}>
                <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
                <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
                <YAxis domain={[0,100]} tick={{fontSize:10,fill:'var(--text-light)'}} label={{value:'%',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
                <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
                {racKeys.map((k,i)=><Bar key={k} dataKey={k} fill={COLORS[resultados.findIndex(r=>r.reservatorio===k)%4]?.stroke||COLORS[0].stroke} fillOpacity={0.75} radius={[3,3,0,0]}/>)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      )}

      {/* Balanço */}
      <ChartCard title="Balanço Hídrico" subtitle="Afluência, Evaporação e Vertimento (hm³/mês)">
        <ResSel resultados={resultados} sel={selBal} onChange={setSelBal}/>
        <div style={{ height:185 }}>
          <ResponsiveContainer>
            <BarChart data={balData} margin={{top:4,right:20,left:0,bottom:0}}>
              <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
              <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
              <YAxis tick={{fontSize:10,fill:'var(--text-light)'}} label={{value:'hm³',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
              <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
              {activeRes(selBal).map((r,i)=>{const gi=resultados.indexOf(r);return[
                <Bar key={`af${gi}`} dataKey={`Afluência(${r.reservatorio})`} fill="#2a9d8f" fillOpacity={0.65} radius={[3,3,0,0]}/>,
                <Bar key={`ev${gi}`} dataKey={`Evap.(${r.reservatorio})`}     fill="#e07b2a" fillOpacity={0.65} radius={[3,3,0,0]}/>,
                <Bar key={`vt${gi}`} dataKey={`Vertimento(${r.reservatorio})`} fill="#264fa3" fillOpacity={0.65} radius={[3,3,0,0]}/>,
              ]})}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {/* Transferências */}
      {hasTransf && (
        <ChartCard title="Transferências entre Reservatórios" subtitle="m³/s mensal">
          <ResSel resultados={resultados} sel={selTr} onChange={setSelTr}/>
          <div style={{ height:180 }}>
            <ResponsiveContainer>
              <BarChart data={trData} margin={{top:4,right:20,left:0,bottom:0}}>
                <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
                <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
                <YAxis tick={{fontSize:10,fill:'var(--text-light)'}} label={{value:'m³/s',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
                <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
                {Object.keys(trData[0]||{}).filter(k=>k!=='data').map((k,i)=><Bar key={k} dataKey={k} fill={k.startsWith('Rec')?'#2a9d8f':'#9b2dca'} fillOpacity={0.7} radius={[3,3,0,0]}/>)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// DETALHAMENTO DAS VAZÕES
// ─────────────────────────────────────────────────────────────────────────────

function VazoesDetail({ resultados, modo }) {
  const [sel, setSel] = useState(0)
  const [page, setPage] = useState(0)
  const PAGE = 18

  if (!resultados?.length) return null

  const r = resultados[sel]
  const dados = r.dados
  const allD  = [...new Set(resultados.flatMap(x => x.dados.map(d => d.Data)))].sort()
  const iv    = Math.max(0, Math.floor(allD.length/12)-1)

  // Série de vazões (afluência)
  const serieData = allD.map(data => {
    const p = { data }
    resultados.forEach(res => {
      const d = res.dados.find(x => x.Data === data)
      if (d) p[res.reservatorio] = parseFloat(d['Vazão (m³/s)'] ?? d['Afluências (hm³/mês)']) || 0
    })
    return p
  })

  // Tabela paginada
  const totalPg = Math.ceil(dados.length / PAGE)
  const pagDados = dados.slice(page * PAGE, (page+1)*PAGE)

  const VCOLS_ALL = [
    { key:'Data',                           label:'Mês/Ano',     mono:true  },
    { key:'Vazão (m³/s)',                   label:'Vazão (m³/s)',mono:true  },
    { key:'Afluências (hm³/mês)',           label:'Afluência (hm³)',mono:true},
    { key:'Evaporação (hm³)',               label:'Evap. (hm³)', mono:true  },
    { key:'Armazenamento Inicial',          label:'Vol. Ini.',   mono:true  },
    { key:'Armazenamento Final',            label:'Vol. Fin.',   mono:true  },
    { key:'Demanda Solicitada (m³/s)',      label:'Dem. Sol.',   mono:true  },
    { key:'Demanda Atendida (m³/s)',        label:'Dem. At.',    mono:true  },
    { key:'Transferência Recebida (m³/s)',  label:'Tr. Rec.',    mono:true  },
    { key:'Transferência Enviada (m³/s)',   label:'Tr. Env.',    mono:true  },
    { key:'Racionamento (%)',               label:'Rac.(%)',     mono:true  },
    { key:'Vertimento (hm³)',               label:'Vertimento',  mono:true  },
    { key:'Falha',                          label:'Falha',       align:'center'},
    { key:'Modo Operação',                  label:'Modo',        align:'center'},
  ]
  const VCOLS = modo==='Série' ? VCOLS_ALL : VCOLS_ALL.filter(c=>!c.key.startsWith('Transferência'))

  function fv(val,key){
    if(val===null||val===undefined||val==='') return '—'
    if(key==='Falha'||key==='Modo Operação'||key==='Data') return val
    const n=parseFloat(val); return isNaN(n)?val:n.toFixed(3)
  }

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
      {/* Tabs reservatório */}
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

      {/* Gráfico série de vazões */}
      <ChartCard title="Série de Vazões Afluentes" subtitle="Vazão mensal afluente a cada reservatório (m³/s)">
        <div style={{ height:220 }}>
          <ResponsiveContainer>
            <LineChart data={serieData} margin={{top:4,right:20,left:0,bottom:0}}>
              <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
              <XAxis dataKey="data" tickFormatter={tickFmt} interval={iv} tick={{fontSize:10,fill:'var(--text-light)'}}/>
              <YAxis tick={{fontSize:10,fill:'var(--text-light)'}} label={{value:'m³/s',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
              <Tooltip content={<CTip/>}/><Legend wrapperStyle={{fontSize:10}}/>
              {resultados.map((res,i)=>(
                <Line key={i} type="monotone" dataKey={res.reservatorio} stroke={COLORS[i%4].stroke} strokeWidth={1.5} dot={false}/>
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {/* Tabela mensal */}
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

// ─────────────────────────────────────────────────────────────────────────────
// ANÁLISE DE GARANTIA (fiel ao Streamlit)
// ─────────────────────────────────────────────────────────────────────────────

function GarantiaAnalise({ resultados, modo, vazaoConjunta, params }) {
  if (!resultados?.length) return null

  const totalMeses = resultados[0].dados.length
  const dfs = resultados.map(r => r.dados)

  // Vazões do sistema = soma das demandas atendidas
  const vazoesSystem = new Array(totalMeses).fill(0)
  dfs.forEach(df => df.forEach((d,t) => { vazoesSystem[t] += parseFloat(d['Demanda Atendida (m³/s)'])||0 }))

  // Falhas conjuntas — usa a mesma lógica de MetricsRow
  const falhasConj = calcFalhasConjuntas(resultados, modo)

  const numFalhas  = falhasConj.filter(Boolean).length
  const garantiaSistema = ((totalMeses - numFalhas) / totalMeses * 100)
  const vazoesSemFalha  = vazoesSystem.filter((_,t) => !falhasConj[t])
  const vazaoMedia   = vazoesSemFalha.length ? vazoesSemFalha.reduce((a,b)=>a+b,0)/vazoesSemFalha.length : 0
  const vazaoMaxima  = vazoesSemFalha.length ? Math.max(...vazoesSemFalha) : 0
  const vazaoMinima  = vazoesSemFalha.length ? Math.min(...vazoesSemFalha) : 0

  // Tabela de permanência/garantia do SISTEMA
  const grouped = {}
  vazoesSystem.forEach((v,t) => {
    if (!falhasConj[t]) {
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
    return { vazao:k, perm, freq:freq.toFixed(2), garantia:cumFreq.toFixed(2) }
  })
  if (numFalhas>0) resumo.push({ vazao:'FALHA', perm:numFalhas, freq:(numFalhas/totalMeses*100).toFixed(2), garantia:'-' })

  // Curva de permanência para gráfico
  const curvData = sortedKeys.map((k,i) => ({
    vazao: k,
    garantia: parseFloat(resumo[i].garantia),
    permanencia: parseFloat(resumo[i].freq),
  }))

  // Demanda nominal sistema
  const demNominal = (params||[]).reduce((s,p)=>s+(p?.demanda_nominal||0),0) + (vazaoConjunta||0)

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:14 }}>

      {/* KPIs sistema */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(150px,1fr))', gap:10 }}>
        <MCard label="Garantia Sistema"  value={`${garantiaSistema.toFixed(2)}%`} sub="Meses sem falha / total" variant={garantiaSistema>=95?'success':garantiaSistema>=80?'yellow':'danger'} icon={Shield}/>
        <MCard label="Meses Simulados"   value={totalMeses}  sub={`${numFalhas} com falha`}     variant="info"    icon={TrendingDown}/>
        <MCard label="Dem. Nominal Total" value={`${demNominal.toFixed(3)}`} sub="m³/s"        variant="default" icon={BarChart3}/>
        <MCard label="Vazão Média"        value={vazaoMedia.toFixed(3)}  sub="m³/s" variant="default" icon={Waves}/>
        <MCard label="Vazão Máxima"       value={vazaoMaxima.toFixed(3)} sub="m³/s" variant="success" icon={Waves}/>
        <MCard label="Vazão Mínima"       value={vazaoMinima.toFixed(3)} sub="m³/s" variant={vazaoMinima>0?'info':'danger'} icon={Waves}/>
      </div>

      {/* Curva de permanência */}
      {curvData.length>1 && (
        <ChartCard title="Curva de Permanência e Garantia" subtitle="Garantia acumulada (%) × Vazão total do sistema (m³/s)">
          <div style={{ height:230 }}>
            <ResponsiveContainer>
              <AreaChart data={curvData} margin={{top:4,right:20,left:0,bottom:0}}>
                <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
                <XAxis dataKey="garantia" type="number" domain={[0,100]} tick={{fontSize:10,fill:'var(--text-light)'}} label={{value:'Garantia Acumulada (%)',position:'insideBottom',offset:-2,fill:'var(--text-light)',fontSize:10}}/>
                <YAxis tick={{fontSize:10,fill:'var(--blue)'}} label={{value:'Vazão (m³/s)',angle:-90,position:'insideLeft',fill:'var(--blue)',fontSize:10}}/>
                <Tooltip content={<CTip/>}/>
                <Area type="monotone" dataKey="vazao" name="Vazão (m³/s)" stroke="#264fa3" fill="#264fa3" fillOpacity={0.15} strokeWidth={2} dot={false}/>
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      )}

      {/* Tabela resumo sistema */}
      <ChartCard title="Análise de Vazões Totais do Sistema" subtitle="Permanência, frequência e garantia acumulada">
        <div style={{ overflowX:'auto' }}>
          <table style={{ width:'100%', borderCollapse:'collapse', fontSize:11.5 }}>
            <thead>
              <tr style={{ background:'var(--bg)' }}>
                {['Vazão Total Sistema (m³/s)','Permanência (meses)','Frequência (%)','Garantia Acumulada (%)'].map(h=>(
                  <th key={h} style={{ padding:'7px 12px', textAlign:'right', fontSize:10, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.05em', color:'var(--text-light)', borderBottom:'1.5px solid var(--border)', whiteSpace:'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {resumo.map((row,i)=>(
                <tr key={i} className="sim-tr" style={{ background:row.vazao==='FALHA'?'var(--red-pale)':i%2===0?'transparent':'rgba(236,220,200,0.15)' }}>
                  <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono', fontWeight:row.vazao==='FALHA'?700:400, color:row.vazao==='FALHA'?'var(--red)':'var(--text)' }}>{row.vazao==='FALHA'?'⚠️ FALHA':parseFloat(row.vazao).toFixed(3)}</td>
                  <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono' }}>{row.perm}</td>
                  <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono' }}>{row.freq}%</td>
                  <td style={{ padding:'6px 12px', textAlign:'right', borderBottom:'1px solid var(--border-light)', fontFamily:'JetBrains Mono', color:row.garantia==='-'?'var(--text-light)':'var(--blue)', fontWeight:row.garantia==='-'?400:600 }}>{row.garantia === '-' ? '—' : `${row.garantia}%`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ChartCard>

      {/* Detalhamento por reservatório */}
      <div style={{ fontSize:13, fontWeight:800, color:'var(--text)', marginTop:4, display:'flex', alignItems:'center', gap:8 }}>
        <Database size={14} color="var(--orange)"/>
        Detalhamento por Reservatório
      </div>

      {resultados.map((r, idx) => {
        const df    = r.dados
        const p     = params?.[idx]
        const demNom = p?.demanda_nominal || 0
        const demConj = idx===0 ? (vazaoConjunta||0) : 0
        const demTot = demNom + demConj

        // Agrupar por (nomeFaixa + rac) para que níveis com rac=0 mas nomes
        // diferentes (ex: "Normal" e "Acima do Teto") sejam contabilizados separadamente
        const gruposVistos = new Set()
        const grupos = []
        df.forEach(d => {
          if (d['Falha'] === 'Sim') return
          const rac      = parseFloat(d['Racionamento (%)']) || 0
          const nome     = d['Modo Operação'] || 'Normal'
          const chave    = `${nome}__${rac}`
          if (!gruposVistos.has(chave)) {
            gruposVistos.add(chave)
            grupos.push({ nome, rac, chave })
          }
        })
        // Ordenar: primeiro pelo racionamento crescente, depois pelo nome
        grupos.sort((a,b) => a.rac - b.rac || a.nome.localeCompare(b.nome))

        let cumG = 0
        const tabelaRes = []

        grupos.forEach(({ nome, rac, chave }) => {
          const filtro = df.filter(d =>
            d['Falha'] === 'Não' &&
            (parseFloat(d['Racionamento (%)']) || 0) === rac &&
            (d['Modo Operação'] || 'Normal') === nome
          )
          if (!filtro.length) return
          const vazAlvo = demTot * (1 - rac / 100)
          const freq    = (filtro.length / totalMeses) * 100
          cumG += freq
          tabelaRes.push({ faixa:nome, rac:rac.toFixed(1), vazAlvo:vazAlvo.toFixed(3), count:filtro.length, freq:freq.toFixed(2), garantia:cumG.toFixed(2) })
        })
        const cntFalha = df.filter(d=>d['Falha']==='Sim').length
        if (cntFalha>0) tabelaRes.push({ faixa:'⚠️ FALHA', rac:'FALHA', vazAlvo:'0.000', count:cntFalha, freq:(cntFalha/totalMeses*100).toFixed(2), garantia:'-' })

        return (
          <Card key={idx} style={{ overflow:'hidden' }}>
            <div style={{ padding:'11px 16px', background:'var(--bg)', borderBottom:'1.5px solid var(--border)', display:'flex', alignItems:'center', justifyContent:'space-between', flexWrap:'wrap', gap:6 }}>
              <div>
                <span style={{ fontSize:13, fontWeight:800, color:'var(--text)' }}>🌊 {r.reservatorio}</span>
                <span style={{ fontSize:11, color:'var(--text-light)', marginLeft:10 }}>Demanda Total: <strong style={{ fontFamily:'JetBrains Mono' }}>{demTot.toFixed(3)} m³/s</strong></span>
              </div>
              {demConj>0 && (
                <span style={{ fontSize:10.5, background:'var(--blue-pale)', color:'var(--blue)', borderRadius:6, padding:'2px 9px', fontWeight:600 }}>
                  {demNom.toFixed(3)} (espec.) + {demConj.toFixed(3)} (conjunta) = {demTot.toFixed(3)} m³/s
                </span>
              )}
            </div>
            <div style={{ overflowX:'auto' }}>
              <table style={{ width:'100%', borderCollapse:'collapse', fontSize:11.5 }}>
                <thead>
                  <tr style={{ background:'var(--bg)' }}>
                    {['Nível Meta','Racionamento (%)','Vazão Total (m³/s)','Meses Responsável','Frequência (%)','Garantia (%)'].map(h=>(
                      <th key={h} style={{ padding:'7px 12px', textAlign:'right', fontSize:10, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.05em', color:'var(--text-light)', borderBottom:'1.5px solid var(--border)', whiteSpace:'nowrap', '&:first-child':{textAlign:'left'} }}>
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

// ─────────────────────────────────────────────────────────────────────────────
// RESULTS TABLE
// ─────────────────────────────────────────────────────────────────────────────

const RCOLS_BASE = [
  {key:'Data',label:'Data',align:'left'},{key:'Armazenamento Inicial',label:'Vol. Ini (hm³)',mono:true},{key:'Afluências (hm³/mês)',label:'Afluência (hm³)',mono:true},{key:'Evaporação (hm³)',label:'Evap. (hm³)',mono:true},{key:'Demanda Solicitada (m³/s)',label:'Dem. Sol.',mono:true},{key:'Demanda Atendida (m³/s)',label:'Dem. At.',mono:true},{key:'Transferência Recebida (m³/s)',label:'Tr. Rec.',mono:true},{key:'Transferência Enviada (m³/s)',label:'Tr. Env.',mono:true},{key:'Racionamento (%)',label:'Rac.(%)',mono:true},{key:'Vertimento (hm³)',label:'Vertimento',mono:true},{key:'Armazenamento Final',label:'Vol. Fin.',mono:true},{key:'Falha',label:'Falha',align:'center'},{key:'Modo Operação',label:'Modo',align:'center'},
]
// No Paralelo não há transferência física — ocultar colunas
function getRCols(modo) {
  if (modo === 'Série') return RCOLS_BASE
  return RCOLS_BASE.filter(c => !c.key.startsWith('Transferência'))
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

// ─────────────────────────────────────────────────────────────────────────────
// PLANO DE SECAS
// ─────────────────────────────────────────────────────────────────────────────

const FAIXAS_COR = {
  'Acima do Teto':{bg:'var(--teal-pale)',t:'var(--teal)'},
  'Normal':{bg:'var(--blue-pale)',t:'var(--blue)'},
  'Atenção':{bg:'var(--yellow-pale)',t:'var(--yellow)'},
  'Alerta':{bg:'var(--orange-pale)',t:'var(--orange-deep)'},
  'Emergência':{bg:'var(--red-pale)',t:'var(--red)'},
}

function PlanoSecasPanel({ api, reservatorios }) {
  const [faixas,setFaixas]=useState(null)           // estado de trabalho da sessão
  const [faixasOriginal,setFaixasOriginal]=useState(null)  // snapshot da BD (nunca perde sem Save na Base)
  const [loading,setLoading]=useState(false)
  const [saving,setSaving]=useState(false)
  const [msg,setMsg]=useState(null)

  const [selRes, setSelRes] = useState(0)
  const reservatorio = reservatorios?.[selRes] || null

  // chave estável — recalculada apenas quando os CODs realmente mudam
  const resKey = useMemo(
    ()=>(reservatorios||[]).map(r=>r.cod).join(','),
    [reservatorios]
  )

  // Ao mudar de hidrossistema: volta ao primeiro e limpa
  useEffect(()=>{
    setSelRes(0)
    setFaixas(null)
    setFaixasOriginal(null)
    setMsg(null)
  },[resKey])

  // Ao mudar de reservatório: busca da BD
  useEffect(()=>{
    if(!reservatorio?.cod) return
    setFaixas(null); setFaixasOriginal(null)
    setLoading(true); setMsg(null)
    api.fetchPlanoSecas(reservatorio.cod)
      .then(d=>{
        setFaixas(JSON.parse(JSON.stringify(d)))
        setFaixasOriginal(JSON.parse(JSON.stringify(d)))
      })
      .catch(()=>{ setFaixas([]); setFaixasOriginal([]) })
      .finally(()=>setLoading(false))
  },[reservatorio?.cod])

  const set=(idx,f,v)=>setFaixas(p=>p.map((x,i)=>i===idx?{...x,[f]:v}:x))
  const add=()=>setFaixas(p=>[...(p||[]),{Faixa:'Novo Nível',Racionamento:0,...Object.fromEntries(MESES.map(m=>[m,100]))}])
  const del=(idx)=>setFaixas(p=>p.filter((_,i)=>i!==idx))

  // Reverter: volta ao snapshot da BD (desfaz edições da sessão)
  const revert=()=>{
    if(!faixasOriginal) return
    setFaixas(JSON.parse(JSON.stringify(faixasOriginal)))
    setMsg({type:'info',text:'Revertido para o estado salvo na base de dados.'})
  }

  // Salvar apenas na sessão: mantém as faixas em memória para a simulação,
  // mas NÃO persiste na BD. As alterações somem ao fechar o browser.
  const saveSession=()=>{
    setMsg({type:'session',text:'Aplicado na sessão. As alterações serão usadas na simulação, mas não foram salvas na base de dados.'})
  }

  // Salvar na BD desactivado — plano gerido apenas na sessão
  // Para persistir, edite directamente o banco_site.db com DB Browser for SQLite

  const hasChanges = faixasOriginal !== null && JSON.stringify(faixas) !== JSON.stringify(faixasOriginal)
  const isSessionOnly = hasChanges  // alterações existem mas ainda não foram à BD

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
                ● Não salvo na base
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
        {msg&&<div style={{marginTop:9,padding:'7px 11px',borderRadius:'var(--radius-xs)',
          background:msg.type==='success'?'var(--teal-pale)':msg.type==='info'?'var(--blue-pale)':msg.type==='session'?'var(--yellow-pale)':'var(--red-pale)',
          color:msg.type==='success'?'var(--teal)':msg.type==='info'?'var(--blue)':msg.type==='session'?'var(--yellow)':'var(--red)',
          fontSize:11.5,fontWeight:600,lineHeight:1.5}}>
          {msg.type==='success'?'✓':msg.type==='session'?'⚡':msg.type==='info'?'ℹ':'✗'} {msg.text}
        </div>}
      </Card>
      <div style={{display:'flex',gap:8,padding:'9px 13px',background:'var(--blue-pale)',borderRadius:'var(--radius-sm)',alignItems:'flex-start'}}>
        <Info size={13} color="var(--blue)" style={{flexShrink:0,marginTop:1}}/>
        <div style={{fontSize:11,color:'var(--blue)',lineHeight:1.6}}>Os valores <strong>JAN…DEZ</strong> são o limite máximo de volume (% da capacidade) que activa o nível nesse mês. <strong>Racionamento</strong> =(%) de redução na demanda. As alterações feitas aqui são válidas apenas para esta sessão — para persistir permanentemente edite o arquivo <strong>banco_site.db</strong>.</div>
      </div>
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
        <Card style={{padding:'28px',textAlign:'center'}}><div style={{fontSize:12,color:'var(--text-light)'}}>Nenhuma faixa definida. Clique em <strong>+ Faixa</strong> para adicionar.</div></Card>
      )}

      {/* Gráfico dos Níveis Meta */}
      {faixas && faixas.length > 0 && <NiveisMeta faixas={faixas}/>}
    </div>
  )
}

// Gráfico de visualização dos Níveis Meta
// Interpolação verde→vermelho: índice 0 = verde (nível superior, volume alto)
//                               índice N-1 = vermelho (nível crítico, volume baixo)
function nivelColor(idx, total) {
  if (total <= 1) return '#2a9d8f'
  // t=0 → verde, t=1 → vermelho, passando por amarelo e laranja
  const t = idx / (total - 1)
  // Verde: #2a9d8f  Amarelo: #d4a017  Laranja: #e07b2a  Vermelho: #d94040
  const stops = [
    [42,157,143],   // verde-teal
    [212,160,23],   // amarelo
    [224,123,42],   // laranja
    [217,64,64],    // vermelho
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
  const n = faixas.length

  // Ordenar as faixas do limite MAIS ALTO para o MAIS BAIXO
  // (as faixas com limites mais altos ficam no topo do gráfico = verde)
  // Usamos a média dos valores mensais como proxy do "nível no gráfico"
  const faixasOrdenadas = [...faixas].sort((a, b) => {
    const mediaA = MESES.reduce((s,m) => s + (parseFloat(a[m])||0), 0) / 12
    const mediaB = MESES.reduce((s,m) => s + (parseFloat(b[m])||0), 0) / 12
    return mediaB - mediaA  // descendente: maior limite = primeiro = verde
  })

  // Cor de cada faixa na ordem ordenada
  const cores = faixasOrdenadas.map((_, i) => nivelColor(i, n))

  // Dados para o AreaChart
  // Cada área representa a FAIXA entre o seu limite e o do nível abaixo
  // Usamos type="number" no eixo X para posicionar correctamente
  // Recharts AreaChart com areas sobrepostas (não empilhadas): cada area vai de 0 ao seu limite
  // A ordem de renderização (de baixo para cima) determina o preenchimento visível entre faixas.
  // Renderizamos da faixa MAIOR para a MENOR — assim a menor fica por cima e "recorta" a maior.
  const data = MESES.map(mes => {
    const ponto = { mes }
    faixasOrdenadas.forEach(f => {
      ponto[f.Faixa] = parseFloat(f[mes]) || 0
    })
    return ponto
  })

  const Tip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    // Mostrar apenas valores únicos (sem duplicatas por sobreposição)
    const vistos = new Set()
    const items  = payload.filter(p => { if(vistos.has(p.name)) return false; vistos.add(p.name); return true })
    return (
      <div style={{ background:'#fff', border:'1.5px solid var(--border)', borderRadius:10, padding:'9px 13px', boxShadow:'var(--shadow)', fontSize:11 }}>
        <div style={{ fontWeight:700, marginBottom:5, color:'var(--text)' }}>{label}</div>
        {items.map((p,i) => (
          <div key={i} style={{ display:'flex', gap:7, alignItems:'center', marginBottom:2 }}>
            <div style={{ width:7, height:7, borderRadius:'50%', background:p.color }}/>
            <span style={{ color:'var(--text-mid)' }}>{p.name}:</span>
            <span style={{ fontWeight:600, fontFamily:'JetBrains Mono', color:'var(--text)' }}>≤ {p.value}%</span>
          </div>
        ))}
      </div>
    )
  }

  return (
    <Card className="sim-fade" style={{ padding:'16px 18px' }}>
      <div style={{ marginBottom:14 }}>
        <div style={{ fontSize:13, fontWeight:800, color:'var(--text)' }}>Limites de Activação por Mês</div>
        <div style={{ fontSize:11, color:'var(--text-light)', marginTop:2 }}>
          Volume máximo (% da capacidade) que activa cada nível — verde = volume alto, vermelho = nível crítico
        </div>
      </div>
      <div style={{ height:260 }}>
        <ResponsiveContainer>
          <AreaChart data={data} margin={{top:4,right:20,left:0,bottom:4}}>
            <CartesianGrid strokeDasharray={"3 3"} stroke="var(--border)"/>
            <XAxis dataKey="mes" tick={{fontSize:10,fill:'var(--text-light)'}}/>
            <YAxis domain={[0,100]} tick={{fontSize:10,fill:'var(--text-light)'}}
              label={{value:'% Cap.',angle:-90,position:'insideLeft',fill:'var(--text-light)',fontSize:10}}/>
            <Tooltip content={<Tip/>}/>
            {/* Renderizar da faixa MAIOR para a MENOR para que as menores
                fiquem por cima e criem o efeito de bandas coloridas entre linhas */}
            {faixasOrdenadas.map((f, i) => (
              <Area
                key={f.Faixa}
                type="monotone"
                dataKey={f.Faixa}
                stroke={cores[i]}
                strokeWidth={2.5}
                fill={cores[i]}
                fillOpacity={0.30}
                dot={{ r:3.5, fill:cores[i], strokeWidth:0 }}
                activeDot={{ r:5, fill:cores[i], strokeWidth:0 }}
                legendType="none"
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Legenda manual com gradiente de cor */}
      <div style={{ display:'flex', gap:8, flexWrap:'wrap', marginTop:12 }}>
        {faixasOrdenadas.map((f, i) => {
          const cor = cores[i]
          const rac = parseFloat(f.Racionamento) || 0
          return (
            <span key={i} style={{
              display:'inline-flex', alignItems:'center', gap:6,
              fontSize:10.5, borderRadius:20, padding:'3px 11px', fontWeight:600,
              background:`${cor}18`, color:cor,
              border:`1.5px solid ${cor}55`,
            }}>
              <span style={{ width:8, height:8, borderRadius:'50%', background:cor, display:'inline-block', flexShrink:0 }}/>
              {f.Faixa}{rac > 0 ? ` — ${rac}% de Racionamento` : ' — Sem Racionamento'}
            </span>
          )
        })}
      </div>
    </Card>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG PANEL
// ─────────────────────────────────────────────────────────────────────────────

// Busca de reservatório com filtragem em tempo real
function ResSearch({ resList, value, onChange }) {
  const [query, setQuery] = useState(value || '')
  const [open,  setOpen]  = useState(false)
  const ref = React.useRef(null)

  // Sync external value changes (e.g. preset)
  useEffect(() => { setQuery(value || '') }, [value])

  // Close on outside click
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

  return (
    <div ref={ref} style={{position:'relative'}}>
      <input
        value={query}
        onChange={e=>{ setQuery(e.target.value); setOpen(true); if(!e.target.value) onChange('') }}
        onFocus={()=>setOpen(true)}
        placeholder="Digite para buscar…"
        style={{width:'100%',padding:'7px 10px',border:'1.5px solid var(--border)',borderRadius:'var(--radius-xs)',background:'#fff',color:'var(--text)',fontSize:12.5,outline:'none',transition:'border-color 0.15s'}}
        onMouseEnter={e=>e.target.style.borderColor='var(--orange)'}
        onMouseLeave={e=>{ if(document.activeElement!==e.target) e.target.style.borderColor='var(--border)' }}
        onFocusCapture={e=>e.target.style.borderColor='var(--orange)'}
        onBlurCapture={e=>e.target.style.borderColor='var(--border)'}
      />
      {open && filtered.length > 0 && (
        <div style={{position:'absolute',top:'100%',left:0,right:0,background:'#fff',border:'1.5px solid var(--border)',borderRadius:'var(--radius-xs)',boxShadow:'var(--shadow)',zIndex:999,maxHeight:200,overflowY:'auto',marginTop:2}}>
          {filtered.map(r=>(
            <div key={r.COD}
              onMouseDown={()=>select(r.CORPO)}
              style={{padding:'7px 11px',fontSize:12,cursor:'pointer',borderBottom:'1px solid var(--border-light)',transition:'background 0.1s'}}
              onMouseEnter={e=>e.currentTarget.style.background='var(--orange-pale)'}
              onMouseLeave={e=>e.currentTarget.style.background='#fff'}>
              <span style={{fontWeight:600,color:'var(--text)'}}>{r.CORPO}</span>
              <span style={{fontSize:10,color:'var(--text-light)',marginLeft:8,fontFamily:'JetBrains Mono'}}>{r.COD}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ResCard({ res, index, resList, onChange, onRemove, modoLocked, modo }) {
  const [open,setOpen]=useState(true)
  const showGatilho = modo !== 'Individual'  // gatilho shown only for index=0 below
  return (
    <div style={{background:'var(--bg)',border:'1.5px solid var(--border)',borderRadius:'var(--radius-sm)',marginBottom:6,overflow:'hidden'}}>
      <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',padding:'8px 10px',cursor:'pointer',borderBottom:open?'1.5px solid var(--border-light)':'none'}} onClick={()=>setOpen(!open)}>
        <div style={{display:'flex',alignItems:'center',gap:7}}>
          <div style={{width:19,height:19,borderRadius:'50%',background:'var(--orange-pale)',border:'1.5px solid var(--orange-light)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:9,fontWeight:800,color:'var(--orange-deep)',flexShrink:0}}>{index+1}</div>
          <span style={{fontSize:12,fontWeight:700,color:'var(--orange-deep)'}}>{res.nome||`Reservatório ${index+1}`}</span>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:4}}>
          {index>0&&<button onClick={e=>{e.stopPropagation();onRemove(index)}} style={{background:'none',border:'none',cursor:'pointer',color:'var(--text-light)',padding:3,borderRadius:4}} onMouseEnter={e=>e.currentTarget.style.color='var(--red)'} onMouseLeave={e=>e.currentTarget.style.color='var(--text-light)'}><Trash2 size={11}/></button>}
          <ChevronDown size={12} color="var(--text-light)" style={{transform:open?'rotate(180deg)':'none',transition:'transform 0.2s'}}/>
        </div>
      </div>
      {open&&(
        <div style={{padding:'10px 10px 12px'}}>
          <div style={{marginBottom:8}}>
            <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Reservatório</div>
            <ResSearch resList={resList} value={res.nome} onChange={val=>{
              const s=resList.find(r=>r.CORPO===val)
              onChange(index,{nome:val,cod:s?.COD||'',capacidade:s?parseFloat(s['CAPAC (m³)']):0,est_evap:s?.['Est. Evap.']||'',volPct:50,vol_inicial:s?parseFloat(s['CAPAC (m³)'])*0.5:0})
            }}/>
            {res.capacidade>0&&<div style={{fontSize:9.5,color:'var(--text-light)',marginTop:2,fontFamily:'JetBrains Mono'}}>Cap: {res.capacidade.toFixed(2)} hm³ · COD: {res.cod}</div>}
          </div>
          <div style={{display:'grid',gridTemplateColumns:showGatilho?'1fr 1fr':'1fr 1fr',gap:6}}>
            <div>
              <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Vol. Inicial (%)</div>
              <FC type="number" min="0" max="100" step="1" value={res.volPct??50} onChange={e=>{const p=parseFloat(e.target.value)||0;onChange(index,{volPct:p,vol_inicial:(res.capacidade*p)/100})}}/>
              {res.capacidade>0&&<div style={{fontSize:9,color:'var(--text-light)',marginTop:2,fontFamily:'JetBrains Mono'}}>= {((res.capacidade*(res.volPct??50))/100).toFixed(2)} hm³</div>}
            </div>
            <div>
              <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Demanda (m³/s)</div>
              <FC type="number" min="0" step="0.1" value={res.demanda} onChange={e=>onChange(index,{demanda:parseFloat(e.target.value)||0})}/>
            </div>
            {showGatilho && index===0 &&(
              <div>
                <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Gatilho Transf. (%)</div>
                <FC type="number" min="0" max="100" step="1" value={res.gatilho} onChange={e=>onChange(index,{gatilho:parseFloat(e.target.value)||0})}/>
              </div>
            )}
            <div>
              <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Est. Evap.</div>
              <div style={{padding:'7px 10px',border:'1.5px solid var(--border-light)',borderRadius:'var(--radius-xs)',background:'var(--bg)',color:'var(--text-light)',fontSize:12.5,fontFamily:'JetBrains Mono'}}>{res.est_evap||'—'}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function ConfigPanel({ resList, presets, onSimulate, loading, onResChange, onReset }) {
  const [items,setItems]=useState([{nome:'',cod:'',capacidade:0,est_evap:'',volPct:50,vol_inicial:0,demanda:0,gatilho:10}])
  const [modo,setModo]=useState('Individual')
  const [modoLocked,setModoLocked]=useState(false)
  const [vazaoConj,setVazaoConj]=useState(0)
  const [mesIni,setMesIni]=useState('JAN'),[anoIni,setAnoIni]=useState(1911)
  const [mesFim,setMesFim]=useState('DEZ'),[anoFim,setAnoFim]=useState(2017)
  const [presetSel,setPresetSel]=useState('')

  const change=(idx,patch)=>setItems(prev=>{const n=prev.map((it,i)=>i===idx?{...it,...patch}:it);onResChange&&onResChange(n);return n})

  const applyPreset=(nome)=>{
    const p=presets.find(x=>x.nome===nome)
    if(!p) return
    setModo(p.modo); setModoLocked(true)
    const ni=p.reservatorios.map(cod=>{
      const f=resList.find(r=>r.COD===cod||r.CORPO===cod)
      return {nome:f?.CORPO||cod,cod:f?.COD||cod,capacidade:f?parseFloat(f['CAPAC (m³)']):0,est_evap:f?.['Est. Evap.']||'',volPct:50,vol_inicial:f?parseFloat(f['CAPAC (m³)'])*0.:0,demanda:0,gatilho:10}
    })
    setItems(ni)
    onResChange&&onResChange(ni)
    onReset&&onReset()   // limpa resultados ao trocar hidrossistema
  }

  const clearPreset=()=>{
    setPresetSel('')
    setModoLocked(false)
    const empty = [{nome:'',cod:'',capacidade:0,est_evap:'',volPct:50,vol_inicial:0,demanda:0.5,gatilho:30}]
    setItems(empty)
    onResChange&&onResChange(empty)
    onReset&&onReset()   // limpa resultados ao limpar preset
  }

  const submit=()=>{
    onSimulate({
      reservatorios:items.map(it=>({nome:String(it.nome||''),cod:String(it.cod||''),capacidade:parseFloat(it.capacidade)||0,est_evap:String(it.est_evap??''),vol_inicial:parseFloat(it.vol_inicial)||0,demanda:parseFloat(it.demanda)||0,gatilho:parseFloat(it.gatilho)||0})),
      // Bug fix: modo Individual nunca envia vazão conjunta
      modo:String(modo),vazao_conjunta:modo==='Individual'?0:(parseFloat(vazaoConj)||0),
      mes_inicial:String(mesIni),ano_inicial:parseInt(anoIni),
      mes_final:String(mesFim),ano_final:parseInt(anoFim),
    })
  }

  return (
    <Card style={{padding:'18px 14px',position:'sticky',top:16}}>
      <div style={{fontSize:14.5,fontWeight:800,color:'var(--text)',marginBottom:2}}>Configuração</div>
      <div style={{fontSize:11,color:'var(--text-light)',marginBottom:14}}>Cenário: <strong style={{color:'var(--orange-deep)'}}>{items[0]?.nome||'Nenhum selecionado'}</strong></div>

      {presets.length>0&&(
        <>
          <Label icon={Zap}>Hidrossistema</Label>
          <div style={{display:'flex',gap:5}}>
            <FC as="select" style={{flex:1}} value={presetSel} onChange={e=>{setPresetSel(e.target.value);applyPreset(e.target.value)}}>
              <option value="">Configuração manual…</option>
              {/* REMOVIDO: modo entre parênteses no label */}
              {presets.map(p=><option key={p.nome} value={p.nome}>{p.nome}</option>)}
            </FC>
            {presetSel&&<button onClick={clearPreset} style={{background:'none',border:'1.5px solid var(--border)',borderRadius:'var(--radius-xs)',padding:'0 8px',cursor:'pointer',color:'var(--text-light)',fontSize:14,transition:'all 0.15s'}} title="Limpar preset" onMouseEnter={e=>e.currentTarget.style.color='var(--red)'} onMouseLeave={e=>e.currentTarget.style.color='var(--text-light)'}><X size={13}/></button>}
          </div>
          {presetSel&&<div style={{marginTop:5,fontSize:10.5,color:'var(--blue)',background:'var(--blue-pale)',borderRadius:5,padding:'3px 9px',display:'inline-flex',alignItems:'center',gap:5}}><Info size={11}/> Modo de operação: <strong>{modo}</strong></div>}
        </>
      )}

      <Label icon={Database}>Reservatórios</Label>
      {items.map((res,i)=>(
        <ResCard key={i} res={res} index={i} resList={resList} onChange={change} onRemove={idx=>setItems(p=>p.filter((_,j)=>j!==idx))} modoLocked={modoLocked} modo={modo}/>
      ))}
      <button onClick={()=>setItems(p=>[...p,{nome:'',cod:'',capacidade:0,est_evap:'',volPct:50,vol_inicial:0,demanda:0.5,gatilho:30}])}
        style={{width:'100%',padding:'6px',background:'none',border:'1.5px dashed var(--border)',borderRadius:'var(--radius-sm)',color:'var(--text-light)',fontSize:11,cursor:'pointer',marginBottom:2,transition:'all 0.15s'}}
        onMouseEnter={e=>{e.currentTarget.style.borderColor='var(--orange)';e.currentTarget.style.color='var(--orange)';e.currentTarget.style.background='var(--orange-pale)'}}
        onMouseLeave={e=>{e.currentTarget.style.borderColor='var(--border)';e.currentTarget.style.color='var(--text-light)';e.currentTarget.style.background='none'}}>
        <Plus size={10} style={{marginRight:4}}/> Adicionar Reservatório
      </button>

      <Label icon={Settings2}>Modo de Operação</Label>
      <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:5,marginBottom:4}}>
        {['Individual','Série','Paralelo'].map(m=>(
          <button key={m} onClick={()=>!modoLocked&&setModo(m)}
            style={{padding:'7px 4px',border:`1.5px solid ${modo===m?'var(--orange)':'var(--border)'}`,borderRadius:'var(--radius-xs)',background:modo===m?'var(--orange-pale)':'none',color:modo===m?'var(--orange-deep)':'var(--text-light)',fontSize:11,fontWeight:700,cursor:modoLocked?'not-allowed':'pointer',transition:'all 0.15s',opacity:modoLocked&&modo!==m?0.4:1}}>
            {m}
          </button>
        ))}
      </div>
      {modo!=='Individual'&&(
        <div style={{marginTop:9}}>
          <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600,textTransform:'uppercase',letterSpacing:'0.05em'}}>Vazão Conjunta (m³/s)</div>
          <FC type="number" min="0" step="0.1" value={vazaoConj} onChange={e=>setVazaoConj(e.target.value)}/>
        </div>
      )}

      <Label icon={Calendar}>Período</Label>
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:6}}>
        <div>
          <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Início</div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:4}}>
            <FC as="select" value={mesIni} onChange={e=>setMesIni(e.target.value)} style={{fontSize:11}}>{MESES.map(m=><option key={m}>{m}</option>)}</FC>
            <FC type="number" value={anoIni} onChange={e=>setAnoIni(e.target.value)} min="1900" max="2100" style={{fontSize:11}} placeholder="Ano"/>
          </div>
        </div>
        <div>
          <div style={{fontSize:10,color:'var(--text-light)',marginBottom:3,fontWeight:600}}>Fim</div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:4}}>
            <FC as="select" value={mesFim} onChange={e=>setMesFim(e.target.value)} style={{fontSize:11}}>{MESES.map(m=><option key={m}>{m}</option>)}</FC>
            <FC type="number" value={anoFim} onChange={e=>setAnoFim(e.target.value)} min="1900" max="2100" style={{fontSize:11}} placeholder="Ano"/>
          </div>
        </div>
      </div>

      <button onClick={submit} disabled={loading||!items[0].nome}
        style={{width:'100%',marginTop:16,padding:12,background:loading||!items[0].nome?'var(--border)':'linear-gradient(135deg,var(--orange),var(--orange-deep))',border:'none',borderRadius:'var(--radius-sm)',color:loading||!items[0].nome?'var(--text-light)':'#fff',fontSize:13,fontWeight:800,cursor:loading||!items[0].nome?'not-allowed':'pointer',boxShadow:loading?'none':'0 4px 18px var(--orange-glow)',transition:'all 0.2s',letterSpacing:'0.02em'}}
        onMouseEnter={e=>{if(!loading)e.currentTarget.style.transform='translateY(-1px)'}}
        onMouseLeave={e=>e.currentTarget.style.transform='none'}>
        {loading?'⏳ Simulando…':'▶ Gerar Simulação'}
      </button>
    </Card>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ROOT
// ─────────────────────────────────────────────────────────────────────────────

// Missing X import reference — add inline
function X({ size=14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
    </svg>
  )
}

export default function SimuladorHidrico({ apiUrl }) {
  const api = useMemo(() => makeApi(apiUrl), [apiUrl])
  const [resList,setResList]=useState([])
  const [presets,setPresets]=useState([])
  const [resultados,setResultados]=useState(null)
  const [simMeta,setSimMeta]=useState(null) // {modo, vazaoConjunta, params}
  const [loading,setLoading]=useState(false)
  const [error,setError]=useState(null)
  const [apiError,setApiError]=useState(null)
  const [activeTab,setActiveTab]=useState('sim')
  const [resultTab,setResultTab]=useState('graficos')
  const [activeRes,setActiveRes]=useState([])

  useEffect(()=>{
    Promise.all([api.fetchReservatorios(),api.fetchPresets()])
      .then(([r,p])=>{setResList(r);setPresets(p)})
      .catch(e=>setApiError(e.message))
  },[api])

  const handleReset=()=>{
    setResultados(null)
    setSimMeta(null)
    setError(null)
    setActiveTab('sim')
    setResultTab('graficos')
  }

  const handleSimulate=async(payload)=>{
    setLoading(true);setError(null)
    try{
      const data=await api.runSimulacao(payload)
      setResultados(data.resultados)
      // guardar meta para análise de garantia
      setSimMeta({
        modo:payload.modo,
        vazaoConjunta:payload.vazao_conjunta,
        params:payload.reservatorios.map(r=>({demanda_nominal:r.demanda, capacidade:r.capacidade}))
      })
      setActiveTab('sim');setResultTab('graficos')
      setTimeout(()=>document.getElementById('sim-anchor')?.scrollIntoView({behavior:'smooth',block:'start'}),200)
    }catch(e){setError(e.message)}
    finally{setLoading(false)}
  }

  const MAIN_TABS=[
    {id:'sim',    label:'▶ Simulação'},
    {id:'secas',  label:'🛡 Níveis Meta'},
  ]
  const RES_TABS=[
    {id:'graficos',  label:'📈 Gráficos'},
    {id:'vazoes',    label:'💧 Vazões'},
    {id:'garantia',  label:'📊 Garantia'},
    {id:'tabela',    label:'📋 Dados'},
  ]

  return (
    <div className="sim-root" style={{minHeight:600,paddingBottom:48}}>
      <style>{CSS}</style>

      {/* Top bar */}
      <div style={{padding:'18px 26px 0',display:'flex',alignItems:'flex-start',justifyContent:'space-between',gap:12,flexWrap:'wrap'}}>
        <div>
          <div style={{display:'flex',alignItems:'center',gap:9,marginBottom:3}}>
            <Waves size={21} color="var(--orange)" strokeWidth={2}/>
            <h2 style={{fontSize:19,fontWeight:800,color:'var(--text)',letterSpacing:'-0.01em',margin:0}}>Simulador de Balanço Hídrico</h2>
          </div>
          <p style={{fontSize:11.5,color:'var(--text-light)',margin:0}}>
            {resultados?.[0]?.reservatorio
              ? <>Cenário: <strong style={{color:'var(--orange-deep)'}}>{resultados[0].reservatorio}</strong> · Série histórica processada.</>
              : 'Configure os reservatórios e clique em Gerar Simulação.'}
          </p>
        </div>
        <div style={{display:'flex',gap:7,alignItems:'center',flexWrap:'wrap'}}>
          <div style={{display:'flex',gap:3,background:'var(--card)',border:'1.5px solid var(--border)',borderRadius:'var(--radius-sm)',padding:3,boxShadow:'var(--shadow-sm)'}}>
            {MAIN_TABS.map(t=><button key={t.id} className={`sim-tab ${activeTab===t.id?'on':'off'}`} onClick={()=>setActiveTab(t.id)}>{t.label}</button>)}
          </div>
          {resultados&&(
            <div style={{display:'flex',gap:6}}>
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

      {/* API error */}
      {apiError&&(
        <div style={{margin:'12px 26px 0',padding:'10px 14px',background:'#fffbea',border:'1.5px solid #f5c842',borderRadius:'var(--radius-sm)',display:'flex',gap:9,alignItems:'flex-start'}}>
          <AlertTriangle size={13} color="#b48a0c" style={{marginTop:1}}/>
          <div style={{fontSize:11,color:'#7a5c00',lineHeight:1.6}}>
            <strong>API não encontrada.</strong> Configure <code style={{fontFamily:'JetBrains Mono',background:'#fef3cd',padding:'1px 4px',borderRadius:3}}>VITE_API_URL</code> ou passe a prop <code style={{fontFamily:'JetBrains Mono',background:'#fef3cd',padding:'1px 4px',borderRadius:3}}>apiUrl</code>.<br/>
            <span style={{fontSize:10,opacity:0.75}}>{apiError}</span>
          </div>
        </div>
      )}

      {/* Main grid */}
      <div style={{padding:'14px 26px 0',display:'grid',gridTemplateColumns:'295px 1fr',gap:16,alignItems:'start'}}>

        <ConfigPanel resList={resList} presets={presets} onSimulate={handleSimulate} loading={loading} onResChange={setActiveRes} onReset={handleReset}/>

        <div style={{display:'flex',flexDirection:'column',gap:12}}>

          {/* ── SIMULAÇÃO TAB ── */}
          {activeTab==='sim'&&(
            <>
              {error&&(
                <div style={{background:'var(--red-pale)',border:'1.5px solid var(--red)',borderRadius:'var(--radius-sm)',padding:'10px 14px',display:'flex',alignItems:'center',gap:9}}>
                  <AlertTriangle size={13} color="var(--red)"/>
                  <span style={{flex:1,fontSize:11.5,color:'var(--red)',fontWeight:500}}>{error}</span>
                  <button onClick={()=>setError(null)} style={{background:'none',border:'none',cursor:'pointer',color:'var(--red)',fontSize:16,lineHeight:1}}>×</button>
                </div>
              )}
              {loading&&(
                <Card style={{padding:'46px 20px',display:'flex',flexDirection:'column',alignItems:'center',gap:12}}>
                  <RefreshCw size={32} color="var(--orange)" className="sim-spin"/>
                  <div style={{fontSize:13,fontWeight:700,color:'var(--text)'}}>Simulando…</div>
                  <div style={{fontSize:11,color:'var(--text-light)'}}>Processando série histórica e calculando balanço hídrico.</div>
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
                  {/* result tabs */}
                  <div style={{display:'flex',gap:3,background:'var(--card)',border:'1.5px solid var(--border)',borderRadius:'var(--radius-sm)',padding:3,width:'fit-content',boxShadow:'var(--shadow-sm)',flexWrap:'wrap'}}>
                    {RES_TABS.map(t=><button key={t.id} className={`sim-tab ${resultTab===t.id?'on':'off'}`} onClick={()=>setResultTab(t.id)}>{t.label}</button>)}
                  </div>

                  <MetricsRow resultados={resultados} modo={simMeta?.modo||'Individual'}/>
                  <MesesAbastecidos resultados={resultados} modo={simMeta?.modo||'Individual'} params={simMeta?.params}/>
                  <FailureDetail resultados={resultados}/>

                  {resultTab==='graficos'  && <Charts resultados={resultados} params={simMeta?.params} modo={simMeta?.modo||'Individual'}/>}
                  {resultTab==='vazoes'    && <VazoesDetail resultados={resultados} modo={simMeta?.modo||'Individual'}/>}
                  {resultTab==='garantia'  && simMeta && <GarantiaAnalise resultados={resultados} modo={simMeta.modo} vazaoConjunta={simMeta.vazaoConjunta} params={simMeta.params}/>}
                  {resultTab==='tabela'    && <ResultsTable resultados={resultados} modo={simMeta?.modo||'Individual'}/>}
                </>
              )}
            </>
          )}

          {/* ── PLANO DE SECAS TAB ── */}
          {activeTab==='secas'&&(
            <PlanoSecasPanel api={api} reservatorios={activeRes}/>
          )}
        </div>
      </div>
    </div>
  )
}
