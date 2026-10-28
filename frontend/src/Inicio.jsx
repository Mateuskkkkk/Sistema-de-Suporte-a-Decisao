import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Area, AreaChart, Bar, BarChart, Brush, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { BarChart3, CloudRain, Crosshair, Download, LineChart as IconeLinha, Minus, Plus, Search, Settings2, ShieldCheck, X } from 'lucide-react'
import { MUNICIPIOS, PROJECAO, projetar } from './mapa/malhaCeara'
import { cssEscuro } from './tema'

const MESES = ['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN', 'JUL', 'AGO', 'SET', 'OUT', 'NOV', 'DEZ']
const MESES_MIN = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']
const ZOOM_MAX = 12

const fmt = (v, casas = 1) => Number(v).toLocaleString('pt-BR', { minimumFractionDigits: casas, maximumFractionDigits: casas })
const raioBase = cap => 3 + 2.6 * Math.log10(1 + Math.max(0, cap || 0))
const semAcento = s => String(s || '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase()

function makeApi(base) {
  const b = base || import.meta.env?.VITE_API_URL || 'http://127.0.0.1:8000'
  const get = async p => {
    const r = await fetch(`${b}${p}`)
    if (!r.ok) {
      const e = await r.json().catch(() => ({}))
      throw new Error(typeof e.detail === 'string' ? e.detail : `Erro ${r.status} ao consultar ${p}`)
    }
    return r.json()
  }
  return {
    acudes: () => get('/api/mapa/acudes'),
    detalhes: cod => get(`/api/acudes/${encodeURIComponent(cod)}/detalhes`),
  }
}

const CSS = `
.ini-root{--teal:#2a9d8f;--card-2:#fff;--text-mid:#5a3c24;--shadow:0 8px 26px rgba(120,70,30,.13);
  --mapa-fundo:#f6ede2;--mapa-mun:#efe3d4;--mapa-borda:#fffaf4;--mapa-contorno:#d9c3ab;--evap:#6fa8d6;
  position:relative;margin:14px 18px 18px;height:calc(100vh - 92px);min-height:520px;border:1.5px solid var(--border);
  border-radius:18px;overflow:hidden;background:var(--mapa-fundo);font-family:'Sora',sans-serif}
${cssEscuro('.ini-root.ini-dark')}
.ini-root.ini-dark{--mapa-fundo:#171310;--mapa-mun:#2a231e;--mapa-borda:#171310;--mapa-contorno:#4a3f36;--evap:#71a9f2}
.ini-svg{position:absolute;inset:0;width:100%;height:100%;cursor:grab;display:block;touch-action:none;user-select:none;-webkit-user-select:none}
.ini-svg.arrastando{cursor:grabbing}
.ini-mun path{fill:var(--mapa-mun);stroke:var(--mapa-borda);stroke-width:1.1;vector-effect:non-scaling-stroke}
.ini-acude{fill:var(--orange);fill-opacity:.85;stroke:var(--card);stroke-width:1.3;cursor:pointer;transition:fill-opacity .12s}
.ini-acude:hover,.ini-acude:focus-visible{fill-opacity:1;stroke:var(--text);outline:none}
.ini-acude.multi{fill:var(--teal)}
.ini-acude.aprox{fill-opacity:.28;stroke:var(--orange-deep);stroke-dasharray:2 2}
.ini-acude.apagado{opacity:.18;pointer-events:none}
.ini-sel{fill:var(--orange-deep);stroke:var(--card);stroke-width:2.5}
.ini-pulso{fill:none;stroke:var(--orange-deep);stroke-width:2;opacity:.35;pointer-events:none}
.ini-flut{position:absolute;background:var(--card);border:1.5px solid var(--border);border-radius:14px;box-shadow:var(--shadow);color:var(--text)}
.ini-busca{left:16px;top:16px;width:min(420px,calc(100% - 92px));padding:11px 12px;z-index:5}
.ini-campo{display:flex;align-items:center;gap:8px;border:1.5px solid var(--border);border-radius:10px;padding:8px 10px;background:var(--card)}
.ini-campo input{border:0;outline:0;background:transparent;color:var(--text);font:500 13px 'Sora',sans-serif;width:100%}
.ini-sug{margin-top:6px;max-height:240px;overflow:auto}
.ini-sug button{display:flex;justify-content:space-between;width:100%;border:0;background:transparent;color:var(--text);text-align:left;padding:7px 8px;border-radius:7px;font:600 12px 'Sora',sans-serif;cursor:pointer}
.ini-sug button:hover{background:var(--orange-pale);color:var(--orange-deep)}
.ini-sug small{color:var(--text-light);font-weight:500}
.ini-chips{display:flex;gap:6px;margin-top:9px;flex-wrap:wrap}
.ini-chip{border:1.5px solid var(--border);background:transparent;color:var(--text-light);border-radius:999px;padding:5px 10px;font:700 11.5px 'Sora',sans-serif;cursor:pointer;white-space:nowrap}
.ini-chip.on{background:var(--orange-pale);border-color:transparent;color:var(--orange-deep)}
.ini-zoom{right:16px;top:16px;display:flex;flex-direction:column;overflow:hidden;z-index:5}
.ini-zoom button{width:42px;height:42px;border:0;border-bottom:1.5px solid var(--border);background:var(--card);color:var(--text-light);display:grid;place-items:center;cursor:pointer}
.ini-zoom button:last-child{border-bottom:0}
.ini-zoom button:hover{color:var(--orange-deep)}
.ini-legenda{right:16px;bottom:62px;padding:12px 14px;font-size:11.5px;color:var(--text-light);z-index:4}
.ini-legenda b{display:block;font-size:10.5px;letter-spacing:.06em;margin-bottom:6px}
.ini-leg{display:flex;align-items:center;gap:8px;margin:5px 0}
.ini-bola{width:12px;height:12px;border-radius:50%;flex:none;border:1.5px solid var(--card);box-shadow:0 0 0 1px var(--border)}
.ini-dica{right:16px;bottom:16px;padding:8px 12px;font-size:11.5px;color:var(--text-light);z-index:4}
.ini-tooltip{position:absolute;pointer-events:none;background:#2b1d14;color:#fff;font-size:12px;padding:6px 9px;border-radius:8px;white-space:nowrap;z-index:6}
.ini-tooltip small{color:#e9c9ad}
.ini-card{width:380px;max-width:calc(100% - 32px);padding:16px 16px 14px;z-index:7;max-height:calc(100% - 32px);overflow:auto}
.ini-card h3{font-size:19px;margin:0;font-weight:800}
.ini-card .local{font-size:12.5px;color:var(--text-light);margin-top:2px}
.ini-fechar{border:0;background:transparent;color:var(--text-light);cursor:pointer;padding:2px;line-height:0}
.ini-dados{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:12px 0 10px}
.ini-dados div{background:var(--bg);border-radius:10px;padding:7px 9px}
.ini-dados small{display:block;font-size:9.5px;color:var(--text-light);text-transform:uppercase;letter-spacing:.04em}
.ini-dados b{font-size:13.5px}
.ini-aviso{font-size:11px;color:var(--orange-deep);background:var(--orange-pale);border-radius:8px;padding:6px 8px;margin-bottom:8px}
.ini-sistemas{font-size:11.5px;color:var(--text-light);margin-bottom:8px}
.ini-mini{border-top:1.5px solid var(--border);padding-top:9px;margin-bottom:10px}
.ini-mini span{font-size:11px;color:var(--text-light)}
.ini-mini text{font-size:8.5px;fill:var(--text-light);text-anchor:middle}
.ini-btserie{display:flex;justify-content:space-between;align-items:center;width:100%;border:1.5px dashed var(--orange);background:var(--orange-pale);color:var(--orange-deep);border-radius:11px;padding:9px 11px;font:800 12.5px 'Sora',sans-serif;cursor:pointer;margin-bottom:10px}
.ini-btserie small{color:var(--text-light);font-weight:600}
.ini-acoes{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.ini-acao{display:flex;gap:9px;align-items:center;border:1.5px solid var(--border);border-radius:11px;padding:9px 10px;background:var(--card);color:var(--text);cursor:pointer;text-align:left;font-family:'Sora',sans-serif}
.ini-acao:hover{border-color:var(--orange)}
.ini-acao .ic{width:28px;height:28px;border-radius:8px;display:grid;place-items:center;flex:none;background:var(--orange-pale);color:var(--orange-deep)}
.ini-acao b{display:block;font-size:12.5px}.ini-acao small{font-size:10.5px;color:var(--text-light)}
.ini-acao.pri{background:linear-gradient(135deg,#ea8a3b,#cc621c);border-color:transparent;color:#fff}
.ini-acao.pri small{color:#ffe2c9}.ini-acao.pri .ic{background:rgba(255,255,255,.22);color:#fff}
.ini-veu{position:absolute;inset:0;background:rgba(30,18,8,.42);z-index:20;display:grid;place-items:center;padding:16px}
.ini-modal{width:min(1320px,100%);max-height:100%;overflow:auto;background:var(--card);color:var(--text);border-radius:18px;padding:22px 24px;box-shadow:0 20px 60px rgba(40,20,5,.32)}
.ini-modal h2{font-size:22px;margin:0}
.ini-mbarra{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin:14px 0 12px}
.ini-abas{display:flex;gap:4px;background:var(--bg);padding:4px;border-radius:11px;flex-wrap:wrap}
.ini-aba{border:0;background:transparent;color:var(--text-light);padding:7px 13px;border-radius:8px;font:700 12.5px 'Sora',sans-serif;cursor:pointer}
.ini-aba.on{background:var(--card);color:var(--orange-deep);box-shadow:0 1px 3px rgba(0,0,0,.1)}
.ini-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
.ini-stats div{background:var(--bg);border-radius:12px;padding:9px 12px}
.ini-stats small{display:block;font-size:10.5px;color:var(--text-light);text-transform:uppercase;letter-spacing:.04em}
.ini-stats b{font-size:17px}.ini-stats span{font-size:11.5px;color:var(--text-light);margin-left:6px}
.ini-mrod{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-top:10px;font-size:12px;color:var(--text-light)}
.ini-bts{display:flex;gap:8px;flex-wrap:wrap}
.ini-bt{display:inline-flex;align-items:center;gap:6px;border:1.5px solid var(--border);background:var(--card);color:var(--text-mid);border-radius:10px;padding:8px 13px;font:700 12.5px 'Sora',sans-serif;cursor:pointer}
.ini-bt.pri{background:linear-gradient(135deg,#ea8a3b,#cc621c);color:#fff;border-color:transparent}
.ini-erro{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);padding:16px 18px;max-width:420px;font-size:13px;z-index:8}
@media (max-width:760px){
  .ini-root{margin:8px;height:calc(100vh - 120px)}
  .ini-card{left:8px!important;right:8px;top:auto!important;bottom:8px;width:auto;max-width:none;max-height:62%}
  .ini-legenda{display:none}
  .ini-stats{grid-template-columns:1fr 1fr}
}`

function GraficoEvaporacao({ valores }) {
  const mx = Math.max(...valores, 1)
  return (
    <svg width="100%" viewBox="0 0 336 70" role="img" aria-label="Evaporação média mensal">
      {valores.map((v, i) => {
        const h = 40 * v / mx
        return (
          <g key={MESES[i]}>
            <rect x={i * 28 + 3} y={54 - h} width="20" height={h} rx="2" fill="var(--evap)" />
            <text x={i * 28 + 13} y={50 - h} style={{ fontSize: 8 }}>{Math.round(v)}</text>
            <text x={i * 28 + 13} y="66">{MESES[i][0]}</text>
          </g>
        )
      })}
    </svg>
  )
}

function PainelSerie({ detalhes, onFechar, onUsarNoSimulador }) {
  const [aba, setAba] = useState('mensal')
  const serie = detalhes.afluencias
  const dados = useMemo(() => {
    const mensal = serie.map(p => ({ rotulo: `${MESES_MIN[p.mes - 1]}/${p.ano}`, ano: p.ano, vazao: p.vazao_m3s }))
    const porAno = {}
    serie.forEach(p => { (porAno[p.ano] = porAno[p.ano] || []).push(p.vazao_m3s) })
    const anual = Object.entries(porAno)
      .filter(([, v]) => v.length === 12)
      .map(([ano, v]) => ({ rotulo: ano, volume: v.reduce((a, b) => a + b, 0) * 2.592 }))
    const media = MESES.map((m, i) => {
      const v = serie.filter(p => p.mes === i + 1).map(p => p.vazao_m3s)
      return { rotulo: m, vazao: v.reduce((a, b) => a + b, 0) / Math.max(1, v.length) }
    })
    const ordenadas = serie.map(p => p.vazao_m3s).sort((a, b) => b - a)
    const passo = Math.max(1, Math.floor(ordenadas.length / 200))
    const permanencia = ordenadas.filter((_, i) => i % passo === 0).map((v, j) => ({
      rotulo: fmt(100 * (j * passo + 1) / ordenadas.length, 1), perm: 100 * (j * passo + 1) / ordenadas.length, vazao: v,
    }))
    return { mensal, anual, media, permanencia }
  }, [serie])

  const q = serie.map(p => p.vazao_m3s)
  const media = q.reduce((a, b) => a + b, 0) / q.length
  const imax = q.indexOf(Math.max(...q))
  const anuais = dados.anual.map(a => a.volume)
  const mediaAnual = anuais.reduce((a, b) => a + b, 0) / Math.max(1, anuais.length)
  const secos = q.filter(v => v < 0.01).length
  const ini = serie[0], fim = serie[serie.length - 1]
  const periodo = `${MESES_MIN[ini.mes - 1]}/${ini.ano}–${MESES_MIN[fim.mes - 1]}/${fim.ano}`

  const exportarCsv = () => {
    const linhas = ['ano;mes;vazao_m3s', ...serie.map(p => `${p.ano};${p.mes};${String(p.vazao_m3s).replace('.', ',')}`)]
    const url = URL.createObjectURL(new Blob(['﻿' + linhas.join('\n')], { type: 'text/csv;charset=utf-8' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `afluencias_${semAcento(detalhes.nome).replace(/\W+/g, '_')}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const eixo = { stroke: 'var(--text-light)', fontSize: 11 }
  const dica = { contentStyle: { background: 'var(--card)', border: '1.5px solid var(--border)', borderRadius: 8, fontSize: 12 } }
  return (
    <div className="ini-veu" onMouseDown={e => { if (e.target === e.currentTarget) onFechar() }}>
      <div className="ini-modal" role="dialog" aria-modal="true" aria-label={`Série de afluências de ${detalhes.nome}`}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
          <div>
            <h2>Série de afluências</h2>
            <div style={{ fontSize: 13, color: 'var(--text-light)', marginTop: 3 }}>
              {detalhes.nome} · {detalhes.municipio} · código {detalhes.cod} · vazão média mensal (m³/s), {periodo}
            </div>
          </div>
          <button className="ini-fechar" onClick={onFechar} aria-label="Fechar"><X size={22} /></button>
        </div>
        <div className="ini-mbarra">
          <div className="ini-abas" role="tablist">
            {[['mensal', 'Mensal'], ['anual', 'Anual'], ['media', 'Média por mês'], ['permanencia', 'Permanência']].map(([id, r]) => (
              <button key={id} role="tab" aria-selected={aba === id} className={`ini-aba ${aba === id ? 'on' : ''}`} onClick={() => setAba(id)}>{r}</button>
            ))}
          </div>
        </div>
        <div className="ini-stats">
          <div><small>Vazão média</small><b>{fmt(media, 2)} m³/s</b></div>
          <div><small>Afluência média anual</small><b>{fmt(mediaAnual, 1)} hm³</b></div>
          <div><small>Máxima mensal</small><b>{fmt(q[imax], 2)} m³/s</b><span>{dados.mensal[imax].rotulo}</span></div>
          <div><small>Meses com Q &lt; 0,01 m³/s</small><b>{secos} ({fmt(100 * secos / q.length, 1)}%)</b></div>
        </div>
        <div style={{ width: '100%', height: 400 }}>
          <ResponsiveContainer>
            {aba === 'mensal' ? (
              <AreaChart data={dados.mensal} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="rotulo" tick={eixo} minTickGap={40} />
                <YAxis tick={eixo} width={46} label={{ value: 'm³/s', angle: -90, position: 'insideLeft', fill: 'var(--text-light)', fontSize: 11 }} />
                <Tooltip {...dica} formatter={v => [`${fmt(v, 3)} m³/s`, 'Vazão']} />
                <Area type="linear" dataKey="vazao" stroke="var(--orange)" fill="var(--orange-pale)" strokeWidth={1.2} isAnimationActive={false} />
                <Brush dataKey="rotulo" height={22} stroke="var(--orange)" travellerWidth={8} />
              </AreaChart>
            ) : aba === 'anual' ? (
              <BarChart data={dados.anual} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="rotulo" tick={eixo} minTickGap={24} />
                <YAxis tick={eixo} width={52} label={{ value: 'hm³/ano', angle: -90, position: 'insideLeft', fill: 'var(--text-light)', fontSize: 11 }} />
                <Tooltip {...dica} formatter={v => [`${fmt(v, 1)} hm³`, 'Afluência anual']} />
                <Bar dataKey="volume" fill="var(--orange)" isAnimationActive={false} />
              </BarChart>
            ) : aba === 'media' ? (
              <BarChart data={dados.media} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="rotulo" tick={eixo} />
                <YAxis tick={eixo} width={46} label={{ value: 'm³/s', angle: -90, position: 'insideLeft', fill: 'var(--text-light)', fontSize: 11 }} />
                <Tooltip {...dica} formatter={v => [`${fmt(v, 3)} m³/s`, 'Vazão média']} />
                <Bar dataKey="vazao" fill="var(--orange)" isAnimationActive={false} />
              </BarChart>
            ) : (
              <LineChart data={dados.permanencia} margin={{ top: 8, right: 16, left: 0, bottom: 14 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="perm" type="number" domain={[0, 100]} tick={eixo} tickFormatter={v => `${v}%`}
                  label={{ value: 'percentual do tempo em que a vazão é igualada ou superada', position: 'insideBottom', offset: -8, fill: 'var(--text-light)', fontSize: 11 }} />
                <YAxis tick={eixo} width={46} label={{ value: 'm³/s', angle: -90, position: 'insideLeft', fill: 'var(--text-light)', fontSize: 11 }} />
                <Tooltip {...dica} labelFormatter={v => `${fmt(v, 1)}% do tempo`} formatter={v => [`${fmt(v, 3)} m³/s`, 'Vazão']} />
                <Line type="monotone" dataKey="vazao" stroke="var(--orange)" dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            )}
          </ResponsiveContainer>
        </div>
        <div className="ini-mrod">
          <span>{aba === 'mensal' ? 'Arraste as alças abaixo do gráfico para aproximar um período · passe o mouse para ver o valor do mês' : 'Passe o mouse sobre o gráfico para ver os valores'}</span>
          <div className="ini-bts">
            <button className="ini-bt" onClick={exportarCsv}><Download size={14} /> Exportar CSV</button>
            <button className="ini-bt pri" onClick={onUsarNoSimulador}>Usar no Simulador →</button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Inicio({ apiUrl, darkMode = false, onAbrirModulo }) {
  const api = useMemo(() => makeApi(apiUrl), [apiUrl])
  const [acudes, setAcudes] = useState([])
  const [erro, setErro] = useState(null)
  const [filtro, setFiltro] = useState('todos')
  const [busca, setBusca] = useState('')
  const [buscaAberta, setBuscaAberta] = useState(false)
  const [selCod, setSelCod] = useState(null)
  const [hover, setHover] = useState(null)
  const [detalhes, setDetalhes] = useState({})
  const [erroDetalhes, setErroDetalhes] = useState(null)
  const [mostrarSerie, setMostrarSerie] = useState(false)
  const [tam, setTam] = useState({ w: 0, h: 0 })
  const [vista, setVista] = useState(null)        // { k, x, y }
  const [arrastando, setArrastando] = useState(false)
  const raiz = useRef(null)
  const ponteiros = useRef(new Map())
  const gesto = useRef(null)

  useEffect(() => {
    api.acudes().then(setAcudes).catch(e => setErro(e.message))
  }, [api])

  // tamanho do quadro do mapa
  useEffect(() => {
    const el = raiz.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setTam({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const vistaInicial = useCallback(() => {
    if (!tam.w || !tam.h) return null
    const k = Math.min(tam.w / PROJECAO.largura, tam.h / PROJECAO.altura) * 0.92
    return { k, x: (tam.w - PROJECAO.largura * k) / 2, y: (tam.h - PROJECAO.altura * k) / 2 }
  }, [tam])
  const k0 = vistaInicial()?.k || 1

  useEffect(() => { if (!vista && tam.w) setVista(vistaInicial()) }, [tam, vista, vistaInicial])

  const zoomEm = useCallback((fator, cx, cy) => {
    setVista(v => {
      if (!v) return v
      const k = Math.min(k0 * ZOOM_MAX, Math.max(k0 * 0.8, v.k * fator))
      const f = k / v.k
      return { k, x: cx - (cx - v.x) * f, y: cy - (cy - v.y) * f }
    })
  }, [k0])

  // roda do mouse: aproxima em torno do cursor (precisa de passive:false para não rolar a página)
  useEffect(() => {
    const el = raiz.current
    if (!el) return
    const roda = e => {
      if (e.target.closest('.ini-flut, .ini-veu')) return
      e.preventDefault()
      const r = el.getBoundingClientRect()
      zoomEm(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top)
    }
    el.addEventListener('wheel', roda, { passive: false })
    return () => el.removeEventListener('wheel', roda)
  }, [zoomEm])

  const aoPressionar = e => {
    if (e.target.closest('.ini-acude')) return
    e.currentTarget.setPointerCapture(e.pointerId)
    ponteiros.current.set(e.pointerId, { x: e.clientX, y: e.clientY })
    gesto.current = { moveu: false }
  }
  const aoMover = e => {
    const p = ponteiros.current
    if (!p.has(e.pointerId) || !vista) return
    const ant = p.get(e.pointerId)
    p.set(e.pointerId, { x: e.clientX, y: e.clientY })
    if (p.size === 2) {                                   // pinça com dois dedos
      const [a, b] = [...p.values()]
      const outro = [...p.entries()].find(([id]) => id !== e.pointerId)[1]
      const dAnt = Math.hypot(ant.x - outro.x, ant.y - outro.y)
      const dNova = Math.hypot(a.x - b.x, a.y - b.y)
      const r = raiz.current.getBoundingClientRect()
      if (dAnt > 0) zoomEm(dNova / dAnt, (a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top)
      gesto.current.moveu = true
      return
    }
    const dx = e.clientX - ant.x, dy = e.clientY - ant.y
    if (Math.abs(dx) + Math.abs(dy) > 0) {
      if (!gesto.current.moveu && Math.hypot(dx, dy) < 3) return
      gesto.current.moveu = true
      setArrastando(true)
      setVista(v => ({ ...v, x: v.x + dx, y: v.y + dy }))
    }
  }
  const aoSoltar = e => {
    ponteiros.current.delete(e.pointerId)
    if (ponteiros.current.size === 0) {
      if (gesto.current && !gesto.current.moveu) setSelCod(null)   // clique no fundo fecha o cartão
      setArrastando(false)
    }
  }

  const visiveis = useMemo(() => acudes.filter(a =>
    filtro === 'todos' || (filtro === 'hidro' ? a.multirreservatorio || a.hidrossistemas.length > 0 : a.niveis_meta)
  ), [acudes, filtro])
  const visivelSet = useMemo(() => new Set(visiveis.map(a => a.cod)), [visiveis])
  const sugestoes = useMemo(() => {
    const t = semAcento(busca.trim())
    if (!t) return []
    return acudes.filter(a => semAcento(a.nome).includes(t) || semAcento(a.municipio).includes(t)).slice(0, 8)
  }, [acudes, busca])

  const tela = useCallback(a => {
    const [px, py] = projetar(a.lon, a.lat)
    return [vista.x + px * vista.k, vista.y + py * vista.k]
  }, [vista])
  const escalaMarcador = vista ? Math.min(1.7, Math.pow(vista.k / k0, 0.3)) : 1

  const sel = acudes.find(a => a.cod === selCod) || null
  const det = sel ? detalhes[sel.cod] : null

  const selecionar = useCallback((a, centralizar = false) => {
    setSelCod(a.cod)
    setErroDetalhes(null)
    if (!detalhes[a.cod]) {
      api.detalhes(a.cod).then(d => setDetalhes(prev => ({ ...prev, [a.cod]: d }))).catch(e => setErroDetalhes(e.message))
    }
    if (centralizar && tam.w) {
      const [px, py] = projetar(a.lon, a.lat)
      setVista(v => {
        const k = Math.max(v.k, k0 * 2.5)
        return { k, x: tam.w * 0.62 - px * k, y: tam.h / 2 - py * k }
      })
    }
  }, [api, detalhes, k0, tam])

  useEffect(() => {
    const esc = e => {
      if (e.key !== 'Escape') return
      if (mostrarSerie) setMostrarSerie(false)
      else setSelCod(null)
    }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [mostrarSerie])

  // posição do cartão: ao lado do marcador, dentro do quadro
  let posCard = null
  if (sel && vista) {
    const [sx, sy] = tela(sel)
    const W = 380, H = 520
    const esquerda = sx + 30 + W > tam.w - 70 ? sx - 30 - W : sx + 30
    posCard = {
      left: Math.max(16, Math.min(esquerda, tam.w - W - 16)),
      top: Math.max(16, Math.min(sy - 160, tam.h - H - 16)),
    }
  }

  const abrir = modulo => onAbrirModulo && sel && onAbrirModulo(modulo, { cod: sel.cod, nome: sel.nome })
  const anosSerie = det?.afluencias?.length ? `${det.afluencias[0].ano}–${det.afluencias[det.afluencias.length - 1].ano}` : '…'

  return (
    <div ref={raiz} className={`ini-root ${darkMode ? 'ini-dark' : ''}`}>
      <style>{CSS}</style>

      {vista && (
        <svg className={`ini-svg ${arrastando ? 'arrastando' : ''}`} onPointerDown={aoPressionar} onPointerMove={aoMover}
          onPointerUp={aoSoltar} onPointerCancel={aoSoltar} role="application" aria-label="Mapa dos açudes do Ceará">
          <g transform={`translate(${vista.x} ${vista.y}) scale(${vista.k})`}>
            <g className="ini-mun">{MUNICIPIOS.map(m => <path key={m.nome} d={m.d}><title>{m.nome}</title></path>)}</g>
          </g>
          {[...acudes].sort((a, b) => b.capacidade_hm3 - a.capacidade_hm3).map(a => {
            if (a.cod === selCod) return null
            const [x, y] = tela(a)
            const cls = ['ini-acude', a.multirreservatorio ? 'multi' : '', a.posicao_aproximada ? 'aprox' : '', visivelSet.has(a.cod) ? '' : 'apagado'].join(' ')
            return (
              <circle key={a.cod} className={cls} cx={x} cy={y} r={raioBase(a.capacidade_hm3) * escalaMarcador}
                role="button" tabIndex={0} aria-label={`${a.nome}, ${a.municipio}`}
                onMouseEnter={() => setHover(a.cod)} onMouseLeave={() => setHover(h => (h === a.cod ? null : h))}
                onClick={() => selecionar(a)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selecionar(a) } }} />
            )
          })}
          {sel && (() => {
            const [x, y] = tela(sel)
            const r = raioBase(sel.capacidade_hm3) * escalaMarcador
            return (
              <g>
                <circle className="ini-pulso" cx={x} cy={y} r={r + 10} />
                <circle className="ini-pulso" cx={x} cy={y} r={r + 19} style={{ opacity: 0.18 }} />
                <circle className="ini-sel" cx={x} cy={y} r={r + 2} />
              </g>
            )
          })()}
        </svg>
      )}

      {hover && hover !== selCod && vista && (() => {
        const a = acudes.find(x => x.cod === hover)
        if (!a) return null
        const [x, y] = tela(a)
        return (
          <div className="ini-tooltip" style={{ left: x + 14, top: y - 14 }}>
            {a.nome} <small>· {fmt(a.capacidade_hm3, 1)} hm³ · {a.municipio}</small>
          </div>
        )
      })()}

      <div className="ini-flut ini-busca">
        <div className="ini-campo">
          <Search size={15} color="var(--text-light)" />
          <input value={busca} placeholder="Buscar açude ou município…" aria-label="Buscar açude ou município"
            onChange={e => { setBusca(e.target.value); setBuscaAberta(true) }} onFocus={() => setBuscaAberta(true)}
            onKeyDown={e => { if (e.key === 'Enter' && sugestoes[0]) { selecionar(sugestoes[0], true); setBuscaAberta(false) } }} />
          {busca && <button className="ini-fechar" onClick={() => setBusca('')} aria-label="Limpar busca"><X size={14} /></button>}
        </div>
        {buscaAberta && sugestoes.length > 0 && (
          <div className="ini-sug">
            {sugestoes.map(a => (
              <button key={a.cod} onClick={() => { selecionar(a, true); setBuscaAberta(false); setBusca(a.nome) }}>
                <span>{a.nome}</span><small>{a.municipio}</small>
              </button>
            ))}
          </div>
        )}
        <div className="ini-chips">
          {[['todos', `Todos (${acudes.length})`], ['hidro', 'Hidrossistemas'], ['niveis', 'Com níveis meta']].map(([id, r]) => (
            <button key={id} className={`ini-chip ${filtro === id ? 'on' : ''}`} aria-pressed={filtro === id} onClick={() => setFiltro(id)}>{r}</button>
          ))}
        </div>
      </div>

      <div className="ini-flut ini-zoom">
        <button onClick={() => zoomEm(1.5, tam.w / 2, tam.h / 2)} aria-label="Aproximar"><Plus size={18} /></button>
        <button onClick={() => zoomEm(1 / 1.5, tam.w / 2, tam.h / 2)} aria-label="Afastar"><Minus size={18} /></button>
        <button onClick={() => setVista(vistaInicial())} aria-label="Ver o estado inteiro"><Crosshair size={16} /></button>
      </div>

      <div className="ini-flut ini-legenda">
        <b>LEGENDA</b>
        <div className="ini-leg"><span className="ini-bola" style={{ background: 'var(--orange)' }} /> Açude monitorado</div>
        <div className="ini-leg"><span className="ini-bola" style={{ background: 'var(--teal)' }} /> Integra hidrossistema (Série/Paralelo)</div>
        <div className="ini-leg"><span className="ini-bola" style={{ background: 'transparent', border: '1.5px dashed var(--orange-deep)', boxShadow: 'none' }} /> Posição aproximada (sede municipal)</div>
        <div className="ini-leg" style={{ marginTop: 8 }}>
          {[1, 50, 2000].map(c => <span key={c} className="ini-bola" style={{ width: raioBase(c) * 2, height: raioBase(c) * 2, background: 'var(--orange-pale)' }} />)}
          <span>tamanho = capacidade</span>
        </div>
      </div>

      <div className="ini-flut ini-dica">Arraste para mover · role ou use + e − para aproximar · clique em um açude</div>

      {sel && posCard && (
        <div className="ini-flut ini-card" style={posCard} role="dialog" aria-label={`Açude ${sel.nome}`}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div><h3>{sel.nome}</h3><div className="local">{sel.municipio} · código {sel.cod}</div></div>
            <button className="ini-fechar" onClick={() => setSelCod(null)} aria-label="Fechar"><X size={18} /></button>
          </div>
          <div className="ini-dados">
            <div><small>Capacidade</small><b>{fmt(sel.capacidade_hm3, 1)} hm³</b></div>
            <div><small>Estação evap.</small><b>{det ? det.estacao_evaporacao : '…'}</b></div>
            <div><small>Série</small><b>{anosSerie}</b></div>
          </div>
          {sel.posicao_aproximada && <div className="ini-aviso">Posição aproximada: o açude está marcado na sede do município.</div>}
          {sel.hidrossistemas.length > 0 && (
            <div className="ini-sistemas">Hidrossistema: {sel.hidrossistemas.map(h => `${h.nome} (${h.operacao})`).join(', ')}</div>
          )}
          <div className="ini-mini">
            <span>Evaporação mensal (mm){det ? ` · estação ${det.estacao_evaporacao} · ${fmt(det.evaporacao_mm.reduce((a, b) => a + b, 0), 0)} mm/ano` : ''}</span>
            {erroDetalhes ? <div style={{ fontSize: 12, color: 'var(--red, #d94040)', marginTop: 6 }}>{erroDetalhes}</div>
              : det ? <GraficoEvaporacao valores={det.evaporacao_mm} /> : <div style={{ height: 70, display: 'grid', placeItems: 'center', fontSize: 12, color: 'var(--text-light)' }}>Carregando…</div>}
          </div>
          <button className="ini-btserie" disabled={!det} onClick={() => setMostrarSerie(true)}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}><BarChart3 size={15} /> Ver série de afluências</span>
            <small>mensal, {anosSerie} →</small>
          </button>
          <div className="ini-acoes">
            <button className="ini-acao pri" onClick={() => abrir('simulador')}><span className="ic"><IconeLinha size={15} /></span><span><b>Simular</b><small>balanço hídrico mês a mês</small></span></button>
            <button className="ini-acao" onClick={() => abrir('otimizador')}><span className="ic"><Settings2 size={15} /></span><span><b>Otimizar</b><small>níveis meta</small></span></button>
            <button className="ini-acao" onClick={() => abrir('vazoes')}><span className="ic"><ShieldCheck size={15} /></span><span><b>Vazões</b><small>de garantia</small></span></button>
            <button className="ini-acao" onClick={() => abrir('previsao')}><span className="ic"><CloudRain size={15} /></span><span><b>Previsão</b><small>de vazões</small></span></button>
          </div>
        </div>
      )}

      {mostrarSerie && det && (
        <PainelSerie detalhes={det} onFechar={() => setMostrarSerie(false)} onUsarNoSimulador={() => { setMostrarSerie(false); abrir('simulador') }} />
      )}

      {erro && (
        <div className="ini-flut ini-erro">
          <b>Não foi possível carregar os açudes.</b>
          <div style={{ color: 'var(--text-light)', marginTop: 6 }}>{erro}</div>
        </div>
      )}
    </div>
  )
}
