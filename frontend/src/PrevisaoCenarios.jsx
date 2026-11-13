import { useEffect, useMemo, useState } from 'react'
import {
  Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { CloudRain, RefreshCw, Search, Shuffle } from 'lucide-react'
import { VARS_ESCURO } from './tema'
import PrevisaoVazoes from './PrevisaoVazoes'

const MESES = ['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN', 'JUL', 'AGO', 'SET', 'OUT', 'NOV', 'DEZ']
const EMISSOES = [
  { mes: 10, rotulo: 'Outubro (ano anterior)' },
  { mes: 11, rotulo: 'Novembro (ano anterior)' },
  { mes: 12, rotulo: 'Dezembro (ano anterior)' },
  { mes: 1, rotulo: 'Janeiro' },
]
const NOMES_CONJUNTOS = {
  nino: 'Niño 3.4',
  atlantico: 'Gradiente do Atlântico (TNA − TSA)',
  ambos: 'Niño 3.4 + gradiente do Atlântico',
  ambos_quadra: 'Niño 3.4 + gradiente + quadra anterior',
}

const ESTILO = `.pc-root{--grade:#ecdcc8;--card-2:#fff;--campo:#fff;--serie-1:#264fa3;--serie-2:#e07b2a;--serie-3:#2a9d8f;--bg:#fdf6ee;--orange:#e07b2a;--orange-pale:#fdebd3;--orange-deep:#c46318;--teal:#2a9d8f;--teal-pale:#d4f5ef;--blue:#264fa3;--red:#d94040;--red-pale:#fde8e8;--text:#1e1208;--text-mid:#5a3c24;--text-light:#9a7055;--border:#ecdcc8;--border-light:#f5ebe0;--card:#fff;--shadow:0 2px 16px rgba(150,90,40,.10);--radius:14px;--radius-sm:9px;--radius-xs:6px;font-family:'Sora',sans-serif;background:var(--bg);color:var(--text)}.pc-dark{${VARS_ESCURO}}.pc-tabbar{display:flex;gap:4px;background:var(--card);border:1.5px solid var(--border);border-radius:9px;padding:3px;width:fit-content}.pc-tab{border:0;border-radius:7px;background:transparent;color:var(--text-light);font:900 12px 'Sora',sans-serif;padding:8px 13px;cursor:pointer}.pc-tab.on{background:var(--orange-pale);color:var(--orange-deep)}.pc-layout{display:grid;grid-template-columns:340px minmax(0,1fr);gap:16px;align-items:start}.pc-btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;border:0;border-radius:9px;padding:10px 13px;font-size:12px;font-weight:900;cursor:pointer;font-family:'Sora',sans-serif;background:linear-gradient(135deg,var(--orange),var(--orange-deep));color:#fff;width:100%}.pc-btn:disabled{cursor:not-allowed;opacity:.6}.pc-section{border-top:1.5px solid var(--border-light);padding-top:12px;margin-top:12px;display:flex;flex-direction:column;gap:10px}.pc-table{width:100%;border-collapse:collapse;font-size:12px}.pc-table th{background:var(--card);color:var(--text-light);text-align:left;padding:7px 8px;border-bottom:1.5px solid var(--border-light)}.pc-table td{padding:7px 8px;border-bottom:1px solid var(--border-light)}.pc-metricas{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}@keyframes pc-spin{to{transform:rotate(360deg)}}.pc-spin{animation:pc-spin 1s linear infinite}@media(max-width:920px){.pc-layout{grid-template-columns:1fr}.pc-tabbar{width:100%}.pc-tab{flex:1}}`

function apiBase(apiUrl) {
  return apiUrl || import.meta.env?.VITE_API_URL || 'http://127.0.0.1:8000'
}

function Card({ children, style }) {
  return <div style={{ background: 'var(--card)', border: '1.5px solid var(--border)', borderRadius: 'var(--radius)', boxShadow: 'var(--shadow)', ...style }}>{children}</div>
}

function Field({ label, children, dica }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <span style={{ fontSize: 10.5, fontWeight: 800, color: 'var(--text-light)', textTransform: 'uppercase' }}>{label}</span>
      {children}
      {dica && <span style={{ fontSize: 10.5, color: 'var(--text-light)' }}>{dica}</span>}
    </label>
  )
}

const estiloControle = {
  width: '100%', border: '1.5px solid var(--border)', borderRadius: 'var(--radius-xs)', background: 'var(--card)',
  color: 'var(--text)', padding: '8px 10px', font: "700 12px 'Sora', sans-serif", outline: 'none',
}

function Metrica({ rotulo, valor, sub }) {
  return (
    <Card style={{ padding: 14 }}>
      <div style={{ fontSize: 10.5, fontWeight: 900, color: 'var(--text-light)', textTransform: 'uppercase' }}>{rotulo}</div>
      <div style={{ fontSize: 21, fontWeight: 900, color: 'var(--orange-deep)', marginTop: 5 }}>{valor}</div>
      {sub && <div style={{ fontSize: 11, color: 'var(--text-light)', marginTop: 2 }}>{sub}</div>}
    </Card>
  )
}

function fmt(v, casas = 1) {
  return v === null || v === undefined || !Number.isFinite(Number(v)) ? '—' : Number(v).toLocaleString('pt-BR', { minimumFractionDigits: casas, maximumFractionDigits: casas })
}

function Dica({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const linha = (nome, v, cor) => (
    <div style={{ display: 'flex', gap: 8, justifyContent: 'space-between' }}>
      <span style={{ color: cor, fontWeight: 800 }}>{nome}</span><span style={{ fontFamily: 'JetBrains Mono' }}>{fmt(v, 2)}</span>
    </div>
  )
  return (
    <div style={{ background: 'var(--card)', border: '1.5px solid var(--border)', borderRadius: 8, padding: '8px 10px', boxShadow: 'var(--shadow)', fontSize: 11, minWidth: 190 }}>
      <div style={{ fontWeight: 900, marginBottom: 4 }}>{label} (hm³)</div>
      {linha('Previsão P10', d.faixa?.[0], 'var(--orange-deep)')}
      {linha('Previsão mediana', d.mediana, 'var(--orange-deep)')}
      {linha('Previsão P90', d.faixa?.[1], 'var(--orange-deep)')}
      {linha('Climatologia mediana', d.climMediana, 'var(--blue)')}
      {d.observado !== undefined && d.observado !== null && linha('Observado', d.observado, 'var(--teal)')}
    </div>
  )
}

function CenariosKnn({ apiUrl, acudeInicial }) {
  const base = apiBase(apiUrl)
  const [reservatorios, setReservatorios] = useState([])
  const [busca, setBusca] = useState('')
  const [reservatorio, setReservatorio] = useState('')
  const [mesEmissao, setMesEmissao] = useState(12)
  const [anoAlvo, setAnoAlvo] = useState('')
  const [k, setK] = useState('')
  const [conjunto, setConjunto] = useState('ambos')
  const [nCenarios, setNCenarios] = useState(1000)
  const [semente, setSemente] = useState(1)
  const [resultado, setResultado] = useState(null)
  const [carregando, setCarregando] = useState(false)
  const [erro, setErro] = useState('')

  useEffect(() => {
    fetch(`${base}/api/reservatorios`)
      .then(r => r.json())
      .then(lista => {
        const nomes = [...new Set(lista.map(r => r.CORPO))].sort((a, b) => a.localeCompare(b, 'pt-BR'))
        setReservatorios(nomes)
        const inicial = nomes.includes(acudeInicial?.nome) ? acudeInicial.nome : (nomes.includes('Carnaubal') ? 'Carnaubal' : nomes[0])
        setReservatorio(inicial || '')
        setBusca(inicial || '')
      })
      .catch(() => setErro('Não foi possível carregar a lista de reservatórios.'))
  }, [base, acudeInicial])

  const filtrados = useMemo(() => reservatorios.filter(n => n.toLowerCase().includes(busca.toLowerCase())).slice(0, 40), [reservatorios, busca])

  async function prever() {
    setCarregando(true)
    setErro('')
    try {
      const corpo = {
        reservatorio, mes_emissao: Number(mesEmissao), conjunto,
        n_cenarios: Number(nCenarios) || 1000, semente: Number(semente) || 1, incluir_cenarios: false,
      }
      if (anoAlvo !== '') corpo.ano_alvo = Number(anoAlvo)
      if (k !== '') corpo.k = Number(k)
      const r = await fetch(`${base}/api/previsao-knn/cenarios`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(corpo),
      })
      const dados = await r.json().catch(() => ({}))
      if (!r.ok) {
        const det = dados.detail
        throw new Error(typeof det === 'string' ? det : (Array.isArray(det) ? det.map(d => d.msg).join(' ') : 'Falha na previsão.'))
      }
      setResultado(dados)
    } catch (e) {
      setResultado(null)
      setErro(e.message)
    } finally {
      setCarregando(false)
    }
  }

  const grafico = useMemo(() => {
    if (!resultado) return []
    const q = resultado.quantis_mensais
    const c = resultado.climatologia.quantis_mensais
    return resultado.meses.map((mes, i) => ({
      mes,
      faixa: [q.p10[i], q.p90[i]],
      mediana: q.p50[i],
      climFaixa: [c.p10[i], c.p90[i]],
      climMediana: c.p50[i],
      observado: resultado.observado ? resultado.observado[i] : null,
    }))
  }, [resultado])

  const quadra = resultado?.quantis_quadra
  const quadraClim = resultado?.climatologia?.quantis_quadra

  return (
    <div className="pc-layout">
      <Card style={{ padding: 16, position: 'sticky', top: 14 }}>
        <Field label="Reservatório">
          <div style={{ position: 'relative' }}>
            <Search size={13} style={{ position: 'absolute', left: 10, top: 11, color: 'var(--text-light)' }} />
            <input value={busca} onChange={e => setBusca(e.target.value)} placeholder="Digite para buscar..." style={{ ...estiloControle, paddingLeft: 30 }} />
            {busca && busca !== reservatorio && filtrados.length > 0 && (
              <div style={{ position: 'absolute', zIndex: 20, left: 0, right: 0, top: 'calc(100% + 4px)', maxHeight: 220, overflow: 'auto', background: 'var(--card)', border: '1.5px solid var(--border)', borderRadius: 'var(--radius-xs)', boxShadow: 'var(--shadow)', padding: 4 }}>
                {filtrados.map(nome => (
                  <button key={nome} type="button" onClick={() => { setReservatorio(nome); setBusca(nome); setResultado(null) }}
                    style={{ display: 'block', width: '100%', textAlign: 'left', border: 0, background: 'transparent', color: 'var(--text)', padding: '7px 9px', borderRadius: 6, cursor: 'pointer', fontWeight: 800 }}>
                    {nome}
                  </button>
                ))}
              </div>
            )}
          </div>
        </Field>

        <div className="pc-section">
          <Field label="Mês de emissão" dica="Os preditores são as médias dos três meses anteriores à emissão.">
            <select value={mesEmissao} onChange={e => setMesEmissao(Number(e.target.value))} style={{ ...estiloControle, cursor: 'pointer' }}>
              {EMISSOES.map(o => <option key={o.mes} value={o.mes}>{o.rotulo}</option>)}
            </select>
          </Field>
          <Field label="Ano-alvo (quadra)" dica="Vazio: o ano mais recente com todos os preditores. Se o ano já está no histórico, ele é retirado antes da previsão.">
            <input type="number" value={anoAlvo} onChange={e => setAnoAlvo(e.target.value)} placeholder="automático" style={estiloControle} />
          </Field>
          <Field label="Preditores">
            <select value={conjunto} onChange={e => setConjunto(e.target.value)} style={{ ...estiloControle, cursor: 'pointer' }}>
              {Object.entries(NOMES_CONJUNTOS).map(([id, nome]) => <option key={id} value={id}>{nome}</option>)}
            </select>
          </Field>
        </div>

        <div className="pc-section">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
            <Field label="k">
              <input type="number" min="1" value={k} onChange={e => setK(e.target.value)} placeholder="√n" style={estiloControle} />
            </Field>
            <Field label="Cenários">
              <input type="number" min="10" max="10000" value={nCenarios} onChange={e => setNCenarios(e.target.value)} style={estiloControle} />
            </Field>
            <Field label="Semente">
              <input type="number" value={semente} onChange={e => setSemente(e.target.value)} style={estiloControle} />
            </Field>
          </div>
        </div>

        <button className="pc-btn" onClick={prever} disabled={carregando || !reservatorio} style={{ marginTop: 16 }}>
          {carregando ? <RefreshCw size={14} className="pc-spin" /> : <Shuffle size={14} />}
          {carregando ? 'Calculando...' : 'Gerar cenários'}
        </button>
        <div style={{ marginTop: 10, fontSize: 10.5, lineHeight: 1.45, color: 'var(--text-light)' }}>
          Reamostragem dos k anos mais parecidos (Lall e Sharma, 1996), com pesos 1/j. Cada cenário é a sequência mensal observada de um ano sorteado.
        </div>
      </Card>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minWidth: 0 }}>
        {erro && <div style={{ padding: '9px 13px', borderRadius: 'var(--radius-sm)', fontSize: 12, fontWeight: 800, background: 'var(--red-pale)', color: 'var(--red)' }}>{erro}</div>}
        {resultado?.aviso && <div style={{ padding: '9px 13px', borderRadius: 'var(--radius-sm)', fontSize: 12, fontWeight: 700, background: 'var(--orange-pale)', color: 'var(--orange-deep)' }}>{resultado.aviso}</div>}

        {!resultado && !erro && (
          <Card style={{ padding: 28, textAlign: 'center', color: 'var(--text-light)', fontSize: 12.5 }}>
            Escolha o reservatório, o mês de emissão e os preditores e clique em Gerar cenários.
          </Card>
        )}

        {resultado && (
          <>
            <div className="pc-metricas">
              <Metrica rotulo={`Quadra ${resultado.ano_alvo} prevista`} valor={`${fmt(quadra?.p50)} hm³`} sub={`P10–P90: ${fmt(quadra?.p10)} a ${fmt(quadra?.p90)} hm³`} />
              <Metrica rotulo="Climatologia da quadra" valor={`${fmt(quadraClim?.p50)} hm³`} sub={`P10–P90: ${fmt(quadraClim?.p10)} a ${fmt(quadraClim?.p90)} hm³`} />
              {resultado.quadra_observada !== null && <Metrica rotulo="Quadra observada" valor={`${fmt(resultado.quadra_observada)} hm³`} sub="ano retirado do histórico" />}
              <Metrica rotulo="Histórico" valor={`${resultado.historico.anos_usados} anos`} sub={`${resultado.historico.primeiro_ano}–${resultado.historico.ultimo_ano}; k = ${resultado.k}; ${resultado.historico.anos_excluidos} ano(s) excluído(s)`} />
            </div>

            <Card style={{ padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 900 }}>Afluência mensal prevista — {resultado.reservatorio}, emissão em {resultado.emissao}</div>
              <div style={{ fontSize: 11, color: 'var(--text-light)', marginTop: 2, marginBottom: 10 }}>
                Faixa laranja: 10% a 90% dos cenários, linha laranja: mediana. Azul: climatologia (todos os anos com o mesmo peso).
              </div>
              <div style={{ height: 330 }}>
                <ResponsiveContainer>
                  <ComposedChart data={grafico} margin={{ top: 6, right: 16, left: 0, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--grade)" />
                    <XAxis dataKey="mes" tick={{ fontSize: 10, fill: 'var(--text-light)' }} />
                    <YAxis tick={{ fontSize: 10, fill: 'var(--text-light)' }} label={{ value: 'hm³', angle: -90, position: 'insideLeft', fill: 'var(--text-light)', fontSize: 10 }} />
                    <Tooltip content={<Dica />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Area type="linear" dataKey="climFaixa" name="Climatologia P10–P90" stroke="none" fill="var(--blue)" fillOpacity={0.12} isAnimationActive={false} />
                    <Area type="linear" dataKey="faixa" name="Previsão P10–P90" stroke="none" fill="var(--orange)" fillOpacity={0.28} isAnimationActive={false} />
                    <Line type="linear" dataKey="climMediana" name="Climatologia mediana" stroke="var(--blue)" strokeDasharray="5 4" strokeWidth={1.8} dot={false} isAnimationActive={false} />
                    <Line type="linear" dataKey="mediana" name="Previsão mediana" stroke="var(--orange-deep)" strokeWidth={2.4} dot={{ r: 2.5 }} isAnimationActive={false} />
                    {resultado.observado && <Line type="linear" dataKey="observado" name="Observado" stroke="var(--teal)" strokeWidth={2} dot={{ r: 2.5 }} isAnimationActive={false} />}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 14 }}>
              <Card style={{ padding: 16 }}>
                <div style={{ fontSize: 13, fontWeight: 900, marginBottom: 8 }}>Anos vizinhos</div>
                <table className="pc-table">
                  <thead><tr><th>Ordem</th><th>Ano</th><th>Distância</th><th>Peso</th></tr></thead>
                  <tbody>
                    {resultado.vizinhos.map((v, i) => (
                      <tr key={v.ano}><td>{i + 1}</td><td style={{ fontWeight: 800 }}>{v.ano}</td><td>{fmt(v.distancia, 2)}</td><td>{fmt(v.peso * 100, 1)}%</td></tr>
                    ))}
                  </tbody>
                </table>
              </Card>
              <Card style={{ padding: 16 }}>
                <div style={{ fontSize: 13, fontWeight: 900, marginBottom: 8 }}>Estado no mês de emissão</div>
                <table className="pc-table">
                  <thead><tr><th>Preditor</th><th>Valor</th><th>Padronizado</th></tr></thead>
                  <tbody>
                    {resultado.preditores.map(p => (
                      <tr key={p.nome}><td title={p.descricao}>{p.descricao}</td><td>{fmt(p.valor_atual, 2)}</td><td>{fmt(p.padronizado, 2)}</td></tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ fontSize: 10.5, color: 'var(--text-light)', marginTop: 8 }}>
                  Índices calculados da ERSST v5 (NOAA), anomalias em relação a 1991–2020.
                </div>
              </Card>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default function PainelPrevisao({ apiUrl, acudeInicial, darkMode = false }) {
  const [aba, setAba] = useState('cenarios')
  return (
    <div className={`pc-root ${darkMode ? 'pc-dark' : ''}`} style={{ minHeight: 600 }}>
      <style>{ESTILO}</style>
      <div style={{ padding: '18px 26px 0' }}>
        <div className="pc-tabbar">
          <button className={`pc-tab ${aba === 'cenarios' ? 'on' : ''}`} onClick={() => setAba('cenarios')}>Cenários da quadra (k-NN)</button>
          <button className={`pc-tab ${aba === 'regressao' ? 'on' : ''}`} onClick={() => setAba('regressao')}>Regressão (KNN e XGBoost)</button>
        </div>
      </div>
      {aba === 'cenarios' ? (
        <div style={{ padding: '14px 26px 48px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 4 }}>
            <CloudRain size={22} color="var(--orange)" />
            <h2 style={{ margin: 0, fontSize: 19, fontWeight: 900 }}>Cenários de afluência da quadra chuvosa</h2>
          </div>
          <p style={{ margin: '0 0 14px', color: 'var(--text-light)', fontSize: 11.5 }}>
            Previsão por reamostragem dos anos com estado climático mais parecido, com a climatologia como referência.
          </p>
          <CenariosKnn apiUrl={apiUrl} acudeInicial={acudeInicial} />
        </div>
      ) : (
        <PrevisaoVazoes apiUrl={apiUrl} acudeInicial={acudeInicial} darkMode={darkMode} mode="knn" />
      )}
    </div>
  )
}
