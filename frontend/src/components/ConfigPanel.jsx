import React, { useState, useEffect } from 'react'
import { Plus, Trash2, Database, Calendar, Settings2, Zap, ChevronDown } from 'lucide-react'

const MESES = ['JAN','FEV','MAR','ABR','MAI','JUN','JUL','AGO','SET','OUT','NOV','DEZ']
const ANOS = Array.from({ length: 110 }, (_, i) => 1911 + i)

function Label({ icon: Icon, children }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 7,
      fontSize: 10.5, fontWeight: 700, color: 'var(--text-light)',
      textTransform: 'uppercase', letterSpacing: '0.06em',
      marginBottom: 8, marginTop: 16,
    }}>
      {Icon && <Icon size={12} strokeWidth={2.5} />}
      {children}
    </div>
  )
}

function FormControl({ as = 'input', children, style, ...props }) {
  const base = {
    width: '100%', padding: '8px 11px',
    border: '1.5px solid var(--border)',
    borderRadius: 'var(--radius-xs)',
    background: '#fff', color: 'var(--text)',
    fontSize: 13, fontFamily: 'Sora, sans-serif', outline: 'none',
    transition: 'border-color 0.15s',
    ...style,
  }
  const focusStyle = e => e.target.style.borderColor = 'var(--orange)'
  const blurStyle = e => e.target.style.borderColor = 'var(--border)'

  if (as === 'select') {
    return (
      <select style={{ ...base, appearance: 'none', cursor: 'pointer' }} onFocus={focusStyle} onBlur={blurStyle} {...props}>
        {children}
      </select>
    )
  }
  return <input style={base} onFocus={focusStyle} onBlur={blurStyle} {...props} />
}

function ReservatorioCard({ res, index, reservatorios, onChange, onRemove }) {
  const [open, setOpen] = useState(true)
  const pct = res.capacidade > 0 ? ((res.vol_inicial / res.capacidade) * 100).toFixed(1) : 0

  return (
    <div style={{
      background: 'var(--bg)', border: '1.5px solid var(--border)',
      borderRadius: 'var(--radius-sm)', marginBottom: 8, overflow: 'hidden',
    }}>
      {/* header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '10px 12px', cursor: 'pointer',
        borderBottom: open ? '1.5px solid var(--border-light)' : 'none',
      }} onClick={() => setOpen(!open)}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 22, height: 22, borderRadius: '50%',
            background: 'var(--orange-pale)', border: '1.5px solid var(--orange-light)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 10, fontWeight: 800, color: 'var(--orange-deep)',
            flexShrink: 0,
          }}>{index + 1}</div>
          <span style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--orange-deep)' }}>
            {res.nome || `Reservatório ${index + 1}`}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {index > 0 && (
            <button onClick={e => { e.stopPropagation(); onRemove(index) }} style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--text-light)', padding: 3, borderRadius: 4,
            }}
              onMouseEnter={e => e.currentTarget.style.color = 'var(--red)'}
              onMouseLeave={e => e.currentTarget.style.color = 'var(--text-light)'}
            >
              <Trash2 size={13} />
            </button>
          )}
          <ChevronDown size={14} color="var(--text-light)" style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
        </div>
      </div>

      {open && (
        <div style={{ padding: '12px 12px 14px' }}>
          {/* Reservatório select */}
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-light)', marginBottom: 4, fontWeight: 600 }}>Reservatório</div>
            <FormControl as="select" value={res.nome} onChange={e => {
              const sel = reservatorios.find(r => r.CORPO === e.target.value)
              onChange(index, {
                nome: e.target.value,
                cod: sel?.COD || '',
                capacidade: sel ? parseFloat(sel['CAPAC (m³)']) : 0,
                est_evap: sel?.['Est. Evap.'] || '',
                vol_inicial: sel ? parseFloat(sel['CAPAC (m³)']) * 0.5 : 0,
              })
            }}>
              <option value="">Selecione...</option>
              {reservatorios.map(r => <option key={r.COD} value={r.CORPO}>{r.CORPO}</option>)}
            </FormControl>
            {res.capacidade > 0 && (
              <div style={{ fontSize: 10.5, color: 'var(--text-light)', marginTop: 3, fontFamily: 'Sora, sans-serif' }}>
                Cap. Máx: {res.capacidade.toFixed(2)} hm³ • COD: {res.cod}
              </div>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {/* Vol Inicial */}
            <div>
              <div style={{ fontSize: 10.5, color: 'var(--text-light)', marginBottom: 4, fontWeight: 600 }}>Vol. Inicial (%)</div>
              <FormControl
                type="number" min="0" max="100" step="1"
                value={res.volPct ?? 50}
                onChange={e => {
                  const p = parseFloat(e.target.value) || 0
                  onChange(index, { volPct: p, vol_inicial: (res.capacidade * p) / 100 })
                }}
              />
              {res.capacidade > 0 && (
                <div style={{ fontSize: 10, color: 'var(--text-light)', marginTop: 3, fontFamily: 'Sora, sans-serif' }}>
                  = {((res.capacidade * (res.volPct ?? 50)) / 100).toFixed(2)} hm³
                </div>
              )}
            </div>

            {/* Demanda */}
            <div>
              <div style={{ fontSize: 10.5, color: 'var(--text-light)', marginBottom: 4, fontWeight: 600 }}>Demanda (m³/s)</div>
              <FormControl
                type="number" min="0" step="0.1"
                value={res.demanda}
                onChange={e => onChange(index, { demanda: parseFloat(e.target.value) || 0 })}
              />
            </div>

            {/* Gatilho */}
            <div>
              <div style={{ fontSize: 10.5, color: 'var(--text-light)', marginBottom: 4, fontWeight: 600 }}>Gatilho (%)</div>
              <FormControl
                type="number" min="0" max="100" step="1"
                value={res.gatilho}
                onChange={e => onChange(index, { gatilho: parseFloat(e.target.value) || 0 })}
              />
            </div>

            {/* Est Evap */}
            <div>
              <div style={{ fontSize: 10.5, color: 'var(--text-light)', marginBottom: 4, fontWeight: 600 }}>Est. Evap.</div>
              <FormControl
                type="text"
                value={res.est_evap}
                onChange={e => onChange(index, { est_evap: e.target.value })}
                placeholder="Código"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default function ConfigPanel({ reservatorios: resList, presets, onSimulate, loading }) {
  const [items, setItems] = useState([{
    nome: '', cod: '', capacidade: 0, est_evap: '',
    volPct: 50, vol_inicial: 0, demanda: 0.5, gatilho: 30,
  }])
  const [modo, setModo] = useState('Individual')
  const [vazaoConjunta, setVazaoConjunta] = useState(0)
  const [mesIni, setMesIni] = useState('JAN')
  const [anoIni, setAnoIni] = useState(1911)
  const [mesFim, setMesFim] = useState('DEZ')
  const [anoFim, setAnoFim] = useState(1915)
  const [presetSel, setPresetSel] = useState('')

  const handleChange = (idx, patch) => {
    setItems(prev => prev.map((it, i) => i === idx ? { ...it, ...patch } : it))
  }

  const addReservatorio = () => {
    setItems(prev => [...prev, { nome: '', cod: '', capacidade: 0, est_evap: '', volPct: 50, vol_inicial: 0, demanda: 0.5, gatilho: 30 }])
  }

  const removeReservatorio = (idx) => {
    setItems(prev => prev.filter((_, i) => i !== idx))
  }

  const applyPreset = (nome) => {
    const p = presets.find(x => x.nome === nome)
    if (!p) return
    setModo(p.modo)
    const newItems = p.reservatorios.map(cod => {
      const found = resList.find(r => r.COD === cod || r.CORPO === cod)
      return {
        nome: found?.CORPO || cod,
        cod: found?.COD || cod,
        capacidade: found ? parseFloat(found['CAPAC (m³)']) : 0,
        est_evap: found?.['Est. Evap.'] || '',
        volPct: 50,
        vol_inicial: found ? parseFloat(found['CAPAC (m³)']) * 0.5 : 0,
        demanda: 0.5, gatilho: 30,
      }
    })
    setItems(newItems)
  }

  const handleSubmit = () => {
    const payload = {
      reservatorios: items.map(it => ({
        nome: String(it.nome || ''),
        cod: String(it.cod || ''),
        capacidade: parseFloat(it.capacidade) || 0,
        est_evap: String(it.est_evap ?? ''),
        vol_inicial: parseFloat(it.vol_inicial) || 0,
        demanda: parseFloat(it.demanda) || 0,
        gatilho: parseFloat(it.gatilho) || 0,
      })),
      modo: String(modo),
      vazao_conjunta: parseFloat(vazaoConjunta) || 0,
      mes_inicial: String(mesIni),
      ano_inicial: parseInt(anoIni),
      mes_final: String(mesFim),
      ano_final: parseInt(anoFim),
    }
    console.log('Payload:', JSON.stringify(payload, null, 2))
    onSimulate(payload)
  }

  const nomeAtual = items[0]?.nome || 'Nenhum selecionado'

  return (
    <div style={{
      background: 'var(--card)', border: '1.5px solid var(--border)',
      borderRadius: 'var(--radius)', padding: '22px 18px',
      boxShadow: 'var(--shadow)', position: 'sticky', top: 20,
    }}>
      <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--text)', marginBottom: 3 }}>Configuração</div>
      <div style={{ fontSize: 11.5, color: 'var(--text-light)', marginBottom: 18 }}>
        Cenário para <strong style={{ color: 'var(--orange-deep)' }}>{nomeAtual}</strong>
      </div>

      {/* Preset */}
      {presets.length > 0 && (
        <>
          <Label icon={Zap}>Hidrossistema (Preset)</Label>
          <FormControl as="select" value={presetSel} onChange={e => { setPresetSel(e.target.value); applyPreset(e.target.value) }}>
            <option value="">Configuração manual...</option>
            {presets.map(p => <option key={p.nome} value={p.nome}>{p.nome} ({p.modo})</option>)}
          </FormControl>
        </>
      )}

      {/* Reservatórios */}
      <Label icon={Database}>Reservatórios Selecionados</Label>
      {items.map((res, i) => (
        <ReservatorioCard
          key={i} res={res} index={i}
          reservatorios={resList}
          onChange={handleChange}
          onRemove={removeReservatorio}
        />
      ))}
      <button onClick={addReservatorio} style={{
        width: '100%', padding: '8px',
        background: 'none', border: '1.5px dashed var(--border)',
        borderRadius: 'var(--radius-sm)', color: 'var(--text-light)',
        fontSize: 12, cursor: 'pointer', marginBottom: 4, transition: 'all 0.15s',
      }}
        onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--orange)'; e.currentTarget.style.color = 'var(--orange)'; e.currentTarget.style.background = 'var(--orange-pale)' }}
        onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-light)'; e.currentTarget.style.background = 'none' }}
      >
        <Plus size={12} style={{ marginRight: 5 }} /> Adicionar Reservatório
      </button>

      {/* Modo Operação */}
      <Label icon={Settings2}>Modo de Operação</Label>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6, marginBottom: 4 }}>
        {['Individual', 'Série', 'Paralelo'].map(m => (
          <button key={m} onClick={() => setModo(m)} style={{
            padding: '8px 4px',
            border: `1.5px solid ${modo === m ? 'var(--orange)' : 'var(--border)'}`,
            borderRadius: 'var(--radius-xs)',
            background: modo === m ? 'var(--orange-pale)' : 'none',
            color: modo === m ? 'var(--orange-deep)' : 'var(--text-light)',
            fontSize: 11.5, fontWeight: 700, cursor: 'pointer',
            transition: 'all 0.15s',
          }}>{m}</button>
        ))}
      </div>

      {/* Vazão Conjunta */}
      {modo !== 'Individual' && (
        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 10.5, color: 'var(--text-light)', marginBottom: 4, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Vazão Conjunta (m³/s)
          </div>
          <FormControl
            type="number" min="0" step="0.1" value={vazaoConjunta}
            onChange={e => setVazaoConjunta(e.target.value)}
          />
        </div>
      )}

      {/* Período */}
      <Label icon={Calendar}>Período</Label>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <div>
          <div style={{ fontSize: 10.5, color: 'var(--text-light)', marginBottom: 4, fontWeight: 600 }}>Início</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
            <FormControl as="select" value={mesIni} onChange={e => setMesIni(e.target.value)} style={{ fontSize: 11.5 }}>
              {MESES.map(m => <option key={m}>{m}</option>)}
            </FormControl>
            <FormControl as="select" value={anoIni} onChange={e => setAnoIni(e.target.value)} style={{ fontSize: 11.5 }}>
              {ANOS.map(a => <option key={a}>{a}</option>)}
            </FormControl>
          </div>
        </div>
        <div>
          <div style={{ fontSize: 10.5, color: 'var(--text-light)', marginBottom: 4, fontWeight: 600 }}>Fim</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
            <FormControl as="select" value={mesFim} onChange={e => setMesFim(e.target.value)} style={{ fontSize: 11.5 }}>
              {MESES.map(m => <option key={m}>{m}</option>)}
            </FormControl>
            <FormControl as="select" value={anoFim} onChange={e => setAnoFim(e.target.value)} style={{ fontSize: 11.5 }}>
              {ANOS.map(a => <option key={a}>{a}</option>)}
            </FormControl>
          </div>
        </div>
      </div>

      {/* Simulate button */}
      <button onClick={handleSubmit} disabled={loading || !items[0].nome} style={{
        width: '100%', marginTop: 20, padding: 13,
        background: loading ? 'var(--border)' : 'linear-gradient(135deg, var(--orange), var(--orange-deep))',
        border: 'none', borderRadius: 'var(--radius-sm)',
        color: loading ? 'var(--text-light)' : '#fff',
        fontSize: 13.5, fontWeight: 800, cursor: loading ? 'not-allowed' : 'pointer',
        boxShadow: loading ? 'none' : '0 4px 18px var(--orange-glow)',
        transition: 'all 0.2s', letterSpacing: '0.02em',
      }}
        onMouseEnter={e => { if (!loading) e.currentTarget.style.transform = 'translateY(-1px)' }}
        onMouseLeave={e => { e.currentTarget.style.transform = 'none' }}
      >
        {loading ? '⏳ Simulando...' : '▶ Gerar Simulação'}
      </button>
    </div>
  )
}
