import React, { useState } from 'react'
import { Download, ChevronLeft, ChevronRight } from 'lucide-react'

const COLS = [
  { key: 'Data', label: 'Data', align: 'left' },
  { key: 'Armazenamento Inicial', label: 'Vol. Ini (hm³)', mono: true, colorClass: 'vol-ini' },
  { key: 'Afluências (hm³/mês)', label: 'Afluência (hm³)', mono: true },
  { key: 'Evaporação (hm³)', label: 'Evap. (hm³)', mono: true },
  { key: 'Demanda Solicitada (m³/s)', label: 'Dem. Sol. (m³/s)', mono: true },
  { key: 'Demanda Atendida (m³/s)', label: 'Dem. At. (m³/s)', mono: true },
  { key: 'Retirada Total (m³/s)', label: 'Retirada Total (m³/s)', mono: true },
  { key: 'Transferência Recebida (m³/s)', label: 'Transf. Rec.', mono: true },
  { key: 'Transferência Enviada (m³/s)', label: 'Transf. Env.', mono: true },
  { key: 'Racionamento (%)', label: 'Rac. (%)', mono: true },
  { key: 'Vertimento (hm³)', label: 'Vertimento', mono: true },
  { key: 'Armazenamento Final', label: 'Vol. Fin. (hm³)', mono: true, colorClass: 'vol-fin' },
  { key: 'Falha', label: 'Falha', align: 'center' },
  { key: 'Modo Operação', label: 'Modo', align: 'center' },
]

const PAGE_SIZE = 15

function fmt(val, key) {
  if (val === null || val === undefined || val === '') return '—'
  if (key === 'Falha') return val
  if (key === 'Modo Operação' || key === 'Data') return val
  const n = parseFloat(val)
  if (isNaN(n)) return val
  return n.toFixed(2)
}

function exportCSV(dados, nome) {
  const headers = COLS.map(c => c.label).join(',')
  const rows = dados.map(d => COLS.map(c => fmt(d[c.key], c.key)).join(','))
  const csv = [headers, ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = `simulacao_${nome}.csv`; a.click()
}

function ReservatorioTable({ resultado }) {
  const [page, setPage] = useState(0)
  const { dados, reservatorio } = resultado
  const totalPages = Math.ceil(dados.length / PAGE_SIZE)
  const pageDados = dados.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  return (
    <div style={{
      background: 'var(--card)', border: '1.5px solid var(--border)',
      borderRadius: 'var(--radius)', boxShadow: 'var(--shadow)', overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '14px 20px', borderBottom: '1.5px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'var(--bg)',
      }}>
        <div>
          <span style={{ fontSize: 13.5, fontWeight: 800, color: 'var(--text)' }}>
            {reservatorio}
          </span>
          <span style={{ fontSize: 11, color: 'var(--text-light)', marginLeft: 10 }}>
            {dados.length} registros
          </span>
        </div>
        <button onClick={() => exportCSV(dados, reservatorio)} style={{
          display: 'flex', alignItems: 'center', gap: 6,
          background: 'none', border: '1.5px solid var(--border)',
          borderRadius: 'var(--radius-xs)', padding: '6px 12px',
          fontSize: 11, fontWeight: 600, color: 'var(--text-mid)', cursor: 'pointer',
          transition: 'all 0.15s',
        }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--orange)'; e.currentTarget.style.color = 'var(--orange)' }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-mid)' }}
        >
          <Download size={12} /> Exportar CSV
        </button>
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11.5 }}>
          <thead>
            <tr style={{ background: 'var(--bg)' }}>
              {COLS.map(c => (
                <th key={c.key} style={{
                  padding: '9px 13px',
                  textAlign: c.align || 'right',
                  fontSize: 10, fontWeight: 700,
                  textTransform: 'uppercase', letterSpacing: '0.06em',
                  color: 'var(--text-light)',
                  borderBottom: '1.5px solid var(--border)',
                  whiteSpace: 'nowrap',
                }}>{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageDados.map((d, ri) => {
              const isFail = d['Falha'] === 'Sim'
              const hasRac = parseFloat(d['Racionamento (%)']) > 0
              return (
                <tr key={ri} style={{
                  background: isFail ? 'rgba(217,64,64,0.04)' : 'transparent',
                  transition: 'background 0.12s',
                }}
                  onMouseEnter={e => e.currentTarget.style.background = 'var(--orange-pale)'}
                  onMouseLeave={e => e.currentTarget.style.background = isFail ? 'rgba(217,64,64,0.04)' : 'transparent'}
                >
                  {COLS.map(c => {
                    const raw = d[c.key]
                    const val = fmt(raw, c.key)
                    let color = 'var(--text-mid)'
                    let fontWeight = 400

                    if (c.colorClass === 'vol-ini') color = 'var(--blue-light)', fontWeight = 600
                    if (c.colorClass === 'vol-fin') color = 'var(--blue)', fontWeight = 700
                    if (c.key === 'Falha') {
                      color = val === 'Sim' ? 'var(--red)' : 'var(--teal)'
                      fontWeight = 700
                    }
                    if (c.key === 'Racionamento (%)' && hasRac) color = 'var(--yellow)', fontWeight = 600
                    if (c.key === 'Data') color = 'var(--text)', fontWeight = 600

                    return (
                      <td key={c.key} style={{
                        padding: '8px 13px',
                        textAlign: c.align || 'right',
                        borderBottom: '1px solid var(--border-light)',
                        fontFamily: c.mono ? 'JetBrains Mono' : 'Sora',
                        color, fontWeight,
                        whiteSpace: 'nowrap',
                      }}>
                        {c.key === 'Falha'
                          ? <span style={{
                            display: 'inline-flex', alignItems: 'center', gap: 4,
                            padding: '2px 8px', borderRadius: 20,
                            background: val === 'Sim' ? 'var(--red-pale)' : 'var(--teal-pale)',
                            fontSize: 10.5,
                          }}>
                            {val === 'Sim' ? '✗ Sim' : '✓ Não'}
                          </span>
                          : c.key === 'Modo Operação'
                            ? <span style={{
                              display: 'inline-flex', alignItems: 'center',
                              padding: '2px 8px', borderRadius: 20,
                              background: val === 'Normal' ? 'var(--blue-pale)' : val?.includes('FALHA') ? 'var(--red-pale)' : 'var(--yellow-pale)',
                              color: val === 'Normal' ? 'var(--blue)' : val?.includes('FALHA') ? 'var(--red)' : 'var(--yellow)',
                              fontSize: 10.5, fontWeight: 600,
                            }}>{val}</span>
                            : val
                        }
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{
          padding: '12px 20px', borderTop: '1.5px solid var(--border)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span style={{ fontSize: 11, color: 'var(--text-light)' }}>
            Página {page + 1} de {totalPages} ({dados.length} registros)
          </span>
          <div style={{ display: 'flex', gap: 6 }}>
            <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0} style={{
              display: 'flex', alignItems: 'center', gap: 4,
              padding: '5px 10px', border: '1.5px solid var(--border)',
              borderRadius: 'var(--radius-xs)', background: 'none',
              fontSize: 11, fontWeight: 600, color: 'var(--text-mid)',
              cursor: page === 0 ? 'not-allowed' : 'pointer', opacity: page === 0 ? 0.4 : 1,
            }}>
              <ChevronLeft size={12} /> Anterior
            </button>
            <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1} style={{
              display: 'flex', alignItems: 'center', gap: 4,
              padding: '5px 10px', border: '1.5px solid var(--border)',
              borderRadius: 'var(--radius-xs)', background: 'none',
              fontSize: 11, fontWeight: 600, color: 'var(--text-mid)',
              cursor: page >= totalPages - 1 ? 'not-allowed' : 'pointer', opacity: page >= totalPages - 1 ? 0.4 : 1,
            }}>
              Próxima <ChevronRight size={12} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default function ResultsTable({ resultados }) {
  const [activeRes, setActiveRes] = useState(0)
  if (!resultados?.length) return null

  return (
    <div>
      {/* Tabs for multiple reservatórios */}
      {resultados.length > 1 && (
        <div style={{ display: 'flex', gap: 6, marginBottom: 14 }}>
          {resultados.map((r, i) => (
            <button key={i} onClick={() => setActiveRes(i)} style={{
              padding: '7px 16px', borderRadius: 20,
              border: `1.5px solid ${activeRes === i ? 'var(--orange)' : 'var(--border)'}`,
              background: activeRes === i ? 'var(--orange-pale)' : 'var(--card)',
              color: activeRes === i ? 'var(--orange-deep)' : 'var(--text-light)',
              fontSize: 12, fontWeight: 700, cursor: 'pointer', transition: 'all 0.15s',
            }}>
              {r.reservatorio}
            </button>
          ))}
        </div>
      )}
      <ReservatorioTable resultado={resultados[activeRes]} />
    </div>
  )
}
