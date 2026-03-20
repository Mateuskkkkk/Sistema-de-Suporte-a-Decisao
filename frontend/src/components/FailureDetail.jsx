import React from 'react'
import { AlertCircle, CheckCircle2, Info } from 'lucide-react'

export default function FailureDetail({ resultados }) {
  if (!resultados?.length) return null

  const falhas = []
  resultados.forEach(r => {
    r.dados.forEach(d => {
      if (d['Falha'] === 'Sim') {
        falhas.push({
          reservatorio: r.reservatorio,
          data: d.Data,
          volIni: parseFloat(d['Armazenamento Inicial'] || 0).toFixed(2),
          demSol: parseFloat(d['Demanda Solicitada (m³/s)'] || 0).toFixed(3),
          demAt: parseFloat(d['Demanda Atendida (m³/s)'] || 0).toFixed(3),
          rac: parseFloat(d['Racionamento (%)'] || 0).toFixed(1),
          modo: d['Modo Operação'],
        })
      }
    })
  })

  return (
    <div style={{
      background: 'var(--card)', border: `1.5px solid ${falhas.length > 0 ? 'var(--red-pale)' : 'var(--teal-pale)'}`,
      borderRadius: 'var(--radius)', padding: '18px 22px', boxShadow: 'var(--shadow)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: falhas.length ? 16 : 0 }}>
        {falhas.length > 0
          ? <AlertCircle size={18} color="var(--red)" />
          : <CheckCircle2 size={18} color="var(--teal)" />
        }
        <div>
          <div style={{ fontSize: 13.5, fontWeight: 800, color: 'var(--text)' }}>Detalhamento das Falhas</div>
          {falhas.length === 0 && (
            <div style={{ fontSize: 12, color: 'var(--teal)', marginTop: 2, fontWeight: 600 }}>
              ✓ Nenhuma falha registrada no período.
            </div>
          )}
        </div>
      </div>

      {falhas.length > 0 && (
        <div style={{ maxHeight: 280, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
          {falhas.map((f, i) => (
            <div key={i} style={{
              background: 'var(--red-pale)', borderRadius: 'var(--radius-xs)',
              padding: '10px 14px', display: 'flex', alignItems: 'center',
              justifyContent: 'space-between', gap: 10, flexWrap: 'wrap',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{
                  background: 'var(--red)', color: '#fff', borderRadius: 6,
                  padding: '1px 8px', fontSize: 10.5, fontWeight: 700,
                  fontFamily: 'JetBrains Mono',
                }}>{f.data}</div>
                <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--red)' }}>{f.reservatorio}</span>
              </div>
              <div style={{ display: 'flex', gap: 14, fontSize: 11, color: 'var(--text-mid)', flexWrap: 'wrap' }}>
                <span>Vol: <strong style={{ fontFamily: 'JetBrains Mono' }}>{f.volIni} hm³</strong></span>
                <span>Solicit.: <strong style={{ fontFamily: 'JetBrains Mono' }}>{f.demSol} m³/s</strong></span>
                <span>Atend.: <strong style={{ fontFamily: 'JetBrains Mono', color: 'var(--red)' }}>{f.demAt} m³/s</strong></span>
                {parseFloat(f.rac) > 0 && <span>Rac: <strong style={{ fontFamily: 'JetBrains Mono' }}>{f.rac}%</strong></span>}
                <span style={{
                  background: 'rgba(217,64,64,0.15)', borderRadius: 4,
                  padding: '1px 6px', fontSize: 10.5, fontWeight: 600, color: 'var(--red)',
                }}>{f.modo}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
