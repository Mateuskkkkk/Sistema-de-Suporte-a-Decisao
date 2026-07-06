import { useState } from 'react'
import SimuladorHidrico from './Simulador'
import OtimizadorMeta from './OtimizadorMeta'

export default function App() {
  const [view, setView] = useState('simulador')
  const [curvasOtimizadas, setCurvasOtimizadas] = useState(null)
  const apiUrl = import.meta.env.VITE_API_URL

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      <header style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: 12,
        padding: '12px 26px',
        background: '#fffaf4',
        borderBottom: '1.5px solid var(--border)',
        position: 'sticky',
        top: 0,
        zIndex: 20,
      }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 900, color: 'var(--text)' }}>Sistema de Suporte à Decisão</div>
          <div style={{ fontSize: 11, color: 'var(--text-light)' }}>
            Simulação histórica e otimização de níveis meta no mesmo fluxo.
          </div>
        </div>
        <nav style={{ display: 'flex', gap: 4, background: '#fff', border: '1.5px solid var(--border)', borderRadius: 9, padding: 3 }}>
          {[
            ['simulador', 'Simulador'],
            ['otimizador', 'Otimizador'],
          ].map(([id, label]) => (
            <button
              key={id}
              onClick={() => setView(id)}
              style={{
                padding: '8px 14px',
                border: 0,
                borderRadius: 7,
                background: view === id ? 'var(--orange-pale)' : 'transparent',
                color: view === id ? 'var(--orange-deep)' : 'var(--text-light)',
                fontSize: 12,
                fontWeight: 900,
                cursor: 'pointer',
              }}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      {view === 'simulador' ? (
        <SimuladorHidrico apiUrl={apiUrl} curvasOtimizadas={curvasOtimizadas} />
      ) : (
        <OtimizadorMeta
          apiUrl={apiUrl}
          onApplyCurvas={(payload) => {
            setCurvasOtimizadas(payload)
            setView('simulador')
          }}
        />
      )}
    </div>
  )
}
