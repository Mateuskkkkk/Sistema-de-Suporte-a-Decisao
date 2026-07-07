import { useState } from 'react'
import { Moon, Sun } from 'lucide-react'
import SimuladorHidrico from './Simulador'
import OtimizadorMeta from './OtimizadorMeta'

export default function App() {
  const [view, setView] = useState('simulador')
  const [curvasOtimizadas, setCurvasOtimizadas] = useState(null)
  const [darkMode, setDarkMode] = useState(false)
  const apiUrl = import.meta.env.VITE_API_URL

  return (
    <div className={`app-shell ${darkMode ? 'app-dark' : ''}`} style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)' }}>
      <style>{`.app-shell{--bg:#fdf6ee;--orange:#e07b2a;--orange-pale:#fdebd3;--orange-deep:#c46318;--text:#1e1208;--text-light:#9a7055;--border:#ecdcc8;--card:#fffaf4;font-family:'Sora',sans-serif}.app-shell.app-dark{--bg:#160f0a;--orange-pale:#4a2a14;--orange-deep:#f5a654;--text:#fff5ec;--text-light:#b68b6f;--border:#4a3325;--card:#211711}.theme-btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;border:1.5px solid var(--border);border-radius:8px;background:var(--card);color:var(--text-light);padding:8px 12px;font-size:12px;font-weight:900;cursor:pointer}.theme-btn:hover{color:var(--orange-deep);border-color:var(--orange-deep)}`}</style>
      <header style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: 12,
        padding: '12px 26px',
        background: 'var(--card)',
        borderBottom: '1.5px solid var(--border)',
        position: 'sticky',
        top: 0,
        zIndex: 20,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
          <img
            src="/favicon.svg"
            alt=""
            aria-hidden="true"
            style={{ width: 32, height: 32, flex: '0 0 auto' }}
          />
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 15, fontWeight: 900, color: 'var(--text)' }}>Sistema de Suporte à Decisão</div>
            <div style={{ fontSize: 11, color: 'var(--text-light)' }}>
              Simulação de Série Histórica e Otimização de Níveis Meta.
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <nav style={{ display: 'flex', gap: 4, background: 'var(--card)', border: '1.5px solid var(--border)', borderRadius: 9, padding: 3 }}>
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
          <button className="theme-btn" onClick={() => setDarkMode(v => !v)}>
            {darkMode ? <Sun size={14} /> : <Moon size={14} />}
            {darkMode ? 'Modo Claro' : 'Modo Escuro'}
          </button>
        </div>
      </header>

      {view === 'simulador' ? (
        <SimuladorHidrico apiUrl={apiUrl} curvasOtimizadas={curvasOtimizadas} darkMode={darkMode} />
      ) : (
        <OtimizadorMeta
          apiUrl={apiUrl}
          darkMode={darkMode}
          onApplyCurvas={(payload) => {
            setCurvasOtimizadas(payload)
            setView('simulador')
          }}
        />
      )}
    </div>
  )
}
