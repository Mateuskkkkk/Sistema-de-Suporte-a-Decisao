import { useRef, useState } from 'react'
import { Moon, Sun } from 'lucide-react'
import Inicio from './Inicio'
import SimuladorHidrico from './Simulador'
import OtimizadorMeta from './OtimizadorMeta'
import PrevisaoVazoes from './PrevisaoVazoes'
import ChartExportMenu from './components/ChartExportMenu'
import { VARS_ESCURO } from './tema'

// Campos em que o valor é selecionado ao receber o foco, para que o usuário
// digite o novo valor sem precisar apagar o anterior.
const TIPOS_SELECIONAVEIS = new Set(['text', 'number', 'search', 'email', 'tel', 'url'])

export default function App() {
  const [view, setView] = useState('inicio')
  // reservatório escolhido no mapa da tela inicial, pré-selecionado no módulo aberto
  const [acudeInicial, setAcudeInicial] = useState(null)
  const [curvasOtimizadas, setCurvasOtimizadas] = useState(null)
  const [darkMode, setDarkMode] = useState(false)
  // na versão desktop a interface e a API ficam no mesmo endereço
  const apiUrl = import.meta.env.MODE === 'desktop' ? '.' : import.meta.env.VITE_API_URL
  const campoSelecionado = useRef(null)

  const selecionarAoFocar = e => {
    const el = e.target
    if (el.tagName !== 'INPUT' || !TIPOS_SELECIONAVEIS.has(el.type) || el.readOnly) return
    el.select()
    campoSelecionado.current = el
  }
  // o clique que deu o foco terminaria desfazendo a seleção ao soltar o botão
  const manterSelecao = e => {
    if (campoSelecionado.current !== e.target) return
    e.preventDefault()
    campoSelecionado.current = null
  }
  const liberarSelecao = () => { campoSelecionado.current = null }

  return (
    <div className={`app-shell ${darkMode ? 'app-dark' : ''}`} style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)' }}
      onFocus={selecionarAoFocar} onMouseUp={manterSelecao} onKeyDown={liberarSelecao} onBlur={liberarSelecao}>
      <style>{`.app-shell{--bg:#fdf6ee;--orange:#e07b2a;--orange-pale:#fdebd3;--orange-deep:#c46318;--text:#1e1208;--text-light:#9a7055;--border:#ecdcc8;--card:#fffaf4;font-family:'Sora',sans-serif}.app-shell.app-dark{${VARS_ESCURO}}.brand-icon{width:34px;height:34px;border-radius:10px;display:flex;align-items:center;justify-content:center;background:#fff7ed;border:1.5px solid var(--border);box-shadow:0 1px 8px rgba(150,90,40,.14);flex:0 0 auto}.app-dark .brand-icon{background:var(--card-2);border-color:var(--border);box-shadow:0 2px 14px rgba(0,0,0,.35)}.theme-btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;border:1.5px solid var(--border);border-radius:8px;background:var(--card);color:var(--text-light);padding:8px 12px;font-size:12px;font-weight:900;cursor:pointer}.theme-btn:hover{color:var(--orange-deep);border-color:var(--orange-deep)}.app-shell button:focus-visible{outline:2px solid var(--orange-deep);outline-offset:2px}@media (max-width:760px){.app-header{padding:10px 12px!important;flex-wrap:wrap}.app-header>div{flex:1 1 100%;min-width:0}.app-header>div:last-child{flex-wrap:nowrap!important}.app-header nav{flex:1 1 auto;min-width:0;overflow-x:auto}.app-header nav button{flex:1 0 auto;padding:8px 10px!important}.app-subtitulo{display:none}.theme-btn .theme-rotulo{display:none}}`}</style>
      <header className="app-header" style={{
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
          <span className="brand-icon">
            <img
              src="/favicon.svg"
              alt=""
              aria-hidden="true"
              style={{ width: 27, height: 27, display: 'block' }}
            />
          </span>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 15, fontWeight: 900, color: 'var(--text)' }}>Sistema de Suporte à Decisão</div>
            <div className="app-subtitulo" style={{ fontSize: 11, color: 'var(--text-light)' }}>
              Simulação de Série Histórica e Otimização de Níveis Meta.
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <nav aria-label="Módulos do sistema" style={{ display: 'flex', gap: 4, background: 'var(--card)', border: '1.5px solid var(--border)', borderRadius: 9, padding: 3 }}>
            {[
              ['inicio', 'Início'],
              ['simulador', 'Simulador'],
              ['otimizador', 'Otimizador'],
              ['vazoes', 'Vazões'],
              ['previsao', 'Previsão'],
            ].map(([id, label]) => (
              <button
                key={id}
                aria-current={view === id ? 'page' : undefined}
                onClick={() => { setAcudeInicial(null); setView(id) }}
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
          <button className="theme-btn" aria-pressed={darkMode} aria-label={darkMode ? 'Modo Claro' : 'Modo Escuro'} onClick={() => setDarkMode(v => !v)}>
            {darkMode ? <Sun size={14} /> : <Moon size={14} />}
            <span className="theme-rotulo">{darkMode ? 'Modo Claro' : 'Modo Escuro'}</span>
          </button>
        </div>
      </header>

      {view === 'inicio' ? (
        <Inicio
          apiUrl={apiUrl}
          darkMode={darkMode}
          onAbrirModulo={(modulo, acude) => {
            setAcudeInicial({ ...acude, id: Date.now() })
            setView(modulo)
          }}
        />
      ) : view === 'simulador' ? (
        <SimuladorHidrico
          apiUrl={apiUrl}
          acudeInicial={acudeInicial}
          curvasOtimizadas={curvasOtimizadas}
          darkMode={darkMode}
          onOpenOtimizador={() => setView('otimizador')}
        />
      ) : view === 'otimizador' ? (
        <OtimizadorMeta
          apiUrl={apiUrl}
          acudeInicial={acudeInicial}
          darkMode={darkMode}
          onApplyCurvas={(payload) => {
            setAcudeInicial(null)
            setCurvasOtimizadas(payload)
            setView('simulador')
          }}
        />
      ) : view === 'vazoes' ? (
        <PrevisaoVazoes apiUrl={apiUrl} acudeInicial={acudeInicial} darkMode={darkMode} mode="qxx" />
      ) : (
        <PrevisaoVazoes apiUrl={apiUrl} acudeInicial={acudeInicial} darkMode={darkMode} mode="knn" />
      )}
      <ChartExportMenu />
    </div>
  )
}
