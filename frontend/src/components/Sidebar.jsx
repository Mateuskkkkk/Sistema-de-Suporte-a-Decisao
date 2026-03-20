import React from 'react'
import {
  Activity, AlertTriangle, FileText, Zap, Droplets,
  BarChart2, Cpu, Users, Waves
} from 'lucide-react'

const NAV_ITEMS = [
  { icon: Activity, label: 'Monitoramento do Estado de Seca' },
  { icon: AlertTriangle, label: 'Implementação dos Planos de Secas' },
  { icon: FileText, label: 'Planos de Ação' },
  { icon: Zap, label: 'Impactos' },
  { icon: Droplets, label: 'Usos da Água' },
  { icon: BarChart2, label: 'Balanço Hídrico' },
  { icon: Cpu, label: 'Simulador', active: true },
  { icon: Users, label: 'Responsáveis' },
]

export default function Sidebar() {
  return (
    <aside style={{
      width: 220,
      minHeight: '100vh',
      background: 'var(--sidebar-bg)',
      borderRight: '1.5px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      position: 'fixed',
      top: 0, left: 0, bottom: 0,
      zIndex: 100,
    }}>
      {/* Logo */}
      <div style={{
        padding: '22px 18px 16px',
        borderBottom: '1.5px solid var(--border)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 4 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 10,
            background: 'linear-gradient(135deg, var(--orange), var(--orange-deep))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 3px 10px var(--orange-glow)',
          }}>
            <Waves size={18} color="#fff" strokeWidth={2.5} />
          </div>
          <div>
            <div style={{ fontSize: 12.5, fontWeight: 800, color: 'var(--orange-deep)', letterSpacing: '0.01em', lineHeight: 1.2 }}>
              HidroSim
            </div>
            <div style={{ fontSize: 9.5, color: 'var(--text-light)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
              Balanço Hídrico
            </div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: '14px 10px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {NAV_ITEMS.map(({ icon: Icon, label, active }) => (
          <button
            key={label}
            style={{
              display: 'flex', alignItems: 'center', gap: 9,
              padding: '9px 11px',
              borderRadius: 'var(--radius-sm)',
              background: active ? 'var(--orange-pale)' : 'none',
              border: active ? '1.5px solid var(--orange-light)' : '1.5px solid transparent',
              color: active ? 'var(--orange-deep)' : 'var(--text-mid)',
              fontSize: 12, fontWeight: active ? 600 : 500,
              cursor: 'pointer', textAlign: 'left',
              transition: 'all 0.15s',
              lineHeight: 1.3,
            }}
            onMouseEnter={e => { if (!active) { e.currentTarget.style.background = 'var(--orange-pale)'; e.currentTarget.style.color = 'var(--orange)' } }}
            onMouseLeave={e => { if (!active) { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = 'var(--text-mid)' } }}
          >
            <Icon size={15} strokeWidth={active ? 2.5 : 2} style={{ flexShrink: 0 }} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      {/* Footer */}
      <div style={{
        padding: '14px 18px',
        borderTop: '1.5px solid var(--border)',
        fontSize: 10.5, color: 'var(--text-light)',
        textAlign: 'center',
      }}>
        API: <span style={{ fontFamily: 'JetBrains Mono', fontSize: 10 }}>localhost:8000</span>
      </div>
    </aside>
  )
}
