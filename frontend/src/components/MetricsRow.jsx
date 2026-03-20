import React from 'react'
import { CheckCircle2, AlertCircle, TrendingDown, Droplets, BarChart3 } from 'lucide-react'

function MetricCard({ label, value, sub, variant = 'default', icon: Icon, delay = 0 }) {
  const colors = {
    default: { accent: 'var(--orange)', bg: 'var(--orange-pale)', text: 'var(--orange)' },
    success: { accent: 'var(--teal)', bg: 'var(--teal-pale)', text: 'var(--teal)' },
    danger: { accent: 'var(--red)', bg: 'var(--red-pale)', text: 'var(--red)' },
    info: { accent: 'var(--blue)', bg: 'var(--blue-pale)', text: 'var(--blue)' },
    yellow: { accent: 'var(--yellow)', bg: 'var(--yellow-pale)', text: 'var(--yellow)' },
  }
  const c = colors[variant]

  return (
    <div className={`animate-fade-in animate-delay-${delay}`} style={{
      background: 'var(--card)', border: '1.5px solid var(--border)',
      borderRadius: 'var(--radius)', padding: '18px 20px',
      boxShadow: 'var(--shadow)', position: 'relative', overflow: 'hidden',
    }}>
      {/* top bar */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 3,
        background: c.accent, borderRadius: '4px 4px 0 0',
      }} />

      <div style={{
        display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 10,
      }}>
        <div style={{
          fontSize: 10.5, fontWeight: 700, color: 'var(--text-light)',
          textTransform: 'uppercase', letterSpacing: '0.06em',
        }}>{label}</div>
        {Icon && (
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: c.bg, display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Icon size={14} color={c.accent} strokeWidth={2.5} />
          </div>
        )}
      </div>

      <div style={{
        fontSize: 30, fontWeight: 800, color: c.text,
        fontFamily: 'JetBrains Mono', lineHeight: 1, marginBottom: 6,
      }}>{value}</div>

      <div style={{ fontSize: 11.5, color: 'var(--text-light)', lineHeight: 1.4 }}>{sub}</div>
    </div>
  )
}

export default function MetricsRow({ resultados }) {
  if (!resultados || resultados.length === 0) return null

  // Aggregate across all reservatórios
  let totalMeses = 0
  let totalFalhas = 0
  let totalRac = 0
  let racMeses = 0
  let somaAtend = 0
  let somaSolic = 0
  let totalVertHm3 = 0
  let totalEvapHm3 = 0

  resultados.forEach(r => {
    r.dados.forEach(d => {
      totalMeses++
      if (d['Falha'] === 'Sim') totalFalhas++
      const rac = parseFloat(d['Racionamento (%)']) || 0
      if (rac > 0) { totalRac += rac; racMeses++ }
      somaAtend += parseFloat(d['Demanda Atendida (m³/s)']) || 0
      somaSolic += parseFloat(d['Demanda Solicitada (m³/s)']) || 0
      totalVertHm3 += parseFloat(d['Vertimento (hm³)']) || 0
      totalEvapHm3 += parseFloat(d['Evaporação (hm³)']) || 0
    })
  })

  const freqFalha = totalMeses > 0 ? ((totalFalhas / totalMeses) * 100).toFixed(1) : '0.0'
  const atendPct = somaSolic > 0 ? ((somaAtend / somaSolic) * 100).toFixed(1) : '100.0'
  const racMedio = racMeses > 0 ? (totalRac / racMeses).toFixed(1) : '0.0'
  const semFalha = parseFloat(freqFalha) === 0

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))', gap: 14 }}>
      {/* Frequência de Falha */}
      <div className="animate-fade-in" style={{
        background: 'var(--card)', border: `1.5px solid ${semFalha ? 'var(--teal-pale)' : 'var(--red-pale)'}`,
        borderRadius: 'var(--radius)', padding: '18px 20px',
        boxShadow: 'var(--shadow)', position: 'relative', overflow: 'hidden',
      }}>
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: 3,
          background: semFalha ? 'var(--teal)' : 'var(--red)',
          borderRadius: '4px 4px 0 0',
        }} />
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 10 }}>
          <div style={{ fontSize: 10.5, fontWeight: 700, color: 'var(--text-light)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Frequência de Falha
          </div>
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: semFalha ? 'var(--teal-pale)' : 'var(--red-pale)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            {semFalha
              ? <CheckCircle2 size={14} color="var(--teal)" strokeWidth={2.5} />
              : <AlertCircle size={14} color="var(--red)" strokeWidth={2.5} />
            }
          </div>
        </div>
        <div style={{
          fontSize: 30, fontWeight: 800,
          color: semFalha ? 'var(--teal)' : 'var(--red)',
          fontFamily: 'JetBrains Mono', lineHeight: 1, marginBottom: 6,
        }}>{freqFalha}%</div>
        <div style={{ fontSize: 11.5, color: 'var(--text-light)' }}>
          {semFalha
            ? <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}><CheckCircle2 size={12} color="var(--teal)" /> Demanda atendida integralmente (100%).</span>
            : `${totalFalhas} mês(es) com falha de ${totalMeses} total.`
          }
        </div>
      </div>

      <MetricCard
        label="Atendimento Médio"
        value={`${atendPct}%`}
        sub="Demanda atendida vs solicitada"
        variant={parseFloat(atendPct) >= 95 ? 'success' : parseFloat(atendPct) >= 80 ? 'yellow' : 'danger'}
        icon={BarChart3}
        delay={1}
      />

      <MetricCard
        label="Meses Analisados"
        value={totalMeses}
        sub={`${resultados.length} reservatório(s)`}
        variant="info"
        icon={TrendingDown}
        delay={2}
      />

      <MetricCard
        label="Rac. Médio"
        value={`${racMedio}%`}
        sub={`em ${racMeses} mês(es) com restrição`}
        variant={parseFloat(racMedio) > 20 ? 'danger' : parseFloat(racMedio) > 0 ? 'yellow' : 'success'}
        icon={AlertCircle}
        delay={3}
      />

      <MetricCard
        label="Evaporação Total"
        value={`${totalEvapHm3.toFixed(1)}`}
        sub="hm³ no período"
        variant="default"
        icon={Droplets}
        delay={4}
      />

      <MetricCard
        label="Vertimento Total"
        value={`${totalVertHm3.toFixed(1)}`}
        sub="hm³ de excedente"
        variant={totalVertHm3 > 0 ? 'info' : 'success'}
        icon={Droplets}
        delay={5}
      />
    </div>
  )
}
