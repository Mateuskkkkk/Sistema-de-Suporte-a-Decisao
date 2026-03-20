import React, { useState } from 'react'
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

const COLORS = [
  { stroke: '#264fa3', fill: '#264fa3', fillOp: 0.15 },
  { stroke: '#e07b2a', fill: '#e07b2a', fillOp: 0.15 },
  { stroke: '#2a9d8f', fill: '#2a9d8f', fillOp: 0.15 },
  { stroke: '#9b2dca', fill: '#9b2dca', fillOp: 0.15 },
]

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#fff', border: '1.5px solid var(--border)',
      borderRadius: 10, padding: '10px 14px', boxShadow: 'var(--shadow)',
      fontSize: 11.5,
    }}>
      <div style={{ fontWeight: 700, marginBottom: 6, color: 'var(--text)' }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 3 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: p.color }} />
          <span style={{ color: 'var(--text-mid)' }}>{p.name}:</span>
          <span style={{ fontWeight: 600, fontFamily: 'JetBrains Mono', color: 'var(--text)' }}>
            {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
          </span>
        </div>
      ))}
    </div>
  )
}

function ChartCard({ title, subtitle, children }) {
  return (
    <div className="animate-fade-in" style={{
      background: 'var(--card)', border: '1.5px solid var(--border)',
      borderRadius: 'var(--radius)', padding: '20px 22px',
      boxShadow: 'var(--shadow)',
    }}>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--text)' }}>{title}</div>
        {subtitle && <div style={{ fontSize: 11.5, color: 'var(--text-light)', marginTop: 2 }}>{subtitle}</div>}
      </div>
      {children}
    </div>
  )
}

function ChartToggle({ options, value, onChange }) {
  return (
    <div style={{ display: 'flex', gap: 4, marginBottom: 14 }}>
      {options.map(o => (
        <button key={o} onClick={() => onChange(o)} style={{
          padding: '5px 12px', borderRadius: 20,
          border: `1.5px solid ${value === o ? 'var(--orange)' : 'var(--border)'}`,
          background: value === o ? 'var(--orange-pale)' : 'none',
          color: value === o ? 'var(--orange-deep)' : 'var(--text-light)',
          fontSize: 11, fontWeight: 600, cursor: 'pointer', transition: 'all 0.15s',
        }}>{o}</button>
      ))}
    </div>
  )
}

export default function Charts({ resultados }) {
  const [chartType, setChartType] = useState('Volume')

  if (!resultados?.length) return null

  // Build unified dataset (keyed by Data)
  const allDatas = [...new Set(resultados.flatMap(r => r.dados.map(d => d.Data)))].sort()

  const volumeData = allDatas.map(data => {
    const point = { data }
    resultados.forEach((r, i) => {
      const d = r.dados.find(x => x.Data === data)
      if (d) {
        point[`Vol. Inicial (${r.reservatorio})`] = parseFloat(d['Armazenamento Inicial']) || 0
        point[`Vol. Final (${r.reservatorio})`] = parseFloat(d['Armazenamento Final']) || 0
        point[`Afluência (${r.reservatorio})`] = parseFloat(d['Afluências (hm³/mês)']) || 0
      }
    })
    return point
  })

  const demandaData = allDatas.map(data => {
    const point = { data }
    resultados.forEach(r => {
      const d = r.dados.find(x => x.Data === data)
      if (d) {
        point[`Solicitada (${r.reservatorio})`] = parseFloat(d['Demanda Solicitada (m³/s)']) || 0
        point[`Atendida (${r.reservatorio})`] = parseFloat(d['Demanda Atendida (m³/s)']) || 0
      }
    })
    return point
  })

  const racData = allDatas.map(data => {
    const point = { data }
    resultados.forEach(r => {
      const d = r.dados.find(x => x.Data === data)
      if (d) {
        point[r.reservatorio] = parseFloat(d['Racionamento (%)']) || 0
      }
    })
    return point
  })

  const balanceData = allDatas.map(data => {
    const point = { data }
    resultados.forEach(r => {
      const d = r.dados.find(x => x.Data === data)
      if (d) {
        point[`Evap. (${r.reservatorio})`] = parseFloat(d['Evaporação (hm³)']) || 0
        point[`Vertimento (${r.reservatorio})`] = parseFloat(d['Vertimento (hm³)']) || 0
        point[`Afluência (${r.reservatorio})`] = parseFloat(d['Afluências (hm³/mês)']) || 0
      }
    })
    return point
  })

  // Tick formatter: show only every N months
  const tickFormatter = (val) => {
    if (!val) return ''
    const parts = val.split('-')
    return `${parts[1]}/${parts[0]?.slice(2)}`
  }
  const interval = Math.max(0, Math.floor(allDatas.length / 12) - 1)

  const volKeys = volumeData[0] ? Object.keys(volumeData[0]).filter(k => k !== 'data' && k.startsWith('Vol. Final')) : []
  const afluKeys = volumeData[0] ? Object.keys(volumeData[0]).filter(k => k !== 'data' && k.startsWith('Afluência')) : []
  const demSolicKeys = demandaData[0] ? Object.keys(demandaData[0]).filter(k => k !== 'data' && k.startsWith('Solicitada')) : []
  const demAtendKeys = demandaData[0] ? Object.keys(demandaData[0]).filter(k => k !== 'data' && k.startsWith('Atendida')) : []
  const racKeys = racData[0] ? Object.keys(racData[0]).filter(k => k !== 'data') : []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* Volume Chart */}
      <ChartCard title="Histórico de Simulação" subtitle="Volume Armazenado e Afluência ao longo do tempo">
        <div style={{ height: 280 }}>
          <ResponsiveContainer>
            <AreaChart data={volumeData} margin={{ top: 4, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="data" tickFormatter={tickFormatter} interval={interval} tick={{ fontSize: 10.5, fill: 'var(--text-light)' }} />
              <YAxis yAxisId="vol" tick={{ fontSize: 10.5, fill: 'var(--blue)' }} label={{ value: 'Volume (hm³)', angle: -90, position: 'insideLeft', fill: 'var(--blue)', fontSize: 10 }} />
              <YAxis yAxisId="afl" orientation="right" tick={{ fontSize: 10.5, fill: 'var(--teal)' }} label={{ value: 'Afluência (hm³)', angle: 90, position: 'insideRight', fill: 'var(--teal)', fontSize: 10 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {volKeys.map((k, i) => (
                <Area key={k} yAxisId="vol" type="monotone" dataKey={k}
                  stroke={COLORS[i % COLORS.length].stroke}
                  fill={COLORS[i % COLORS.length].fill}
                  fillOpacity={COLORS[i % COLORS.length].fillOp}
                  strokeWidth={2} dot={false} />
              ))}
              {afluKeys.map((k, i) => (
                <Line key={k} yAxisId="afl" type="monotone" dataKey={k}
                  stroke={COLORS[(i + 2) % COLORS.length].stroke}
                  strokeWidth={1.5} dot={false} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {/* Demanda Chart */}
      <ChartCard title="Demanda: Solicitada vs Atendida" subtitle="Comparativo mensal (m³/s)">
        <div style={{ height: 220 }}>
          <ResponsiveContainer>
            <LineChart data={demandaData} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="data" tickFormatter={tickFormatter} interval={interval} tick={{ fontSize: 10.5, fill: 'var(--text-light)' }} />
              <YAxis tick={{ fontSize: 10.5, fill: 'var(--text-light)' }} label={{ value: 'm³/s', angle: -90, position: 'insideLeft', fill: 'var(--text-light)', fontSize: 10 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {demSolicKeys.map((k, i) => (
                <Line key={k} type="monotone" dataKey={k}
                  stroke={COLORS[i % COLORS.length].stroke}
                  strokeWidth={2} strokeDasharray="5 3" dot={false} />
              ))}
              {demAtendKeys.map((k, i) => (
                <Line key={k} type="monotone" dataKey={k}
                  stroke={COLORS[i % COLORS.length].stroke}
                  strokeWidth={2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {/* Racionamento Chart */}
      {racKeys.length > 0 && (
        <ChartCard title="Racionamento Mensal" subtitle="Percentual de restrição aplicado (%)">
          <div style={{ height: 200 }}>
            <ResponsiveContainer>
              <BarChart data={racData} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="data" tickFormatter={tickFormatter} interval={interval} tick={{ fontSize: 10.5, fill: 'var(--text-light)' }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10.5, fill: 'var(--text-light)' }} label={{ value: '%', angle: -90, position: 'insideLeft', fill: 'var(--text-light)', fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {racKeys.map((k, i) => (
                  <Bar key={k} dataKey={k}
                    fill={COLORS[i % COLORS.length].stroke}
                    fillOpacity={0.75}
                    radius={[3, 3, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      )}

      {/* Balanço Chart */}
      <ChartCard title="Balanço Hídrico" subtitle="Evaporação, Vertimento e Afluência (hm³/mês)">
        <div style={{ height: 200 }}>
          <ResponsiveContainer>
            <BarChart data={balanceData} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="data" tickFormatter={tickFormatter} interval={interval} tick={{ fontSize: 10.5, fill: 'var(--text-light)' }} />
              <YAxis tick={{ fontSize: 10.5, fill: 'var(--text-light)' }} label={{ value: 'hm³', angle: -90, position: 'insideLeft', fill: 'var(--text-light)', fontSize: 10 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {Object.keys(balanceData[0] || {}).filter(k => k !== 'data' && k.startsWith('Afluência')).map((k, i) => (
                <Bar key={k} dataKey={k} fill="#2a9d8f" fillOpacity={0.6} radius={[3, 3, 0, 0]} />
              ))}
              {Object.keys(balanceData[0] || {}).filter(k => k !== 'data' && k.startsWith('Evap.')).map((k, i) => (
                <Bar key={k} dataKey={k} fill="#e07b2a" fillOpacity={0.6} radius={[3, 3, 0, 0]} />
              ))}
              {Object.keys(balanceData[0] || {}).filter(k => k !== 'data' && k.startsWith('Vertimento')).map((k, i) => (
                <Bar key={k} dataKey={k} fill="#264fa3" fillOpacity={0.6} radius={[3, 3, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>
    </div>
  )
}
