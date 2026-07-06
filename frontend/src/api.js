const BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export async function fetchReservatorios() {
  const res = await fetch(`${BASE_URL}/api/reservatorios`)
  if (!res.ok) throw new Error('Falha ao carregar reservatórios')
  return res.json()
}

export async function fetchPresets() {
  const res = await fetch(`${BASE_URL}/api/presets`)
  if (!res.ok) throw new Error('Falha ao carregar presets')
  return res.json()
}

export async function runSimulacao(payload) {
  const res = await fetch(`${BASE_URL}/api/simular`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    // FastAPI validation errors return detail as array of objects
    if (Array.isArray(err.detail)) {
      const msgs = err.detail.map(e => {
        const field = e.loc ? e.loc.join(' → ') : ''
        return field ? `${field}: ${e.msg}` : e.msg
      }).join(' | ')
      throw new Error(`Erro de validação: ${msgs}`)
    }
    throw new Error(typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail) || 'Erro na simulação')
  }
  return res.json()
}
