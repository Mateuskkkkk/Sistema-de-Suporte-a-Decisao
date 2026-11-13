import { useEffect, useState } from 'react'
import { Download, FileImage, LoaderCircle } from 'lucide-react'

function exportName(value) {
  return String(value || 'grafico')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-zA-Z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .toLowerCase() || 'grafico'
}

function chartBackground(element) {
  const root = element.closest('.sim-root, .pv-root, .app-shell') || element
  return getComputedStyle(root).getPropertyValue('--card').trim() || '#ffffff'
}

// Cores dos gráficos podem vir como var(--...), que só existem dentro da página.
// Antes de exportar, troca essas referências pelas cores calculadas pelo navegador.
const ATRIBUTOS_COR = [['fill', 'fill'], ['stroke', 'stroke'], ['stop-color', 'stopColor'], ['color', 'color']]
function resolverCores(original, copia) {
  const origem = [original, ...original.querySelectorAll('*')]
  const destino = [copia, ...copia.querySelectorAll('*')]
  origem.forEach((el, i) => {
    const alvo = destino[i]
    if (!alvo || !el.getAttribute) return
    const calculado = getComputedStyle(el)
    for (const [attr, prop] of ATRIBUTOS_COR) {
      const valor = el.getAttribute(attr)
      if (valor && valor.includes('var(')) alvo.setAttribute(attr, calculado[prop])
    }
    const estilo = el.getAttribute('style')
    if (estilo && estilo.includes('var(')) {
      for (const [attr, prop] of ATRIBUTOS_COR) if (estilo.includes(attr)) alvo.style[prop] = calculado[prop]
    }
  })
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

export async function elementToPngDataUrl(element) {
  if (!element) return null
  const { default: html2canvas } = await import('html2canvas')
  if (document.fonts?.ready) await document.fonts.ready
  const canvas = await html2canvas(element, {
    backgroundColor: chartBackground(element),
    logging: false,
    scale: Math.max(2, Math.min(3, window.devicePixelRatio || 1)),
    useCORS: true,
    onclone: (_doc, copia) => resolverCores(element, copia),
  })
  return canvas.toDataURL('image/png')
}

export async function downloadElementAsPng(element, filename) {
  const dataUrl = await elementToPngDataUrl(element)
  if (!dataUrl) return
  const link = document.createElement('a')
  link.href = dataUrl
  link.download = `${exportName(filename)}.png`
  document.body.appendChild(link)
  link.click()
  link.remove()
}

export function downloadElementAsSvg(element, filename) {
  const source = element?.querySelector('svg')
  if (!source) throw new Error('SVG do grafico nao encontrado.')
  const clone = source.cloneNode(true)
  resolverCores(source, clone)
  const bounds = source.getBoundingClientRect()
  const width = Math.max(1, Math.round(bounds.width))
  const height = Math.max(1, Math.round(bounds.height))
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  clone.setAttribute('width', String(width))
  clone.setAttribute('height', String(height))
  clone.setAttribute('viewBox', clone.getAttribute('viewBox') || `0 0 ${width} ${height}`)
  clone.style.fontFamily = getComputedStyle(element).fontFamily

  const background = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
  background.setAttribute('width', '100%')
  background.setAttribute('height', '100%')
  background.setAttribute('fill', chartBackground(element))
  clone.insertBefore(background, clone.firstChild)

  const xml = new XMLSerializer().serializeToString(clone)
  downloadBlob(new Blob([xml], { type: 'image/svg+xml;charset=utf-8' }), `${exportName(filename)}.svg`)
}

function inferChartName(element) {
  const named = element.closest('[data-chart-name]')
  return named?.dataset.chartName || `grafico_${new Date().toISOString().slice(0, 10)}`
}

export default function ChartExportMenu() {
  const [menu, setMenu] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const onContextMenu = event => {
      const chart = event.target.closest?.('.recharts-responsive-container')
      if (!chart) return
      event.preventDefault()
      setMenu({
        element: chart,
        name: inferChartName(chart),
        x: Math.min(event.clientX, window.innerWidth - 190),
        y: Math.min(event.clientY, window.innerHeight - 104),
      })
    }
    const close = event => {
      if (!event.target.closest?.('.chart-export-menu')) setMenu(null)
    }
    const onKey = event => event.key === 'Escape' && setMenu(null)
    document.addEventListener('contextmenu', onContextMenu)
    document.addEventListener('pointerdown', close)
    window.addEventListener('blur', close)
    window.addEventListener('resize', close)
    window.addEventListener('scroll', close, true)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('contextmenu', onContextMenu)
      document.removeEventListener('pointerdown', close)
      window.removeEventListener('blur', close)
      window.removeEventListener('resize', close)
      window.removeEventListener('scroll', close, true)
      document.removeEventListener('keydown', onKey)
    }
  }, [])

  if (!menu) return null

  const run = async action => {
    setBusy(true)
    try {
      await action()
      setMenu(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      className="chart-export-menu"
      role="menu"
      style={{
        position: 'fixed', left: menu.x, top: menu.y, zIndex: 10000,
        width: 184, padding: 5, border: '1.5px solid var(--border)', borderRadius: 7,
        background: 'var(--card)', color: 'var(--text)', boxShadow: '0 10px 28px rgba(0,0,0,.22)',
      }}
    >
      <button
        type="button"
        role="menuitem"
        disabled={busy}
        onClick={() => run(() => downloadElementAsPng(menu.element, menu.name))}
        style={menuButtonStyle}
      >
        {busy ? <LoaderCircle size={14} className="chart-export-spin" /> : <Download size={14} />}
        Exportar PNG
      </button>
      <button
        type="button"
        role="menuitem"
        disabled={busy}
        onClick={() => run(() => downloadElementAsSvg(menu.element, menu.name))}
        style={menuButtonStyle}
      >
        <FileImage size={14} /> Exportar SVG
      </button>
      <style>{`.chart-export-menu button:hover{background:var(--orange-pale)!important;color:var(--orange-deep)!important}.chart-export-spin{animation:chartExportSpin 1s linear infinite}@keyframes chartExportSpin{to{transform:rotate(360deg)}}`}</style>
    </div>
  )
}

const menuButtonStyle = {
  width: '100%', display: 'flex', alignItems: 'center', gap: 8,
  border: 0, borderRadius: 5, padding: '8px 9px', background: 'transparent',
  color: 'inherit', fontSize: 11.5, fontWeight: 800, cursor: 'pointer', textAlign: 'left',
}
