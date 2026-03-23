import { useState } from 'react'
import SimuladorHidrico from './Simulador'
import Otimizador from './Otimizador'

export default function App() {
  const [pagina, setPagina] = useState('simulador')

  return (
    <div>
      {/* Barra de navegação */}
      <div style={{
        display:'flex', gap:6, padding:'10px 20px',
        borderBottom:'1.5px solid #ecdcc8', background:'#fffaf4',
        position:'sticky', top:0, zIndex:200
      }}>
        {[
          { id:'simulador',  label:'🌊 Simulador'   },
          { id:'otimizador', label:'⚙️ Optimizador' },
        ].map(p => (
          <button key={p.id} onClick={() => setPagina(p.id)} style={{
            padding:'7px 16px', borderRadius:9, border:'none', cursor:'pointer',
            background: pagina===p.id ? '#fdebd3' : 'none',
            color:      pagina===p.id ? '#c46318' : '#9a7055',
            fontWeight:700, fontSize:12, transition:'all .15s',
          }}>{p.label}</button>
        ))}
      </div>

      {pagina === 'simulador'  && <SimuladorHidrico apiUrl={import.meta.env.VITE_API_URL}/>}
      {pagina === 'otimizador' && <Otimizador       apiUrl={import.meta.env.VITE_API_URL}/>}
    </div>
  )
}
