import SimuladorHidrico from './Simulador'

export default function App() {
  return (
    <div>
      <SimuladorHidrico apiUrl={import.meta.env.VITE_API_URL}/>
    </div>
  )
}
