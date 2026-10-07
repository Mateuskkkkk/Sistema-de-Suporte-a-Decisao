// Cores do modo escuro, compartilhadas por todas as telas.
//
// Em vez de preto puro, o fundo usa camadas de grafite quente: página, cartão e
// campos têm tons próprios, o que separa as áreas sem depender de bordas fortes.
// As cores das séries dos gráficos foram escolhidas para fundo escuro: são
// distinguíveis entre si (inclusive com daltonismo) e têm contraste ≥ 3:1 sobre
// o cartão. Os estados de seca mantêm o significado do modo claro.
export const VARS_ESCURO = [
  'color-scheme:dark',
  '--bg:#14110f', '--card:#1c1815', '--card-2:#241f1b', '--campo:#16130f',
  '--text:#f4eee7', '--text-mid:#d6c8b9', '--text-light:#a89888',
  '--border:#3a3029', '--border-light:#2a231e', '--grade:#2e2722',
  '--orange:#ef8a3e', '--orange-light:#f5a65e', '--orange-deep:#ffa863',
  '--orange-pale:#3a2616', '--orange-glow:rgba(239,138,62,.24)',
  '--teal:#35bfa9', '--teal-pale:#12302b',
  '--blue:#71a9f2', '--blue-light:#93bdf5', '--blue-pale:#172840',
  '--red:#f07170', '--red-pale:#3b1c1c',
  '--yellow:#f2bd3a', '--yellow-pale:#372c12',
  '--serie-1:#3987e5', '--serie-2:#d95926', '--serie-3:#199e70', '--serie-4:#9085e9',
  '--faixa-normal:#35bfa9', '--faixa-alerta:#f2bd3a', '--faixa-seca:#ec835a', '--faixa-severa:#ea5f5e', '--faixa-colapso:#c23c3c',
  '--sucesso:#3ccf8e', '--erro:#f07170',
  '--shadow-sm:0 1px 3px rgba(0,0,0,.45)', '--shadow:0 6px 24px rgba(0,0,0,.40)',
].join(';')

// Regras comuns do modo escuro para uma raiz de tela (ex.: '.sim-root.app-dark').
export function cssEscuro(raiz) {
  return `${raiz}{${VARS_ESCURO}}
${raiz} input,${raiz} select,${raiz} textarea{background:var(--campo)!important;color:var(--text)!important;border-color:var(--border)!important}
${raiz} input::placeholder{color:#7d7064}
${raiz} option{background:var(--card);color:var(--text)}
${raiz} .recharts-default-tooltip{background:var(--card-2)!important;border-color:var(--border)!important;color:var(--text)!important;box-shadow:var(--shadow)}
${raiz} ::-webkit-scrollbar-thumb{background:#4a3f36}`
}
