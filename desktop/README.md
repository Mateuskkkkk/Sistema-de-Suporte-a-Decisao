# Versão desktop (Windows)

Gera um programa que abre o SSD numa janela própria, sem precisar de internet,
de servidor ou de rodar comandos. Por dentro, o programa sobe a API num endereço
local (`127.0.0.1`) e mostra a interface numa janela nativa.

## Por que é leve

A janela usa o componente de navegador que **já vem instalado no sistema**
(WebView2 no Windows 10 e 11), por meio do [pywebview](https://pywebview.flowrl.com/).
O programa não carrega um navegador próprio, como fazem os aplicativos em
Electron (cerca de 150 MB a mais e um navegador inteiro em memória).

O tamanho que sobra vem das bibliotecas de cálculo do Python: Numba/LLVM, SciPy, NumPy, pandas,
scikit-learn e XGBoost. Para reduzir, a versão desktop usa o `xgboost-cpu`, sem
as bibliotecas de GPU que o pacote padrão traz. Num teste no Linux, isso
reduziu a pasta de 1,1 GB para 420 MB (180 MB em .zip).

## Como gerar

Precisa de **Python 3.10+** e **Node.js 18+** instalados no Windows.

### Automático

Dê dois cliques em `desktop\construir_windows.bat` (ou rode no terminal, na
raiz do repositório). Ao final, o programa estará em:

```
dist\SSD-Reservatorios\SSD-Reservatorios.exe
```

### Passo a passo

Na raiz do repositório:

```bat
python -m venv .venv-desktop
.venv-desktop\Scripts\activate
pip install -r desktop\requirements-desktop.txt

cd frontend
npm install
npm run build:desktop
cd ..

pyinstaller desktop\ssd_desktop.spec --noconfirm --clean
```

Para testar sem empacotar: `python desktop\ssd_desktop.py`.

## Como distribuir

Compacte a pasta `dist\SSD-Reservatorios` inteira em um .zip. Quem recebe
descompacta e abre `SSD-Reservatorios.exe`. O .exe precisa ficar dentro da
pasta, junto com a subpasta `_internal`.

Se quiser um instalador com atalho no menu Iniciar, use o
[Inno Setup](https://jrsoftware.org/isinfo.php) apontando para essa pasta.

## Observações

- **Primeira simulação de cada sessão:** leva alguns segundos a mais, porque o
  Numba compila o motor de cálculo ao abrir o programa.
- **Aviso do Windows (SmartScreen):** como o programa não tem assinatura
  digital, o Windows pode mostrar "O Windows protegeu o computador". Basta
  clicar em "Mais informações" → "Executar assim mesmo".
- **WebView2:** já vem no Windows 10 e 11. Em computadores sem ele, a Microsoft
  oferece o instalador gratuito "WebView2 Runtime".
- **Sem janela nativa:** `SSD-Reservatorios.exe --navegador` abre a interface no
  navegador padrão (por exemplo, o Firefox). `--porta=8765` fixa a porta local.
- **Ícone:** salve um `desktop\icone.ico` antes de gerar e ele será usado no .exe.
- O executável é gerado para o sistema em que foi construído: para Windows,
  gere no Windows.
