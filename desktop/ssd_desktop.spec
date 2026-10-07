# Especificação do PyInstaller para a versão desktop.
# Uso (na raiz do repositório):  pyinstaller desktop/ssd_desktop.spec --noconfirm
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

raiz = Path(SPECPATH).parent

datas = [
    (str(raiz / "frontend" / "dist"), "frontend/dist"),
    (str(raiz / "backend" / "banco_site.db"), "backend"),
]
binaries = []
hiddenimports = ["main", "simulador", "optimizer_engine", "forecast_engine"]

# pacotes com bibliotecas nativas ou arquivos de dados que precisam ir inteiros
for pacote in ["xgboost", "pyswarms"]:
    d, b, h = collect_all(pacote)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(raiz / "desktop" / "ssd_desktop.py")],
    pathex=[str(raiz / "backend")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "notebook", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SSD-Reservatorios",
    console=False,  # sem janela de terminal
    icon=str(raiz / "desktop" / "icone.ico") if (raiz / "desktop" / "icone.ico").exists() else None,
)

# pasta única (onedir): abre bem mais rápido que um .exe único, que precisaria
# extrair todas as bibliotecas a cada execução
coll = COLLECT(exe, a.binaries, a.datas, name="SSD-Reservatorios")
