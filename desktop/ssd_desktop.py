"""Versão desktop do Sistema de Suporte à Decisão.

Sobe a API (FastAPI) numa porta livre do próprio computador, serve a interface
já compilada (frontend/dist) e abre tudo numa janela nativa por meio do
pywebview. A janela usa o componente de navegador que já vem com o sistema
(WebView2 no Windows, WebKit no macOS e no Linux), então o executável não
carrega um navegador próprio. Sem pywebview, ou com a opção --navegador, a
interface abre no navegador padrão.
"""
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def pasta_recursos():
    # no executável, o PyInstaller extrai os arquivos em sys._MEIPASS
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


RAIZ = pasta_recursos()
BACKEND = RAIZ / "backend"
INTERFACE = RAIZ / "frontend" / "dist"

sys.path.insert(0, str(BACKEND))
os.environ.setdefault("BANCO_SITE_DB", str(BACKEND / "banco_site.db"))

# sem console (executável com janela), stdout e stderr não existem
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import uvicorn  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import main as api  # noqa: E402

TITULO = "Sistema de Suporte à Decisão"


def porta_livre():
    # --porta=8765 fixa a porta; sem ela, o sistema escolhe uma livre
    for arg in sys.argv[1:]:
        if arg.startswith("--porta="):
            return int(arg.split("=", 1)[1])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def criar_app():
    if not (INTERFACE / "index.html").is_file():
        raise SystemExit(
            f"Interface não encontrada em {INTERFACE}.\n"
            "Compile antes com: cd frontend && npm run build:desktop"
        )
    # as rotas /api/... já existem; o restante é servido pela interface
    api.app.mount("/", StaticFiles(directory=INTERFACE, html=True), name="interface")
    return api.app


def iniciar_servidor(app, porta, tempo_limite=30.0):
    config = uvicorn.Config(app, host="127.0.0.1", port=porta, log_level="warning", log_config=None)
    servidor = uvicorn.Server(config)
    threading.Thread(target=servidor.run, daemon=True).start()
    inicio = time.monotonic()
    while not servidor.started:
        if time.monotonic() - inicio > tempo_limite:
            raise SystemExit("O servidor interno não iniciou a tempo.")
        time.sleep(0.05)
    return servidor


def abrir_janela(url):
    try:
        import webview
    except ImportError:
        return False
    webview.settings["ALLOW_DOWNLOADS"] = True  # exportar Excel, PNG e configurações
    webview.create_window(TITULO, url, width=1440, height=900, min_size=(1000, 650))
    webview.start()
    return True


def executar():
    servidor = iniciar_servidor(criar_app(), porta_livre())
    url = f"http://127.0.0.1:{servidor.config.port}/"

    if "--navegador" not in sys.argv and abrir_janela(url):
        servidor.should_exit = True  # a janela foi fechada
        return

    webbrowser.open(url)
    print(f"{TITULO} em {url}\nFeche esta janela (ou Ctrl+C) para encerrar.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        servidor.should_exit = True


if __name__ == "__main__":
    executar()
