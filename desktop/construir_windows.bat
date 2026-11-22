@echo off
rem Gera a versao desktop para Windows em dist\SSD-Reservatorios\
rem Requisitos: Python 3.10+ e Node.js 18+ instalados.
setlocal
cd /d "%~dp0\.."

echo [1/4] Ambiente Python...
if not exist .venv-desktop python -m venv .venv-desktop || goto :erro
call .venv-desktop\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r desktop\requirements-desktop.txt || goto :erro

echo [2/4] Interface...
pushd frontend
call npm install || goto :erro
call npm run build:desktop || goto :erro
popd

echo [3/4] Empacotando...
pyinstaller desktop\ssd_desktop.spec --noconfirm --clean || goto :erro

echo [4/4] Pronto: dist\SSD-Reservatorios\SSD-Reservatorios.exe
goto :fim

:erro
echo.
echo Falhou. Veja a mensagem acima.
exit /b 1

:fim
endlocal
