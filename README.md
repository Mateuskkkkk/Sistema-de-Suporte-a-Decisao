# Sistema de Suporte à Decisão

Aplicacao unificada para simulacao historica de reservatorios e otimizacao de niveis meta.

## Estrutura

- `backend/`: API FastAPI unica.
- `backend/main.py`: rotas do simulador e integracao do motor historico.
- `backend/optimizer_engine.py`: rotas e motor PSO do otimizador.
- `frontend/`: interface React com abas para Simulador e Otimizador.

## Como Rodar

### Backend

```bash
cd backend
..\.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Se ainda nao existir `.venv`:

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r backend\requirements.txt
```

### Frontend

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Acesse `http://127.0.0.1:5173/`.

## Fluxo Integrado

1. Abra a aba `Otimizador`.
2. Calcule as curvas guia por PSO.
3. Clique em `Aplicar no Simulador`.
4. O app volta para o `Simulador` e carrega as curvas na aba `Niveis Meta`.
5. Rode a simulacao; as curvas otimizadas entram como `plano_secas_custom`.

O simulador tambem usa o mesmo passo mensal de dinamica de volume do otimizador, garantindo consistencia entre a simulacao historica e as curvas otimizadas.
