# Sistema de Suporte à Decisão

Sistema web para análise da operação de reservatórios, desenvolvido como artefato computacional do TCC **Desenvolvimento e avaliação de um sistema de suporte à decisão para operação de reservatórios no Ceará**.

## Escopo do TCC

A versão acadêmica integra três módulos:

- **Simulador de balanço hídrico mensal**;
- **Otimizador de níveis meta por PSO**;
- **Calculadora de vazões regularizadas por garantia de atendimento**.

O simulador representa operações **Individual**, **Série** e **Paralelo**, com afluência, evaporação, demanda, armazenamento, vertimento, transferências, falhas e regras de níveis meta.

O módulo de previsão de afluências permanece preservado no histórico do projeto para continuidade futura, mas não integra a navegação principal da branch `tcc-roteiro-verificacao-validacao` e está fora do escopo metodológico do TCC.

## Funcionalidades

- simulação mensal de reservatórios;
- carregamento de reservatórios e hidrossistemas cadastrados;
- operações Individual, Série e Paralelo;
- níveis meta com estados Normal, Alerta, Seca e Seca Severa;
- racionamento por estado operacional;
- otimização de níveis meta por Particle Swarm Optimization (PSO);
- aplicação das curvas otimizadas diretamente no simulador;
- cálculo de vazões regularizadas para diferentes garantias;
- gráficos, indicadores e tabelas mensais;
- exportação de resultados.

## Arquitetura

```text
Sistema-de-Suporte-a-Decisao/
├── backend/
│   ├── main.py
│   ├── optimizer_engine.py
│   ├── forecast_engine.py        # preservado, fora do escopo do TCC
│   ├── test_simulator_engine.py
│   ├── test_optimizer_engine.py
│   ├── test_tcc_protocol.py
│   ├── banco_site.db
│   └── requirements.txt
├── frontend/
│   ├── src/
│   └── package.json
├── docs/
│   └── PROTOCOLO_TCC.md
├── LICENSE
└── README.md
```

A camada de apresentação utiliza React/Vite, a API utiliza FastAPI, as rotinas de cálculo são implementadas em Python e os dados são armazenados em SQLite. A comunicação entre interface e servidor utiliza HTTP/JSON.

## Protocolo de verificação e avaliação

O TCC distingue **verificação da implementação** e **validação hidrológica**.

Sem volumes observados e entradas históricas operacionais completas, os resultados devem ser apresentados como:

- verificação numérica;
- verificação funcional;
- testes de integração;
- avaliação computacional do PSO;
- análise de sensibilidade.

A validação hidrológica somente deve ser utilizada quando o modelo for confrontado com volumes observados usando as entradas efetivamente praticadas no período.

O protocolo completo está em [`docs/PROTOCOLO_TCC.md`](docs/PROTOCOLO_TCC.md).

## Testes do protocolo

A branch do TCC inclui casos controlados para:

- volume constante;
- retirada isolada;
- vertimento;
- falha por indisponibilidade;
- racionamento;
- resíduo do balanço;
- monotonicidade da garantia com a demanda;
- monotonicidade da vazão regularizada;
- verificação de vazões imediatamente abaixo e acima de `Q*`.

A partir da pasta `backend/`:

```bash
python -m unittest test_tcc_protocol.py
```

Para executar a suíte principal do escopo:

```bash
python -m unittest test_simulator_engine.py test_optimizer_engine.py test_tcc_protocol.py
```

## Como usar

### Simulador

1. Abra a aplicação no navegador.
2. Entre em **Simulador**.
3. Escolha o reservatório ou hidrossistema.
4. Ajuste volume inicial, demanda, período e modo de operação.
5. Configure níveis meta e transferências quando aplicável.
6. Execute a simulação.
7. Analise gráficos, tabelas e indicadores.

### Otimizador

1. Entre em **Otimizador**.
2. Escolha o reservatório.
3. Configure período histórico, demanda, frequências e parâmetros do PSO.
4. Execute a otimização.
5. Analise curvas, frequências e função objetivo.
6. Use **Aplicar no Simulador** para testar a política encontrada.

### Vazões regularizadas

1. Entre em **Vazões**.
2. Selecione reservatório e período.
3. Informe o volume inicial.
4. Execute o cálculo.
5. Analise a relação entre garantia requerida e maior demanda constante atendida.

## Como rodar localmente

### Pré-requisitos

- Python 3.10 ou superior;
- Node.js 18 ou superior;
- npm.

### Backend

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r backend\requirements.txt
cd backend
..\.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Frontend

Em outro terminal:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Para desenvolvimento local, crie `frontend/.env` com:

```env
VITE_API_URL=http://127.0.0.1:8000
```

## Evidências ainda necessárias para o fechamento do TCC

Antes da versão final do trabalho, o protocolo prevê consolidar:

- memória de cálculo independente de aproximadamente 12 meses;
- 20 a 30 execuções do PSO com sementes diferentes;
- estatísticas da função objetivo;
- gráfico de convergência e distribuição das execuções;
- busca exaustiva em problema reduzido;
- tabela de integração entre otimizador e simulador;
- verificação completa das vazões vizinhas de `Q*`;
- análise de sensibilidade;
- tabela de testes da interface e da API.

## Licença

Este projeto está licenciado sob a **Apache License 2.0**. Consulte [LICENSE](LICENSE).

## Citação acadêmica

```text
MARTINS, F. M. B. Sistema de Suporte à Decisão: simulação da operação de reservatórios,
otimização de níveis meta e cálculo de vazões regularizadas. GitHub, 2026.
```
