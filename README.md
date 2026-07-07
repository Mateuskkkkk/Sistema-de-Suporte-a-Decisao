# Sistema de Suporte à Decisão

Sistema web para simulação de balanço hídrico e otimização de níveis meta de reservatórios. A aplicação reúne, em uma única interface, o simulador histórico e o otimizador por séries históricas, permitindo calcular curvas de operação e aplicá-las diretamente no simulador.

## Funcionalidades

- Simulação mensal de reservatórios com afluência, evaporação, demanda, vertimento e falhas.
- Carregamento de hidrossistemas pré-configurados.
- Operação individual, em série ou em paralelo.
- Otimização de níveis meta com curvas de Alerta, Seca e Seca Severa.
- Aplicação das curvas otimizadas diretamente no simulador.
- Gráficos com faixas de criticidade hídrica.
- Tabelas de permanência, volumes históricos e resultados mensais.
- Exportação dos resultados da otimização em CSV e planilha Excel.
- Modo claro e modo escuro no topo da aplicação.

## Estrutura do Projeto

```text
Sistema-Hidrico-Unificado/
├── backend/
│   ├── main.py
│   ├── optimizer_engine.py
│   ├── banco_site.db
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── public/
│   │   └── favicon.svg
│   ├── src/
│   └── package.json
├── LICENSE
└── README.md
```

## Como Usar

### Simulador

1. Abra a aplicação no navegador.
2. Entre na aba **Simulador**.
3. Escolha um hidrossistema pré-configurado ou selecione reservatórios manualmente.
4. Ajuste volume inicial, demanda, período e modo de operação.
5. Clique em **Gerar Simulação**.
6. Analise os gráficos, tabelas e indicadores de atendimento.

### Otimizador

1. Entre na aba **Otimizador**.
2. Escolha o reservatório ou hidrossistema.
3. Configure período histórico, mês inicial, demandas e permanências desejadas.
4. Execute a otimização.
5. Analise as curvas de Alerta, Seca e Seca Severa.
6. Use as opções de exportação para salvar os resultados.

### Aplicar Curvas no Simulador

1. Rode uma otimização.
2. Clique em **Aplicar no Simulador**.
3. A aplicação volta para o simulador com as curvas carregadas.
4. Execute a simulação para avaliar o comportamento do sistema com os níveis meta otimizados.

## Como Rodar Localmente

### Pré-requisitos

- Python 3.10 ou superior.
- Node.js 18 ou superior.
- npm.

### 1. Clonar o Repositório

```bash
git clone https://github.com/Mateuskkkkk/Sistema-de-Suporte-a-Decisao.git
cd Sistema-de-Suporte-a-Decisao
```

Se o repositório local estiver com o nome `Sistema-Hidrico-Unificado`, entre nessa pasta normalmente:

```bash
cd Sistema-Hidrico-Unificado
```

### 2. Criar e Preparar o Ambiente Python

No Windows:

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r backend\requirements.txt
```

### 3. Rodar o Backend

```bash
cd backend
..\.venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

A API ficará disponível em:

```text
http://127.0.0.1:8000
```

### 4. Rodar o Frontend

Em outro terminal:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

A interface ficará disponível em:

```text
http://127.0.0.1:5173
```

## Configuração da API no Frontend

O frontend lê a variável `VITE_API_URL`. Para desenvolvimento local, use:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Você pode criar um arquivo `frontend/.env` com esse conteúdo. Em produção, substitua pelo endereço público da API, por exemplo:

```env
VITE_API_URL=https://sua-api.onrender.com
```

## Licença

Este projeto está licenciado sob a **Apache License 2.0**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

## Citação Acadêmica

Caso este sistema seja utilizado em trabalhos, artigos, relatórios ou pesquisas, recomenda-se citar o repositório e o autor do projeto. Uma forma simples de citação é:

```text
MARTINS, F. M. B. . Sistema de Suporte à Decisão: simulador de balanço hídrico e otimizador de níveis meta. GitHub, 2026. Disponível em: https://github.com/Mateuskkkkk/Sistema-de-Suporte-a-Decisao. Acesso em: x xxx. xxxx.
```
