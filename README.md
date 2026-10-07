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
- Indicadores de desempenho de confiabilidade, resiliência e vulnerabilidade (Hashimoto, Stedinger e Loucks, 1982).
- Indicadores do sistema: falha da demanda conjunta (modo Paralelo) e meses/volume transferido (modo Série).
- Histerese opcional no gatilho de transferência, para evitar liga-desliga mês a mês.
- Cenários hidrológicos: série histórica, fatores fixos ou personalizados de afluência, seca repetida e reamostragem anual com semente.
- Comparação lado a lado de dois cenários (fixe um resultado e simule outro).
- Salvar e abrir a configuração da simulação em arquivo JSON.
- Marcação dos meses com falha e dos períodos com racionamento no gráfico de volume.
- Validação dos campos com mensagens em português, no formulário e na API.

As opções novas vêm desligadas por padrão: com os valores padrão, o simulador reproduz exatamente os resultados da versão usada no TCC.

## Estrutura do Projeto

```text
Sistema-Hidrico-Unificado/
├── backend/
│   ├── main.py               # API (FastAPI) e acesso ao banco
│   ├── simulador.py          # motor de balanço hídrico e indicadores
│   ├── optimizer_engine.py   # otimizador de níveis meta
│   ├── banco_site.db         # séries de vazões, evaporação e CAV
│   ├── test_*.py             # testes automatizados
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

### 5. Rodar os Testes

```bash
cd backend
python -m unittest discover -p "test_*.py"
```

Os testes incluem os resultados de referência do TCC (por exemplo, 142 meses de falha em Mundaú e 257 meses de transferência em Fogareiro–Quixeramobim).

## Variáveis de Ambiente do Backend

- `BANCO_SITE_DB`: caminho alternativo para o banco SQLite (padrão: `backend/banco_site.db`, independente da pasta de onde o servidor é iniciado).
- `CORS_ORIGINS`: lista de origens permitidas separadas por vírgula, por exemplo `https://meu-site.com`. Sem ela, qualquer origem é aceita, sem credenciais.

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
