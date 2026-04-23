
# 🌊 Simulador de Balanço Hídrico - SSD

Este é um **Sistema de Suporte à Decisão (SSD)** desenvolvido para a simulação e análise do balanço hídrico de reservatórios. O sistema permite modelar operações em níveis individuais ou integrados (Série e Paralelo), sendo uma ferramenta essencial para o planejamento de recursos hídricos.

---

## 🚀 Como Executar o Projeto Localmemnte

O projeto é dividido em duas partes: **Backend** (API em Python) e **Frontend** (Interface em React).

### 📋 Pré-requisitos

Antes de começar, você precisará ter instalado:
* [Python 3.9+](https://www.python.org/)
* [Node.js 18+](https://nodejs.org/)

---

## ⚙️ Configuração do Back-end (API)

1. **Acesse a pasta do servidor:**
   ```bash
   cd backend

2. **Instale as dependências necessárias:**
   ```bash
   pip install fastapi uvicorn pandas numpy scipy pydantic
   ```

3. **Banco de Dados:**
   Certifique-se de que o arquivo `banco_site.db` está localizado na raiz da pasta `backend`. Sem ele, a API não conseguirá consultar os dados dos açudes.

4. **Inicie o servidor:**
   ```bash
   uvicorn main:app --reload
   ```
   *A API estará rodando em: `http://localhost:8000`*

---

## 💻 Configuração do Front-end (Interface)

1. **Acesse a pasta do cliente:**
   ```bash
   cd frontend
   ```

2. **Instale as dependências do Node:**
   ```bash
   npm install
   ```

3. **Variáveis de Ambiente:**
   * Localize o arquivo `.env.example` na raiz do frontend.
   * Crie uma cópia dele e renomeie para `.env`.
   * Verifique se o conteúdo aponta para a sua API local:
     ```env
     VITE_API_URL=http://localhost:8000
     ```

4. **Inicie a aplicação:**
   ```bash
   npm run dev
   ```
   *Abra o navegador no endereço indicado (geralmente `http://localhost:5173`)*

---

## 📊 Funcionalidades Principais

* **Modos de Operação:** Suporte para simulação Individual, em Série (transferência física) e Paralelo (vazão conjunta).
* **Níveis Meta:** Definição personalizada de faixas de volume (Normal, Alerta, Seca, etc.) e regras de racionamento.
* **Análise de Garantia:** Geração automática de Curvas de Permanência e estatísticas de atendimento de demanda.
* **Balanço Detalhado:** Cálculos mensais de evaporação (baseados em curvas cota-área-volume), afluência e vertimento.
* **Exportação de Dados:** Gere relatórios completos em formato **Excel (.xlsx)** ou **JSON**.

---


Desenvolvido por: Mateus
