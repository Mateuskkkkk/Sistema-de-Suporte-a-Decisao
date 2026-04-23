# Sistema-de-Suporte-a-Decisao
🌊 Simulador de Balanço Hídrico
Este projeto é um Sistema de Suporte à Decisão (SSD) para simulação de balanço hídrico de reservatórios, permitindo análises em modos Individual, Série e Paralelo.

🛠️ Pré-requisitos
Antes de começar, você precisará ter instalado em sua máquina:

Node.js (versão 18 ou superior)

Python (versão 3.9 ou superior)

📁 Estrutura do Projeto
O repositório está dividido em duas partes principais:

/frontend: Interface em React.

/backend: API em FastAPI.

🚀 Como Rodar o Back-end (API)
Navegue até a pasta do servidor:

Bash
cd backend
(Opcional) Crie um ambiente virtual:

Bash
python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
Instale as dependências:

Bash
pip install fastapi uvicorn pandas numpy scipy pydantic
Banco de Dados: Certifique-se de que o arquivo banco_site.db está na raiz da pasta backend.

Inicie o servidor:

Bash
uvicorn main:app --reload
A API estará disponível em: http://localhost:8000

💻 Como Rodar o Front-end (Interface)
Navegue até a pasta do frontend:

Bash
cd frontend
Instale as dependências do projeto:

Bash
npm install
Configuração de Ambiente:

Localize o arquivo .env.example.

Crie uma cópia e renomeie para .env.

Certifique-se de que a variável VITE_API_URL aponta para a sua API local:

Plaintext
VITE_API_URL=http://localhost:8000
Inicie o modo de desenvolvimento:

Bash
npm run dev
Acesse no navegador através da URL indicada no terminal (geralmente http://localhost:5173)
