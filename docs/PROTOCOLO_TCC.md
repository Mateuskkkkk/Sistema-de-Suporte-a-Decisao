# Protocolo de verificação e avaliação do TCC

Este documento organiza as evidências computacionais requeridas pelo roteiro estrutural do TCC **Desenvolvimento e avaliação de um sistema de suporte à decisão para operação de reservatórios no Ceará**.

## Escopo acadêmico ativo

O TCC considera três contribuições integradas:

1. simulador mensal de balanço hídrico, com operações Individual, Série e Paralelo;
2. otimizador de níveis meta por PSO;
3. cálculo de vazões regularizadas por garantia de atendimento.

O módulo de previsão de afluências permanece preservado no repositório para continuidade do projeto, mas está fora do escopo do TCC e não integra a navegação principal desta branch.

## Terminologia de avaliação

Na ausência de entradas históricas operacionais completas e de volumes observados independentes, os resultados devem ser apresentados como:

- verificação numérica;
- verificação funcional;
- testes de integração;
- avaliação computacional do PSO;
- análise de sensibilidade.

A expressão **validação hidrológica** deve ser usada apenas quando o modelo for confrontado com volumes observados usando afluências, retiradas, evaporação, transferências e liberações historicamente praticadas.

## Matriz de verificação

| Componente | Procedimento | Evidência esperada |
|---|---|---|
| Balanço hídrico | cálculo independente de meses selecionados | diferença e resíduo mensal |
| Limites físicos | casos controlados | volume entre zero e capacidade |
| Individual | série artificial simples | atendimento e falhas previstos |
| Série | testes acima e abaixo do gatilho | transferência no mês esperado |
| Paralelo | demanda superior à unidade prioritária | redistribuição sem excesso |
| PSO | repetições com sementes diferentes | convergência, estabilidade e viabilidade |
| PSO reduzido | busca exaustiva discretizada | solução igual ou próxima da melhor conhecida |
| Vazões | busca automática e verificação com vizinhos | mesmo intervalo de vazão |
| Integração | curvas do PSO reaplicadas ao simulador | frequências preservadas |
| Interface/API | entradas válidas e inválidas | respostas e mensagens adequadas |

## Casos controlados automatizados

O arquivo `backend/test_tcc_protocol.py` implementa os seguintes casos artificiais:

- volume constante;
- retirada isolada;
- vertimento: 95 hm³ + 10 hm³ em reservatório de 100 hm³;
- falha: 2 hm³ disponíveis para retirada de 5 hm³;
- racionamento de 40% sobre demanda de 100 L/s;
- resíduo do balanço sem transferência;
- monotonicidade da garantia com o aumento da demanda;
- monotonicidade da vazão regularizada em relação à garantia;
- teste de Q* - delta e Q* + delta.

Executar, a partir de `backend/`:

```bash
python -m unittest test_tcc_protocol.py
```

Para executar a suíte disponível:

```bash
python -m unittest test_simulator_engine.py test_optimizer_engine.py test_tcc_protocol.py
```

## Evidências ainda necessárias para a versão final do TCC

1. planilha independente com aproximadamente 12 meses representativos do balanço;
2. 20 a 30 execuções do PSO com sementes diferentes e estatísticas de J;
3. gráfico de convergência do PSO;
4. busca exaustiva em um problema reduzido e discretizado;
5. tabela de frequências do otimizador e do simulador após reaplicação das curvas;
6. análise de sensibilidade alterando um parâmetro por vez;
7. testes de API para período invertido, demanda negativa, volume inicial inválido, reservatório inexistente, ausência de série, garantia fora do intervalo, curvas cruzadas e transferência incompleta.

## Critério de conclusão

A versão final deverá permitir afirmar, com base em evidências reproduzíveis, se:

- o balanço apresenta consistência numérica;
- os três modos de operação respondem como esperado nos casos controlados;
- o PSO produz curvas admissíveis e apresenta estabilidade aceitável entre execuções;
- a calculadora de vazões apresenta comportamento monotônico;
- a integração entre otimizador e simulador preserva as frequências dentro da tolerância definida.
