"""Sensibilidade de Flor do Campo ao fator de evaporação, com a Regra 2 do plano de secas (Tabela de sensibilidade com a Regra 2)."""
from simlib import *

VM1 = [36.6, 34.6, 33.3, 32.5, 32.0, 30.9, 59.7, 56.2, 52.3, 48.3, 43.9, 39.9]
VM2 = [22.2, 20.7, 19.7, 19.2, 19.0, 18.3, 39.9, 37.2, 34.1, 31.0, 27.6, 24.6]
VM3 = [7.9, 7.0, 6.5, 6.3, 6.3, 5.9, 18.3, 16.6, 14.8, 12.9, 11.0, 9.2]
RAC = (8.33, 33.33, 55.00)  # Regra 2: 60 L/s -> 55 / 40 / 27 L/s no açude
FATOR = {"evap": 1.0}

# a leitura das lâminas mudou de nome entre versões do repositório
if hasattr(main, "ler_evaporacao_mensal"):
    _orig = main.ler_evaporacao_mensal
    main.ler_evaporacao_mensal = lambda *a, **k: _orig(*a, **k) * FATOR["evap"]
else:
    _orig = main.carregar_serie_simulador
    def _serie(*a, **k):
        s = _orig(*a, **k); s["evaporacao_mm"] = s["evaporacao_mm"] * FATOR["evap"]; return s
    main.carregar_serie_simulador = _serie

print("fator;falhas_sem;falhas_regra2;meses_racionamento;normal_pct")
for fe in (0.8, 0.9, 1.0, 1.1, 1.2):
    FATOR["evap"] = fe
    sem = simular([res(172, 50, 60)])["Flor do Campo"]
    com = simular([res(172, 50, 60, plano=faixas(VM1, VM2, VM3, rac=RAC))], niveis_meta=True)["Flor do Campo"]
    normal = 100 * (com["Modo Operação"] == "Normal").mean()
    print(f"{fe};{(sem['Falha'] == 'Sim').sum()};{(com['Falha'] == 'Sim').sum()};"
          f"{(com['Racionamento (%)'] > 0).sum()};{normal:.1f}")
