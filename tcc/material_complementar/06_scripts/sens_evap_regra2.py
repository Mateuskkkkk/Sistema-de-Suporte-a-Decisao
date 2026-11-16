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

K = 2.592
print("fator;falhas_sem;falhas_regra2;meses_racionamento;normal_pct;"
      "evap_sobre_afluencia_pct;agua_poupada_hm3;fracao_poupada_evaporada_pct;nao_fornecido_acude_hm3")
for fe in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
    FATOR["evap"] = fe
    sem = simular([res(172, 50, 60)])["Flor do Campo"]
    com = simular([res(172, 50, 60, plano=faixas(VM1, VM2, VM3, rac=RAC))], niveis_meta=True)["Flor do Campo"]
    normal = 100 * (com["Modo Operação"] == "Normal").mean()
    aflu = sem["Afluências (hm³/mês)"].sum()
    ret_sem = (sem["Demanda Atendida (m³/s)"] * K).sum()
    ret_com = (com["Demanda Atendida (m³/s)"] * K).sum()
    poupada = ret_sem - ret_com
    evap_extra = com["Evaporação (hm³)"].sum() - sem["Evaporação (hm³)"].sum()
    nao_fornecido = (com["Demanda Solicitada (m³/s)"] * K).sum() - ret_com
    print(f"{fe};{(sem['Falha'] == 'Sim').sum()};{(com['Falha'] == 'Sim').sum()};"
          f"{(com['Racionamento (%)'] > 0).sum()};{normal:.1f};"
          f"{100 * sem['Evaporação (hm³)'].sum() / aflu:.1f};{poupada:.2f};"
          f"{100 * evap_extra / poupada:.1f};{nao_fornecido:.2f}")
