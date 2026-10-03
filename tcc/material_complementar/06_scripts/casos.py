"""Casos controlados executados diretamente sobre simular_sistema_n (código do repositório)."""
from simlib import *
import json
def serie(n, q=0.0, evap_mm=0.0):
    meses = [MESES[i % 12] for i in range(n)]
    return {"datas": [f"2000-{(i%12)+1:02d}" for i in range(n)], "meses": meses,
            "meses_num": np.array([(i % 12) + 1 for i in range(n)]), "vazoes_m3s": np.full(n, float(q)),
            "evaporacao_mm": np.full(n, float(evap_mm))}
def par(cap, v0, dem, gat=0.0, regras=None, area=1.0, cod="X"):
    return {"cod": cod, "cav_vol": np.array([0.0, 2 * cap]), "cav_area": np.array([area, area]),
            "regras_secas": regras or {}, "nome_faixa_normal": "Normal", "capacidade": cap, "vol_ini": v0,
            "demanda_nominal": dem, "gatilho": gat}
def tab(saida, t):
    return {k: (v[t] if not isinstance(v[t], (np.floating,)) else float(v[t])) for k, v in saida.items()}
C = []
def caso(nome, entrada, esperado, obtido, ok):
    C.append(dict(teste=nome, entrada=entrada, esperado=esperado, obtido=obtido, situacao="Aprovado" if ok else "Reprovado"))
tol = 1e-12
# 1 volume constante
s = main.simular_sistema_n([serie(12)], [par(10, 5, 0)], "Individual", 0)[0]
dv = float(np.abs(s["armazenamento_final"] - 5).max())
caso("Volume constante", "V0 = 5 hm³; I = 0; e = 0; D = 0; 12 meses", "V(t) = 5,000 hm³ em todos os meses",
     f"máx |V − 5| = {dv:.1e} hm³", dv < tol)
# 2 demanda isolada
s = main.simular_sistema_n([serie(12)], [par(10, 5, 0.1)], "Individual", 0)[0]
esp = 5 - 0.2592 * np.arange(1, 13)
dv = float(np.abs(s["armazenamento_final"] - esp).max())
caso("Aplicação isolada da demanda", "V0 = 5 hm³; I = 0; e = 0; D = 0,1 m³/s", "redução de 0,2592 hm³/mês; V(12) = 1,8896 hm³",
     f"V(12) = {s['armazenamento_final'][-1]:.4f} hm³; máx. desvio {dv:.1e}", dv < 1e-12)
# 3 evaporação área constante
s = main.simular_sistema_n([serie(1, evap_mm=200)], [par(10, 5, 0, area=1.0)], "Individual", 0)[0]
caso("Evaporação com área constante", "V0 = 5 hm³; A = 1 km²; e = 200 mm; I = D = 0", "E = 0,2000 hm³; V = 4,8000 hm³",
     f"E = {s['evaporacao_hm3'][0]:.4f} hm³; V = {s['armazenamento_final'][0]:.4f} hm³",
     abs(s['evaporacao_hm3'][0] - 0.2) < tol and abs(s['armazenamento_final'][0] - 4.8) < tol)
# 4 vertimento
s = main.simular_sistema_n([serie(3, q=1.0)], [par(10, 10, 0)], "Individual", 0)[0]
caso("Ocorrência de vertimento", "V0 = Vmáx = 10 hm³; Q = 1 m³/s; e = D = 0", "S = 2,592 hm³/mês; V = 10 hm³",
     f"S = {s['vertimento_hm3'][0]:.3f} hm³/mês; V = {s['armazenamento_final'][-1]:.3f} hm³",
     np.allclose(s['vertimento_hm3'], 2.592, atol=tol) and np.allclose(s['armazenamento_final'], 10, atol=tol))
# 5 falha
s = main.simular_sistema_n([serie(3)], [par(10, 0.5, 0.1)], "Individual", 0)[0]
r = s["demanda_atendida"] * K
caso("Falha por falta de volume", "V0 = 0,5 hm³; I = e = 0; D = 0,1 m³/s (0,2592 hm³/mês)",
     "R = 0,2592; 0,2408; 0 hm³; falhas nos meses 2 e 3; V ≥ 0",
     f"R = {r[0]:.4f}; {r[1]:.4f}; {r[2]:.4f} hm³; falhas = {list(s['falha'])}",
     np.allclose(r, [0.2592, 0.2408, 0], atol=1e-12) and list(s['falha']) == ['Não', 'Sim', 'Sim'] and s['armazenamento_final'].min() >= 0)
# 6 racionamento
regras = {m: [(20.0, 80.0, "Seca Severa"), (40.0, 50.0, "Seca"), (60.0, 20.0, "Alerta")] for m in MESES}
obt = []; ok = True
for v0, est, rac in [(7, "Normal", 0), (5, "Alerta", 20), (3, "Seca", 50), (1, "Seca Severa", 80)]:
    s = main.simular_sistema_n([serie(1)], [par(10, v0, 0.1, regras=regras)], "Individual", 0)[0]
    obt.append(f"{s['modo_operacao'][0]}: {s['demanda_atendida'][0]*1000:.0f} L/s")
    ok &= s['modo_operacao'][0] == est and abs(s['demanda_atendida'][0] - 0.1 * (1 - rac / 100)) < 1e-12
caso("Aplicação de racionamento", "V0 = 70, 50, 30 e 10% de 10 hm³; limites 60/40/20%; r = 20/50/80%; D = 100 L/s",
     "Normal 100; Alerta 80; Seca 50; Seca Severa 20 L/s", "; ".join(obt), ok)
# 6b fronteira
obt = []; ok = True
for pct, est in [(60.0 - 1e-6, "Alerta"), (60.0, "Alerta"), (60.0 + 1e-6, "Normal")]:
    s = main.simular_sistema_n([serie(1)], [par(10, pct / 10, 0.0, regras=regras)], "Individual", 0)[0]
    obt.append(s['modo_operacao'][0]); ok &= s['modo_operacao'][0] == est
caso("Fronteira de nível meta", "V0 = 60% − ε, 60% e 60% + ε (ε = 10⁻⁶ p.p.)", "Alerta; Alerta; Normal", "; ".join(obt), ok)
# 7 transferência em série
s = main.simular_sistema_n([serie(4), serie(4)], [par(10, 3.5, 0.2, gat=30, cod="A"), par(50, 40, 0.0, cod="B")], "Série", 0.1)
rec = s[0]["transferencia_recebida"]
caso("Ativação de transferência (Série)", "Receptor: 10 hm³, V0 = 35%, D = 200 L/s, gatilho 30%; fornecedor: 40 hm³; T = 100 L/s",
     "sem transferência no mês 1; transferência de 100 L/s a partir do mês 2",
     "T recebida (L/s) = " + ", ".join(f"{x*1000:.0f}" for x in rec),
     abs(rec[0]) < tol and np.allclose(rec[1:], 0.1, atol=1e-12) and np.allclose(s[1]["transferencia_enviada"], rec, atol=1e-12))
eps_serie = max(float(np.abs(s[i]["armazenamento_final"] - (s[i]["armazenamento_inicial"] + (s[i]["transferencia_recebida"] - s[i]["transferencia_enviada"]) * K - s[i]["demanda_atendida"] * K)).max()) for i in range(2))
# 8 paralelo
s = main.simular_sistema_n([serie(5), serie(5)], [par(10, 2.5, 0.0, gat=20, cod="A"), par(10, 8, 0.0, cod="B")], "Paralelo", 0.1)
resp = ["R1" if s[0]["demanda_atendida"][t] > 0 else "R2" for t in range(5)]
caso("Redistribuição da demanda conjunta (Paralelo)", "R1: 10 hm³, V0 = 25%, gatilho 20%; R2: 10 hm³, V0 = 80%; Dconj = 100 L/s",
     "R1 atende nos meses 1–2; R2 assume no mês 3", "responsável: " + ", ".join(resp), resp == ["R1", "R1", "R2", "R2", "R2"])
json.dump(dict(casos=C, eps_serie=eps_serie), open("casos.json", "w"), ensure_ascii=False, indent=1)
for c in C: print(c)
print("eps série", eps_serie)
