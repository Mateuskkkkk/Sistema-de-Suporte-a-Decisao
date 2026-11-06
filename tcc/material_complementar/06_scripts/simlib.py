"""Utilitários para executar o simulador do repositório diretamente (sem HTTP)."""
import os, sys
import numpy as np, pandas as pd
BACKEND = "/home/user/Sistema-de-Suporte-a-Decisao/backend"
sys.path.insert(0, BACKEND)
_cwd = os.getcwd(); os.chdir(BACKEND)
import main  # noqa
os.chdir(_cwd)
main.DB_PATH = os.path.join(BACKEND, "banco_site.db")
import sqlite3
MESES = list(main.ordem_meses.keys())

def acude(cod):
    c = sqlite3.connect(main.DB_PATH)
    r = c.execute('select CORPO, COD, "CAPAC (m³)", "Est. Evap." from acudes where COD=?', (str(cod),)).fetchone()
    c.close()
    return dict(nome=main.corrigir_mojibake(r[0]), cod=str(r[1]), capacidade=float(r[2]) / 1e6, est_evap=str(r[3]))

def faixas(vm1, vm2, vm3, rac=(0, 50, 70), nomes=("Alerta", "Seca", "Seca Severa")):
    out = []
    for nome, curva, r in zip(nomes, (vm1, vm2, vm3), rac):
        d = {"Faixa": nome, "Racionamento": float(r), "NomeFaixaNormal": "Normal"}
        d.update({m: float(v) for m, v in zip(MESES, curva)})
        out.append(d)
    return out

def res(cod, vol_pct, demanda_lps, gatilho=0.0, plano=None):
    a = acude(cod)
    a.update(vol_inicial=a["capacidade"] * vol_pct / 100.0, demanda=demanda_lps / 1000.0,
             gatilho=float(gatilho), plano_secas_custom=plano)
    return a

def simular(reservatorios, modo="Individual", conjunta_lps=0.0, ini=("JAN", 1911), fim=("DEZ", 2021),
            niveis_meta=False, cenario=None, atendimento=100.0):
    req = main.SimulacaoRequest(reservatorios=reservatorios, modo=modo, vazao_conjunta=conjunta_lps / 1000.0,
                                atendimento_transferencia=atendimento, mes_inicial=ini[0], ano_inicial=ini[1],
                                mes_final=fim[0], ano_final=fim[1], usar_niveis_meta=niveis_meta,
                                cenario_hidrossistema=cenario)
    out = main.processar_simulacao_api(req)
    return {r["reservatorio"]: pd.DataFrame(r["dados"]) for r in out["resultados"]}

K = 2.592  # hm³/mês por m³/s

def indicadores(df, cap):
    sol = df["Demanda Solicitada (m³/s)"] * K
    apl = sol * (1 - df["Racionamento (%)"] / 100)
    at = df["Demanda Atendida (m³/s)"] * K
    deficit = (apl - at).clip(lower=0)
    return {
        "meses": len(df),
        "falhas": int((df["Falha"] == "Sim").sum()),
        "atend_aplicada_pct": 100 * at.sum() / apl.sum() if apl.sum() > 0 else 100.0,
        "atend_nominal_pct": 100 * at.sum() / sol.sum() if sol.sum() > 0 else 100.0,
        "deficit_hm3": float(deficit.sum()),
        "deficit_rac_hm3": float((sol - at).clip(lower=0).sum()),
        "vmin_hm3": float(df["Armazenamento Final"].min()),
        "vmin_pct": 100 * float(df["Armazenamento Final"].min()) / cap,
        "vmed_pct": 100 * float(df["Armazenamento Final"].mean()) / cap,
        "vert_hm3": float(df["Vertimento (hm³)"].sum()),
        "evap_hm3": float(df["Evaporação (hm³)"].sum()),
        "meses_transf_rec": int((df["Transferência Recebida (m³/s)"] > 1e-12).sum()),
        "meses_transf_env": int((df["Transferência Enviada (m³/s)"] > 1e-12).sum()),
        "estados": df["Modo Operação"].value_counts().to_dict(),
    }

def residuo(df):
    """ε = Vf − (Vi + I + Tr − Te − R − E − S)."""
    I = df["Afluências (hm³/mês)"]
    tr = (df["Transferência Recebida (m³/s)"] - df["Transferência Enviada (m³/s)"]) * K
    R = df["Demanda Atendida (m³/s)"] * K
    return df["Armazenamento Final"] - (df["Armazenamento Inicial"] + I + tr - R - df["Evaporação (hm³)"] - df["Vertimento (hm³)"])
