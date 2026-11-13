"""Rotas da previsão de afluências por reamostragem k-NN.

O cálculo está em previsao_knn.py. Este módulo lê o banco (afluências e índices
climáticos), monta o histórico, chama a previsão e, na rota de simulação, roda
o simulador existente (simulador.simular_sistema_n) com os cenários previstos
no lugar da série histórica. Nenhuma rota baixa dados: os índices vêm da tabela
indices_climaticos, preenchida por scripts/atualizar_indices_noaa.py.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError, model_validator

import previsao_knn as pk
from simulador import simular_sistema_n

router = APIRouter(prefix="/api/previsao-knn", tags=["previsao-knn"])

ANO_INICIAL_PADRAO = 1912
CONJUNTO_PADRAO = "ambos"


# ---------------------------------------------------------------------------
# Leitura do banco
# ---------------------------------------------------------------------------
def _conexao():
    from main import get_db_connection  # importação tardia: main inclui este roteador

    return get_db_connection()


def ler_indices(conexao) -> Dict[str, Dict[pk.Mes, float]]:
    try:
        linhas = conexao.execute("SELECT ano, mes, indice, valor FROM indices_climaticos").fetchall()
    except sqlite3.OperationalError:
        raise HTTPException(
            status_code=503,
            detail="A tabela indices_climaticos não existe. Rode scripts/atualizar_indices_noaa.py.",
        )
    indices: Dict[str, Dict[pk.Mes, float]] = {}
    for ano, mes, indice, valor in linhas:
        if valor is not None and np.isfinite(valor):
            indices.setdefault(indice, {})[(int(ano), int(mes))] = float(valor)
    for nome in ("nino34", "tna", "tsa"):
        if nome not in indices:
            raise HTTPException(status_code=503, detail=f"Índice {nome} ausente na tabela indices_climaticos.")
    return indices


def ler_fonte_indices(conexao) -> List[dict]:
    try:
        linhas = conexao.execute(
            "SELECT indice, descricao, fonte, periodo_base, primeiro_mes, ultimo_mes, atualizado_em FROM indices_climaticos_fonte"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    campos = ("indice", "descricao", "fonte", "periodo_base", "primeiro_mes", "ultimo_mes", "atualizado_em")
    return [dict(zip(campos, linha)) for linha in linhas]


def ler_vazoes_hm3(conexao, nome: str):
    from main import ler_vazoes_brutas, resolver_nome_reservatorio

    nome_exato = resolver_nome_reservatorio(conexao, nome)
    vazoes = ler_vazoes_brutas(conexao, nome_exato)
    return nome_exato, {k: v * pk.HM3_POR_M3S for k, v in vazoes.items()}


# ---------------------------------------------------------------------------
# Requisições
# ---------------------------------------------------------------------------
class CenariosRequest(BaseModel):
    reservatorio: str
    mes_emissao: int = Field(12, ge=1, le=12)
    ano_alvo: Optional[int] = None
    k: Optional[int] = Field(None, ge=1)
    n_cenarios: int = Field(pk.N_CENARIOS_PADRAO, ge=10, le=10000)
    semente: int = pk.SEMENTE_PADRAO
    conjunto: Optional[str] = None
    preditores: Optional[List[str]] = None
    ano_inicial_historico: int = ANO_INICIAL_PADRAO
    ano_final_historico: Optional[int] = None
    incluir_cenarios: bool = True

    @model_validator(mode="after")
    def validar(self):
        if self.preditores is None:
            nome = self.conjunto or CONJUNTO_PADRAO
            if nome not in pk.CONJUNTOS:
                raise ValueError(f"Conjunto de preditores desconhecido: {nome}.")
            self.preditores = list(pk.CONJUNTOS[nome])
        try:
            pk.validar_preditores(self.preditores, self.mes_emissao)
        except ValueError as erro:
            raise ValueError(str(erro))
        if self.ano_final_historico is not None and self.ano_final_historico < self.ano_inicial_historico:
            raise ValueError("O ano final do histórico é anterior ao inicial.")
        return self


class SimularPrevisaoRequest(CenariosRequest):
    reservatorios: List[dict]
    modo: str = "Individual"
    vazao_conjunta: float = 0.0
    atendimento_transferencia: float = 100.0
    usar_niveis_meta: bool = False
    cenario_hidrossistema: Optional[str] = None
    incluir_cenarios: bool = False


# ---------------------------------------------------------------------------
# Previsão
# ---------------------------------------------------------------------------
def _ano_alvo_padrao(mes_emissao, preditores, indices, vazoes) -> Optional[int]:
    ano_corrente = dt.date.today().year
    for ano in range(ano_corrente + 1, ANO_INICIAL_PADRAO - 1, -1):
        if pk.vetor_estado(ano, mes_emissao, preditores, indices, vazoes) is not None:
            return ano
    return None


def calcular_previsao(req: CenariosRequest, conexao, anos_permitidos: Optional[set] = None) -> dict:
    indices = ler_indices(conexao)
    nome, vazoes = ler_vazoes_hm3(conexao, req.reservatorio)
    preditores = tuple(req.preditores)
    ano_alvo = req.ano_alvo
    aviso = None
    if ano_alvo is None:
        ano_alvo = _ano_alvo_padrao(req.mes_emissao, preditores, indices, vazoes)
        if ano_alvo is None:
            raise HTTPException(status_code=400, detail="Não há ano com os preditores disponíveis para essa emissão.")
        if ano_alvo <= dt.date.today().year - 1:
            aviso = (f"O ano-alvo mais recente com todos os preditores é {ano_alvo}: "
                     "a série de afluências ou os índices do banco não cobrem os anos seguintes.")
    x = pk.vetor_estado(ano_alvo, req.mes_emissao, preditores, indices, vazoes)
    if x is None:
        raise HTTPException(
            status_code=400,
            detail=f"Os preditores de {ano_alvo} não estão disponíveis para a emissão em {pk.MESES_NOME[req.mes_emissao - 1]}.",
        )

    ano_fim = req.ano_final_historico or max(a for a, _ in vazoes)
    anos = [a for a in range(req.ano_inicial_historico, ano_fim + 1) if a != ano_alvo]
    if anos_permitidos is not None:
        anos = [a for a in anos if a in anos_permitidos]
    hist = pk.montar_historico(anos, req.mes_emissao, preditores, indices, vazoes)
    n = len(hist["anos"])
    if n < 3:
        raise HTTPException(status_code=400, detail="Histórico insuficiente: menos de 3 anos com dados completos.")
    if req.k is not None and req.k > n:
        raise HTTPException(status_code=400, detail=f"k = {req.k} é maior que o número de anos disponíveis ({n}).")

    iq = pk.indices_quadra(req.mes_emissao)
    prev = pk.prever(hist["X"], hist["anos"], hist["S"], x, k=req.k, n_cenarios=req.n_cenarios,
                     semente=req.semente, indices_quadra_horizonte=iq)
    clim = pk.prever(hist["X"], hist["anos"], hist["S"], None, n_cenarios=req.n_cenarios,
                     semente=req.semente, climatologia=True, indices_quadra_horizonte=iq)
    observado = pk.sequencia(ano_alvo, req.mes_emissao, vazoes)
    _, _, media, desvio = pk.padronizar(hist["X"], x)

    resposta = {
        "reservatorio": nome,
        "ano_alvo": ano_alvo,
        "aviso": aviso,
        "mes_emissao": req.mes_emissao,
        "emissao": f"{pk.MESES_NOME[req.mes_emissao - 1]}/{ano_alvo + pk.emissao_relativa(req.mes_emissao)[0]}",
        "meses": pk.rotulos_horizonte(req.mes_emissao, ano_alvo),
        "indices_quadra": iq,
        "unidade": "hm³",
        "preditores": [{"nome": p, "descricao": pk.PREDITORES[p], "valor_atual": float(v),
                        "padronizado": float((v - m) / d)} for p, v, m, d in zip(preditores, x, media, desvio)],
        "historico": {
            "anos_usados": n,
            "primeiro_ano": int(hist["anos"][0]),
            "ultimo_ano": int(hist["anos"][-1]),
            "anos_excluidos": len(hist["excluidos"]),
            "excluidos": hist["excluidos"],
        },
        "k": prev["k"],
        "n_cenarios": req.n_cenarios,
        "semente": req.semente,
        "vizinhos": prev["vizinhos"],
        "quantis_mensais": prev["quantis_mensais"],
        "quantis_acumulados": prev["quantis_acumulados"],
        "quantis_quadra": prev["quantis_quadra"],
        "climatologia": {
            "quantis_mensais": clim["quantis_mensais"],
            "quantis_acumulados": clim["quantis_acumulados"],
            "quantis_quadra": clim["quantis_quadra"],
        },
        "observado": None if observado is None else observado.tolist(),
        "quadra_observada": None if observado is None else float(observado[iq].sum()),
        "fonte_indices": ler_fonte_indices(conexao),
    }
    if req.incluir_cenarios:
        resposta["cenarios"] = prev["cenarios"]
        resposta["anos_sorteados"] = prev["anos_sorteados"]
    return {"resposta": resposta, "_prev": prev, "_clim": clim, "_hist": hist}


@router.get("/opcoes")
def opcoes():
    conexao = _conexao()
    try:
        fontes = ler_fonte_indices(conexao)
    finally:
        conexao.close()
    return pk.para_json({
        "meses_emissao_validados": [10, 11, 12, 1],
        "conjuntos": {k: list(v) for k, v in pk.CONJUNTOS.items()},
        "preditores": pk.PREDITORES,
        "conjunto_padrao": CONJUNTO_PADRAO,
        "n_cenarios_padrao": pk.N_CENARIOS_PADRAO,
        "semente_padrao": pk.SEMENTE_PADRAO,
        "ano_inicial_historico": ANO_INICIAL_PADRAO,
        "fonte_indices": fontes,
    })


@router.post("/cenarios")
def gerar_cenarios(req: CenariosRequest):
    conexao = _conexao()
    try:
        resultado = calcular_previsao(req, conexao)
    finally:
        conexao.close()
    return pk.para_json(resultado["resposta"])


# ---------------------------------------------------------------------------
# Simulação com os cenários
# ---------------------------------------------------------------------------
def _simular_por_ano(anos_unicos, req, reservatorios, params, sequencias, evaporacoes, rotulos, meses_num):
    """Roda o simulador uma vez por ano sorteado; cenários com o mesmo ano têm o mesmo resultado."""
    resultados = {}
    for ano in anos_unicos:
        series = []
        for i, r in enumerate(reservatorios):
            series.append({
                "nome_reservatorio": r.nome,
                "datas": rotulos,
                "meses": [pk.MESES_NOME[m - 1] for m in meses_num],
                "meses_num": meses_num,
                "vazoes_m3s": sequencias[i][ano] / pk.HM3_POR_M3S,
                "evaporacao_mm": evaporacoes[i][meses_num - 1],
            })
        saidas = simular_sistema_n(series, params, req.modo, req.vazao_conjunta, req.atendimento_transferencia,
                                   req.cenario_hidrossistema if req.usar_niveis_meta else None)
        resultados[ano] = [
            {"volume": np.asarray(s["armazenamento_final"], dtype=float),
             "falha": np.asarray(s["falha"]) == "Sim"}
            for s in saidas
        ]
    return resultados


def _resumir(anos_sorteados, por_ano, reservatorios):
    resumo = []
    for i, r in enumerate(reservatorios):
        volumes = np.array([por_ano[a][i]["volume"] for a in anos_sorteados])
        falhas = np.array([por_ano[a][i]["falha"] for a in anos_sorteados])
        garantia = 1.0 - falhas.mean(axis=1)
        resumo.append({
            "reservatorio": r.nome,
            "quantis_volume_hm3": pk.quantis(volumes),
            "quantis_volume_percentual": pk.quantis(volumes / float(r.capacidade) * 100.0),
            "garantia_por_cenario": garantia,
            "garantia": {
                "media": float(garantia.mean()),
                **{k: v[0] for k, v in pk.quantis(garantia[:, None]).items()},
            },
            "probabilidade_de_falha": float((falhas.any(axis=1)).mean()),
        })
    return resumo


@router.post("/simular")
def simular_com_previsao(req: SimularPrevisaoRequest):
    from main import Reservatorio, carregar_regras_simulador, ler_cav, ler_evaporacao_mensal

    try:
        reservatorios = [Reservatorio(**r) for r in req.reservatorios]
    except ValidationError as erro:
        raise HTTPException(status_code=422, detail=erro.errors(include_url=False))
    if not reservatorios:
        raise HTTPException(status_code=400, detail="Informe ao menos um reservatório.")

    conexao = _conexao()
    try:
        # sequências de todos os reservatórios; o histórico só usa anos completos em todos
        horiz = pk.horizonte(req.mes_emissao)
        sequencias, anos_comuns = [], None
        for r in reservatorios:
            _, vazoes_r = ler_vazoes_hm3(conexao, r.nome)
            seq_r = {}
            for ano in {a for a, _ in vazoes_r}:
                s = pk.sequencia(ano, req.mes_emissao, vazoes_r)
                if s is not None:
                    seq_r[ano] = s
            sequencias.append(seq_r)
            anos_comuns = set(seq_r) if anos_comuns is None else anos_comuns & set(seq_r)
        resultado = calcular_previsao(req, conexao, anos_permitidos=anos_comuns)
        params, evaporacoes = [], []
        for r in reservatorios:
            cav_vol, cav_area = ler_cav(conexao, r.cod, r.capacidade)
            params.append({
                "cod": str(r.cod),
                "cav_vol": cav_vol,
                "cav_area": cav_area,
                "regras_secas": carregar_regras_simulador(conexao, r, req.usar_niveis_meta),
                "nome_faixa_normal": (
                    r.plano_secas_custom[0].NomeFaixaNormal
                    if r.plano_secas_custom and r.plano_secas_custom[0].NomeFaixaNormal else "Acima do Teto"
                ),
                "capacidade": float(r.capacidade),
                "vol_ini": float(r.vol_inicial),
                "demanda_nominal": float(r.demanda),
                "gatilho": float(r.gatilho),
            })
            evaporacoes.append(ler_evaporacao_mensal(conexao, r.est_evap))
    finally:
        conexao.close()

    resposta = resultado["resposta"]
    meses_num = np.array([m for _, m in horiz], dtype=np.int16)
    rotulos = resposta["meses"]
    prev, clim = resultado["_prev"], resultado["_clim"]
    anos_unicos = sorted(set(prev["anos_sorteados"]) | set(clim["anos_sorteados"]))
    try:
        por_ano = _simular_por_ano(anos_unicos, req, reservatorios, params, sequencias, evaporacoes, rotulos, meses_num)
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    resposta["simulacao"] = {
        "modo": req.modo,
        "previsao": _resumir(prev["anos_sorteados"], por_ano, reservatorios),
        "climatologia": _resumir(clim["anos_sorteados"], por_ano, reservatorios),
    }
    return pk.para_json(resposta)
