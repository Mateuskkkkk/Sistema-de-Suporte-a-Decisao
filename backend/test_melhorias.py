"""Testes das funcionalidades da versão 1.1 e dos resultados de referência do TCC.

Os valores de referência (142 falhas em Mundaú, 42 meses de falha da demanda
conjunta em Carnaubal–Barragem do Batalhão, 257 meses de transferência em
Fogareiro–Quixeramobim) são os publicados no TCC e precisam continuar sendo
reproduzidos com as opções padrão.
"""
import math
import unittest

import numpy as np
from fastapi.testclient import TestClient

import main
import simulador

PLANO_FQ = main.FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID


def acude(nome):
    return next(r for r in main.listar_reservatorios() if r["CORPO"] == nome)


def reservatorio(nome, vol_pct, demanda_lps, gatilho=0.0):
    a = acude(nome)
    cap = float(a["capacidade_hm3"])
    return main.Reservatorio(
        nome=a["CORPO"], cod=str(a["COD"]), capacidade=cap, est_evap=str(a["Est. Evap."]),
        vol_inicial=cap * vol_pct / 100.0, demanda=demanda_lps / 1000.0, gatilho=gatilho,
    )


def simular(reservatorios, modo="Individual", conjunta_lps=0.0, ano_final=2021, **extra):
    req = main.SimulacaoRequest(
        reservatorios=reservatorios, modo=modo, vazao_conjunta=conjunta_lps / 1000.0,
        mes_inicial="JAN", ano_inicial=1911, mes_final="DEZ", ano_final=ano_final, **extra,
    )
    return main.processar_simulacao_api(req)


def fogareiro_quixeramobim():
    return simular(
        [reservatorio("Fogareiro", 100, 272), reservatorio("Quixeramobim", 100, 342, gatilho=30)],
        modo="Série", conjunta_lps=500, ano_final=2019, usar_niveis_meta=True,
        cenario_hidrossistema=PLANO_FQ,
    )


class ResultadosDeReferenciaTests(unittest.TestCase):
    def test_mundau_sem_racionamento(self):
        ind = simular([reservatorio("Mundaú", 50, 250)])["resultados"][0]["indicadores"]
        self.assertEqual(ind["meses"], 1332)
        self.assertEqual(ind["meses_falha"], 142)
        self.assertAlmostEqual(ind["atendimento_demanda_aplicada_percent"], 91.77, places=2)

    def test_carnaubal_batalhao_falha_da_demanda_conjunta(self):
        r = simular(
            [reservatorio("Carnaubal", 50, 0, gatilho=10), reservatorio("Barragem do Batalhão", 50, 0)],
            modo="Paralelo", conjunta_lps=160,
        )
        sistema = r["indicadores_sistema"]
        self.assertEqual(sistema["meses_falha_demanda_conjunta"], 42)
        self.assertEqual(sistema["falhas_sistemicas"], 0)
        self.assertEqual(sistema["meses_por_unidade_responsavel"], [1202, 130])
        self.assertAlmostEqual(sistema["atendimento_demanda_conjunta_percent"], 97.57, places=2)
        responsavel = [d["Responsável Demanda Conjunta"] for d in r["resultados"][1]["dados"]]
        self.assertEqual(responsavel.count("Sim"), 130)

    def test_fogareiro_quixeramobim_transferencia(self):
        sistema = fogareiro_quixeramobim()["indicadores_sistema"]
        self.assertEqual(sistema["meses_com_transferencia"], 257)
        self.assertAlmostEqual(sistema["volume_transferido_hm3"], 301.58, places=2)


class IndicadoresDesempenhoTests(unittest.TestCase):
    def saida(self, falhas, aplicada, atendida):
        n = len(falhas)
        return {
            "falha": np.array(["Sim" if f else "Não" for f in falhas], dtype=object),
            "demanda_aplicada": np.array(aplicada, dtype=float),
            "demanda_atendida": np.array(atendida, dtype=float),
            "demanda_solicitada": np.array(aplicada, dtype=float),
            "racionamento": np.zeros(n),
        }

    def test_confiabilidade_resiliencia_vulnerabilidade(self):
        # dois eventos: meses 1-2 (déficits de 50% e 100%) e mês 4 (déficit de 20%)
        falhas = [False, True, True, False, True, False]
        aplicada = [1.0] * 6
        atendida = [1.0, 0.5, 0.0, 1.0, 0.8, 1.0]
        ind = simulador.indicadores_desempenho(self.saida(falhas, aplicada, atendida))
        self.assertAlmostEqual(ind["confiabilidade_percent"], 50.0)
        self.assertAlmostEqual(ind["resiliencia_percent"], 2 / 3 * 100, places=3)
        self.assertAlmostEqual(ind["vulnerabilidade_percent"], 60.0)
        self.assertEqual(ind["eventos_falha"], 2)
        self.assertEqual(ind["duracao_maxima_falha_meses"], 2)

    def test_sem_falhas(self):
        ind = simulador.indicadores_desempenho(self.saida([False] * 3, [1.0] * 3, [1.0] * 3))
        self.assertEqual(ind["confiabilidade_percent"], 100.0)
        self.assertEqual(ind["vulnerabilidade_percent"], 0.0)


class CenariosHidrologicosTests(unittest.TestCase):
    def test_fator_personalizado(self):
        base = simular([reservatorio("Mundaú", 50, 250)], ano_final=1915)["resultados"][0]["dados"]
        oitenta = simular(
            [reservatorio("Mundaú", 50, 250)], ano_final=1915,
            cenario_hidrologico="fator_personalizado", fator_afluencia_percent=80,
        )["resultados"][0]["dados"]
        for a, b in zip(base, oitenta):
            self.assertTrue(math.isclose(b["Vazão (m³/s)"], 0.8 * a["Vazão (m³/s)"], rel_tol=1e-12))

    def test_seca_repetida_usa_os_anos_da_seca(self):
        dados = simular(
            [reservatorio("Mundaú", 50, 250)], ano_final=1925,
            cenario_hidrologico="seca_repetida", seca_ano_inicial=2012, seca_ano_final=2014,
        )["resultados"][0]["dados"]
        anos = [d["Ano de Origem da Vazão"] for d in dados[::12]]
        self.assertEqual(anos[:6], [2012, 2013, 2014, 2012, 2013, 2014])

    def test_reamostragem_reprodutivel_e_coerente_entre_reservatorios(self):
        res = [reservatorio("Carnaubal", 50, 0, gatilho=10), reservatorio("Barragem do Batalhão", 50, 0)]
        a = simular(res, modo="Paralelo", conjunta_lps=160, cenario_hidrologico="reamostragem_anual", semente=7)
        b = simular(res, modo="Paralelo", conjunta_lps=160, cenario_hidrologico="reamostragem_anual", semente=7)
        self.assertEqual(a, b)
        anos0 = [d["Ano de Origem da Vazão"] for d in a["resultados"][0]["dados"]]
        anos1 = [d["Ano de Origem da Vazão"] for d in a["resultados"][1]["dados"]]
        self.assertEqual(anos0, anos1)


class ValidacaoApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cliente = TestClient(main.app)

    def post(self, **alteracoes):
        res = {"nome": "Mundaú", "cod": "61", "capacidade": 21.3, "est_evap": "9",
               "vol_inicial": 10.65, "demanda": 0.25, "gatilho": 10}
        res.update(alteracoes.pop("res", {}))
        corpo = {"reservatorios": [res], "modo": "Individual", "vazao_conjunta": 0,
                 "mes_inicial": "JAN", "ano_inicial": 1911, "mes_final": "DEZ", "ano_final": 1912}
        corpo.update(alteracoes)
        return self.cliente.post("/api/simular", json=corpo)

    def test_entradas_invalidas_sao_rejeitadas_com_mensagem_em_portugues(self):
        casos = [
            ({"res": {"demanda": -1}}, "demanda não pode ser negativa"),
            ({"res": {"vol_inicial": 30}}, "maior que a capacidade"),
            ({"res": {"gatilho": 150}}, "gatilho deve estar entre"),
            ({"ano_inicial": 1920}, "período inicial deve ser anterior"),
            ({"cenario_hidrologico": "seca_repetida"}, "anos inicial e final da seca"),
        ]
        for alteracoes, trecho in casos:
            resposta = self.post(**alteracoes)
            self.assertEqual(resposta.status_code, 422)
            self.assertIn(trecho, str(resposta.json()["detail"]))

    def test_nome_ambiguo_nao_mistura_series(self):
        resposta = self.post(res={"nome": "São José"})
        self.assertEqual(resposta.status_code, 400)
        self.assertIn("mais de um reservatório", resposta.json()["detail"])

    def test_mensagens_sem_acentuacao_corrompida(self):
        resposta = self.post(ano_inicial=1800, ano_final=1800)
        self.assertEqual(resposta.status_code, 404)
        self.assertNotIn("Ã", resposta.json()["detail"])
        self.assertIn("Não há vazões", resposta.json()["detail"])


if __name__ == "__main__":
    unittest.main()
