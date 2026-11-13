import math
import unittest

import numpy as np

from previsao_knn import (
    area_roc,
    crps_discreto,
    emissao_relativa,
    horizonte,
    indices_quadra,
    k_padrao,
    meses_anteriores,
    montar_historico,
    para_json,
    pesos_lall_sharma,
    prever,
    probabilidades,
)


def dados_sinteticos(anos=range(1950, 2001), zeros=False):
    """Índices e afluências sintéticos, com relação entre o gradiente e a quadra."""
    rng = np.random.default_rng(7)
    indices = {"nino34": {}, "tna": {}, "tsa": {}}
    vazoes = {}
    for a in range(min(anos) - 2, max(anos) + 2):
        g = rng.normal()
        for m in range(1, 13):
            indices["nino34"][(a, m)] = float(rng.normal())
            indices["tna"][(a, m)] = g / 2
            indices["tsa"][(a, m)] = -g / 2
    for a in range(min(anos) - 1, max(anos) + 2):
        g_ant = indices["tna"][(a - 1, 12)] - indices["tsa"][(a - 1, 12)]
        for m in range(1, 13):
            base = 5.0 if m in (2, 3, 4, 5) else 0.3
            v = max(0.0, base * math.exp(-0.5 * g_ant + rng.normal(0, 0.3)))
            if zeros and m in (8, 9, 10):
                v = 0.0
            vazoes[(a, m)] = v
    return indices, vazoes


class CalendarioTests(unittest.TestCase):
    def test_emissoes_e_horizontes(self):
        self.assertEqual(emissao_relativa(12), (-1, 12))
        self.assertEqual(emissao_relativa(1), (0, 1))
        self.assertEqual(meses_anteriores(1), [(-1, 10), (-1, 11), (-1, 12)])
        self.assertEqual(meses_anteriores(12), [(-1, 9), (-1, 10), (-1, 11)])
        self.assertEqual(horizonte(10)[0], (-1, 10))
        self.assertEqual(horizonte(10)[-1], (0, 12))
        self.assertEqual(len(horizonte(10)), 15)
        self.assertEqual(len(horizonte(1)), 12)
        self.assertEqual([horizonte(1)[i][1] for i in indices_quadra(1)], [2, 3, 4, 5])


class PesosTests(unittest.TestCase):
    def test_pesos_somam_um(self):
        for k in (1, 2, 5, 8, 10, 37, 110):
            w = pesos_lall_sharma(k)
            self.assertAlmostEqual(w.sum(), 1.0, places=12)
            self.assertTrue(np.all(np.diff(w) < 0) or k == 1)
            self.assertAlmostEqual(w[0] / w[1] if k > 1 else 2.0, 2.0, places=12)

    def test_k_padrao(self):
        self.assertEqual(k_padrao(71), 8)
        self.assertEqual(k_padrao(110), 10)
        self.assertEqual(k_padrao(72), 8)  # raiz(72) = 8,49
        self.assertEqual(k_padrao(1), 1)

    def test_probabilidades_da_previsao_somam_um(self):
        indices, vazoes = dados_sinteticos()
        h = montar_historico(range(1950, 2001), 12, ("nino34", "gradiente"), indices, vazoes)
        p = probabilidades(h["X"][1:], h["anos"][1:], h["X"][0])
        self.assertAlmostEqual(p["prob"].sum(), 1.0, places=12)


class ClimatologiaTests(unittest.TestCase):
    def test_k_igual_a_n_com_pesos_iguais_reproduz_a_climatologia(self):
        indices, vazoes = dados_sinteticos()
        h = montar_historico(range(1950, 2001), 1, ("nino34", "gradiente"), indices, vazoes)
        X, anos, S = h["X"][1:], h["anos"][1:], h["S"][1:]
        n = len(anos)
        knn = probabilidades(X, anos, h["X"][0], k=n, pesos="iguais")
        clim = probabilidades(X, anos, None, climatologia=True)
        np.testing.assert_allclose(knn["prob"], clim["prob"])
        np.testing.assert_allclose(clim["prob"], np.full(n, 1.0 / n))
        # com as mesmas probabilidades, a distribuição prevista é a mesma
        alvo = S[:, indices_quadra(1)].sum(axis=1)
        y = float(alvo.mean())
        self.assertAlmostEqual(crps_discreto(alvo, knn["prob"], y), crps_discreto(alvo, clim["prob"], y), places=12)


class ReproducibilidadeTests(unittest.TestCase):
    def test_semente_fixa_reproduz_os_cenarios(self):
        indices, vazoes = dados_sinteticos()
        h = montar_historico(range(1950, 2001), 11, ("nino34", "gradiente", "quadra_anterior"), indices, vazoes)
        args = (h["X"][1:], h["anos"][1:], h["S"][1:], h["X"][0])
        a = prever(*args, n_cenarios=500, semente=42)
        b = prever(*args, n_cenarios=500, semente=42)
        c = prever(*args, n_cenarios=500, semente=43)
        np.testing.assert_array_equal(a["cenarios"], b["cenarios"])
        self.assertEqual(a["anos_sorteados"], b["anos_sorteados"])
        self.assertNotEqual(a["anos_sorteados"], c["anos_sorteados"])

    def test_cenarios_sao_sequencias_reais_e_preservam_zeros(self):
        indices, vazoes = dados_sinteticos(zeros=True)
        h = montar_historico(range(1950, 2001), 10, ("gradiente",), indices, vazoes)
        r = prever(h["X"][1:], h["anos"][1:], h["S"][1:], h["X"][0], n_cenarios=200, semente=3)
        por_ano = {int(a): s for a, s in zip(h["anos"], h["S"])}
        for cen, ano in zip(r["cenarios"], r["anos_sorteados"]):
            np.testing.assert_array_equal(cen, por_ano[ano])
        # agosto a outubro do ano anterior estão no horizonte de outubro e são zero nos dados
        posicoes = [i for i, (dy, m) in enumerate(horizonte(10)) if m in (8, 9, 10)]
        self.assertTrue(np.all(r["cenarios"][:, posicoes] == 0.0))


class DeixarUmDeForaTests(unittest.TestCase):
    def test_ano_retirado_nunca_aparece_entre_os_vizinhos(self):
        indices, vazoes = dados_sinteticos()
        h = montar_historico(range(1950, 2001), 12, ("nino34", "gradiente"), indices, vazoes)
        n = len(h["anos"])
        for i in range(n):
            treino = np.arange(n) != i
            for k in (1, k_padrao(n - 1), n - 1):
                r = prever(h["X"][treino], h["anos"][treino], h["S"][treino], h["X"][i], k=k, n_cenarios=50)
                anos_viz = {v["ano"] for v in r["vizinhos"]}
                self.assertNotIn(int(h["anos"][i]), anos_viz)
                self.assertNotIn(int(h["anos"][i]), r["anos_sorteados"])

    def test_padronizacao_usa_so_o_historico(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        anos = np.array([1, 2, 3, 4])
        # um valor atual extremo não altera a ordem dos vizinhos dada pelo histórico
        a = probabilidades(X, anos, np.array([100.0]), k=2)
        self.assertEqual([int(anos[i]) for i in a["ordem"]], [4, 3])


class MetricasTests(unittest.TestCase):
    def test_crps_de_previsao_perfeita_e_zero(self):
        self.assertAlmostEqual(crps_discreto(np.array([3.0]), np.array([1.0]), 3.0), 0.0)

    def test_crps_de_dois_pontos(self):
        # X uniforme em {0, 1}, y = 0: E|X-y| = 0,5; E|X-X'|/2 = 0,25
        self.assertAlmostEqual(crps_discreto(np.array([0.0, 1.0]), np.array([0.5, 0.5]), 0.0), 0.25)

    def test_area_roc(self):
        self.assertAlmostEqual(area_roc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]), 1.0)
        self.assertAlmostEqual(area_roc([0.5, 0.5], [True, False]), 0.5)
        self.assertTrue(math.isnan(area_roc([0.5], [True])))


class JsonTests(unittest.TestCase):
    def test_nan_e_inf_viram_none(self):
        saida = para_json({"a": np.array([1.0, np.nan, np.inf]), "b": np.float64(-np.inf), "c": np.int64(3)})
        self.assertEqual(saida, {"a": [1.0, None, None], "b": None, "c": 3})


class ApiTests(unittest.TestCase):
    """Rotas com o banco real; exigem a tabela indices_climaticos preenchida."""

    @classmethod
    def setUpClass(cls):
        import sqlite3

        from fastapi.testclient import TestClient

        import main

        con = sqlite3.connect(main.DB_PATH)
        try:
            con.execute("SELECT 1 FROM indices_climaticos LIMIT 1")
        except sqlite3.OperationalError:
            raise unittest.SkipTest("tabela indices_climaticos ausente")
        finally:
            con.close()
        cls.cliente = TestClient(main.app)

    def test_cenarios_de_um_ano_do_historico(self):
        r = self.cliente.post("/api/previsao-knn/cenarios", json={
            "reservatorio": "Carnaubal", "mes_emissao": 12, "ano_alvo": 2016, "n_cenarios": 200})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertNotIn(2016, [v["ano"] for v in d["vizinhos"]])
        self.assertAlmostEqual(sum(v["peso"] for v in d["vizinhos"]), 1.0, places=9)
        self.assertEqual(len(d["cenarios"]), 200)
        self.assertEqual(len(d["cenarios"][0]), len(d["meses"]))
        self.assertEqual(d["meses"][0], "DEZ/2015")
        self.assertIsNotNone(d["quadra_observada"])

    def test_preditor_invalido_para_a_emissao(self):
        r = self.cliente.post("/api/previsao-knn/cenarios", json={
            "reservatorio": "Carnaubal", "mes_emissao": 12, "preditores": ["acumulada"]})
        self.assertEqual(r.status_code, 422)

    def test_simulacao_com_cenarios(self):
        res = [{"nome": "Carnaubal", "cod": "53", "capacidade": 46.621, "est_evap": "4",
                "vol_inicial": 2.0, "demanda": 0.16, "gatilho": 10}]
        r = self.cliente.post("/api/previsao-knn/simular", json={
            "reservatorio": "Carnaubal", "mes_emissao": 1, "ano_alvo": 2013, "n_cenarios": 100, "reservatorios": res})
        self.assertEqual(r.status_code, 200)
        s = r.json()["simulacao"]["previsao"][0]
        self.assertEqual(len(s["garantia_por_cenario"]), 100)
        for valor in s["garantia_por_cenario"]:
            self.assertTrue(0.0 <= valor <= 1.0)
        self.assertEqual(len(s["quantis_volume_hm3"]["p50"]), 12)


if __name__ == "__main__":
    unittest.main()
