"""Testes da otimização multiobjetivo (NSGA-II) dos níveis meta."""
import json
import unittest

import numpy as np
from fastapi.testclient import TestClient

import main
import otimizador_pareto as op


def mundau(**extra):
    base = dict(reservatorio="Mundaú", durb_m3s=0.25, frac_durb=[1, 1, 0.8, 0.5], frac_dsup=[1, 0.8, 0.5, 0],
                ninicio=7, seed=1, populacao=40, geracoes=20)
    base.update(extra)
    return op.ParetoPayload(**base)


def rodar(payload):
    eventos = [json.loads(e[6:]) for e in op.pareto_generator(payload)]
    return eventos


class ParetoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eventos = rodar(mundau())
        cls.r = cls.eventos[-1]

    def test_sucesso_com_progresso(self):
        self.assertEqual(self.r["status"], "sucesso", self.r.get("mensagem"))
        self.assertTrue(any(e["status"] == "progresso" for e in self.eventos[:-1]))
        self.assertGreater(len(self.r["solucoes"]), 3)

    def test_solucoes_nao_dominadas(self):
        pts = [(s["meses_falha"], s["objetivo_entrega"]) for s in self.r["solucoes"]]
        for i, a in enumerate(pts):
            for j, b in enumerate(pts):
                if i != j:
                    domina = b[0] <= a[0] and b[1] <= a[1] and (b[0] < a[0] or b[1] < a[1])
                    self.assertFalse(domina, f"solução {j} domina {i}")

    def test_curvas_respeitam_restricoes(self):
        for s in self.r["solucoes"]:
            niveis = sorted(s["niveis_meta"])
            self.assertTrue(np.all(np.diff(niveis) >= op.FOLGA_ENTRE_NIVEIS - 1e-9))
            self.assertGreaterEqual(min(min(c) for c in s["matriz_curvas"]), op.VOLUME_META_MIN - 1e-9)
            # largura mínima padrão (5% do volume) entre curvas vizinhas, em todos os meses
            curvas = np.array(s["matriz_curvas"])
            self.assertGreaterEqual(np.min(curvas[:-1] - curvas[1:]), 0.05 - 1e-9)

    def test_destaques(self):
        d, sol = self.r["destaques"], self.r["solucoes"]
        self.assertEqual(sol[d["seguranca_total"]]["meses_falha"], 0)
        melhor = min(s["objetivo_entrega"] for s in sol)
        self.assertEqual(sol[d["maxima_entrega"]]["objetivo_entrega"], melhor)
        self.assertIn(d["equilibrio"], range(len(sol)))

    def test_referencia_igual_ao_simulador(self):
        # sem curvas-guia, o resultado é o do Simulador para 250 L/s (TCC: 142 falhas, 91,77%)
        ref = self.r["referencia_sem_racionamento"]
        self.assertEqual(ref["meses_falha"], 142)
        self.assertAlmostEqual(ref["entregue_percent"], 91.772, places=2)
        self.assertAlmostEqual(ref["deficit_hm3"], 71.016, places=2)

    def test_volumes_por_solucao(self):
        for s in self.r["solucoes"]:
            self.assertEqual(len(s["volumes_historicos"]), self.r["meses"])
            self.assertLessEqual(max(s["volumes_historicos"]), self.r["capacidade_hm3"] + 1e-6)
            menor = 100.0 * min(s["volumes_historicos"]) / self.r["capacidade_hm3"]
            self.assertAlmostEqual(menor, s["volume_minimo_percent"], delta=0.05)

    def test_reprodutivel_com_semente(self):
        outra = rodar(mundau())[-1]
        self.assertEqual([(s["meses_falha"], s["entregue_hm3"]) for s in outra["solucoes"]],
                         [(s["meses_falha"], s["entregue_hm3"]) for s in self.r["solucoes"]])

    def test_largura_minima_configuravel(self):
        r = rodar(mundau(largura_minima=0.10))[-1]
        self.assertEqual(r["status"], "sucesso")
        for s in r["solucoes"]:
            curvas = np.array(s["matriz_curvas"])
            self.assertGreaterEqual(np.min(curvas[:-1] - curvas[1:]), 0.10 - 1e-9)

    def test_criterio_total_maximiza_volume(self):
        r = rodar(mundau(criterio_entrega="total"))[-1]
        self.assertEqual(r["status"], "sucesso")
        for s in r["solucoes"]:
            self.assertAlmostEqual(s["objetivo_entrega"], -s["entregue_hm3"], places=3)
        pts = [(s["meses_falha"], -s["entregue_hm3"]) for s in r["solucoes"]]
        for i, a in enumerate(pts):
            self.assertFalse(any(b[0] <= a[0] and b[1] <= a[1] and b != a for b in pts), f"solução {i} dominada")

    def test_gradual_afasta_curvas(self):
        # o déficit quadrático favorece o racionamento escalonado: faixas mais largas que no critério "total"
        def largura_media(r):
            return np.mean([np.mean(np.array(s["matriz_curvas"])[:-1] - np.array(s["matriz_curvas"])[1:])
                            for s in r["solucoes"]])
        gradual = rodar(mundau(largura_minima=0.0, frac_durb=[1, 0.9, 0.8, 0.5]))[-1]
        total = rodar(mundau(largura_minima=0.0, frac_durb=[1, 0.9, 0.8, 0.5], criterio_entrega="total"))[-1]
        self.assertGreater(largura_media(gradual), 2 * largura_media(total))

    def test_erro_vira_mensagem(self):
        r = rodar(mundau(durb_m3s=0.0))[-1]
        self.assertEqual(r["status"], "erro")
        self.assertIn("demanda", r["mensagem"])

    def test_rota(self):
        cliente = TestClient(main.app)
        corpo = mundau(populacao=20, geracoes=5).model_dump()
        resposta = cliente.post("/api/otimizador/pareto", json=corpo)
        self.assertEqual(resposta.status_code, 200)
        ultimo = [linha for linha in resposta.text.split("\n\n") if linha.startswith("data: ")][-1]
        self.assertEqual(json.loads(ultimo[6:])["status"], "sucesso")


if __name__ == "__main__":
    unittest.main()
