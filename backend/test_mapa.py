"""Testes das rotas usadas pela tela inicial em mapa."""
import unittest

from fastapi.testclient import TestClient

import main


class TestMapa(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def test_lista_todos_os_acudes_com_coordenadas_no_ceara(self):
        acudes = self.client.get("/api/mapa/acudes").json()
        self.assertEqual(len(acudes), 155)
        for a in acudes:
            self.assertTrue(-8.0 < a["lat"] < -2.7, a["nome"])
            self.assertTrue(-41.6 < a["lon"] < -37.2, a["nome"])
        aproximados = {a["nome"] for a in acudes if a["posicao_aproximada"]}
        self.assertIn("Fogareiro", aproximados)
        self.assertNotIn("Flor do Campo", aproximados)

    def test_marca_hidrossistemas_multirreservatorio(self):
        acudes = {a["nome"]: a for a in self.client.get("/api/mapa/acudes").json()}
        self.assertTrue(acudes["Carnaubal"]["multirreservatorio"])
        self.assertFalse(acudes["Flor do Campo"]["multirreservatorio"])

    def test_detalhes_trazem_evaporacao_da_estacao_e_serie_completa(self):
        d = self.client.get("/api/acudes/172/detalhes").json()
        self.assertEqual(d["nome"], "Flor do Campo")
        self.assertEqual(d["estacao_evaporacao"], "Crateús")
        self.assertAlmostEqual(sum(d["evaporacao_mm"]), 2904.4, places=1)
        self.assertEqual(len(d["afluencias"]), 1335)
        self.assertEqual((d["afluencias"][0]["ano"], d["afluencias"][0]["mes"]), (1910, 10))
        media = sum(p["vazao_m3s"] for p in d["afluencias"]) / len(d["afluencias"])
        self.assertAlmostEqual(media, 0.65, places=2)

    def test_acude_inexistente(self):
        self.assertEqual(self.client.get("/api/acudes/99999/detalhes").status_code, 404)


if __name__ == "__main__":
    unittest.main()
