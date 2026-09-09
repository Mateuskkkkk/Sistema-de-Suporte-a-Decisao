import unittest

import numpy as np

import main


class TccControlledCasesTests(unittest.TestCase):
    """Casos controlados do protocolo de verificação do TCC.

    Estes testes usam entradas artificiais simples e verificam resultados
    esperados sem reutilizar a lógica de preparação da interface.
    """

    @staticmethod
    def cav_sem_evaporacao(capacidade=100.0):
        return (
            np.array([0.0, float(capacidade)], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        )

    def test_volume_constante(self):
        cav_vol, cav_area = self.cav_sem_evaporacao()
        vol, retirada, vertimento, evaporacao = main.dinamica_mensal_fast(
            50.0, 0.0, 0.0, 0.0, 0.0, 100.0, cav_vol, cav_area
        )
        self.assertAlmostEqual(vol, 50.0, places=10)
        self.assertAlmostEqual(retirada, 0.0, places=10)
        self.assertAlmostEqual(vertimento, 0.0, places=10)
        self.assertAlmostEqual(evaporacao, 0.0, places=10)

    def test_retirada_isolada(self):
        cav_vol, cav_area = self.cav_sem_evaporacao()
        vol, retirada, vertimento, evaporacao = main.dinamica_mensal_fast(
            50.0, 0.0, 0.0, 5.0, 0.0, 100.0, cav_vol, cav_area
        )
        self.assertAlmostEqual(vol, 45.0, places=10)
        self.assertAlmostEqual(retirada, 5.0, places=10)
        self.assertAlmostEqual(vertimento, 0.0, places=10)
        self.assertAlmostEqual(evaporacao, 0.0, places=10)

    def test_vertimento_controlado(self):
        cav_vol, cav_area = self.cav_sem_evaporacao()
        vol, retirada, vertimento, _ = main.dinamica_mensal_fast(
            95.0, 10.0, 0.0, 0.0, 0.0, 100.0, cav_vol, cav_area
        )
        self.assertAlmostEqual(vol, 100.0, places=10)
        self.assertAlmostEqual(retirada, 0.0, places=10)
        self.assertAlmostEqual(vertimento, 5.0, places=10)

    def test_falha_por_indisponibilidade(self):
        cav_vol, cav_area = self.cav_sem_evaporacao()
        vol, retirada, vertimento, _ = main.dinamica_mensal_fast(
            2.0, 0.0, 0.0, 5.0, 0.0, 100.0, cav_vol, cav_area
        )
        self.assertAlmostEqual(vol, 0.0, places=10)
        self.assertAlmostEqual(retirada, 2.0, places=10)
        self.assertAlmostEqual(vertimento, 0.0, places=10)
        self.assertAlmostEqual(5.0 - retirada, 3.0, places=10)

    def test_racionamento_de_quarenta_porcento(self):
        demanda_lps = 100.0
        racionamento = 40.0
        demanda_operacional_lps = demanda_lps * (1.0 - racionamento / 100.0)
        self.assertAlmostEqual(demanda_operacional_lps, 60.0, places=10)

    def test_residuo_do_balanco_sem_transferencia(self):
        cav_vol, cav_area = self.cav_sem_evaporacao()
        vol_ini = 40.0
        afluencia = 7.0
        retirada_pedida = 4.0
        vol_fin, retirada, vertimento, evaporacao = main.dinamica_mensal_fast(
            vol_ini,
            afluencia,
            0.0,
            retirada_pedida,
            0.0,
            100.0,
            cav_vol,
            cav_area,
        )
        residuo = vol_fin - (
            vol_ini + afluencia - retirada - evaporacao - vertimento
        )
        self.assertAlmostEqual(residuo, 0.0, places=10)


class TccRegularizedFlowTests(unittest.TestCase):
    @staticmethod
    def synthetic_problem():
        cap = 20.0
        cav_vol = np.array([0.0, cap], dtype=float)
        cav_area = np.array([0.0, 0.0], dtype=float)
        aflu_hm3 = np.array([2.592] * 24, dtype=float)  # 1 m3/s mensal
        evap_m = np.zeros(24, dtype=float)
        return aflu_hm3, evap_m, cap, cav_vol, cav_area

    def test_garantia_cai_ou_permanece_ao_aumentar_demanda(self):
        aflu, evap, cap, cav_vol, cav_area = self.synthetic_problem()
        g_baixa, _ = main.simular_garantia_demanda(
            0.5, aflu, evap, cap, cav_vol, cav_area, 100.0
        )
        g_alta, _ = main.simular_garantia_demanda(
            1.5, aflu, evap, cap, cav_vol, cav_area, 100.0
        )
        self.assertGreaterEqual(g_baixa, g_alta)

    def test_vazao_regularizada_e_monotonica_com_a_garantia(self):
        aflu, evap, cap, cav_vol, cav_area = self.synthetic_problem()
        q90, _, _ = main.buscar_vazao_por_garantia(
            0.90, aflu, evap, cap, cav_vol, cav_area, 100.0
        )
        q95, _, _ = main.buscar_vazao_por_garantia(
            0.95, aflu, evap, cap, cav_vol, cav_area, 100.0
        )
        q99, _, _ = main.buscar_vazao_por_garantia(
            0.99, aflu, evap, cap, cav_vol, cav_area, 100.0
        )
        self.assertGreaterEqual(q90 + 1e-7, q95)
        self.assertGreaterEqual(q95 + 1e-7, q99)

    def test_vizinhos_da_vazao_limite(self):
        aflu, evap, cap, cav_vol, cav_area = self.synthetic_problem()
        alvo = 0.95
        q_star, g_star, _ = main.buscar_vazao_por_garantia(
            alvo, aflu, evap, cap, cav_vol, cav_area, 100.0
        )
        delta = max(1e-4, q_star * 1e-3)
        g_menos, _ = main.simular_garantia_demanda(
            max(0.0, q_star - delta), aflu, evap, cap, cav_vol, cav_area, 100.0
        )
        g_mais, _ = main.simular_garantia_demanda(
            q_star + delta, aflu, evap, cap, cav_vol, cav_area, 100.0
        )
        self.assertGreaterEqual(g_star + 1e-7, alvo)
        self.assertGreaterEqual(g_menos + 1e-7, alvo)
        self.assertLessEqual(g_mais, g_menos + 1e-7)


if __name__ == "__main__":
    unittest.main()
