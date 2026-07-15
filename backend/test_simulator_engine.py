import math
import unittest

import main


class SimulatorNumpyEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = {
            row["CORPO"]: row
            for row in main.listar_reservatorios()
            if row["CORPO"] in {"Acarape do Meio", "Adauto Bezerra"}
        }
        cls.acarape = rows["Acarape do Meio"]
        cls.adauto = rows["Adauto Bezerra"]

    @staticmethod
    def reservoir(row, demand, trigger=30.0, custom=None):
        capacity = float(row["capacidade_hm3"])
        return main.Reservatorio(
            nome=row["CORPO"],
            cod=str(row["COD"]),
            capacidade=capacity,
            est_evap=str(row["Est. Evap."]),
            vol_inicial=capacity * 0.5,
            demanda=demand,
            gatilho=trigger,
            plano_secas_custom=custom,
        )

    @staticmethod
    def request(reservoirs, mode, use_meta=False, scenario="historico"):
        return main.SimulacaoRequest(
            reservatorios=reservoirs,
            modo=mode,
            vazao_conjunta=0.0 if mode == "Individual" else 0.05,
            mes_inicial="JAN",
            ano_inicial=1911,
            mes_final="DEZ",
            ano_final=1911,
            cenario_hidrologico=scenario,
            usar_niveis_meta=use_meta,
        )

    def test_all_modes_keep_the_monthly_api_contract(self):
        reservoirs = [
            self.reservoir(self.acarape, 0.2),
            self.reservoir(self.adauto, 0.1),
        ]
        expected_keys = {
            "Data",
            "Vazão (m³/s)",
            "Afluências (hm³/mês)",
            "Armazenamento Inicial",
            "Armazenamento Final",
            "Demanda Solicitada (m³/s)",
            "Demanda Atendida (m³/s)",
            "Racionamento (%)",
            "Transferência Recebida (m³/s)",
            "Transferência Enviada (m³/s)",
            "Evaporação (hm³)",
            "Vertimento (hm³)",
            "Falha",
            "Modo Operação",
        }

        for mode in ("Individual", "Série", "Paralelo"):
            selected = reservoirs[:1] if mode == "Individual" else reservoirs
            result = main.processar_simulacao_api(self.request(selected, mode))
            self.assertEqual(len(result["resultados"]), len(selected))
            for item in result["resultados"]:
                self.assertEqual(len(item["dados"]), 12)
                self.assertTrue(expected_keys.issubset(item["dados"][0]))

    def test_series_accepts_accented_or_ascii_mode_and_records_transfer(self):
        reservoirs = [
            self.reservoir(self.acarape, 0.2, trigger=90.0),
            self.reservoir(self.adauto, 0.1),
        ]
        accented = main.processar_simulacao_api(self.request(reservoirs, "Série"))
        ascii_mode = main.processar_simulacao_api(self.request(reservoirs, "Serie"))
        self.assertEqual(accented, ascii_mode)

        receiver = accented["resultados"][0]["dados"][0]
        sender = accented["resultados"][1]["dados"][0]
        self.assertGreater(receiver["Transferência Recebida (m³/s)"], 0.0)
        self.assertTrue(math.isclose(
            receiver["Transferência Recebida (m³/s)"],
            sender["Transferência Enviada (m³/s)"],
            abs_tol=1e-12,
        ))

    def test_custom_meta_rule_reduces_the_served_demand(self):
        custom = [main.FaixaCustom(
            Faixa="Seca",
            Racionamento=50,
            JAN=100,
            FEV=100,
            MAR=100,
            ABR=100,
            MAI=100,
            JUN=100,
            JUL=100,
            AGO=100,
            SET=100,
            OUT=100,
            NOV=100,
            DEZ=100,
        )]
        reservoir = self.reservoir(self.acarape, 0.2, custom=custom)
        result = main.processar_simulacao_api(self.request([reservoir], "Individual", use_meta=True))
        row = result["resultados"][0]["dados"][0]
        self.assertEqual(row["Racionamento (%)"], 50.0)
        self.assertEqual(row["Modo Operação"], "Seca")
        self.assertTrue(math.isclose(row["Demanda Atendida (m³/s)"], 0.1, abs_tol=1e-12))

    def test_zero_inflow_scenario_zeros_both_inflow_fields(self):
        reservoir = self.reservoir(self.acarape, 0.2)
        result = main.processar_simulacao_api(
            self.request([reservoir], "Individual", scenario="afluencia_zero")
        )
        for row in result["resultados"][0]["dados"]:
            self.assertEqual(row["Vazão (m³/s)"], 0.0)
            self.assertEqual(row["Afluências (hm³/mês)"], 0.0)


if __name__ == "__main__":
    unittest.main()
