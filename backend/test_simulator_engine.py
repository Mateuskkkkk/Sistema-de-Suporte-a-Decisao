import math
import unittest

import main


class SimulatorNumpyEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selected = {
            "Flor do Campo",
            "Carnaubal",
            "Barragem do Batalhão",
            "Quixeramobim",
            "Fogareiro",
        }
        rows = {
            row["CORPO"]: row
            for row in main.listar_reservatorios()
            if row["CORPO"] in selected
        }
        cls.flor = rows["Flor do Campo"]
        cls.carnaubal = rows["Carnaubal"]
        cls.batalhao = rows["Barragem do Batalhão"]
        cls.quixeramobim = rows["Quixeramobim"]
        cls.fogareiro = rows["Fogareiro"]

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
        expected_keys = {
            "Data",
            "Vazão (m³/s)",
            "Afluências (hm³/mês)",
            "Armazenamento Inicial",
            "Armazenamento Final",
            "Demanda Solicitada (m³/s)",
            "Demanda Atendida (m³/s)",
            "Retirada Total (m³/s)",
            "Racionamento (%)",
            "Transferência Recebida (m³/s)",
            "Transferência Enviada (m³/s)",
            "Evaporação (hm³)",
            "Vertimento (hm³)",
            "Falha",
            "Modo Operação",
        }

        cases = (
            ("Individual", [self.reservoir(self.flor, 0.2)]),
            ("Série", [
                self.reservoir(self.quixeramobim, 0.2, trigger=90.0),
                self.reservoir(self.fogareiro, 0.1),
            ]),
            ("Paralelo", [
                self.reservoir(self.carnaubal, 0.2),
                self.reservoir(self.batalhao, 0.1),
            ]),
        )

        for mode, selected in cases:
            result = main.processar_simulacao_api(self.request(selected, mode))
            self.assertEqual(len(result["resultados"]), len(selected))
            for item in result["resultados"]:
                self.assertEqual(len(item["dados"]), 12)
                self.assertTrue(expected_keys.issubset(item["dados"][0]))

    def test_series_accepts_accented_or_ascii_mode_and_records_transfer(self):
        reservoirs = [
            self.reservoir(self.quixeramobim, 0.2, trigger=90.0),
            self.reservoir(self.fogareiro, 0.1),
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
        reservoir = self.reservoir(self.flor, 0.2, custom=custom)
        result = main.processar_simulacao_api(self.request([reservoir], "Individual", use_meta=True))
        row = result["resultados"][0]["dados"][0]
        self.assertEqual(row["Racionamento (%)"], 50.0)
        self.assertEqual(row["Modo Operação"], "Seca")
        self.assertTrue(math.isclose(row["Demanda Atendida (m³/s)"], 0.1, abs_tol=1e-12))

    def test_zero_inflow_scenario_zeros_both_inflow_fields(self):
        reservoir = self.reservoir(self.flor, 0.2)
        result = main.processar_simulacao_api(
            self.request([reservoir], "Individual", scenario="afluencia_zero")
        )
        for row in result["resultados"][0]["dados"]:
            self.assertEqual(row["Vazão (m³/s)"], 0.0)
            self.assertEqual(row["Afluências (hm³/mês)"], 0.0)

    def pgps_request(self, fogareiro_percent, quixeramobim_hm3=None, atendimento=100.0):
        quixeramobim = self.reservoir(self.quixeramobim, 0.342, trigger=30.0)
        fogareiro = self.reservoir(self.fogareiro, 0.272, trigger=0.0)
        quixeramobim.vol_inicial = (
            quixeramobim.capacidade * 0.1
            if quixeramobim_hm3 is None
            else quixeramobim_hm3
        )
        fogareiro.vol_inicial = fogareiro.capacidade * fogareiro_percent / 100.0
        return main.SimulacaoRequest(
            reservatorios=[quixeramobim, fogareiro],
            modo="Serie",
            vazao_conjunta=0.5,
            atendimento_transferencia=atendimento,
            mes_inicial="JAN",
            ano_inicial=1911,
            mes_final="JAN",
            ano_final=1911,
            cenario_hidrologico="afluencia_zero",
            usar_niveis_meta=True,
            cenario_hidrossistema=main.FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID,
        )

    def test_pgps_scenario_transfers_the_full_normal_flow(self):
        result = main.processar_simulacao_api(self.pgps_request(100.0))
        receiver = result["resultados"][0]["dados"][0]
        sender = result["resultados"][1]["dados"][0]

        self.assertEqual(receiver["Modo Operação"], "Normal")
        self.assertAlmostEqual(receiver["Demanda Atendida (m³/s)"], 0.342)
        self.assertAlmostEqual(sender["Demanda Atendida (m³/s)"], 0.272)
        self.assertAlmostEqual(receiver["Transferência Recebida (m³/s)"], 0.5)
        self.assertAlmostEqual(sender["Transferência Enviada (m³/s)"], 0.5)
        self.assertAlmostEqual(sender["Retirada Total (m³/s)"], 0.772)

    def test_pgps_scenario_transfers_the_full_severe_flow(self):
        result = main.processar_simulacao_api(self.pgps_request(20.0))
        receiver = result["resultados"][0]["dados"][0]
        sender = result["resultados"][1]["dados"][0]

        self.assertEqual(receiver["Modo Operação"], "Seca Severa")
        self.assertEqual(sender["Modo Operação"], "Seca Severa")
        self.assertAlmostEqual(receiver["Demanda Atendida (m³/s)"], 0.0705)
        self.assertAlmostEqual(sender["Demanda Atendida (m³/s)"], 0.006)
        self.assertAlmostEqual(receiver["Transferência Recebida (m³/s)"], 0.085)
        self.assertAlmostEqual(sender["Transferência Enviada (m³/s)"], 0.085)
        self.assertAlmostEqual(sender["Retirada Total (m³/s)"], 0.091)

    def test_pgps_transfer_attendance_scales_the_requested_flow(self):
        result = main.processar_simulacao_api(self.pgps_request(100.0, atendimento=50.0))
        receiver = result["resultados"][0]["dados"][0]
        sender = result["resultados"][1]["dados"][0]
        self.assertAlmostEqual(receiver["Transferência Recebida (m³/s)"], 0.25)
        self.assertAlmostEqual(sender["Transferência Enviada (m³/s)"], 0.25)
        self.assertAlmostEqual(sender["Retirada Total (m³/s)"], 0.522)

    def test_pgps_does_not_transfer_above_the_thirty_percent_trigger(self):
        result = main.processar_simulacao_api(
            self.pgps_request(100.0, quixeramobim_hm3=4.0)
        )
        receiver = result["resultados"][0]["dados"][0]
        sender = result["resultados"][1]["dados"][0]
        self.assertEqual(receiver["Transferência Recebida (m³/s)"], 0.0)
        self.assertEqual(sender["Transferência Enviada (m³/s)"], 0.0)

    def test_pgps_curves_and_trigger_use_volume_percentages(self):
        preset = next(
            item for item in main.listar_presets()
            if item.get("cenario_hidrossistema") == main.FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID
        )
        self.assertEqual(preset["defaults"]["16"]["gatilho_percent"], 30.0)
        alerta = next(
            faixa for faixa in preset["niveis_meta"]["faixas"]
            if faixa["Faixa"] == "Alerta"
        )
        self.assertEqual(alerta["JAN"], 49.15)
        self.assertEqual(
            main.estado_fogareiro_quixeramobim(58.0, 118.0, 1),
            "Alerta",
        )


if __name__ == "__main__":
    unittest.main()
