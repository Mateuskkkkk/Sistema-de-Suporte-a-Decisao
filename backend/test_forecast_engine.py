import unittest

from fastapi import HTTPException

from forecast_engine import (
    ForecastBaseRequest,
    copeland_scores,
    list_indicators,
    normalized_scores,
    validate_request,
)


class ForecastEngineTests(unittest.TestCase):
    def test_relative_importance_sums_to_one_hundred(self):
        result = normalized_scores({"nino34": 3.0, "amo": 1.0, "amm": 0.0})

        self.assertAlmostEqual(sum(result.values()), 100.0)
        self.assertEqual(result["nino34"], 75.0)
        self.assertEqual(result["amo"], 25.0)

    def test_copeland_favors_consistent_first_place(self):
        result = copeland_scores(
            [
                ["nino34", "tna", "tsa", "dipolo", "amm", "amo"],
                ["nino34", "tsa", "tna", "dipolo", "amo", "amm"],
                ["nino34", "amm", "amo", "dipolo", "tna", "tsa"],
            ]
        )

        self.assertGreater(result["nino34"], max(
            value for indicator, value in result.items() if indicator != "nino34"
        ))

    def test_climate_catalog_has_all_supported_indicators(self):
        catalog = list_indicators()["indicadores"]

        self.assertEqual(
            set(catalog),
            {"nino34", "tna", "tsa", "dipolo", "amm", "amo"},
        )
        self.assertTrue(all(item["meses"] > 0 for item in catalog.values()))

    def test_invalid_horizon_is_rejected(self):
        request = ForecastBaseRequest(reservatorio="Mundaú", horizonte=13)

        with self.assertRaises(HTTPException):
            validate_request(request)


if __name__ == "__main__":
    unittest.main()
