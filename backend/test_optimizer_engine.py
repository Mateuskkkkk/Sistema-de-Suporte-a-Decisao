import unittest

import numpy as np

from optimizer_engine import (
    calcular_erro_garantias,
    diferenciar_retiradas_equivalentes,
    engine_simulacao_temporal,
)


class EquivalentTargetLevelTests(unittest.TestCase):
    def test_distinct_attendances_are_not_adjusted(self):
        withdrawals = np.array([1.0, 0.9, 0.8, 0.5])

        adjusted = diferenciar_retiradas_equivalentes(withdrawals)

        np.testing.assert_array_equal(adjusted, withdrawals)

    def test_equal_attendances_receive_distinct_internal_references(self):
        withdrawals = np.array([1.0, 1.0, 0.8, 0.5])

        references = diferenciar_retiradas_equivalentes(withdrawals)

        np.testing.assert_allclose(references, [1.0, 0.9999, 0.8, 0.5])
        np.testing.assert_array_equal(withdrawals, [1.0, 1.0, 0.8, 0.5])

    def test_each_equivalent_state_keeps_its_own_required_guarantee(self):
        guarantees = np.array([0.92, 0.92, 0.97, 0.99])
        required = np.array([0.9, 0.95, 0.98, 1.0])

        error = calcular_erro_garantias(guarantees, required)
        expected = (
            ((0.92 - 0.9) / 0.9) ** 2
            + ((0.92 - 0.95) / 0.95) ** 2
            + ((0.97 - 0.98) / 0.98) ** 2
            + ((0.99 - 1.0) / 1.0) ** 2
        )

        self.assertAlmostEqual(error, expected)

    def test_distinct_references_can_produce_distinct_guarantees(self):
        curves = np.repeat(np.array([0.1, 0.2, 0.3])[:, None], 12, axis=1)
        withdrawals = np.array([1.0, 1.0, 0.8, 0.5])
        references = diferenciar_retiradas_equivalentes(withdrawals)
        inflows = np.zeros(1)
        evaporation = np.zeros(1)
        cav_volume = np.array([0.0, 2.0])
        cav_area = np.array([0.0, 0.0])

        guarantees = engine_simulacao_temporal(
            curves, inflows, evaporation, references,
            1.9999, cav_volume, cav_area, 1,
        )

        np.testing.assert_array_equal(guarantees, [0.0, 1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
