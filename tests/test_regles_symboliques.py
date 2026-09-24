import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import regles_symboliques as rules  # noqa: E402


class SymbolicRulesTests(unittest.TestCase):
    def test_unknown_appliance_is_neutral(self):
        prediction = 123.4
        self.assertEqual(
            rules.appliquer_regles("UnknownAppliance", delta=500, hour=8, pred=prediction),
            prediction,
        )

    def test_currently_disabled_rules_are_neutral(self):
        for appliance in rules.SEUILS_PLAID:
            with self.subTest(appliance=appliance):
                prediction = 75.0
                self.assertEqual(
                    rules.appliquer_regles(appliance, delta=400, hour=7, pred=prediction),
                    prediction,
                )

    def test_registered_rule_contract_is_applied(self):
        original = rules.REGLES_PAR_APPAREIL.copy()
        try:
            rules.REGLES_PAR_APPAREIL["TestAppliance"] = {
                "condition": lambda delta, hour, pred: delta > 100 and hour < 12,
                "boost": lambda delta, pred: pred + delta / 10,
            }
            self.assertEqual(
                rules.appliquer_regles("TestAppliance", delta=200, hour=8, pred=30),
                50,
            )
            self.assertEqual(
                rules.appliquer_regles("TestAppliance", delta=50, hour=8, pred=30),
                30,
            )
        finally:
            rules.REGLES_PAR_APPAREIL.clear()
            rules.REGLES_PAR_APPAREIL.update(original)

    def test_plaid_reference_schema_is_present(self):
        required = {"P_steady_mean_W", "P_steady_std_W", "P_peak_mean_W"}
        for appliance, reference in rules.SEUILS_PLAID.items():
            with self.subTest(appliance=appliance):
                self.assertTrue(required.issubset(reference))
                self.assertGreaterEqual(reference["P_steady_mean_W"], 0)
                self.assertGreaterEqual(reference["P_peak_mean_W"], 0)


if __name__ == "__main__":
    unittest.main()
