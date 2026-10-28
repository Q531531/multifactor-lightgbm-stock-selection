import unittest

import numpy as np
import pandas as pd

from original5_strategy.data import panel_from_long, panel_to_long


class DataTests(unittest.TestCase):
    def test_long_panel_round_trip(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
                "asset": ["A", "B", "A"],
                "open": [10.0, 20.0, 11.0],
                "high": [10.5, 20.5, 11.5],
                "low": [9.5, 19.5, 10.5],
                "close": [10.2, 20.2, 11.2],
                "volume": [1000.0, 2000.0, 1200.0],
                "amount": [10.0, 40.0, 13.0],
                "factor": [1.0, 1.0, 1.0],
                "member": [True, True, True],
            }
        )
        panel = panel_from_long(frame)
        self.assertEqual(panel["open"].shape, (2, 2))
        self.assertTrue(np.isnan(panel["open"][1, 1]))
        self.assertFalse(panel["member"][1, 1])
        rebuilt = panel_to_long(panel)
        self.assertEqual(len(rebuilt), 4)
        self.assertEqual(rebuilt.loc[0, "asset"], "A")

    def test_duplicates_are_rejected(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "asset": ["A", "A"],
                **{
                    name: [1.0, 1.0]
                    for name in (
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "amount",
                        "factor",
                    )
                },
                "member": [True, True],
            }
        )
        with self.assertRaises(ValueError):
            panel_from_long(frame)


if __name__ == "__main__":
    unittest.main()
