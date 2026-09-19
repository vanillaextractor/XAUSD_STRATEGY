"""
tests/test_data_prep.py - Unit tests for data preparation
"""

import unittest
import numpy as np
import pandas as pd
from data_prep import compute_garman_klass_var, resample_to_5min


class TestDataPrep(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 1-minute dataframe with a weekend gap
        idx1 = pd.date_range("2023-01-06 16:50:00", "2023-01-06 16:55:00", freq="1min")
        # Weekend gap to Sunday 18:00
        idx2 = pd.date_range("2023-01-08 18:00:00", "2023-01-08 18:10:00", freq="1min")
        idx = idx1.union(idx2)
        
        # Friday close = 1800, Sunday open = 1820 (jump of +20)
        opens = [1800.0] * len(idx1) + [1820.0] * len(idx2)
        highs = [1801.0] * len(idx1) + [1821.0] * len(idx2)
        lows = [1799.0] * len(idx1) + [1819.0] * len(idx2)
        closes = [1800.5] * len(idx1) + [1820.5] * len(idx2)
        
        self.df_m1 = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows, "Close": closes
        }, index=idx)

    def test_garman_klass_variance(self):
        gk = compute_garman_klass_var(self.df_m1)
        self.assertTrue((gk >= 0.0).all())
        self.assertFalse(gk.isna().any())

    def test_resampling_and_weekend_gap(self):
        df_5m = resample_to_5min(self.df_m1)
        
        # Verify gap detection
        self.assertTrue(df_5m["is_gap"].any())
        
        # Identify the Sunday opening bar
        sunday_bar = df_5m.loc[pd.Timestamp("2023-01-08 18:00:00")]
        self.assertTrue(sunday_bar["is_gap"])
        
        # Raw C2C return would be log(1820.5 / 1800.5) ~ +1.1%
        raw_ret = np.log(1820.5 / 1800.5)
        # Intrabar return on Sunday bar should be log(1820.5 / 1820.0) ~ +0.027%
        expected_ret = np.log(1820.5 / 1820.0)
        
        self.assertAlmostEqual(sunday_bar["return_c2c_raw"], raw_ret, places=5)
        self.assertAlmostEqual(sunday_bar["return"], expected_ret, places=5)
        self.assertNotAlmostEqual(sunday_bar["return"], raw_ret, places=3)
        self.assertFalse(df_5m["return"].isna().any())


if __name__ == "__main__":
    unittest.main()
