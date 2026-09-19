"""
tests/test_pdv_features.py - Unit tests for TSPL kernels, R1/R2 causality, and RV targets
"""

import unittest
import numpy as np
import pandas as pd
from pdv_features import (
    compute_tspl_weights,
    compute_r1_r2,
    compute_forward_rv,
    compute_trailing_rv
)


class TestPDVFeatures(unittest.TestCase):
    def test_tspl_weights_sum_to_one(self):
        for alpha in [0.5, 1.0, 1.8]:
            for delta in [0.1, 1.0, 10.0]:
                for N in [50, 500]:
                    w = compute_tspl_weights(alpha, delta, N)
                    self.assertEqual(len(w), N)
                    self.assertAlmostEqual(np.sum(w), 1.0, places=7)
                    self.assertTrue((w > 0).all())
                    # Power law decay check: w[0] > w[-1]
                    self.assertGreater(w[0], w[-1])

    def test_r1_r2_strict_causality(self):
        # Verify that R1_t has ZERO dependence on return at time t
        N = 10
        returns = np.array([0.01, -0.02, 0.015, -0.005, 0.03, 0.01, -0.01], dtype=np.float64)
        r1, r2 = compute_r1_r2(returns, alpha1=1.0, delta1=1.0, alpha2=1.0, delta2=1.0, N=N)
        
        # At index 0, there are no prior bars, so r1[0] and r2[0] MUST be 0.0
        self.assertEqual(r1[0], 0.0)
        self.assertEqual(r2[0], 0.0)
        
        # If we alter returns[4], r1[4] and r2[4] must NOT change!
        returns_mod = returns.copy()
        returns_mod[4] = 999.0
        r1_mod, r2_mod = compute_r1_r2(returns_mod, alpha1=1.0, delta1=1.0, alpha2=1.0, delta2=1.0, N=N)
        
        self.assertEqual(r1[4], r1_mod[4])
        self.assertEqual(r2[4], r2_mod[4])
        
        # But r1[5] MUST change because returns[4] is the lag-1 return for index 5!
        self.assertNotEqual(r1[5], r1_mod[5])
        self.assertNotEqual(r2[5], r2_mod[5])

    def test_forward_rv_causality_and_values(self):
        gk_var = pd.Series([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float64)
        h = 2
        rv_fwd = compute_forward_rv(gk_var, h=h)
        
        # At t=0: future bars are t=1 (4.0) and t=2 (9.0) -> sum=13.0, sqrt = sqrt(13)
        self.assertAlmostEqual(rv_fwd.iloc[0], np.sqrt(13.0), places=7)
        # At t=1: future bars are t=2 (9.0) and t=3 (16.0) -> sum=25.0, sqrt = 5.0
        self.assertAlmostEqual(rv_fwd.iloc[1], 5.0, places=7)
        # At t=2: future bars are t=3 (16.0) and t=4 (25.0) -> sum=41.0, sqrt = sqrt(41)
        self.assertAlmostEqual(rv_fwd.iloc[2], np.sqrt(41.0), places=7)
        # At t=3, 4: not enough future bars -> NaN
        self.assertTrue(pd.isna(rv_fwd.iloc[3]))
        self.assertTrue(pd.isna(rv_fwd.iloc[4]))

    def test_trailing_rv_causality_and_values(self):
        gk_var = pd.Series([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float64)
        h = 2
        rv_trail = compute_trailing_rv(gk_var, h=h)
        
        # At t=0: only 1 bar available -> NaN
        self.assertTrue(pd.isna(rv_trail.iloc[0]))
        # At t=1: bars t=0 (1.0) and t=1 (4.0) -> sum=5.0, sqrt = sqrt(5)
        self.assertAlmostEqual(rv_trail.iloc[1], np.sqrt(5.0), places=7)
        # At t=2: bars t=1 (4.0) and t=2 (9.0) -> sum=13.0, sqrt = sqrt(13)
        self.assertAlmostEqual(rv_trail.iloc[2], np.sqrt(13.0), places=7)


if __name__ == "__main__":
    unittest.main()
