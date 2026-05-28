from __future__ import annotations

import unittest

import numpy as np

from video_stabilizer.stats import robust_aggregate_stats


class TestRobustReference(unittest.TestCase):
    def test_trim_mean_excludes_extreme_frame(self) -> None:
        base = np.array([100.0, 20.0, 128.0, 10.0, 128.0, 10.0, 50.0], dtype=np.float32)
        stats_list = [base.copy() for _ in range(8)]
        stats_list.append(
            np.array([255.0, 80.0, 200.0, 50.0, 200.0, 50.0, 255.0], dtype=np.float32)
        )
        agg = robust_aggregate_stats(stats_list, percentile_low=5, percentile_high=95)
        self.assertIsNotNone(agg)
        self.assertAlmostEqual(float(agg[0]), 100.0, delta=15.0)


if __name__ == "__main__":
    unittest.main()
