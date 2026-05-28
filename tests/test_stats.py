from __future__ import annotations

import unittest

import numpy as np

from video_stabilizer.stats import (
    clamp_mitchell_t,
    find_stats_segment_index,
    get_stats_and_coeffs,
    interpolate_stats_for_frame,
    interpolate_stats_tail,
    robust_aggregate_stats,
)
from video_stabilizer.stats_constants import STAT_DIM


def _sample7(
    y=100.0, ys=20.0, u=128.0, us=10.0, v=128.0, vs=10.0, s=50.0
) -> np.ndarray:
    return np.array([y, ys, u, us, v, vs, s], dtype=np.float32)


class TestStats7D(unittest.TestCase):
    def test_get_stats_returns_7_elements(self) -> None:
        y = np.full((32, 32), 100, dtype=np.uint8)
        u = np.full((16, 16), 128, dtype=np.uint8)
        v = np.full((16, 16), 128, dtype=np.uint8)
        stats = get_stats_and_coeffs(y, u, v, 256)
        self.assertEqual(stats.shape, (STAT_DIM,))

    def test_robust_aggregate_rejects_outlier(self) -> None:
        normal = [_sample7() for _ in range(10)]
        outlier = _sample7(y=250.0, ys=80.0, u=200.0, us=50.0, v=200.0, vs=50.0, s=200.0)
        agg = robust_aggregate_stats(normal + [outlier], percentile_low=10, percentile_high=90)
        self.assertIsNotNone(agg)
        self.assertLess(float(agg[0]), 150.0)


class TestStatsInterpolation(unittest.TestCase):
    def _buffer(self) -> list[tuple[int, np.ndarray]]:
        return [
            (0, _sample7(100, 20, 128, 10, 128, 10, 50)),
            (10, _sample7(110, 22, 130, 11, 126, 11, 55)),
            (20, _sample7(120, 24, 132, 12, 124, 12, 60)),
            (30, _sample7(130, 26, 134, 13, 122, 13, 65)),
        ]

    def test_find_segment_inclusive_upper_bound(self) -> None:
        buf = self._buffer()
        self.assertEqual(find_stats_segment_index(buf, 10), 1)
        self.assertEqual(find_stats_segment_index(buf, 15), 1)
        self.assertEqual(find_stats_segment_index(buf, 0), 0)

    def test_find_segment_miss_before_first(self) -> None:
        buf = self._buffer()
        self.assertEqual(find_stats_segment_index(buf, -1), -1)

    def test_clamp_mitchell_t(self) -> None:
        self.assertEqual(clamp_mitchell_t(-0.5), 0.0)
        self.assertEqual(clamp_mitchell_t(2.5, t_max=1.0), 1.0)
        self.assertAlmostEqual(clamp_mitchell_t(0.4), 0.4)

    def test_interpolate_at_keyframe_boundary(self) -> None:
        buf = self._buffer()
        result = interpolate_stats_for_frame(10, buf, sampling_n=10, t_max=1.0)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, buf[1][1], rtol=1e-4)

    def test_interpolate_mid_segment(self) -> None:
        buf = self._buffer()
        result = interpolate_stats_for_frame(15, buf, sampling_n=10, t_max=1.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (STAT_DIM,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertGreaterEqual(result[0], buf[1][1][0])
        self.assertLessEqual(result[0], buf[2][1][0])

    def test_interpolate_tail_clamps_t(self) -> None:
        buf = self._buffer()
        result_far = interpolate_stats_tail(100, buf, sampling_n=10, t_max=1.0)
        self.assertIsNotNone(result_far)
        result_at_one = interpolate_stats_tail(30, buf, sampling_n=10, t_max=1.0)
        self.assertIsNotNone(result_at_one)
        np.testing.assert_allclose(result_far, result_at_one, rtol=1e-4, atol=1e-4)

    def test_interpolate_returns_none_with_insufficient_buffer(self) -> None:
        self.assertIsNone(interpolate_stats_for_frame(5, [(0, np.zeros(7))], 10))


if __name__ == "__main__":
    unittest.main()
