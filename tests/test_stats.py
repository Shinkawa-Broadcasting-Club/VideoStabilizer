from __future__ import annotations

import unittest

import numpy as np

from video_stabilizer.stats import (
    clamp_mitchell_t,
    find_stats_segment_index,
    interpolate_stats_for_frame,
    interpolate_stats_tail,
)


class TestStatsInterpolation(unittest.TestCase):
    def _buffer(self) -> list[tuple[int, np.ndarray]]:
        return [
            (0, np.array([100.0, 20.0, 50.0], dtype=np.float32)),
            (10, np.array([110.0, 22.0, 55.0], dtype=np.float32)),
            (20, np.array([120.0, 24.0, 60.0], dtype=np.float32)),
            (30, np.array([130.0, 26.0, 65.0], dtype=np.float32)),
        ]

    def test_find_segment_inclusive_upper_bound(self):
        buf = self._buffer()
        self.assertEqual(find_stats_segment_index(buf, 10), 1)
        self.assertEqual(find_stats_segment_index(buf, 15), 1)
        self.assertEqual(find_stats_segment_index(buf, 0), 0)

    def test_find_segment_miss_before_first(self):
        buf = self._buffer()
        self.assertEqual(find_stats_segment_index(buf, -1), -1)

    def test_clamp_mitchell_t(self):
        self.assertEqual(clamp_mitchell_t(-0.5), 0.0)
        self.assertEqual(clamp_mitchell_t(2.5, t_max=1.0), 1.0)
        self.assertAlmostEqual(clamp_mitchell_t(0.4), 0.4)

    def test_interpolate_at_keyframe_boundary(self):
        buf = self._buffer()
        result = interpolate_stats_for_frame(10, buf, sampling_n=10, t_max=1.0)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, buf[1][1], rtol=1e-4)

    def test_interpolate_mid_segment(self):
        buf = self._buffer()
        result = interpolate_stats_for_frame(15, buf, sampling_n=10, t_max=1.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertGreaterEqual(result[0], buf[1][1][0])
        self.assertLessEqual(result[0], buf[2][1][0])

    def test_interpolate_tail_clamps_t(self):
        buf = self._buffer()
        result_far = interpolate_stats_tail(100, buf, sampling_n=10, t_max=1.0)
        self.assertIsNotNone(result_far)
        result_at_one = interpolate_stats_tail(30, buf, sampling_n=10, t_max=1.0)
        self.assertIsNotNone(result_at_one)
        np.testing.assert_allclose(result_far, result_at_one, rtol=1e-4, atol=1e-4)

    def test_interpolate_returns_none_with_insufficient_buffer(self):
        self.assertIsNone(interpolate_stats_for_frame(5, [(0, np.zeros(3))], 10))


if __name__ == "__main__":
    unittest.main()
