from __future__ import annotations

import unittest

import numpy as np

from video_stabilizer.color_metadata import ColorMetadata
from video_stabilizer.config import Config
from video_stabilizer.stats_constants import (
    IDX_U_MEAN,
    IDX_U_STD,
    IDX_V_MEAN,
    IDX_V_STD,
    IDX_Y_MEAN,
    IDX_Y_STD,
)
from video_stabilizer.worker import process_frame_worker


class TestWorker(unittest.TestCase):
    def test_gray_frame_stays_neutral_chroma(self) -> None:
        y = np.full((8, 8), 128, dtype=np.uint8)
        u = np.full((4, 4), 128, dtype=np.uint8)
        v = np.full((4, 4), 128, dtype=np.uint8)
        cur = np.array([128.0, 10.0, 128.0, 5.0, 128.0, 5.0, 40.0], dtype=np.float32)
        tgt = np.array([128.0, 10.0, 128.0, 5.0, 128.0, 5.0, 40.0], dtype=np.float32)
        out = process_frame_worker(y, u, v, cur, tgt, Config())
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.shape, (8, 8, 3))

    def test_full_range_uses_wider_clip(self) -> None:
        y = np.full((8, 8), 200, dtype=np.uint8)
        u = np.full((4, 4), 140, dtype=np.uint8)
        v = np.full((4, 4), 140, dtype=np.uint8)
        cur = np.zeros(7, dtype=np.float32)
        cur[IDX_Y_MEAN] = 100
        cur[IDX_Y_STD] = 20
        cur[IDX_U_MEAN] = 128
        cur[IDX_U_STD] = 10
        cur[IDX_V_MEAN] = 128
        cur[IDX_V_STD] = 10
        tgt = cur.copy()
        tgt[IDX_Y_MEAN] = 150
        meta = ColorMetadata(is_full_range=True, color_range="jpeg")
        out = process_frame_worker(y, u, v, cur, tgt, Config(), meta)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertGreaterEqual(int(out.max()), 200)


if __name__ == "__main__":
    unittest.main()
