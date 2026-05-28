"""E2E-style integration test with mocked video I/O (CI-safe)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from video_stabilizer.config import Config
from video_stabilizer.job_runner import run_batch
from video_stabilizer.stats_constants import STAT_DIM


class _FakeStream:
    frames = 2
    duration = 2
    time_base = 1
    average_rate = 30
    width = 8
    height = 8


class _FakeContainer:
    def __init__(self) -> None:
        self.streams = SimpleNamespace(video=[_FakeStream()])

    def decode(self, _stream):
        for _ in range(2):
            yield object()

    def close(self) -> None:
        return None


class TestE2EBatch(unittest.TestCase):
    def test_run_batch_success_mocked(self) -> None:
        ref_stats = np.array(
            [100.0, 20.0, 128.0, 10.0, 128.0, 10.0, 50.0], dtype=np.float32
        )
        fake_y = np.full((8, 8), 64, dtype=np.uint8)
        fake_u = np.full((4, 4), 128, dtype=np.uint8)
        fake_v = np.full((4, 4), 128, dtype=np.uint8)
        fake_out = np.full((8, 8, 3), 32, dtype=np.uint8)

        class _Writer:
            def isOpened(self) -> bool:
                return True

            def write(self, _f) -> None:
                return None

            def release(self) -> None:
                return None

        with (
            patch(
                "video_stabilizer.job_runner.analyze_reference_video",
                return_value=ref_stats,
            ),
            patch("video_stabilizer.pipeline.av.open", return_value=_FakeContainer()),
            patch("video_stabilizer.pipeline.cv2.VideoWriter", return_value=_Writer()),
            patch("video_stabilizer.pipeline.cv2.VideoWriter_fourcc", return_value=0),
            patch(
                "video_stabilizer.pipeline.extract_yuv_planes",
                return_value=(fake_y, fake_u, fake_v),
            ),
            patch(
                "video_stabilizer.pipeline.get_stats_and_coeffs",
                return_value=ref_stats,
            ),
            patch(
                "video_stabilizer.pipeline.process_frame_worker",
                return_value=fake_out,
            ),
            patch(
                "video_stabilizer.pipeline.finalize_output_with_audio",
                return_value=True,
            ),
            patch(
                "video_stabilizer.job_runner.collect_targets",
                return_value=["in.mp4"],
            ),
            patch("video_stabilizer.job_runner.build_output_path", return_value="out.mp4"),
            patch("video_stabilizer.job_runner.resolve_collision", return_value="out.mp4"),
            patch("video_stabilizer.job_runner.save_manifest"),
        ):
            result = run_batch(
                "ref.mp4",
                ["in.mp4"],
                Config(use_gui_progress=True, collision_policy="overwrite"),
            )

        self.assertEqual(result.ok_count, 1)
        self.assertEqual(result.fail_count, 0)
        self.assertEqual(ref_stats.shape[0], STAT_DIM)


if __name__ == "__main__":
    unittest.main()
