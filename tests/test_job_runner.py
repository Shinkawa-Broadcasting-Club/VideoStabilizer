from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from video_stabilizer.config import Config
from video_stabilizer.job_runner import run_batch
from video_stabilizer.manifest import BatchManifest, JobEntry, config_hash


class TestJobRunnerResume(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config(collision_policy="overwrite", resume_mode="skip_done")
        self.cfg_hash = config_hash(self.cfg.to_serializable_dict())
        self.ref_stats = np.array(
            [100.0, 20.0, 128.0, 10.0, 128.0, 10.0, 50.0], dtype=np.float32
        )

    def test_skip_done_uses_existing_manifest_status(self) -> None:
        existing = BatchManifest(
            config_hash=self.cfg_hash,
            jobs=[JobEntry("in.mp4", "out.mp4", status="done")],
        )
        with (
            patch("video_stabilizer.job_runner.collect_targets", return_value=["in.mp4"]),
            patch("video_stabilizer.job_runner.analyze_reference_video", return_value=self.ref_stats),
            patch("video_stabilizer.job_runner.build_output_path", return_value="out.mp4"),
            patch("video_stabilizer.job_runner.resolve_collision", return_value="out.mp4"),
            patch("video_stabilizer.job_runner.load_manifest", return_value=existing),
            patch("video_stabilizer.job_runner.save_manifest"),
            patch("video_stabilizer.job_runner.os.path.isfile", return_value=True),
            patch("video_stabilizer.job_runner.process_target_video") as mock_process,
        ):
            result = run_batch(
                "ref.mp4",
                ["in.mp4"],
                self.cfg,
                manifest_path="manifest.json",
            )

        self.assertEqual(result.skip_count, 1)
        self.assertEqual(result.ok_count, 0)
        mock_process.assert_not_called()

    def test_retry_failed_reuses_manifest_and_runs(self) -> None:
        cfg = Config(collision_policy="overwrite", resume_mode="retry_failed")
        existing = BatchManifest(
            config_hash=config_hash(cfg.to_serializable_dict()),
            jobs=[JobEntry("in.mp4", "out.mp4", status="failed", error="old")],
        )
        with (
            patch("video_stabilizer.job_runner.collect_targets", return_value=["in.mp4"]),
            patch("video_stabilizer.job_runner.analyze_reference_video", return_value=self.ref_stats),
            patch("video_stabilizer.job_runner.build_output_path", return_value="out.mp4"),
            patch("video_stabilizer.job_runner.resolve_collision", return_value="out.mp4"),
            patch("video_stabilizer.job_runner.load_manifest", return_value=existing),
            patch("video_stabilizer.job_runner.save_manifest"),
            patch("video_stabilizer.job_runner.os.path.isfile", return_value=False),
            patch("video_stabilizer.job_runner.process_target_video", return_value=True),
        ):
            result = run_batch(
                "ref.mp4",
                ["in.mp4"],
                cfg,
                manifest_path="manifest.json",
            )

        self.assertEqual(result.ok_count, 1)
        self.assertEqual(existing.jobs[0].status, "done")


if __name__ == "__main__":
    unittest.main()

