from __future__ import annotations

import unittest

from video_stabilizer.manifest import JobEntry, should_skip_job


class TestManifest(unittest.TestCase):
    def test_skip_done_when_output_exists(self) -> None:
        entry = JobEntry("in.mp4", "out.mp4", status="done")
        self.assertTrue(should_skip_job(entry, "skip_done", output_exists=True))

    def test_retry_failed_does_not_skip_failed(self) -> None:
        entry = JobEntry("in.mp4", "out.mp4", status="failed")
        self.assertFalse(should_skip_job(entry, "retry_failed", output_exists=False))

    def test_run_all_never_skips(self) -> None:
        entry = JobEntry("in.mp4", "out.mp4", status="done")
        self.assertFalse(should_skip_job(entry, "run_all", output_exists=True))


if __name__ == "__main__":
    unittest.main()
