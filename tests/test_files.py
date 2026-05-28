from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from video_stabilizer.files import build_output_path, collect_targets, resolve_collision


class TestFiles(unittest.TestCase):
    def test_collect_targets_from_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.mp4"
            p2 = Path(tmp) / "b.txt"
            p1.write_bytes(b"x")
            p2.write_bytes(b"x")
            found = collect_targets([str(tmp)], (".mp4",))
            self.assertEqual(len(found), 1)
            self.assertTrue(found[0].endswith("a.mp4"))

    def test_resolve_collision_rename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.mp4")
            Path(out).write_bytes(b"x")
            resolved = resolve_collision(out, "rename")
            self.assertIsNotNone(resolved)
            assert resolved is not None
            self.assertIn("(2)", resolved)

    def test_resolve_collision_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.mp4")
            Path(out).write_bytes(b"x")
            self.assertIsNone(resolve_collision(out, "skip"))

    def test_build_output_path_unified_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "clip.mp4")
            Path(inp).write_bytes(b"x")
            out = build_output_path(
                inp,
                output_dir=os.path.join(tmp, "out"),
                output_subdir="ignored",
                prefix="c_",
                suffix="",
            )
            self.assertTrue(out.endswith(os.path.join("out", "c_clip.mp4")))


if __name__ == "__main__":
    unittest.main()
