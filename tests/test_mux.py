"""Unit tests for audio remux helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from video_stabilizer.mux import (
    _add_stream_from_template,
    _mux_temp_path,
    finalize_output_with_audio,
    remux_preserve_audio,
    remux_video_only,
)


class TestMuxPaths(unittest.TestCase):
    def test_mux_temp_uses_final_extension(self) -> None:
        path = _mux_temp_path(r"D:/out/corrected_target.MP4")
        self.assertTrue(path.endswith(".vs-mux.mp4"))
        self.assertNotIn(".tmp", path)

    def test_mux_temp_defaults_to_mp4(self) -> None:
        path = _mux_temp_path(r"D:/out/corrected_target")
        self.assertTrue(path.endswith(".vs-mux.mp4"))


class TestMux(unittest.TestCase):
    def test_add_stream_from_template_uses_compat_method(self) -> None:
        out_container = MagicMock()
        in_stream = MagicMock()
        expected = MagicMock()
        out_container.add_stream_from_template.return_value = expected
        self.assertIs(_add_stream_from_template(out_container, in_stream), expected)
        out_container.add_stream_from_template.assert_called_once_with(in_stream)

    @patch("video_stabilizer.mux.remux_video_only", return_value=True)
    @patch("video_stabilizer.mux.av.open")
    def test_remux_no_audio_uses_video_only_remux(
        self, mock_open: MagicMock, mock_video_only: MagicMock
    ) -> None:
        source = MagicMock()
        source.streams.audio = []
        mock_open.return_value = source

        self.assertTrue(remux_preserve_audio("src.mp4", "vid.avi", "out.mp4"))
        mock_video_only.assert_called_once_with("vid.avi", "out.mp4")

    @patch("video_stabilizer.mux.remux_preserve_audio", return_value=True)
    @patch("video_stabilizer.mux.os.replace")
    def test_finalize_mux_success(self, mock_replace: MagicMock, mock_remux: MagicMock) -> None:
        self.assertTrue(
            finalize_output_with_audio(
                "src.mp4",
                "vid.avi",
                "out.mp4",
                preserve_audio=True,
            )
        )
        mock_remux.assert_called_once()
        mux_arg = mock_remux.call_args[0][2]
        self.assertTrue(mux_arg.endswith(".vs-mux.mp4"))
        mock_replace.assert_called_once()

    @patch("video_stabilizer.mux.remux_video_only", return_value=True)
    @patch("video_stabilizer.mux.os.replace")
    def test_finalize_without_audio_uses_video_remux(
        self, mock_replace: MagicMock, mock_video_only: MagicMock
    ) -> None:
        self.assertTrue(
            finalize_output_with_audio(
                "src.mp4",
                "vid.avi",
                "out.mp4",
                preserve_audio=False,
            )
        )
        mock_video_only.assert_called_once()
        mux_arg = mock_video_only.call_args[0][1]
        self.assertTrue(mux_arg.endswith(".vs-mux.mp4"))
        mock_replace.assert_called_once()

    @patch("video_stabilizer.mux.remux_preserve_audio", return_value=False)
    def test_finalize_mux_failure_returns_false(self, _mock_remux: MagicMock) -> None:
        self.assertFalse(
            finalize_output_with_audio(
                "src.mp4",
                "vid.avi",
                "out.mp4",
                preserve_audio=True,
            )
        )

    @patch("video_stabilizer.mux.remux_video_only", return_value=False)
    def test_finalize_video_only_failure_returns_false(self, _mock_remux: MagicMock) -> None:
        self.assertFalse(
            finalize_output_with_audio(
                "src.mp4",
                "vid.avi",
                "out.mkv",
                preserve_audio=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
