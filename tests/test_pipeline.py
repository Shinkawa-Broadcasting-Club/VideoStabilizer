from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from video_stabilizer.config import Config
from video_stabilizer.pipeline import (
    _FrameWriteState,
    _NullProgress,
    _drain_all_futures,
    _video_only_temp_path,
    process_target_video,
    _write_next_ready,
)


class _MockWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame)


class TestFrameWriteState(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(on_frame_failure="hold")
        self.pbar = MagicMock()
        self.writer = _MockWriter()
        self.state = _FrameWriteState(self.writer, self.pbar, self.config, 4, 4)

    def test_writes_good_frame(self) -> None:
        frame = np.ones((4, 4, 3), dtype=np.uint8)
        self.assertTrue(self.state.resolve(frame, 0))
        self.assertEqual(len(self.writer.frames), 1)

    def test_hold_repeats_last_good_on_failure(self) -> None:
        good = np.full((4, 4, 3), 42, dtype=np.uint8)
        self.state.resolve(good, 0)
        self.assertTrue(self.state.resolve(None, 1))
        self.assertEqual(len(self.writer.frames), 2)
        np.testing.assert_array_equal(self.writer.frames[1], good)

    def test_black_policy(self) -> None:
        state = _FrameWriteState(
            _MockWriter(), MagicMock(), Config(on_frame_failure="black"), 2, 2
        )
        self.assertTrue(state.resolve(None, 0))
        self.assertEqual(state._black.shape, (2, 2, 3))  # noqa: SLF001

    def test_abort_policy(self) -> None:
        state = _FrameWriteState(
            _MockWriter(), MagicMock(), Config(on_frame_failure="abort"), 2, 2
        )
        self.assertFalse(state.resolve(None, 0))
        self.assertTrue(state.aborted)


class TestVideoWriterPaths(unittest.TestCase):
    def test_video_only_temp_uses_avi(self) -> None:
        path = _video_only_temp_path(r"D:/out/corrected_target.MP4")
        self.assertTrue(path.endswith(".vs-video-only.avi"))
        self.assertNotIn(".MP4.vs-video-only", path)


class TestDrainHelpers(unittest.TestCase):
    def test_drain_all_fills_gaps_with_hold(self) -> None:
        config = Config(on_frame_failure="hold")
        writer = _MockWriter()
        pbar = MagicMock()
        state = _FrameWriteState(writer, pbar, config, 2, 2)
        good = np.full((2, 2, 3), 7, dtype=np.uint8)
        state.resolve(good, 0)
        writer.frames.clear()

        futures: dict = {}
        next_idx = _drain_all_futures(futures, 0, expected_frames=3, write_state=state)
        self.assertEqual(next_idx, 3)
        self.assertEqual(len(writer.frames), 3)
        np.testing.assert_array_equal(writer.frames[0], good)

    def test_write_next_ready_skips_none_with_hold(self) -> None:
        import concurrent.futures

        config = Config(on_frame_failure="hold")
        writer = _MockWriter()
        state = _FrameWriteState(writer, MagicMock(), config, 2, 2)
        good = np.full((2, 2, 3), 1, dtype=np.uint8)

        fut_ok = concurrent.futures.Future()
        fut_ok.set_result(good)
        fut_fail = concurrent.futures.Future()
        fut_fail.set_result(None)

        futures = {0: fut_ok, 1: fut_fail}
        state.resolve(good, 0)
        next_idx = _write_next_ready(futures, 1, state)
        self.assertEqual(next_idx, 2)
        self.assertEqual(len(writer.frames), 2)


class _FakeStream:
    frames = 3
    duration = 3
    time_base = 1
    average_rate = 30
    width = 4
    height = 4


class _FakeStreamMissingRate:
    frames = 0
    duration = None
    time_base = None
    average_rate = None
    guessed_rate = None
    base_rate = None
    width = 4
    height = 4


class _FakeContainer:
    def __init__(self) -> None:
        self.streams = SimpleNamespace(video=[_FakeStream()])

    def decode(self, _stream):
        for _ in range(3):
            yield object()

    def close(self) -> None:
        return None


class _FakeContainerMissingRate:
    def __init__(self) -> None:
        self.streams = SimpleNamespace(video=[_FakeStreamMissingRate()])

    def decode(self, _stream):
        for _ in range(3):
            yield object()

    def close(self) -> None:
        return None


class _FakeVideoWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        return None


class TestProcessTargetVideo(unittest.TestCase):
    def test_process_target_video_writes_all_frames(self) -> None:
        fake_writer = _FakeVideoWriter()
        fake_y = np.full((4, 4), 64, dtype=np.uint8)
        fake_u = np.full((2, 2), 128, dtype=np.uint8)
        fake_v = np.full((2, 2), 128, dtype=np.uint8)
        fake_stats = np.array(
            [100.0, 20.0, 128.0, 10.0, 128.0, 10.0, 50.0], dtype=np.float32
        )
        fake_output = np.full((4, 4, 3), 32, dtype=np.uint8)

        with (
            patch("video_stabilizer.pipeline.av.open", return_value=_FakeContainer()),
            patch("video_stabilizer.pipeline.cv2.VideoWriter", return_value=fake_writer),
            patch("video_stabilizer.pipeline.cv2.VideoWriter_fourcc", return_value=0),
            patch(
                "video_stabilizer.pipeline.extract_yuv_planes",
                return_value=(fake_y, fake_u, fake_v),
            ),
            patch(
                "video_stabilizer.pipeline.get_stats_and_coeffs",
                return_value=fake_stats,
            ),
            patch(
                "video_stabilizer.pipeline.process_frame_worker",
                return_value=fake_output,
            ),
            patch(
                "video_stabilizer.pipeline.finalize_output_with_audio",
                return_value=True,
            ) as mock_finalize,
        ):
            ok = process_target_video("in.mp4", "out.mp4", fake_stats, Config())

        self.assertTrue(ok)
        self.assertEqual(len(fake_writer.frames), 3)
        mock_finalize.assert_called_once()
        self.assertEqual(mock_finalize.call_args.kwargs.get("preserve_audio"), True)

    def test_process_target_video_fails_when_finalize_fails(self) -> None:
        fake_writer = _FakeVideoWriter()
        fake_y = np.full((4, 4), 64, dtype=np.uint8)
        fake_u = np.full((2, 2), 128, dtype=np.uint8)
        fake_v = np.full((2, 2), 128, dtype=np.uint8)
        fake_stats = np.array(
            [100.0, 20.0, 128.0, 10.0, 128.0, 10.0, 50.0], dtype=np.float32
        )
        fake_output = np.full((4, 4, 3), 32, dtype=np.uint8)

        with (
            patch("video_stabilizer.pipeline.av.open", return_value=_FakeContainer()),
            patch("video_stabilizer.pipeline.cv2.VideoWriter", return_value=fake_writer),
            patch("video_stabilizer.pipeline.cv2.VideoWriter_fourcc", return_value=0),
            patch(
                "video_stabilizer.pipeline.extract_yuv_planes",
                return_value=(fake_y, fake_u, fake_v),
            ),
            patch(
                "video_stabilizer.pipeline.get_stats_and_coeffs",
                return_value=fake_stats,
            ),
            patch(
                "video_stabilizer.pipeline.process_frame_worker",
                return_value=fake_output,
            ),
            patch(
                "video_stabilizer.pipeline.finalize_output_with_audio",
                return_value=False,
            ),
            patch("video_stabilizer.pipeline._remove_partial_output") as mock_remove,
        ):
            ok = process_target_video("in.mp4", "out.mp4", fake_stats, Config())

        self.assertFalse(ok)
        mock_remove.assert_any_call("out.mp4")


class TestProgressAndCancel(unittest.TestCase):
    def test_progress_callback_invoked(self) -> None:
        fake_writer = _FakeVideoWriter()
        fake_y = np.full((4, 4), 64, dtype=np.uint8)
        fake_u = np.full((2, 2), 128, dtype=np.uint8)
        fake_v = np.full((2, 2), 128, dtype=np.uint8)
        fake_stats = np.array(
            [100.0, 20.0, 128.0, 10.0, 128.0, 10.0, 50.0], dtype=np.float32
        )
        fake_output = np.full((4, 4, 3), 32, dtype=np.uint8)
        calls: list[tuple[int, int | None]] = []

        def progress_cb(cur: int, tot: int | None) -> None:
            calls.append((cur, tot))

        with (
            patch("video_stabilizer.pipeline.av.open", return_value=_FakeContainer()),
            patch("video_stabilizer.pipeline.cv2.VideoWriter", return_value=fake_writer),
            patch("video_stabilizer.pipeline.cv2.VideoWriter_fourcc", return_value=0),
            patch(
                "video_stabilizer.pipeline.extract_yuv_planes",
                return_value=(fake_y, fake_u, fake_v),
            ),
            patch(
                "video_stabilizer.pipeline.get_stats_and_coeffs",
                return_value=fake_stats,
            ),
            patch(
                "video_stabilizer.pipeline.process_frame_worker",
                return_value=fake_output,
            ),
            patch(
                "video_stabilizer.pipeline.finalize_output_with_audio",
                return_value=True,
            ),
        ):
            ok = process_target_video(
                "in.mp4",
                "out.mp4",
                fake_stats,
                Config(use_gui_progress=True),
                progress_cb=progress_cb,
            )

        self.assertTrue(ok)
        self.assertGreater(len(calls), 0)

    def test_cancel_event_aborts(self) -> None:
        fake_writer = _FakeVideoWriter()
        cancel = threading.Event()
        cancel.set()
        fake_stats = np.zeros(7, dtype=np.float32)

        with (
            patch("video_stabilizer.pipeline.av.open", return_value=_FakeContainer()),
            patch("video_stabilizer.pipeline.cv2.VideoWriter", return_value=fake_writer),
            patch("video_stabilizer.pipeline.cv2.VideoWriter_fourcc", return_value=0),
            patch(
                "video_stabilizer.pipeline.extract_yuv_planes",
                return_value=None,
            ),
            patch(
                "video_stabilizer.pipeline.finalize_output_with_audio",
                return_value=True,
            ),
        ):
            ok = process_target_video(
                "in.mp4",
                "out.mp4",
                fake_stats,
                Config(use_gui_progress=True),
                cancel_event=cancel,
            )

        self.assertFalse(ok)

    def test_missing_rate_metadata_uses_fallback_fps(self) -> None:
        fake_writer = _FakeVideoWriter()
        fake_y = np.full((4, 4), 64, dtype=np.uint8)
        fake_u = np.full((2, 2), 128, dtype=np.uint8)
        fake_v = np.full((2, 2), 128, dtype=np.uint8)
        fake_stats = np.array(
            [100.0, 20.0, 128.0, 10.0, 128.0, 10.0, 50.0], dtype=np.float32
        )
        fake_output = np.full((4, 4, 3), 32, dtype=np.uint8)

        with (
            patch(
                "video_stabilizer.pipeline.av.open",
                return_value=_FakeContainerMissingRate(),
            ),
            patch("video_stabilizer.pipeline.cv2.VideoWriter", return_value=fake_writer),
            patch("video_stabilizer.pipeline.cv2.VideoWriter_fourcc", return_value=0),
            patch(
                "video_stabilizer.pipeline.extract_yuv_planes",
                return_value=(fake_y, fake_u, fake_v),
            ),
            patch(
                "video_stabilizer.pipeline.get_stats_and_coeffs",
                return_value=fake_stats,
            ),
            patch(
                "video_stabilizer.pipeline.process_frame_worker",
                return_value=fake_output,
            ),
            patch(
                "video_stabilizer.pipeline.finalize_output_with_audio",
                return_value=True,
            ),
        ):
            ok = process_target_video("in.mp4", "out.mp4", fake_stats, Config())

        self.assertTrue(ok)


class TestFrameWriteStateProgress(unittest.TestCase):
    def test_null_progress_does_not_crash(self) -> None:
        writer = _MockWriter()
        state = _FrameWriteState(
            writer,  # type: ignore[arg-type]
            _NullProgress(),
            Config(),
            2,
            2,
            progress_cb=lambda c, t: None,
        )
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertTrue(state.resolve(frame, 0))


if __name__ == "__main__":
    unittest.main()
