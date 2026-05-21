# 補正済み映像と元音声の remux

from __future__ import annotations

import logging
import os

import av

logger = logging.getLogger(__name__)


def _add_stream_from_template(
    out_container: av.container.OutputContainer,
    in_stream: av.stream.Stream,
) -> av.stream.Stream:
    """Add output stream from an input stream template (PyAV compatibility helper)."""
    # PyAV 17 uses add_stream_from_template; add_stream(template=...) is not supported.
    if hasattr(out_container, "add_stream_from_template"):
        return out_container.add_stream_from_template(in_stream)
    return out_container.add_stream(template=in_stream)


def _mux_temp_path(output_path: str) -> str:
    """Temporary mux path with the same container extension as the final output."""
    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".mp4"
    return base + ".vs-mux" + ext.lower()


def _packet_time_seconds(packet: av.Packet, stream: av.stream.Stream) -> float | None:
    if packet.pts is None:
        return None
    return float(packet.pts * stream.time_base)


def _mux_video_packets(
    video_only_path: str,
    output_path: str,
    out_container: av.container.OutputContainer,
) -> bool:
    video_in: av.container.InputContainer | None = None
    try:
        video_in = av.open(video_only_path)
        in_v = video_in.streams.video[0]
        if in_v is None:
            logger.error("No video stream in corrected file: %s", video_only_path)
            return False

        out_v = _add_stream_from_template(out_container, in_v)
        for packet in video_in.demux(in_v):
            if packet.dts is None and packet.pts is None:
                continue
            packet.stream = out_v
            out_container.mux(packet)
        return True
    except Exception:
        logger.exception("Failed to mux video from %s -> %s", video_only_path, output_path)
        return False
    finally:
        if video_in is not None:
            video_in.close()


def remux_video_only(video_only_path: str, output_path: str) -> bool:
    """Remux corrected video into a container matching output_path extension."""
    out_container: av.container.OutputContainer | None = None
    try:
        out_container = av.open(output_path, mode="w")
        return _mux_video_packets(video_only_path, output_path, out_container)
    except Exception:
        logger.exception("Failed to open output container: %s", output_path)
        return False
    finally:
        if out_container is not None:
            out_container.close()


def remux_preserve_audio(source_path: str, video_only_path: str, output_path: str) -> bool:
    """Mux corrected video with audio copied from the source file.

    Uses stream copy (no re-encode). Trims audio to the corrected video duration.
    If the source has no audio, remuxes video only into the target container.
    """
    source: av.container.InputContainer | None = None
    video_in: av.container.InputContainer | None = None
    out_container: av.container.OutputContainer | None = None

    try:
        source = av.open(source_path)
        audio_streams = source.streams.audio
        if not audio_streams:
            logger.info("No audio in source; remuxing video only: %s", source_path)
            source.close()
            source = None
            return remux_video_only(video_only_path, output_path)

        video_in = av.open(video_only_path)
        in_v = video_in.streams.video[0]
        if in_v is None:
            logger.error("No video stream in corrected file: %s", video_only_path)
            return False

        in_a = audio_streams[0]
        last_video_time: float | None = None

        out_container = av.open(output_path, mode="w")
        out_v = _add_stream_from_template(out_container, in_v)
        out_a = _add_stream_from_template(out_container, in_a)

        for packet in video_in.demux(in_v):
            if packet.dts is None and packet.pts is None:
                continue
            packet.stream = out_v
            out_container.mux(packet)
            pts_time = _packet_time_seconds(packet, in_v)
            if pts_time is not None:
                last_video_time = pts_time if last_video_time is None else max(last_video_time, pts_time)

        for packet in source.demux(in_a):
            if packet.dts is None and packet.pts is None:
                continue
            pts_time = _packet_time_seconds(packet, in_a)
            if last_video_time is not None and pts_time is not None and pts_time > last_video_time:
                break
            packet.stream = out_a
            out_container.mux(packet)

        return True
    except Exception:
        logger.exception(
            "Failed to remux audio from %s with video %s -> %s",
            source_path,
            video_only_path,
            output_path,
        )
        return False
    finally:
        if out_container is not None:
            out_container.close()
        if video_in is not None:
            video_in.close()
        if source is not None:
            source.close()


def finalize_output_with_audio(
    source_path: str,
    video_only_path: str,
    output_path: str,
    *,
    preserve_audio: bool,
) -> bool:
    """Write final output in a container matching output_path extension."""
    if os.path.abspath(video_only_path) == os.path.abspath(output_path):
        return True

    mux_tmp = _mux_temp_path(output_path)
    _remove_file(mux_tmp)
    try:
        if preserve_audio:
            ok = remux_preserve_audio(source_path, video_only_path, mux_tmp)
        else:
            ok = remux_video_only(video_only_path, mux_tmp)

        if not ok:
            logger.error("Failed to finalize output: %s", output_path)
            return False

        os.replace(mux_tmp, output_path)
        return True
    except OSError:
        logger.exception("Failed to move mux output to %s", output_path)
        return False
    finally:
        _remove_file(mux_tmp)


def _remove_file(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        logger.exception("Failed to remove file: %s", path)
