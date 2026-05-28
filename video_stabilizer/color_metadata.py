# 色メタデータの読み取りとフォールバック

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import av


@dataclass(frozen=True)
class ColorMetadata:
    """Per-video color metadata used for YUV<->RGB and output mux."""

    color_range: str = "mpeg"  # mpeg=limited/tv, jpeg=full
    color_primaries: str = "bt709"
    colorspace: str = "bt709"
    color_trc: str = "bt709"
    is_full_range: bool = False

    @property
    def y_offset(self) -> float:
        return 0.0 if self.is_full_range else 16.0

    @property
    def y_scale(self) -> float:
        return 255.0 / 255.0 if self.is_full_range else 255.0 / 219.0

    @property
    def uv_offset(self) -> float:
        return 0.0 if self.is_full_range else 128.0

    @property
    def uv_scale(self) -> float:
        return 255.0 / 255.0 if self.is_full_range else 255.0 / 224.0

    @property
    def clip_min(self) -> int:
        return 0 if self.is_full_range else 16

    @property
    def clip_max(self) -> int:
        return 255 if self.is_full_range else 235


def _enum_name(value: object | None, default: str) -> str:
    if value is None:
        return default
    name = getattr(value, "name", None)
    if name:
        return str(name).lower()
    return str(value).lower()


def _infer_primaries_from_height(height: int) -> str:
    return "bt601" if height < 720 else "bt709"


def read_color_metadata(stream: av.stream.Stream) -> ColorMetadata:
    """Read color metadata from a PyAV video stream with sensible fallbacks."""
    height = getattr(stream, "height", None) or 0
    ctx = getattr(stream, "codec_context", None)

    if ctx is None:
        return ColorMetadata(
            color_range="mpeg",
            color_primaries=_infer_primaries_from_height(height),
            colorspace=_infer_primaries_from_height(height),
            color_trc=_infer_primaries_from_height(height),
            is_full_range=False,
        )

    range_name = _enum_name(getattr(ctx, "color_range", None), "mpeg")
    is_full = range_name in ("jpeg", "pc", "full")

    primaries = _enum_name(getattr(ctx, "color_primaries", None), "")
    if not primaries or primaries == "unspecified":
        primaries = _infer_primaries_from_height(height)

    colorspace = _enum_name(getattr(ctx, "colorspace", None), "")
    if not colorspace or colorspace == "unspecified":
        colorspace = primaries if primaries in ("bt601", "bt709", "bt2020") else "bt709"

    trc = _enum_name(getattr(ctx, "color_trc", None), "")
    if not trc or trc == "unspecified":
        trc = primaries if primaries in ("bt601", "bt709", "bt2020") else "bt709"

    return ColorMetadata(
        color_range="jpeg" if is_full else "mpeg",
        color_primaries=primaries,
        colorspace=colorspace,
        color_trc=trc,
        is_full_range=is_full,
    )


def apply_metadata_to_stream(
    out_stream: av.stream.Stream,
    metadata: ColorMetadata,
) -> None:
    """Best-effort copy of color metadata onto an output stream."""
    ctx = out_stream.codec_context
    try:
        import av

        range_map = {
            "mpeg": av.video.reformatter.ColorRange.MPEG,
            "jpeg": av.video.reformatter.ColorRange.JPEG,
        }
        prim_map = {
            "bt601": av.video.reformatter.ColorPrimaries.BT601,
            "bt709": av.video.reformatter.ColorPrimaries.BT709,
            "bt2020": av.video.reformatter.ColorPrimaries.BT2020,
        }
        space_map = {
            "bt601": av.video.reformatter.Colorspace.BT601,
            "bt709": av.video.reformatter.Colorspace.BT709,
            "bt2020": av.video.reformatter.Colorspace.BT2020,
        }
        trc_map = {
            "bt601": av.video.reformatter.ColorTransferCharacteristic.BT601,
            "bt709": av.video.reformatter.ColorTransferCharacteristic.BT709,
            "bt2020": av.video.reformatter.ColorTransferCharacteristic.BT2020,
            "smpte2084": av.video.reformatter.ColorTransferCharacteristic.SMPTE2084,
        }
        if metadata.color_range in range_map:
            ctx.color_range = range_map[metadata.color_range]
        if metadata.color_primaries in prim_map:
            ctx.color_primaries = prim_map[metadata.color_primaries]
        if metadata.colorspace in space_map:
            ctx.colorspace = space_map[metadata.colorspace]
        if metadata.color_trc in trc_map:
            ctx.color_trc = trc_map[metadata.color_trc]
    except Exception:
        pass
