# 参照/ターゲット映像の 7D 色統計と、フレーム間補間（Mitchell-Netravali）
#
# 7D = [Y_mean, Y_std, U_mean, U_std, V_mean, V_std, S_mean]
# サンプリング間隔 N フレームごとに統計を取り、間のフレームは Mitchell 補間で
# 「その瞬間のルック」を推定する。短尺動画は EMA で直接追従する（pipeline 側）。

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numexpr as ne
import numpy as np

from video_stabilizer.color import _numexpr_lock, worker_yuv_to_float_bgr
from video_stabilizer.stats_constants import (
    IDX_S_MEAN,
    IDX_U_MEAN,
    IDX_U_STD,
    IDX_V_MEAN,
    IDX_V_STD,
    IDX_Y_MEAN,
    IDX_Y_STD,
    STAT_DIM,
    STD_INDICES,
)

if TYPE_CHECKING:
    import av as _av

logger = logging.getLogger(__name__)

_MN_C1 = np.float32(-7 / 18)
_MN_C2 = np.float32(5 / 6)
_MN_C3 = np.float32(-1 / 2)
_MN_C4 = np.float32(1 / 18)
_MN_C5 = np.float32(7 / 6)
_MN_C6 = np.float32(-2)
_MN_C7 = np.float32(8 / 9)
_MN_C8 = np.float32(-7 / 6)
_MN_C9 = np.float32(3 / 2)
_MN_C10 = np.float32(7 / 18)
_MN_C11 = np.float32(-1 / 3)


def get_optimal_sampling_interval(fps: float, target_sec: float) -> int:
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    return max(2, round(fps * target_sec))


def extract_yuv_planes(frame: _av.VideoFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        fmt = str(frame.format.name)
        if not fmt.startswith("yuv420"):
            frame = frame.reformat(format="yuv420p")
        y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(
            frame.planes[0].height, frame.planes[0].line_size
        )[:, : frame.planes[0].width].copy()
        u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(
            frame.planes[1].height, frame.planes[1].line_size
        )[:, : frame.planes[1].width].copy()
        v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(
            frame.planes[2].height, frame.planes[2].line_size
        )[:, : frame.planes[2].width].copy()
        return y, u, v
    except Exception:
        logger.exception("extract_yuv_planes failed; skipping frame")
        return None


def get_stats_and_coeffs(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    stats_max_dim: int,
    *,
    y_offset: float = 16.0,
    y_scale: float = 255.0 / 219.0,
    uv_offset: float = 128.0,
    uv_scale: float = 255.0 / 224.0,
) -> np.ndarray:
    """Return 7D stats: [Y_mean, Y_std, U_mean, U_std, V_mean, V_std, S_mean]."""
    # 長辺を stats_max_dim に縮小してから平均/標準偏差を取る（全画素走査を避ける）。
    # Y/U/V は limited/full range のオフセット・スケールを反映した上で集計し、
    # S_mean は HSV の S チャンネル平均（彩度の代表値）。
    max_dim = max(y.shape)
    scale = min(float(stats_max_dim) / max_dim, 1.0)
    new_w = max(int(y.shape[1] * scale), 1)
    new_h = max(int(y.shape[0] * scale), 1)
    small_y = cv2.resize(y, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_u = cv2.resize(u, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_v = cv2.resize(v, (new_w, new_h), interpolation=cv2.INTER_AREA)
    bgr_small = worker_yuv_to_float_bgr(
        small_y,
        small_u,
        small_v,
        y_offset=y_offset,
        y_scale=y_scale,
        uv_offset=uv_offset,
        uv_scale=uv_scale,
    )
    bgr_uint8 = np.clip(bgr_small, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2HSV)
    y_mean = float(np.mean(small_y))
    y_std = float(np.std(small_y)) + 1e-6
    u_mean = float(np.mean(small_u))
    u_std = float(np.std(small_u)) + 1e-6
    v_mean = float(np.mean(small_v))
    v_std = float(np.std(small_v)) + 1e-6
    s_mean = float(np.mean(hsv[:, :, 1])) + 1e-6
    return np.array(
        [y_mean, y_std, u_mean, u_std, v_mean, v_std, s_mean],
        dtype=np.float32,
    )


def guard_std_stats(stats: np.ndarray, min_std: float) -> np.ndarray:
    out = stats.copy()
    for idx in STD_INDICES:
        out[idx] = max(float(out[idx]), min_std)
    return out


def robust_aggregate_stats(
    stats_list: list[np.ndarray],
    *,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    min_std: float = 1e-6,
) -> np.ndarray | None:
    """Trim-mean aggregate over per-frame stats (IQR-style outlier rejection)."""
    if not stats_list:
        return None
    arr = np.stack(stats_list, axis=0)
    # 各次元ごとにパーセンタイル境界を取り、全次元が範囲内のフレームだけ残す。
    # シーン切替やフラッシュフレームなど、参照全体の代表値から外れる統計を除外する。
    lo = np.percentile(arr, percentile_low, axis=0)
    hi = np.percentile(arr, percentile_high, axis=0)
    mask = np.all((arr >= lo) & (arr <= hi), axis=1)
    filtered = arr[mask] if np.any(mask) else arr
    mean = np.mean(filtered, axis=0).astype(np.float32)
    return guard_std_stats(mean, min_std)


def clamp_mitchell_t(t: float, t_max: float = 1.0) -> float:
    return float(min(max(t, 0.0), t_max))


def find_stats_segment_index(
    stats_buffer: list[tuple[int, np.ndarray]],
    frame_idx: int,
) -> int:
    for j in range(len(stats_buffer) - 2, -1, -1):
        if stats_buffer[j][0] <= frame_idx <= stats_buffer[j + 1][0]:
            return j
    return -1


def interpolate_stats_for_frame(
    frame_idx: int,
    stats_buffer: list[tuple[int, np.ndarray]],
    sampling_n: int,
    t_max: float = 1.0,
) -> np.ndarray | None:
    # stats_buffer は (frame_idx, 7D stats) のキーフレーム列。
    # frame_idx が属する区間 [p1, p2] を見つけ、Mitchell 4点補間で滑らかな統計を返す。
    if len(stats_buffer) < 2:
        return None

    target_j = find_stats_segment_index(stats_buffer, frame_idx)
    if target_j == -1:
        return None

    p1 = stats_buffer[target_j][1]
    p2 = stats_buffer[target_j + 1][1]
    span = stats_buffer[target_j + 1][0] - stats_buffer[target_j][0]
    if span > 0:
        t = (frame_idx - stats_buffer[target_j][0]) / float(span)
    else:
        t = 0.0
    # t_max で補間係数をクランプし、キーフレーム間の急激な統計ジャンプを抑える。
    t = clamp_mitchell_t(t, t_max)

    linear = (1.0 - t) * p1 + t * p2

    if target_j + 2 < len(stats_buffer):
        p0 = stats_buffer[target_j - 1][1] if target_j >= 1 else p1
        p3 = stats_buffer[target_j + 2][1]
        mitchell = mitchell_netravali(t, [p0, p1, p2, p3])
        # Mitchell はオーバーシュートしうるため、4点の min/max 箱からはみ出したら線形にフォールバック。
        if not np.all(np.isfinite(mitchell)):
            return linear
        lo = np.minimum.reduce([p0, p1, p2, p3])
        hi = np.maximum.reduce([p0, p1, p2, p3])
        if np.any(mitchell < lo) or np.any(mitchell > hi):
            return linear
        return mitchell

    return linear


def interpolate_stats_tail(
    frame_idx: int,
    stats_buffer: list[tuple[int, np.ndarray]],
    sampling_n: int,
    t_max: float = 1.0,
) -> np.ndarray | None:
    # 末尾バッファ用: 最後のキーフレーム以降にフレームが残る場合、
    # p3 を p1→p2 の勾配で外挿して Mitchell 補間を継続する。
    if len(stats_buffer) < 2:
        return None

    last_j = len(stats_buffer) - 2
    p1 = stats_buffer[last_j][1]
    p2 = stats_buffer[last_j + 1][1]
    p0 = stats_buffer[last_j - 1][1] if last_j >= 1 else p1
    p3_extrapolated = p2 + (p2 - p1)

    span = stats_buffer[last_j + 1][0] - stats_buffer[last_j][0]
    if span > 0:
        t = (frame_idx - stats_buffer[last_j][0]) / float(span)
    else:
        t = 0.0
    t = clamp_mitchell_t(t, t_max)

    linear = (1.0 - t) * p1 + t * p2
    mitchell = mitchell_netravali(t, [p0, p1, p2, p3_extrapolated])
    if not np.all(np.isfinite(mitchell)):
        return linear
    lo = np.minimum.reduce([p0, p1, p2, p3_extrapolated])
    hi = np.maximum.reduce([p0, p1, p2, p3_extrapolated])
    if np.any(mitchell < lo) or np.any(mitchell > hi):
        return linear
    return mitchell


def mitchell_netravali(t: float, p: list[np.ndarray]) -> np.ndarray:
    # B=1/3, C=1/3 の Mitchell-Netravali カーネルを 7D ベクトルごとに評価。
    # 係数は定数化して numexpr で一括計算（Python ループより高速）。
    t = np.float32(t)
    expr = (
        "p0 * (t * (t * (_MN_C1 * t + _MN_C2) + _MN_C3) + _MN_C4) + "
        "p1 * ((_MN_C5 * t + _MN_C6) * t**2 + _MN_C7) + "
        "p2 * (t * (t * (_MN_C8 * t + _MN_C9) + _MN_C3) + _MN_C4) + "
        "p3 * ((_MN_C10 * t + _MN_C11) * t**2)"
    )
    local_dict = {
        "t": t,
        "p0": p[0],
        "p1": p[1],
        "p2": p[2],
        "p3": p[3],
        "_MN_C1": _MN_C1,
        "_MN_C2": _MN_C2,
        "_MN_C3": _MN_C3,
        "_MN_C4": _MN_C4,
        "_MN_C5": _MN_C5,
        "_MN_C6": _MN_C6,
        "_MN_C7": _MN_C7,
        "_MN_C8": _MN_C8,
        "_MN_C9": _MN_C9,
        "_MN_C10": _MN_C10,
        "_MN_C11": _MN_C11,
    }
    with _numexpr_lock:
        return ne.evaluate(expr, local_dict=local_dict)


def init_ema_with_warmup(
    warmup_buffer: list,
    ema_alpha: float,
    stats_max_dim: int,
    *,
    y_offset: float = 16.0,
    y_scale: float = 255.0 / 219.0,
    uv_offset: float = 128.0,
    uv_scale: float = 255.0 / 224.0,
) -> np.ndarray | None:
    ema_stats = None
    for y, u, v in warmup_buffer:
        s = get_stats_and_coeffs(
            y,
            u,
            v,
            stats_max_dim,
            y_offset=y_offset,
            y_scale=y_scale,
            uv_offset=uv_offset,
            uv_scale=uv_scale,
        )
        if ema_stats is None:
            ema_stats = s
        else:
            ema_stats = ema_alpha * s + (1.0 - ema_alpha) * ema_stats
    return ema_stats
