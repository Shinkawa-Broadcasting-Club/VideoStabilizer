import cv2
import numpy as np
import numexpr as ne
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import concurrent.futures
import os
import math
import av

cv2.setNumThreads(0)
ne.set_num_threads(1)

# === LUT定義や各種関数は以前のまま配置 ===
_LUT_EXP = np.zeros(1024, dtype=np.float32)
for i in range(-255, 769):
    idx = i + 255
    x = float(i)
    if x > 226.0: val = 226.0 + 9.0 * (1.0 - math.exp(-(x - 226.0) / 9.0))
    elif x < 26.0: val = 26.0 - 10.0 * (1.0 - math.exp((x - 26.0) / 10.0))
    else: val = x
    _LUT_EXP[idx] = val

def apply_exponential_compression_fast(img_float):
    idx = np.clip(np.round(img_float) + 255, 0, 1023).astype(np.int32)
    return _LUT_EXP[idx]

def select_file(title):
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    root.destroy(); return file_path

def select_folder(title):
    root = tk.Tk(); root.withdraw()
    folder_path = filedialog.askdirectory(title=title)
    root.destroy(); return folder_path

def get_optimal_sampling_interval(fps, target_sec=0.5):
    if fps <= 0 or np.isnan(fps): fps = 30.0 
    return max(2, round(fps * target_sec))

def extract_yuv_planes(frame):
    fmt = str(frame.format.name)
    if not fmt.startswith('yuv420'): frame = frame.reformat(format='yuv420p')
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(frame.planes[0].height, frame.planes[0].line_size)[:, :frame.planes[0].width].copy()
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(frame.planes[1].height, frame.planes[1].line_size)[:, :frame.planes[1].width].copy()
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(frame.planes[2].height, frame.planes[2].line_size)[:, :frame.planes[2].width].copy()
    return y, u, v

def worker_yuv_to_float_bgr(y, u, v):
    h, w = y.shape
    u_resized = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
    v_resized = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
    y_f = y.astype(np.float32); u_f = u_resized.astype(np.float32); v_f = v_resized.astype(np.float32)
    c16, c128 = np.float32(16.0), np.float32(128.0)
    c_y, c_uv = np.float32(255.0 / 219.0), np.float32(255.0 / 224.0)
    y_f = ne.evaluate("(y_f - c16) * c_y")
    u_f = ne.evaluate("(u_f - c128) * c_uv")
    v_f = ne.evaluate("(v_f - c128) * c_uv")
    c_rv, c_gu, c_gv, c_bu = np.float32(1.5748), np.float32(0.1873), np.float32(0.4681), np.float32(1.8556)
    r = ne.evaluate("y_f + c_rv * v_f")
    g = ne.evaluate("y_f - c_gu * u_f - c_gv * v_f")
    b = ne.evaluate("y_f + c_bu * u_f")
    return np.dstack((b, g, r))

def get_stats_and_coeffs(y, u, v, ref_avg_stats=None):
    max_dim = max(y.shape)
    scale = min(256.0 / max_dim, 1.0) 
    new_w = max(int(y.shape[1] * scale), 1)
    new_h = max(int(y.shape[0] * scale), 1)
    small_y = cv2.resize(y, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_u = cv2.resize(u, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_v = cv2.resize(v, (new_w, new_h), interpolation=cv2.INTER_AREA)
    bgr_small = worker_yuv_to_float_bgr(small_y, small_u, small_v)
    bgr_uint8 = np.clip(bgr_small, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2HSV)
    y_mean = np.mean(small_y); y_std = np.std(small_y) + 1e-6; s_mean = np.mean(hsv[:,:,1]) + 1e-6
    return np.array([y_mean, y_std, s_mean], dtype=np.float32)

_MN_C1 = np.float32(-7/18); _MN_C2 = np.float32(5/6); _MN_C3 = np.float32(-1/2); _MN_C4 = np.float32(1/18)
_MN_C5 = np.float32(7/6); _MN_C6 = np.float32(-2); _MN_C7 = np.float32(8/9); _MN_C8 = np.float32(-7/6)
_MN_C9 = np.float32(3/2); _MN_C10 = np.float32(7/18); _MN_C11 = np.float32(-1/3)

def mitchell_netravali(t, p):
    t = np.float32(t)
    expr = (
        "p0 * (t * (t * (_MN_C1 * t + _MN_C2) + _MN_C3) + _MN_C4) + "
        "p1 * ((_MN_C5 * t + _MN_C6) * t**2 + _MN_C7) + "
        "p2 * (t * (t * (_MN_C8 * t + _MN_C9) + _MN_C3) + _MN_C4) + "
        "p3 * ((_MN_C10 * t + _MN_C11) * t**2)"
    )
    local_dict = {
        "t": t, "p0": p[0], "p1": p[1], "p2": p[2], "p3": p[3],
        "_MN_C1": _MN_C1, "_MN_C2": _MN_C2, "_MN_C3": _MN_C3, "_MN_C4": _MN_C4,
        "_MN_C5": _MN_C5, "_MN_C6": _MN_C6, "_MN_C7": _MN_C7, "_MN_C8": _MN_C8,
        "_MN_C9": _MN_C9, "_MN_C10": _MN_C10, "_MN_C11": _MN_C11
    }
    return ne.evaluate(expr, local_dict=local_dict)

def process_frame_worker(y, u, v, current_stats, target_stats):
    try:
        if y is None: return None
        c_mean = np.float32(current_stats[0]); t_mean = np.float32(target_stats[0])
        contrast_ratio = np.float32(np.clip(target_stats[1] / current_stats[1], 0.5, 2.0))
        sat_ratio = np.float32(np.clip(target_stats[2] / current_stats[2], 0.5, 2.0))
        h, w = y.shape; h_uv, w_uv = u.shape 
        c128, c16, c1, c0 = np.float32(128.0), np.float32(16.0), np.float32(1.0), np.float32(0.0)
        y_f = y.astype(np.float32)
        y_basic = ne.evaluate("(y_f - c_mean) * contrast_ratio + t_mean")
        y_comp = apply_exponential_compression_fast(y_basic) 
        y_old_norm = ne.evaluate("y_f - c16")
        y_old_norm = ne.evaluate("where(y_old_norm < c1, c1, y_old_norm)") 
        y_new_norm = ne.evaluate("y_comp - c16")
        y_new_norm = ne.evaluate("where(y_new_norm < c0, c0, y_new_norm)")
        y_ratio = ne.evaluate("y_new_norm / y_old_norm")
        y_ratio = np.clip(y_ratio, 0.5, 2.0)
        y_ratio_down = cv2.resize(y_ratio, (w_uv, h_uv), interpolation=cv2.INTER_AREA)
        u_f = u.astype(np.float32); v_f = v.astype(np.float32)
        uv_scale = ne.evaluate("sat_ratio * y_ratio_down")
        u_c = ne.evaluate("(u_f - c128) * uv_scale")
        v_c = ne.evaluate("(v_f - c128) * uv_scale")
        c_max = ne.evaluate("where(abs(u_c) > abs(v_c), abs(u_c), abs(v_c))")
        c_max_safe = ne.evaluate("where(c_max == 0, c1, c_max)") 
        c_thresh = np.float32(80.0); c_diff = np.float32(32.0) 
        c_new = ne.evaluate("where(c_max > c_thresh, c_thresh + c_diff * (c1 - exp(-(c_max - c_thresh) / c_diff)), c_max)")
        chroma_scale = ne.evaluate("c_new / c_max_safe")
        u_comp = ne.evaluate("u_c * chroma_scale + c128"); v_comp = ne.evaluate("v_c * chroma_scale + c128")
        u_up = cv2.resize(u_comp, (w, h), interpolation=cv2.INTER_LINEAR)
        v_up = cv2.resize(v_comp, (w, h), interpolation=cv2.INTER_LINEAR)
        c_y = np.float32(255.0 / 219.0); c_uv = np.float32(255.0 / 224.0)
        y_norm = ne.evaluate("(y_comp - c16) * c_y")
        u_norm = ne.evaluate("(u_up - c128) * c_uv")
        v_norm = ne.evaluate("(v_up - c128) * c_uv")
        c_rv, c_gu, c_gv, c_bu = np.float32(1.5748), np.float32(0.1873), np.float32(0.4681), np.float32(1.8556)
        r = ne.evaluate("y_norm + c_rv * v_norm")
        g = ne.evaluate("y_norm - c_gu * u_norm - c_gv * v_norm")
        b = ne.evaluate("y_norm + c_bu * u_norm")
        bgr = np.dstack((b, g, r))
        return np.clip(bgr, 16, 235).astype(np.uint8)
    except Exception as e:
        print(f"Worker Error: {e}")
        return None

def init_ema_with_warmup(warmup_buffer, ref_avg_stats, ema_alpha):
    ema_stats = None
    for y, u, v in warmup_buffer:
        s = get_stats_and_coeffs(y, u, v, ref_avg_stats)
        if ema_stats is None: ema_stats = s
        else: ema_stats = ema_alpha * s + (1.0 - ema_alpha) * ema_stats
    return ema_stats


# ==========================================
# 【修正】メインロジックを関数化してスコープを保護
# ==========================================
def main():
    ref_path = select_file("参照映像(Lookの元)を選択してください")
    if not ref_path: return

    ref_stats_list = []
    print("\n--- 参照映像を解析中 ---")
    
    ref_container = av.open(ref_path)
    ref_stream = ref_container.streams.video[0]
    total_ref = ref_stream.frames if ref_stream.frames > 0 else int(float(ref_stream.duration * ref_stream.time_base) * ref_stream.average_rate)
    
    pbar_ref = tqdm(total=total_ref if total_ref > 0 else None, desc="Ref Analysis")
    for idx, frame in enumerate(ref_container.decode(ref_stream)):
        if idx % 2 == 0:
            y, u, v = extract_yuv_planes(frame)
            s = get_stats_and_coeffs(y, u, v) 
            if s is not None: ref_stats_list.append(s)
        pbar_ref.update(1)
    pbar_ref.close()
    ref_container.close()
    
    if not ref_stats_list:
        print("エラー: 参照映像から統計情報を取得できませんでした。")
        return
    ref_avg_stats = np.mean(ref_stats_list, axis=0)
    ref_avg_stats[1] = max(ref_avg_stats[1], 1e-6)
    ref_avg_stats[2] = max(ref_avg_stats[2], 1e-6)

    target_folder = select_folder("適用先の映像が入っているフォルダを選択してください")
    if not target_folder: return

    valid_exts = ('.mp4', '.avi', '.mov', '.mkv')
    target_files = [f for f in os.listdir(target_folder) if f.lower().endswith(valid_exts)]
    if not target_files: return

    out_dir = os.path.join(target_folder, "output_corrected")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n合計 {len(target_files)} 個の動画ファイルを処理します。")

    for file_name in target_files:
        app_path = os.path.join(target_folder, file_name)
        out_path = os.path.join(out_dir, f"corrected_{file_name}")
        
        container = av.open(app_path)
        stream = container.streams.video[0]
        total_app_frames = stream.frames if stream.frames > 0 else int(float(stream.duration * stream.time_base) * stream.average_rate)
        if total_app_frames <= 0: total_app_frames = 1000 
        
        fps = float(stream.average_rate)
        if fps <= 0 or math.isnan(fps): fps = 30.0
        
        w, h = stream.width, stream.height
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        N = get_optimal_sampling_interval(fps, target_sec=0.5)
        is_short_video = total_app_frames <= (N * 3)
        
        print(f"\n処理開始: {file_name} ({w}x{h} @ {fps:.2f}fps)")
        if is_short_video: print(f"※ ショート動画({total_app_frames}F)適用: 全フレームダイレクトEMA処理モード")
        else: print(f"最適サンプリング間隔: {N}フレームごと (Mitchell補間モード)")

        stats_buffer = []
        frames_buffer = []      
        warmup_buffer = []      
        ema_stats = None
        ema_alpha = 0.2
        
        max_workers = os.cpu_count() or 4
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        futures = {}
        
        # main() 関数のローカル変数として定義されるため、nonlocal が機能する
        next_frame_to_write = 0
        max_queue_size = max_workers * 5 

        pbar = tqdm(total=total_app_frames if total_app_frames > 1000 else None, desc=file_name[:20])
        f_idx = 0
        
        def write_completed_frames():
            nonlocal next_frame_to_write
            while next_frame_to_write in futures and futures[next_frame_to_write].done():
                res_frame = futures[next_frame_to_write].result()
                if res_frame is not None:
                    out.write(res_frame)
                del futures[next_frame_to_write]
                next_frame_to_write += 1
                pbar.update(1)
        
        for frame in container.decode(stream):
            target_yuv = extract_yuv_planes(frame)
            
            if is_short_video:
                s = get_stats_and_coeffs(target_yuv[0], target_yuv[1], target_yuv[2], ref_avg_stats)
                if ema_stats is None: ema_stats = s
                else: ema_stats = ema_alpha * s + (1 - ema_alpha) * ema_stats
                
                future = executor.submit(process_frame_worker, target_yuv[0], target_yuv[1], target_yuv[2], ema_stats, ref_avg_stats)
                futures[f_idx] = future
            else:
                if f_idx % N == 0:
                    s = get_stats_and_coeffs(target_yuv[0], target_yuv[1], target_yuv[2], ref_avg_stats)
                    if s is not None: stats_buffer.append((f_idx, s))
                
                frames_buffer.append((f_idx, target_yuv))
                
                while len(frames_buffer) > N * 2.5:
                    buf_f_idx, buf_target = frames_buffer[0]
                    interp_s = None
                    target_j = -1
                    
                    for j in range(len(stats_buffer) - 1):
                        if stats_buffer[j][0] <= buf_f_idx < stats_buffer[j+1][0]:
                            target_j = j; break
                            
                    if target_j != -1 and target_j + 2 < len(stats_buffer):
                        p1 = stats_buffer[target_j][1]
                        p2 = stats_buffer[target_j+1][1]
                        p0 = stats_buffer[target_j-1][1] if target_j >= 1 else p1
                        p3 = stats_buffer[target_j+2][1]
                        
                        t = (buf_f_idx - stats_buffer[target_j][0]) / float(N)
                        interp_s = mitchell_netravali(t, [p0, p1, p2, p3])
                        
                        frames_buffer.pop(0)
                        warmup_buffer.append(buf_target)
                        if len(warmup_buffer) > N: warmup_buffer.pop(0)
                    else:
                        break
                    
                    if interp_s is not None:
                        future = executor.submit(process_frame_worker, buf_target[0], buf_target[1], buf_target[2], interp_s, ref_avg_stats)
                        futures[buf_f_idx] = future
            
            write_completed_frames()
            
            while len(futures) > max_queue_size:
                if next_frame_to_write in futures:
                    res_frame = futures[next_frame_to_write].result(timeout=10)
                    if res_frame is not None: out.write(res_frame)
                    del futures[next_frame_to_write]
                    next_frame_to_write += 1
                    pbar.update(1)
                else:
                    break

            f_idx += 1

        if not is_short_video:
            while frames_buffer:
                buf_f_idx, buf_target = frames_buffer.pop(0)
                interp_s = None
                
                if len(stats_buffer) >= 2:
                    last_j = len(stats_buffer) - 2
                    p1 = stats_buffer[last_j][1]
                    p2 = stats_buffer[last_j+1][1]
                    p0 = stats_buffer[last_j-1][1] if last_j >= 1 else p1
                    p3_extrapolated = p2 + (p2 - p1)
                    t = (buf_f_idx - stats_buffer[last_j][0]) / float(N)
                    interp_s = mitchell_netravali(t, [p0, p1, p2, p3_extrapolated])
                else:
                    if ema_stats is None:
                        ema_stats = init_ema_with_warmup(warmup_buffer, ref_avg_stats, ema_alpha)
                        if ema_stats is None:
                            ema_stats = get_stats_and_coeffs(buf_target[0], buf_target[1], buf_target[2], ref_avg_stats)
                    s = get_stats_and_coeffs(buf_target[0], buf_target[1], buf_target[2], ref_avg_stats)
                    ema_stats = ema_alpha * s + (1 - ema_alpha) * ema_stats
                    interp_s = ema_stats
                
                future = executor.submit(process_frame_worker, buf_target[0], buf_target[1], buf_target[2], interp_s, ref_avg_stats)
                futures[buf_f_idx] = future

        while len(futures) > 0:
            if next_frame_to_write in futures:
                res_frame = futures[next_frame_to_write].result()
                if res_frame is not None:
                    out.write(res_frame)
                del futures[next_frame_to_write]
                next_frame_to_write += 1
                pbar.update(1)
            else:
                next_frame_to_write += 1

        pbar.close()
        executor.shutdown(wait=True)
        out.release()
        container.close()

    print("\n--- すべての処理が完了しました ---")

# スクリプトとして直接実行された時のみ main() を呼び出す
if __name__ == '__main__':
    main()