import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import os
import math
import av
import subprocess
import queue
import threading
from collections import deque
import numba
from numba import njit, prange
import tempfile
import shutil

# OpenCVの内部スレッドを停止し、NumbaのSIMDエンジンに一本化する
cv2.setNumThreads(0)
# Numba用のスレッド数確保
numba.set_num_threads(max(1, os.cpu_count() - 2))

# --- Numba JIT & SIMD カーネル群 ---

@njit(parallel=True, fastmath=True, cache=True)
def yuv_to_bgr_numba(y_plane, u_plane, v_plane, bgr_out):
    H, W = y_plane.shape
    for h in prange(H):
        h_uv = h >> 1
        for w in range(W):
            w_uv = w >> 1
            y = y_plane[h, w]
            u = u_plane[h_uv, w_uv]
            v = v_plane[h_uv, w_uv]
            
            y_f = (y - 16.0) * 1.1643835616438356
            u_f = (u - 128.0) * 1.1383928571428572
            v_f = (v - 128.0) * 1.1383928571428572
            
            b = y_f + 1.8556 * u_f
            g = y_f - 0.1873 * u_f - 0.4681 * v_f
            r = y_f + 1.5748 * v_f
            
            bgr_out[h, w, 0] = b
            bgr_out[h, w, 1] = g
            bgr_out[h, w, 2] = r

@njit(parallel=True, fastmath=True, cache=True)
def apply_color_correction_numba(bgr_in, bgr_out, b_gains, g_gains, r_gains, stretch_ratios, target_means, current_means, sat_ratios):
    """
    【次元フラット化】
    S(フレーム数)とH(高さ)のループを1つに統合(S * H)することで、フレーム境界での
    スレッド同期待ちを完全に破壊。全コアが息継ぎなしで全フレームを一気に計算します。
    """
    S, H, W, _ = bgr_in.shape
    for i in prange(S * H):
        s = i // H
        h = i % H
        
        lin_bg, lin_gg, lin_rg = b_gains[s], g_gains[s], r_gains[s]
        stretch = stretch_ratios[s]
        t_mean = target_means[s]
        c_mean = current_means[s]
        sr = sat_ratios[s]
        
        for w in range(W):
            b = bgr_in[s, h, w, 0]
            g = bgr_in[s, h, w, 1]
            r = bgr_in[s, h, w, 2]
            
            # 0.0〜1.0正規化
            bn = max(0.0, (b - 16.0) / 219.0)
            gn = max(0.0, (g - 16.0) / 219.0)
            rn = max(0.0, (r - 16.0) / 219.0)
            
            # ホワイトバランスの適用
            bn *= lin_bg
            gn *= lin_gg
            rn *= lin_rg
            
            # 【Z-scoreマッピング】
            # ラプラシアン加重平均を基準にして引き算し、加重標準偏差の比率で引き伸ばし、ターゲットの加重平均を足す
            bn = (bn - c_mean) * stretch + t_mean
            gn = (gn - c_mean) * stretch + t_mean
            rn = (rn - c_mean) * stretch + t_mean
            
            # スケール復元 (16〜235)
            bc = bn * 219.0 + 16.0
            gc = gn * 219.0 + 16.0
            rc = rn * 219.0 + 16.0

            # --- ソフトクリップ (極端な引き伸ばしによる0未満や255超えを滑らかに丸め込む) ---
            sm_b = 1.0 if bc < 29.0 else 0.0
            hm_b = 1.0 if bc > 226.0 else 0.0
            bm = sm_b * (29.0 - 13.0 * (1.0 - math.exp((bc - 29.0) / 13.0))) + \
                 hm_b * (226.0 + 9.0 * (1.0 - math.exp(-(bc - 226.0) / 9.0))) + \
                 (1.0 - sm_b - hm_b) * bc
                 
            sm_g = 1.0 if gc < 29.0 else 0.0
            hm_g = 1.0 if gc > 226.0 else 0.0
            gm = sm_g * (29.0 - 13.0 * (1.0 - math.exp((gc - 29.0) / 13.0))) + \
                 hm_g * (226.0 + 9.0 * (1.0 - math.exp(-(gc - 226.0) / 9.0))) + \
                 (1.0 - sm_g - hm_g) * gc

            sm_r = 1.0 if rc < 29.0 else 0.0
            hm_r = 1.0 if rc > 226.0 else 0.0
            rm = sm_r * (29.0 - 13.0 * (1.0 - math.exp((rc - 29.0) / 13.0))) + \
                 hm_r * (226.0 + 9.0 * (1.0 - math.exp(-(rc - 226.0) / 9.0))) + \
                 (1.0 - sm_r - hm_r) * rc
                 
            # --- 彩度調整 ---
            luma = 0.114 * bm + 0.587 * gm + 0.299 * rm
            bf = luma + (bm - luma) * sr
            gf = luma + (gm - luma) * sr
            rf = luma + (rm - luma) * sr
            
            bgr_out[s, h, w, 0] = max(16.0, min(235.0, bf))
            bgr_out[s, h, w, 1] = max(16.0, min(235.0, gf))
            bgr_out[s, h, w, 2] = max(16.0, min(235.0, rf))

def warmup_numba_jit():
    print("\n--- LLVMコンパイラをウォームアップ中 (JIT最適化) ---")
    y = np.zeros((32, 32), dtype=np.uint8)
    u = np.zeros((16, 16), dtype=np.uint8)
    v = np.zeros((16, 16), dtype=np.uint8)
    out_bgr = np.zeros((32, 32, 3), dtype=np.float32)
    yuv_to_bgr_numba(y, u, v, out_bgr)
    
    bgr_in = np.zeros((1, 32, 32, 3), dtype=np.float32)
    bgr_out = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    gains = np.ones(1, dtype=np.float32)
    apply_color_correction_numba(bgr_in, bgr_out, gains, gains, gains, gains, gains, gains, gains)
    print("最適化完了。")

# --- 帯域MAX I/Oモジュール ---

def fast_file_transfer(src, dst, desc_text, buffer_size=64*1024*1024):
    file_size = os.path.getsize(src)
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=desc_text) as pbar:
            while True:
                buf = fsrc.read(buffer_size)
                if not buf:
                    break
                fdst.write(buf)
                pbar.update(len(buf))

# --- ヘルパー群 ---

class TensorMemoryPool:
    def __init__(self, N, H, W):
        self.N = N
        self.bgr_batch = np.empty((N, H, W, 3), dtype=np.float32)
        self.out_batch = np.empty((N, H, W, 3), dtype=np.uint8)

def select_file(title):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    root.destroy()
    return file_path

def select_folder(title):
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path

def get_optimal_sampling_interval(fps, target_sec=0.5):
    if fps <= 0 or np.isnan(fps): fps = 30.0 
    return max(1, round(fps * target_sec))

def extract_yuv_planes(frame):
    fmt_name = str(frame.format.name)
    if fmt_name.startswith('yuv420'):
        y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(frame.planes[0].height, frame.planes[0].line_size)[:, :frame.planes[0].width].copy()
        u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(frame.planes[1].height, frame.planes[1].line_size)[:, :frame.planes[1].width].copy()
        v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(frame.planes[2].height, frame.planes[2].line_size)[:, :frame.planes[2].width].copy()
        return True, (y, u, v)
    else:
        return False, frame.to_ndarray(format='bgr24')

def yuv_to_float_bgr_fast(frame, out_bgr_float):
    fmt_name = str(frame.format.name)
    if not fmt_name.startswith('yuv420'):
        arr = frame.to_ndarray(format='bgr24')
        out_bgr_float[:] = arr.astype(np.float32)
        return out_bgr_float
    def extract_plane(plane):
        arr = np.frombuffer(plane, dtype=np.uint8)
        return arr.reshape(plane.height, plane.line_size)[:, :plane.width]
    y, u, v = extract_plane(frame.planes[0]), extract_plane(frame.planes[1]), extract_plane(frame.planes[2])
    yuv_to_bgr_numba(y, u, v, out_bgr_float)
    return out_bgr_float

def get_stats_fast(img_uint8, p_norm=6, sigma=1.2):
    if img_uint8 is None or img_uint8.size == 0: return None
    h, w = img_uint8.shape[:2]
    scale = 512 / max(h, w)
    small = cv2.resize(img_uint8, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else img_uint8
    
    img_float = small.astype(np.float32) / 255.0
    img_blurred = cv2.GaussianBlur(img_float, (0, 0), sigma) if sigma > 0 else img_float
    
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:,:,0]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # 【最重要】ラプラシアンフィルタで全方向のエッジを取得（2次微分）
    laplacian = cv2.Laplacian(img_blurred, cv2.CV_32F, ksize=3)
    grad2_mag = np.abs(laplacian)
    
    # ラプラシアンの絶対値をウェイトマップとして採用 (ゼロ割りを防ぐために微小な値1e-5を加算)
    weight_map = np.mean(grad2_mag, axis=2) + 1e-5
    weight_sum = np.sum(weight_map)
    
    # ラプラシアン加重平均 (のっぺりした面を無視し、輪郭やテクスチャを重視した平均明るさ)
    brightness = np.sum(l_channel * weight_map) / weight_sum
    
    # ラプラシアン加重標準偏差 (エッジ部分のディテールに基づいた真のコントラスト)
    contrast = np.sqrt(np.sum(weight_map * (l_channel - brightness)**2) / weight_sum)
    
    # 彩度と色相もエッジベースで加重平均
    sat = np.sum(hsv[:,:,1] * weight_map) / weight_sum
    hue = np.sum(hsv[:,:,0] * weight_map) / weight_sum
    
    gamma = np.median(l_channel) / 128.0
    
    # RGBイルミナント推定（Gray-Edgeアルゴリズムは維持）
    gx = cv2.Sobel(img_blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad1_mag = np.sqrt(gx**2 + gy**2)

    wp_est = np.power(np.mean(np.power(img_float, p_norm), axis=(0, 1)), 1.0/p_norm)
    ge1_est = np.power(np.mean(np.power(grad1_mag, p_norm), axis=(0, 1)), 1.0/p_norm)
    # ラプラシアン(grad2_mag)は上で算出済みなので再利用
    ge2_est = np.power(np.mean(np.power(grad2_mag, p_norm), axis=(0, 1)), 1.0/p_norm)
    
    wp_est = np.clip(wp_est, 1e-6, None)
    ge1_est = np.clip(ge1_est, 1e-6, None)
    ge2_est = np.clip(ge2_est, 1e-6, None)
    
    wp_weight = wp_est / np.sum(wp_est)
    ge1_weight = ge1_est / np.sum(ge1_est)
    ge2_weight = ge2_est / np.sum(ge2_est)
    
    illuminant = (wp_weight + ge1_weight + ge2_weight) / 3.0
    illuminant_scaled = illuminant * 255.0
    
    return np.array([brightness, contrast, hue, sat, gamma, illuminant_scaled[0], illuminant_scaled[1], illuminant_scaled[2]])

def mitchell_netravali(t, p):
    B, C = 1/3, 1/3
    def k(x):
        x = abs(x)
        if x < 1: return ((12 - 9*B - 6*C) * x**3 + (-18 + 12*B + 6*C) * x**2 + (6 - 2*B)) / 6
        elif x < 2: return ((-B - 6*C) * x**3 + (6*B + 30*C) * x**2 + (-12*B - 48*C) * x + (8*B + 24*C)) / 6
        return 0
    return p[0]*k(t+1) + p[1]*k(t) + p[2]*k(t-1) + p[3]*k(t-2)

def process_segment_batch_zero_alloc(seg_frames, current_stats_list, target_stats, pool):
    S = len(seg_frames)
    for i in range(S): pool.bgr_batch[i] = seg_frames[i]
    current_stats_array = np.array(current_stats_list, dtype=np.float32)
    target_stats_array = np.array(target_stats, dtype=np.float32)
    
    c_b, c_c, c_s, c_bg, c_gg, c_rg = current_stats_array[:, 0], current_stats_array[:, 1], current_stats_array[:, 3], current_stats_array[:, 5], current_stats_array[:, 6], current_stats_array[:, 7]
    
    # 光の色（ホワイトバランス）ゲイン
    raw_b = target_stats_array[5] / (c_bg + 1e-6)
    raw_g = target_stats_array[6] / (c_gg + 1e-6)
    raw_r = target_stats_array[7] / (c_rg + 1e-6)
    avg_gain = (raw_b + raw_g + raw_r) / 3.0
    linear_b_gains = np.clip(raw_b / avg_gain, 0.5, 2.0).astype(np.float32)
    linear_g_gains = np.clip(raw_g / avg_gain, 0.5, 2.0).astype(np.float32)
    linear_r_gains = np.clip(raw_r / avg_gain, 0.5, 2.0).astype(np.float32)
    
    # ラプラシアン加重平均ベースのZ-scoreパラメータ
    t_means = np.full(S, target_stats_array[0] / 255.0, dtype=np.float32)
    c_means = (c_b / 255.0).astype(np.float32)
    
    # ラプラシアン加重標準偏差(コントラスト)の比率 = ストレッチ係数
    stretch_ratios = np.clip(target_stats_array[1] / (c_c + 1e-6), 0.5, 4.0).astype(np.float32)
    
    # 彩度の比率
    sat_ratios = np.clip(target_stats_array[3] / (c_s + 1e-6), 0.5, 3.0).astype(np.float32)

    apply_color_correction_numba(pool.bgr_batch[:S], pool.out_batch[:S], 
                                 linear_b_gains, linear_g_gains, linear_r_gains, 
                                 stretch_ratios, t_means, c_means, sat_ratios)
    return pool.out_batch[:S]

def frame_generator_simple(video_path, H, W):
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO' 
    bgr_float_buffer = np.empty((H, W, 3), dtype=np.float32)
    for frame in container.decode(stream):
        yuv_to_float_bgr_fast(frame, bgr_float_buffer)
        bgr_uint8 = np.clip(bgr_float_buffer, 0, 255).astype(np.uint8)
        yield bgr_float_buffer, bgr_uint8
    container.close()

def merge_audio_ffmpeg(temp_video_path, original_video_path, final_output_path):
    cmd = [
        'ffmpeg', '-y', '-i', temp_video_path, '-i', original_video_path,
        '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0?',
        '-shortest', final_output_path
    ]
    try:
        startupinfo = subprocess.STARTUPINFO() if os.name == 'nt' else None
        if startupinfo: startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo)
        return res.returncode == 0
    except Exception:
        return False

# --- メインロジック ---

if __name__ == '__main__':
    warmup_numba_jit()
    
    ref_path = select_file("参照映像(Lookの元)を選択してください")
    if not ref_path: exit()

    ref_stats_list = []
    ref_temp_path = os.path.join(tempfile.gettempdir(), "ref_temp.mp4")
    fast_file_transfer(ref_path, ref_temp_path, "参照映像の読込 (帯域MAX)")
    
    ref_container = av.open(ref_temp_path)
    ref_stream = ref_container.streams.video[0]
    total_ref = ref_stream.frames if ref_stream.frames > 0 else int(float(ref_stream.duration * ref_stream.time_base) * ref_stream.average_rate)
    ref_H, ref_W = ref_stream.height, ref_stream.width
    ref_container.close()
    
    pbar_ref = tqdm(total=total_ref if total_ref > 0 else None, desc="Ref Analysis")
    for idx, (_, bgr_uint8) in enumerate(frame_generator_simple(ref_temp_path, ref_H, ref_W)):
        if idx % 2 == 0:
            s = get_stats_fast(bgr_uint8)
            if s is not None: ref_stats_list.append(s)
        pbar_ref.update(1)
    pbar_ref.close()
    
    if os.path.exists(ref_temp_path): os.remove(ref_temp_path)
    
    if not ref_stats_list:
        print("エラー: 参照映像から取得できませんでした。")
        exit()
    ref_avg_stats = np.mean(ref_stats_list, axis=0)

    target_folder = select_folder("適用先の映像が入っているフォルダを選択してください")
    if not target_folder: exit()

    target_files = [f for f in os.listdir(target_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not target_files: exit()

    out_dir = os.path.join(target_folder, "output_corrected")
    os.makedirs(out_dir, exist_ok=True)
    local_temp_dir = tempfile.gettempdir()
    
    for file_name in target_files:
        app_path = os.path.join(target_folder, file_name)
        base_name, _ = os.path.splitext(file_name)
        out_path = os.path.join(out_dir, f"corrected_{base_name}.mp4")
        
        temp_in_path = os.path.join(local_temp_dir, f"in_{base_name}.mp4")
        temp_out_vid_path = os.path.join(local_temp_dir, f"out_vid_{base_name}.mp4")
        temp_out_final_path = os.path.join(local_temp_dir, f"out_final_{base_name}.mp4")
        
        print(f"\n==============================================")
        print(f"ファイル処理開始: {file_name}")
        
        fast_file_transfer(app_path, temp_in_path, f"受信 ({base_name})")
        
        container = av.open(temp_in_path)
        stream = container.streams.video[0]
        total_app_frames = stream.frames if stream.frames > 0 else int(float(stream.duration * stream.time_base) * stream.average_rate)
        if total_app_frames <= 0: total_app_frames = 1000
        fps = float(stream.average_rate)
        if fps <= 0 or math.isnan(fps): fps = 30.0
        w, h = stream.width, stream.height
        container.close()

        out = cv2.VideoWriter(temp_out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        N = get_optimal_sampling_interval(fps, target_sec=0.5)

        # 【施策】並行処理バッファの区間数を「8」に拡大（約4秒分をまとめて巨大バッチ化）
        BATCH_SEGMENTS = 8
        MAX_BATCH_SIZE = N * (BATCH_SEGMENTS + 4) # 端数処理も含めたバッファ最大確保サイズ
        
        pool = TensorMemoryPool(MAX_BATCH_SIZE, h, w)

        PREFETCH_SIZE = max(180, MAX_BATCH_SIZE * 2)
        decode_q = queue.Queue(maxsize=PREFETCH_SIZE)
        write_q = queue.Queue(maxsize=MAX_BATCH_SIZE * 2)
        
        free_in = queue.Queue()
        free_out = queue.Queue()
        # 巨大バッチに耐えられるだけのメモリを確保
        for _ in range(MAX_BATCH_SIZE * 2 + 30):
            free_in.put(np.empty((h, w, 3), dtype=np.float32))
            free_out.put(np.empty((h, w, 3), dtype=np.uint8))

        def reader_worker():
            read_container = av.open(temp_in_path)
            read_stream = read_container.streams.video[0]
            read_stream.thread_type = 'AUTO'
            try:
                for f_idx, frame in enumerate(read_container.decode(read_stream)):
                    is_yuv, data = extract_yuv_planes(frame)
                    decode_q.put((is_yuv, data, f_idx))
            finally:
                decode_q.put(None)
                read_container.close()

        def writer_worker():
            while True:
                item = write_q.get()
                if item is None: break
                out.write(item)
                free_out.put(item)

        reader_thread = threading.Thread(target=reader_worker, daemon=True)
        writer_thread = threading.Thread(target=writer_worker, daemon=True)
        reader_thread.start(); writer_thread.start()

        stats_buffer = []
        frames_buffer = deque()
        pbar = tqdm(total=total_app_frames if total_app_frames > 1000 else None, desc="演算中 (CPU MAX)")

        try:
            while True:
                item = decode_q.get()
                if item is None: break
                is_yuv, data, f_idx = item
                
                bgr_float = free_in.get()
                if is_yuv:
                    y, u, v = data
                    yuv_to_bgr_numba(y, u, v, bgr_float)
                else:
                    bgr_float[:] = data.astype(np.float32)
                
                frames_buffer.append(bgr_float)
                
                if f_idx % N == 0:
                    bgr_uint8 = np.clip(bgr_float, 0, 255).astype(np.uint8)
                    s = get_stats_fast(bgr_uint8)
                    if s is not None:
                        if f_idx == 0: stats_buffer.append((-N, s))
                        stats_buffer.append((f_idx, s))
                
                # BATCH_SEGMENTS (8区間) たまったら一気に巨大バッチとして処理
                if len(stats_buffer) >= BATCH_SEGMENTS + 3:
                    total_seg_size = 0
                    all_seg_frames = []
                    all_interp_stats = []
                    
                    for k in range(1, BATCH_SEGMENTS + 1):
                        start_idx, end_idx = stats_buffer[k][0], stats_buffer[k+1][0]
                        seg_size = end_idx - start_idx
                        seg_frames = [frames_buffer.popleft() for _ in range(seg_size)]
                        pts = [stats_buffer[k-1+i][1] for i in range(4)]
                        
                        interp_stats_list = [mitchell_netravali(i / seg_size, pts) for i in range(seg_size)]
                        
                        all_seg_frames.extend(seg_frames)
                        all_interp_stats.extend(interp_stats_list)
                        total_seg_size += seg_size
                        
                    # CPU全コアを限界駆動
                    result_batch = process_segment_batch_zero_alloc(all_seg_frames, all_interp_stats, ref_avg_stats, pool)
                    
                    for i in range(total_seg_size):
                        out_frame = free_out.get()
                        out_frame[:] = result_batch[i]
                        write_q.put(out_frame)
                        free_in.put(all_seg_frames[i])
                        pbar.update(1)
                        
                    for _ in range(BATCH_SEGMENTS):
                        stats_buffer.pop(0)

            # 端数処理ループもまとめて巨大バッチ化
            while frames_buffer:
                while len(stats_buffer) < 4:
                    if not stats_buffer: break
                    stats_buffer.append((stats_buffer[-1][0] + N, stats_buffer[-1][1]))
                if len(stats_buffer) < 4: break
                    
                remaining_segments = len(stats_buffer) - 3
                if remaining_segments <= 0: break
                
                total_seg_size = 0
                all_seg_frames = []
                all_interp_stats = []
                
                for k in range(1, remaining_segments + 1):
                    start_idx, end_idx = stats_buffer[k][0], stats_buffer[k+1][0]
                    seg_size = min(end_idx - start_idx, len(frames_buffer) - total_seg_size)
                    if seg_size <= 0: continue
                    
                    seg_frames = [frames_buffer.popleft() for _ in range(seg_size)]
                    pts = [stats_buffer[k-1+i][1] for i in range(4)]
                    denom = (end_idx - start_idx) if (end_idx - start_idx) > 0 else 1
                    interp_stats_list = [mitchell_netravali(i / denom, pts) for i in range(seg_size)]
                    
                    all_seg_frames.extend(seg_frames)
                    all_interp_stats.extend(interp_stats_list)
                    total_seg_size += seg_size
                
                if total_seg_size > 0:
                    result_batch = process_segment_batch_zero_alloc(all_seg_frames, all_interp_stats, ref_avg_stats, pool)
                    for i in range(total_seg_size):
                        out_frame = free_out.get()
                        out_frame[:] = result_batch[i]
                        write_q.put(out_frame)
                        free_in.put(all_seg_frames[i])
                        pbar.update(1)
                
                for _ in range(remaining_segments):
                    stats_buffer.pop(0)

        finally:
            write_q.put(None)
            writer_thread.join()
            pbar.close()
            out.release()

        print("音声を結合中...")
        success = merge_audio_ffmpeg(temp_out_vid_path, temp_in_path, temp_out_final_path)
        final_src_path = temp_out_final_path if success else temp_out_vid_path

        fast_file_transfer(final_src_path, out_path, f"送信 ({base_name})")

        for temp_file in [temp_in_path, temp_out_vid_path, temp_out_final_path]:
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except: pass

    print("\n--- すべての処理が完了しました ---")