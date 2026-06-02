import cv2
import numpy as np
import os
import time
from skimage.metrics import structural_similarity as ssim_fn
from video_enhance import enhance_video
from video_enhance_hq import enhance_video_hq
from model_router import select_model

TMP_DIR = 'benchmark_tmp'


def _downscale_video(input_path, output_path, scale=2):
    """把影片縮小 scale 倍，模擬低畫質輸入"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w // scale, h // scale)
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(cv2.resize(frame, (w // scale, h // scale),
                             interpolation=cv2.INTER_CUBIC))
    cap.release()
    out.release()


def _calc_metrics(ref_path, enhanced_path):
    """計算 PSNR / SSIM，ref 自動 bicubic 放大到 enhanced 尺寸"""
    cap_r = cv2.VideoCapture(ref_path)
    cap_e = cv2.VideoCapture(enhanced_path)
    enh_w = int(cap_e.get(cv2.CAP_PROP_FRAME_WIDTH))
    enh_h = int(cap_e.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = min(int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap_e.get(cv2.CAP_PROP_FRAME_COUNT)))
    psnr_list, ssim_list = [], []
    for _ in range(total):
        ret_r, fr = cap_r.read()
        ret_e, fe = cap_e.read()
        if not ret_r or not ret_e:
            break
        fr_up = cv2.resize(fr, (enh_w, enh_h), interpolation=cv2.INTER_CUBIC)
        psnr_list.append(cv2.PSNR(fr_up, fe))
        ssim_list.append(ssim_fn(
            cv2.cvtColor(fr_up, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(fe, cv2.COLOR_BGR2GRAY)
        ))
    cap_r.release()
    cap_e.release()
    return np.mean(psnr_list), np.mean(ssim_list), len(psnr_list)


def benchmark(original_path, run_fast=True, run_hq=True):
    """
    一鍵流程：縮小原始影片 → 分別跑快速/精緻模式 → 計算 PSNR/SSIM → 輸出對比表
    original_path: 原始高畫質影片（作為 ground truth）
    """
    os.makedirs(TMP_DIR, exist_ok=True)
    low_res = os.path.join(TMP_DIR, 'lowres_input.mp4')
    fast_out = os.path.join(TMP_DIR, 'fast_output.mp4')
    hq_out = os.path.join(TMP_DIR, 'hq_output.mp4')

    print(f"原始影片：{original_path}")
    print("─" * 44)

    try:
        print("[1/3] 產生低畫質輸入（bicubic 縮小 2x）...")
        _downscale_video(original_path, low_res, scale=2)

        results = {}

        if run_fast:
            print("[2/3] 快速模式（FSRCNN x2）處理中...")
            model_path, model_name, scale = select_model()
            t0 = time.time()
            enhance_video(low_res, fast_out, model_path, model_name, scale)
            elapsed = time.time() - t0
            psnr, ssim_val, frames = _calc_metrics(original_path, fast_out)
            results['fast'] = {'psnr': psnr, 'ssim': ssim_val,
                               'frames': frames, 'time': elapsed}

        if run_hq:
            print("[3/3] 精緻模式（Real-ESRGAN + GFPGAN）處理中...")
            t0 = time.time()
            enhance_video_hq(low_res, hq_out, weight=0.3)
            elapsed = time.time() - t0
            psnr, ssim_val, frames = _calc_metrics(original_path, hq_out)
            results['hq'] = {'psnr': psnr, 'ssim': ssim_val,
                             'frames': frames, 'time': elapsed}

    finally:
        for p in [low_res, fast_out, hq_out]:
            if os.path.exists(p):
                os.remove(p)
        try:
            os.rmdir(TMP_DIR)
        except OSError:
            pass

    # ── 輸出對比表 ──
    print("\n" + "═" * 44)
    print("  評測結果（與原始高畫質影片比較）")
    print("═" * 44)
    if 'fast' in results:
        r = results['fast']
        print(f"  快速模式  FSRCNN x2")
        print(f"    PSNR   : {r['psnr']:.2f} dB")
        print(f"    SSIM   : {r['ssim']:.4f}")
        print(f"    幀數   : {r['frames']} 幀")
        print(f"    耗時   : {r['time']:.1f} 秒")
        print()
    if 'hq' in results:
        r = results['hq']
        print(f"  精緻模式  Real-ESRGAN + GFPGAN")
        print(f"    PSNR   : {r['psnr']:.2f} dB")
        print(f"    SSIM   : {r['ssim']:.4f}")
        print(f"    幀數   : {r['frames']} 幀")
        print(f"    耗時   : {r['time']:.1f} 秒")
    print("═" * 44)

    return results


if __name__ == '__main__':
    benchmark(
        original_path='0528.1.mp4',
        run_fast=True,
        run_hq=True
    )
