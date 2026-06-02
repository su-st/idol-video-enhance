"""
重新跑快速模式（純 Real-ESRGAN）與精緻模式（Real-ESRGAN+GFPGAN）benchmark
- 輸出影片存到 outputs/
- PSNR/SSIM 以 0528.2.mp4 為基準（bicubic 縮小 → 增強 → 與原片比較）
"""
import cv2
import numpy as np
import os
import time
from skimage.metrics import structural_similarity as ssim_fn
from video_enhance_hq import enhance_video_fast, enhance_video_hq

SOURCE = '0528.2.mp4'
FAST_OUT = os.path.join('outputs', 'benchmark_fast.mp4')
HQ_OUT   = os.path.join('outputs', 'benchmark_hq.mp4')
TMP_LOW  = 'tmp_lowres_bench.mp4'


def downscale(src, dst, scale=2):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (w // scale, h // scale))
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        out.write(cv2.resize(fr, (w // scale, h // scale),
                             interpolation=cv2.INTER_CUBIC))
    cap.release()
    out.release()


def calc_metrics(ref_path, enh_path):
    cap_r = cv2.VideoCapture(ref_path)
    cap_e = cv2.VideoCapture(enh_path)
    ew = int(cap_e.get(cv2.CAP_PROP_FRAME_WIDTH))
    eh = int(cap_e.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = min(int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap_e.get(cv2.CAP_PROP_FRAME_COUNT)))
    psnr_list, ssim_list = [], []
    for _ in range(total):
        ret_r, fr = cap_r.read()
        ret_e, fe = cap_e.read()
        if not ret_r or not ret_e:
            break
        fr_up = cv2.resize(fr, (ew, eh), interpolation=cv2.INTER_CUBIC)
        psnr_list.append(cv2.PSNR(fr_up, fe))
        ssim_list.append(ssim_fn(
            cv2.cvtColor(fr_up, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(fe,    cv2.COLOR_BGR2GRAY)
        ))
    cap_r.release()
    cap_e.release()
    return np.mean(psnr_list), np.mean(ssim_list), len(psnr_list)


def main():
    os.makedirs('outputs', exist_ok=True)
    print(f"來源影片：{SOURCE}")
    print("─" * 48)

    # Step 1: 縮小原始影片作為低畫質輸入
    print("[1/6] 產生低畫質輸入（bicubic 縮小 2x）...")
    downscale(SOURCE, TMP_LOW, scale=2)

    results = {}

    # Step 2: bicubic 基準（直接把低畫質輸入 bicubic 放大回原尺寸）
    print("[2/6] 計算 bicubic 基準 PSNR/SSIM...")
    cap_src = cv2.VideoCapture(SOURCE)
    src_w = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_src.release()
    TMP_BICUBIC = 'tmp_bicubic_bench.mp4'
    cap_low = cv2.VideoCapture(TMP_LOW)
    fps_low = cap_low.get(cv2.CAP_PROP_FPS)
    bic_out = cv2.VideoWriter(TMP_BICUBIC, cv2.VideoWriter_fourcc(*'mp4v'),
                              fps_low, (src_w, src_h))
    while True:
        ret, fr = cap_low.read()
        if not ret:
            break
        bic_out.write(cv2.resize(fr, (src_w, src_h), interpolation=cv2.INTER_CUBIC))
    cap_low.release()
    bic_out.release()
    psnr_b, ssim_b, frames_b = calc_metrics(SOURCE, TMP_BICUBIC)
    results['bicubic'] = (psnr_b, ssim_b, frames_b)
    if os.path.exists(TMP_BICUBIC):
        os.remove(TMP_BICUBIC)
    print(f"      bicubic PSNR={psnr_b:.2f} dB  SSIM={ssim_b:.4f}")

    # Step 3: 快速模式增強 → outputs/benchmark_fast.mp4
    print(f"[3/6] 快速模式（純 Real-ESRGAN）增強中 → {FAST_OUT}")
    t0 = time.time()
    enhance_video_fast(TMP_LOW, FAST_OUT)
    results['fast_time'] = time.time() - t0
    print(f"      耗時：{results['fast_time']:.1f} 秒")

    # Step 4: 精緻模式增強 → outputs/benchmark_hq.mp4
    print(f"[4/6] 精緻模式（Real-ESRGAN + GFPGAN）增強中 → {HQ_OUT}")
    t0 = time.time()
    enhance_video_hq(TMP_LOW, HQ_OUT, weight=0.3)
    results['hq_time'] = time.time() - t0
    print(f"      耗時：{results['hq_time']:.1f} 秒")

    # Step 5: 計算 PSNR/SSIM（與原始影片比較）
    print("[5/6] 計算快速模式 PSNR/SSIM...")
    psnr_f, ssim_f, frames_f = calc_metrics(SOURCE, FAST_OUT)
    results['fast'] = (psnr_f, ssim_f, frames_f)

    print("[6/6] 計算精緻模式 PSNR/SSIM...")
    psnr_h, ssim_h, frames_h = calc_metrics(SOURCE, HQ_OUT)
    results['hq'] = (psnr_h, ssim_h, frames_h)

    # 清理臨時低畫質檔
    if os.path.exists(TMP_LOW):
        os.remove(TMP_LOW)

    # 輸出對照表
    print("\n" + "═" * 48)
    print("  評測結果（與原始高畫質影片比較）")
    print("═" * 48)
    psnr_b, ssim_b, frames_b = results['bicubic']
    print(f"  基準：雙三次插值（bicubic x2）：")
    print(f"    PSNR : {psnr_b:.2f} dB")
    print(f"    SSIM : {ssim_b:.4f}")
    print(f"    幀數 : {frames_b} 幀")
    print()
    psnr_f, ssim_f, frames_f = results['fast']
    print(f"  快速模式（純 Real-ESRGAN x2）：")
    print(f"    PSNR : {psnr_f:.2f} dB  ({psnr_f - psnr_b:+.2f} vs bicubic)")
    print(f"    SSIM : {ssim_f:.4f}  ({ssim_f - ssim_b:+.4f} vs bicubic)")
    print(f"    幀數 : {frames_f} 幀")
    print(f"    耗時 : {results['fast_time']:.1f} 秒")
    print()
    psnr_h, ssim_h, frames_h = results['hq']
    print(f"  精緻模式（Real-ESRGAN + GFPGAN）：")
    print(f"    PSNR : {psnr_h:.2f} dB  ({psnr_h - psnr_b:+.2f} vs bicubic)")
    print(f"    SSIM : {ssim_h:.4f}  ({ssim_h - ssim_b:+.4f} vs bicubic)")
    print(f"    幀數 : {frames_h} 幀")
    print(f"    耗時 : {results['hq_time']:.1f} 秒")
    print("═" * 48)
    print(f"\n輸出影片：")
    print(f"  快速：{FAST_OUT}")
    print(f"  精緻：{HQ_OUT}")


if __name__ == '__main__':
    main()
