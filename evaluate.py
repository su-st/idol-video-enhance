import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def evaluate(original_path, enhanced_path):
    """
    比較原始影片與修復後影片的 PSNR / SSIM。
    修復後影片解析度為原始的 2 倍，會先將原始幀 bicubic 放大再比較。
    """
    cap_orig = cv2.VideoCapture(original_path)
    cap_enh = cv2.VideoCapture(enhanced_path)

    total_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    total_enh = int(cap_enh.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(total_orig, total_enh)

    enh_w = int(cap_enh.get(cv2.CAP_PROP_FRAME_WIDTH))
    enh_h = int(cap_enh.get(cv2.CAP_PROP_FRAME_HEIGHT))

    psnr_list = []
    ssim_list = []

    for _ in range(total):
        ret_o, frame_o = cap_orig.read()
        ret_e, frame_e = cap_enh.read()
        if not ret_o or not ret_e:
            break

        # 原始幀放大到與修復後相同尺寸
        frame_o_up = cv2.resize(frame_o, (enh_w, enh_h), interpolation=cv2.INTER_CUBIC)

        # PSNR（在 BGR 上算）
        psnr_val = cv2.PSNR(frame_o_up, frame_e)
        psnr_list.append(psnr_val)

        # SSIM（轉灰階）
        gray_o = cv2.cvtColor(frame_o_up, cv2.COLOR_BGR2GRAY)
        gray_e = cv2.cvtColor(frame_e, cv2.COLOR_BGR2GRAY)
        ssim_val = ssim(gray_o, gray_e)
        ssim_list.append(ssim_val)

    cap_orig.release()
    cap_enh.release()

    if not psnr_list:
        print("錯誤：無法讀取幀，請確認影片路徑正確。")
        return

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    frames = len(psnr_list)

    print(f"平均 PSNR：{avg_psnr:.2f} dB")
    print(f"平均 SSIM：{avg_ssim:.4f}")
    print(f"比較幀數：{frames} 幀")

    return avg_psnr, avg_ssim, frames


if __name__ == '__main__':
    evaluate(
        original_path='uploads/test_input.mp4',
        enhanced_path='outputs/test_enhanced.mp4'
    )
