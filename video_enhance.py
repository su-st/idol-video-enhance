import cv2
import os
import subprocess
import imageio_ffmpeg


def enhance_video(input_path, output_path, model_path, model_name='fsrcnn', scale=2,
                  skip_interval=3, progress_callback=None, cancel_check=None):
    """
    影片畫質增強主函式
    skip_interval: 每 N 幀做一次 SR，其餘用 bicubic（提升速度）
    """
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(model_name, scale)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w * scale, h * scale)
    )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    frame_count = 0
    prev_output = None  # temporal filtering 用

    while True:
        if cancel_check and cancel_check():
            break
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 去雜訊
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        # 2. 對比增強（CLAHE）
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # 3. 銳化
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        # 4. 超解析度（跳幀策略：SR 幀 vs bicubic 幀）
        if frame_count % skip_interval == 0:
            upscaled = sr.upsample(sharpened)
        else:
            upscaled = cv2.resize(sharpened, (w * scale, h * scale),
                                  interpolation=cv2.INTER_CUBIC)

        # 5. Temporal Filtering：與上一幀融合，抑制跨幀閃爍
        if prev_output is not None:
            upscaled = cv2.addWeighted(upscaled, 0.85, prev_output, 0.15, 0)
        prev_output = upscaled

        out.write(upscaled)
        frame_count += 1
        print(f"進度：{frame_count}/{total_frames}")
        if progress_callback and total_frames > 0:
            progress_callback(int(frame_count / total_frames * 90))

    cap.release()
    out.release()

    # 把原始音軌合回輸出影片
    tmp_path = output_path + '.tmp.mp4'
    os.rename(output_path, tmp_path)
    try:
        proc = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), '-y',
                '-i', tmp_path,
                '-i', input_path,
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                output_path
            ],
            capture_output=True
        )
        if proc.returncode == 0:
            os.remove(tmp_path)
        else:
            os.rename(tmp_path, output_path)
    except FileNotFoundError:
        os.rename(tmp_path, output_path)

    print(f"完成！輸出：{output_path}")
    return output_path


if __name__ == '__main__':
    enhance_video(
        input_path='test.mp4',
        output_path='outputs/test_output.mp4',
        model_path='model/FSRCNN_x2.pb'
    )
