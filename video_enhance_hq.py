import os
import cv2
import subprocess
import threading
import torch
import imageio_ffmpeg
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

_upsampler = None
_face_enhancer = None
_model_lock = threading.Lock()


def _is_bad_model_frame(output_frame, source_frame):
    if output_frame is None:
        return True
    if output_frame.ndim != 3 or output_frame.shape[2] != 3:
        return True
    if output_frame.dtype != source_frame.dtype:
        return True

    source_mean = float(source_frame.mean())
    output_mean = float(output_frame.mean())

    # Some GPU/tiling failures return a valid-sized but black or near-black frame.
    if source_mean > 10 and output_mean < max(10, source_mean * 0.2):
        return True
    return source_mean > 3 and output_mean < 3


def _bicubic_upscale(frame):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


def _prepare_video_frame(frame):
    if frame.dtype != 'uint8':
        frame = frame.clip(0, 255).astype('uint8')
    if not frame.flags['C_CONTIGUOUS']:
        frame = frame.copy()
    return frame


def _sample_video_visibility(video_path, max_frames=40):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, False, []

    readable = False
    visible = False
    means = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        positions = {
            0,
            min(10, frame_count - 1),
            frame_count // 4,
            frame_count // 2,
            (frame_count * 3) // 4,
            max(frame_count - 2, 0),
        }
    else:
        positions = set()

    for pos in sorted(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        readable = True
        mean = float(frame.mean())
        means.append(round(mean, 2))
        if mean >= 3:
            visible = True
            break

    if not visible:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        checked = 0
        while checked < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            readable = True
            mean = float(frame.mean())
            means.append(round(mean, 2))
            if mean >= 3:
                visible = True
                break
            checked += 1

    cap.release()
    return readable, visible, means


def _validate_output_video(output_path, source_path):
    out_readable, out_visible, out_means = _sample_video_visibility(output_path)
    src_readable, src_visible, src_means = _sample_video_visibility(source_path)
    print(f"Validate output means={out_means[:10]} source means={src_means[:10]}", flush=True)
    if not out_readable:
        raise RuntimeError("Output video has no readable frames")
    if not out_visible:
        raise RuntimeError("Output video frames are black; refusing to publish audio-only-looking file")


def _publish_checked_output(output_path, input_path):
    try:
        _validate_output_video(output_path, input_path)
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


def _check_cuda():
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1).cuda()
        return True
    except RuntimeError:
        return False


def _load_upsampler():
    global _upsampler
    if _upsampler is None:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        use_gpu = _check_cuda()
        print(f"使用 {'GPU' if use_gpu else 'CPU'} 處理", flush=True)
        _upsampler = RealESRGANer(
            scale=2,
            model_path='model/RealESRGAN_x2plus.pth',
            model=model,
            tile=512,
            tile_pad=32,
            pre_pad=0,
            half=use_gpu
        )
        print("Real-ESRGAN 載入完成", flush=True)


def _load_models():
    global _face_enhancer
    _load_upsampler()
    if _face_enhancer is None:
        _face_enhancer = GFPGANer(
            model_path='model/GFPGANv1.4.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=_upsampler
        )
        print("GFPGAN 載入完成", flush=True)


def _encode_frames_to_file(frames_iter, out_w, out_h, fps, tmp_path):
    """
    Pass 1：OpenCV VideoWriter → MJPEG AVI，再由 FFmpeg 轉成 H264 MP4。
    完全避免 stdin pipe 及 swscaler 的問題。
    """
    avi_path = tmp_path + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(avi_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 無法開啟 {out_w}x{out_h}")
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)

    frame_written = 0
    for frame in frames_iter:
        writer.write(frame)
        frame_written += 1
    writer.release()
    print(f"Pass1 MJPG: {frame_written} 幀 → {avi_path} ({os.path.getsize(avi_path) if os.path.exists(avi_path) else 0} bytes)", flush=True)

    if not os.path.exists(avi_path) or os.path.getsize(avi_path) < 1000:
        raise RuntimeError("MJPEG 中間檔過小")

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), '-y',
        '-i', avi_path,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        tmp_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    err = result.stderr.decode('utf-8', errors='replace')
    if result.returncode != 0:
        print(f"Pass1 FFmpeg 失敗: {err[-300:]}", flush=True)
    else:
        print(f"Pass1 H264 完成: {tmp_path} ({os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0} bytes)", flush=True)
    try:
        os.remove(avi_path)
    except Exception:
        pass
    return result.returncode


def _mux_audio(video_path, audio_source, output_path):
    """Pass 2：把音訊混入已編好的影片（用 copy 模式，速度快）。"""
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), '-y',
        '-i', video_path,
        '-i', audio_source,
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"音訊混入失敗: {result.stderr.decode('utf-8', errors='replace')[-300:]}", flush=True)
    return result.returncode == 0


def enhance_video_hq(input_path, output_path, weight=0.3, progress_callback=None, cancel_check=None):
    """高品質影片增強：Real-ESRGAN（整體）+ GFPGAN（臉部）"""
    _load_models()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 捨入到 4 的倍數：yuv420p 的 chroma 平面高度 = out_h/2，需為偶數
    # 若 h 為奇數（如 719），out_h=1438 → chroma=719（奇數）→ swscaler 失敗輸出全黑
    out_w = ((w * 2) + 3) // 4 * 4
    out_h = ((h * 2) + 3) // 4 * 4

    # GFPGAN 對奇數高度輸入會靜默回傳接近全黑的 frame，跳過改用 Real-ESRGAN
    use_gfpgan = (h % 2 == 0 and w % 2 == 0)
    if not use_gfpgan:
        print(f"影片尺寸 {w}x{h} 含奇數邊長，跳過 GFPGAN 改用 Real-ESRGAN", flush=True)

    tmp_path = output_path + '.video.mp4'
    frame_count = 0

    def frame_generator():
        nonlocal frame_count
        while True:
            if cancel_check and cancel_check():
                break
            ret, frame = cap.read()
            if not ret:
                break

            with _model_lock:
                if use_gfpgan:
                    try:
                        _, _, output_frame = _face_enhancer.enhance(
                            frame, has_aligned=False, only_center_face=False,
                            paste_back=True, weight=weight)
                        if _is_bad_model_frame(output_frame, frame):
                            raise ValueError("GFPGAN 回傳異常暗幀")
                    except Exception:
                        try:
                            output_frame, _ = _upsampler.enhance(frame, outscale=2)
                            if _is_bad_model_frame(output_frame, frame):
                                raise ValueError("Real-ESRGAN returned a bad frame")
                        except Exception:
                            output_frame = _bicubic_upscale(frame)
                else:
                    try:
                        output_frame, _ = _upsampler.enhance(frame, outscale=2)
                        if _is_bad_model_frame(output_frame, frame):
                            raise ValueError("Real-ESRGAN returned a bad frame")
                    except Exception:
                        output_frame = _bicubic_upscale(frame)

            if output_frame.shape[0] != out_h or output_frame.shape[1] != out_w:
                output_frame = cv2.resize(output_frame, (out_w, out_h))
            output_frame = _prepare_video_frame(output_frame)

            frame_count += 1
            print(f"進度：{frame_count}/{total_frames}", flush=True)
            if progress_callback and total_frames > 0:
                progress_callback(int(frame_count / total_frames * 90))

            yield output_frame

    try:
        _encode_frames_to_file(frame_generator(), out_w, out_h, fps, tmp_path)
    finally:
        cap.release()

    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
        if _mux_audio(tmp_path, input_path, output_path):
            os.remove(tmp_path)
        else:
            os.rename(tmp_path, output_path)
    else:
        raise RuntimeError("影片編碼失敗，輸出檔案過小")

    print(f"完成！輸出：{output_path}", flush=True)
    _publish_checked_output(output_path, input_path)
    return output_path


def enhance_video_fast(input_path, output_path, progress_callback=None, cancel_check=None):
    """快速模式：純 Real-ESRGAN x2 放大（不含 GFPGAN 臉部修復）"""
    _load_upsampler()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = ((w * 2) + 3) // 4 * 4
    out_h = ((h * 2) + 3) // 4 * 4

    tmp_path = output_path + '.video.mp4'
    frame_count = 0

    def frame_generator():
        nonlocal frame_count
        while True:
            if cancel_check and cancel_check():
                break
            ret, frame = cap.read()
            if not ret:
                break

            with _model_lock:
                try:
                    output_frame, _ = _upsampler.enhance(frame, outscale=2)
                    if _is_bad_model_frame(output_frame, frame):
                        raise ValueError("Real-ESRGAN returned a bad frame")
                except Exception:
                    output_frame = _bicubic_upscale(frame)

            if output_frame.shape[0] != out_h or output_frame.shape[1] != out_w:
                output_frame = cv2.resize(output_frame, (out_w, out_h))
            output_frame = _prepare_video_frame(output_frame)

            frame_count += 1
            if progress_callback and total_frames > 0:
                progress_callback(int(frame_count / total_frames * 90))

            yield output_frame

    try:
        _encode_frames_to_file(frame_generator(), out_w, out_h, fps, tmp_path)
    finally:
        cap.release()

    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
        if _mux_audio(tmp_path, input_path, output_path):
            os.remove(tmp_path)
        else:
            os.rename(tmp_path, output_path)
    else:
        raise RuntimeError("影片編碼失敗，輸出檔案過小")

    print(f"完成！輸出：{output_path}", flush=True)
    _publish_checked_output(output_path, input_path)
    return output_path


if __name__ == '__main__':
    enhance_video_hq(
        input_path='test_input.mp4',
        output_path='outputs/test_hq.mp4',
        weight=0.3
    )
