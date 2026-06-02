import os

def select_model():
    """快速模式固定使用 FSRCNN x2，回傳 (model_path, model_name, scale)"""
    base_dir = os.path.join(os.path.dirname(__file__), 'model')
    return (os.path.join(base_dir, 'FSRCNN_x2.pb'), 'fsrcnn', 2)


def select_mode(mode='fast', duration=0):
    """
    決定走哪一條處理路線
    'fast'    → FSRCNN x2
    'quality' → Real-ESRGAN + GFPGAN
    'auto'    → 短影片（≤10s）用 HQ，長影片用快速
    """
    if mode in ('hq', 'quality'):
        return 'hq'
    if mode == 'auto':
        return 'hq' if duration <= 10 else 'fast'
    return 'fast'


def get_video_duration(video_path):
    """取得影片秒數"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps > 0:
        return total_frames / fps
    return 0