import os
import csv
import subprocess
import time
import threading
import traceback
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
import imageio_ffmpeg
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from video_enhance_hq import enhance_video_hq, enhance_video_fast
from model_router import select_mode, get_video_duration

app = Flask(__name__)

OUTPUT_FOLDER = 'outputs'
UPLOAD_FOLDER = os.path.join(OUTPUT_FOLDER, '_uploads')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
MAX_CONTENT_LENGTH = 250 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

progress_status = {}
_task_semaphore = threading.Semaphore(1)  # 同時只處理一個，其他自動排隊
CLIENT_TIMEOUT_SECONDS = 120


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_uploaded_video(input_path):
    normalized_path = input_path + '.normalized.mp4'
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), '-y',
        '-i', input_path,
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-movflags', '+faststart',
        normalized_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not os.path.exists(normalized_path):
        err = result.stderr.decode('utf-8', errors='replace')[-500:]
        raise RuntimeError(f'影片正規化失敗：{err}')

    os.replace(normalized_path, input_path)
    return input_path


def process_video_task(task_id, input_path, output_path, route):
    """背景執行影片處理（支援排隊等候）"""
    # 先設為排隊中
    progress_status[task_id] = {
        'status': 'queued',
        'progress': 0,
        'cancelled': False,
        'last_seen': time.time()
    }

    _task_semaphore.acquire()  # 等到輪到自己（其他任務跑完才會放行）

    try:
        # 排隊期間若已取消，直接結束
        if progress_status.get(task_id, {}).get('cancelled', False):
            progress_status[task_id] = {'status': 'cancelled'}
            return

        last_seen = progress_status.get(task_id, {}).get('last_seen', time.time())
        progress_status[task_id] = {
            'status': 'processing',
            'progress': 0,
            'cancelled': False,
            'last_seen': last_seen
        }

        def update_progress(pct):
            progress_status[task_id].update({'status': 'processing', 'progress': pct})

        def should_cancel():
            task = progress_status.get(task_id, {})
            if task.get('cancelled', False):
                return True
            last_seen = task.get('last_seen', time.time())
            if time.time() - last_seen > CLIENT_TIMEOUT_SECONDS:
                task.update({
                    'cancelled': True,
                    'cancel_reason': 'client_timeout'
                })
                return True
            return False

        if route == 'hq':
            enhance_video_hq(input_path, output_path, weight=0.3,
                             progress_callback=update_progress, cancel_check=should_cancel)
        else:
            enhance_video_fast(input_path, output_path,
                               progress_callback=update_progress, cancel_check=should_cancel)

        if should_cancel():
            progress_status[task_id] = {'status': 'cancelled'}
        else:
            progress_status[task_id] = {'status': 'done', 'progress': 100}
    except Exception as e:
        progress_status[task_id] = {'status': 'error', 'message': str(e)}
        if os.path.exists(output_path):
            os.remove(output_path)
    finally:
        _task_semaphore.release()
        if os.path.exists(input_path):
            os.remove(input_path)
        output_path_check = os.path.join(OUTPUT_FOLDER, f"{task_id}_enhanced.mp4")
        if progress_status.get(task_id, {}).get('status') == 'cancelled':
            if os.path.exists(output_path_check):
                os.remove(output_path_check)


@app.route('/')
def index():
    return render_template('index.html')


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return jsonify({'error': f'檔案太大，請上傳 {max_mb}MB 以下的影片'}), 413


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    traceback_text = traceback.format_exc()
    with open('flask_runtime_error.log', 'a', encoding='utf-8') as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(traceback_text)
    return jsonify({'error': str(error)}), 500


def _cleanup_old_outputs(max_age_seconds=7200):
    """刪除超過 max_age_seconds 的 output 檔（預設 2 小時），避免磁碟爆滿"""
    try:
        now = time.time()
        for fname in os.listdir(OUTPUT_FOLDER):
            if not fname.endswith('_enhanced.mp4'):
                continue
            fpath = os.path.join(OUTPUT_FOLDER, fname)
            if now - os.path.getmtime(fpath) > max_age_seconds:
                try:
                    os.remove(fpath)
                except OSError:
                    pass
    except Exception:
        pass


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': '沒有上傳檔案'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '沒有選擇檔案'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '不支援的檔案格式，請上傳 MP4/MOV/AVI'}), 400

    # 順便清理 2 小時前的舊輸出檔
    threading.Thread(target=_cleanup_old_outputs, daemon=True).start()

    task_id = uuid.uuid4().hex
    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, f"{task_id}_{filename}")
    file.save(input_path)
    normalize_uploaded_video(input_path)

    duration = get_video_duration(input_path)
    user_mode = request.form.get('mode', 'auto')
    route = select_mode(user_mode, duration)
    output_path = os.path.join(OUTPUT_FOLDER, f"{task_id}_enhanced.mp4")

    thread = threading.Thread(
        target=process_video_task,
        args=(task_id, input_path, output_path, route)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'task_id': task_id,
        'message': '已加入排隊！',
        'duration': round(duration, 1),
        'mode': route
    })


@app.route('/cancel/<task_id>', methods=['POST'])
def cancel(task_id):
    if task_id not in progress_status:
        return jsonify({'error': 'not found'}), 404
    progress_status[task_id]['cancelled'] = True
    progress_status[task_id]['cancel_reason'] = 'manual'
    return jsonify({'ok': True})


@app.route('/heartbeat/<task_id>', methods=['POST'])
def heartbeat(task_id):
    if task_id not in progress_status:
        return jsonify({'error': 'not found'}), 404
    progress_status[task_id]['last_seen'] = time.time()
    return jsonify({'ok': True})


@app.route('/status/<task_id>')
def status(task_id):
    if task_id not in progress_status:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(progress_status[task_id])


@app.route('/preview/<task_id>')
def preview(task_id):
    output_path = os.path.join(OUTPUT_FOLDER, f"{task_id}_enhanced.mp4")
    if not os.path.exists(output_path):
        return jsonify({'error': '檔案不存在'}), 404
    return send_file(output_path, mimetype='video/mp4')


@app.route('/rate', methods=['POST'])
def rate():
    data = request.get_json(silent=True) or {}
    row = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'task_id': data.get('task_id', ''),
        'ease': data.get('ease', ''),
        'quality': data.get('quality', ''),
        'overall': data.get('overall', ''),
    }
    ratings_file = 'ratings.csv'
    write_header = not os.path.exists(ratings_file)
    with open(ratings_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return jsonify({'ok': True})


@app.route('/download/<task_id>')
def download(task_id):
    output_path = os.path.join(OUTPUT_FOLDER, f"{task_id}_enhanced.mp4")
    if not os.path.exists(output_path):
        return jsonify({'error': '檔案不存在'}), 404
    # 下載後保留檔案，讓 compare 區塊的預覽繼續可用
    # 舊檔由 _cleanup_old_outputs 在下次上傳時自動清理
    return send_file(output_path, as_attachment=True, download_name='enhanced.mp4')


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=False, host='0.0.0.0', port=5000)
