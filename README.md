# 偶像影片畫質修復 Idol Video Enhancement

以 Real-ESRGAN + GFPGAN 為核心的影片超解析度 Web 應用，支援快速模式（Real-ESRGAN x2）與精緻模式（Real-ESRGAN + GFPGAN 人臉修復）。

## 環境需求

- Python 3.9 以上
- NVIDIA GPU（建議，無 GPU 仍可執行但速度較慢）
- CUDA 12.8（搭配 GPU 使用）

## 安裝步驟

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

安裝 PyTorch（擇一）：

```bash
# 有 NVIDIA GPU（推薦）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 無 GPU
pip install torch torchvision
```

### 2. 下載模型檔

請手動下載以下模型並放到對應資料夾：

**`model/` 資料夾（需自行建立）：**

| 檔案 | 下載連結 |
|------|----------|
| `RealESRGAN_x2plus.pth` | [下載](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) |
| `GFPGANv1.4.pth` | [下載](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth) |

**`gfpgan/weights/` 資料夾（需自行建立）：**

| 檔案 | 下載連結 |
|------|----------|
| `detection_Resnet50_Final.pth` | [下載](https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth) |
| `parsing_parsenet.pth` | [下載](https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth) |

下載後的資料夾結構：

```
image_final/
├── model/
│   ├── RealESRGAN_x2plus.pth
│   └── GFPGANv1.4.pth
├── gfpgan/
│   └── weights/
│       ├── detection_Resnet50_Final.pth
│       └── parsing_parsenet.pth
├── app.py
└── ...
```

### 3. 啟動伺服器

```bash
python app.py
```

啟動後開啟瀏覽器連至：[http://127.0.0.1:5000](http://127.0.0.1:5000)

## 使用方式

1. 上傳影片（支援 MP4 / MOV / AVI，最大 250MB）
2. 選擇處理模式：
   - **自動模式**：依影片長度自動選擇
   - **快速模式**：Real-ESRGAN x2 放大，速度優先
   - **精緻模式**：Real-ESRGAN + GFPGAN 人臉修復，畫質優先
3. 按「開始修復」等待處理完成
4. 預覽並下載結果

## 效能評測結果

以 `0528.2.mp4`（135 幀）為測試影片：

| 方法 | PSNR | SSIM | 處理時間 |
|------|------|------|----------|
| 雙三次插值（基準） | 27.56 dB | 0.9390 | — |
| 快速模式（Real-ESRGAN） | 26.58 dB | 0.9363 | 13 秒 |
| 精緻模式（Real-ESRGAN + GFPGAN） | 26.50 dB | 0.9348 | 36 秒 |

> PSNR/SSIM 為像素級指標，Real-ESRGAN 雖數值略低於 bicubic，但能還原更多高頻紋理細節，主觀視覺品質更佳。
