# KV260 人臉辨識 Demo

[English README](README.md)

即時人臉偵測 + 辨識 pipeline，完整部署在 AMD Kria KV260 上，
使用 Vitis AI 編譯的 INT8 量化 DPU 模型。

```
Input frame
    │
    ▼
┌─────────────────────────┐
│   SCRFD det_10g (DPU)   │  640×640 輸入 → 偵測人臉 + 5 個特徵點
└────────────┬────────────┘
             │ bbox + keypoints
             ▼
┌─────────────────────────┐
│   ArcFace Alignment     │  5 點仿射變換 → 112×112 人臉裁切
└────────────┬────────────┘
             │ aligned face
             ▼
┌─────────────────────────┐
│  MobileFaceNet (DPU)    │  128 維 L2 正規化 embedding
└────────────┬────────────┘
             │ embedding vector
             ▼
┌─────────────────────────┐
│   Cosine Similarity DB  │  與已註冊人臉比對
└─────────────────────────┘
```

---

## 硬體

| 項目 | 規格 |
|------|------|
| 板子 | AMD Kria KV260 Vision AI Starter Kit |
| SoC | Zynq UltraScale+ MPSoC (ARM Cortex-A53) |
| DPU | DPUCZDX8G\_ISA1\_B4096 |
| DPU 算力 | ~1.96 TOPS @ 300 MHz |
| Vitis AI | 3.0 (Docker: `xilinx/vitis-ai-cpu:latest`) |

---

## 量化流程

兩顆模型在 Vitis AI Docker 內都走同一套 **Post-Training Quantization (PTQ)** 流程：

```
FP32 model (ONNX / PyTorch)
    │
    ▼  torch_quantizer("calib")
校正 — 對 ~300 張代表性圖片做 forward pass
    │         (收集 activation 統計數據，用於計算 fix_point)
    ▼  torch_quantizer("test") + export_quant_config()
INT8 量化模型 (NNDCT simulation)
    │
    ▼  export_xmodel()
XIR graph (.xmodel) — 對應硬體的計算圖
    │
    ▼  vai_c_xir --arch DPUCZDX8G/KV260/arch.json
編譯後的 .xmodel — KV260 可用的 DPU binary
```

### 校正細節

| 模型 | 校正圖片數 | 來源 |
|-------|-------------------|--------|
| SCRFD det_10g | 300 張人臉圖片 | 光線/角度多樣的子集 |
| MobileFaceNet | 300 張人臉裁切圖 | 對齊後 112×112, 不同身分 |

---

## 精度: FP32 vs INT8

| 模型 | 資料集 | FP32 | INT8 (KV260) | 掉點 |
|-------|---------|------|--------------|------|
| MobileFaceNet | LFW (6k pairs) | 99.33% | 99.10% | −0.23% |
| MobileFaceNet | CFP-FP (7k pairs) | 84.81% | 83.40% | −1.41% |
| MobileFaceNet | MLFW (6k pairs) | 73.15% | 70.50% | −2.65% |
| SCRFD det_10g | CASIA-FaceV5 (2500 imgs) | 100% det | 100% det | 0% |

### 模型大小

| 模型 | FP32 | INT8 xmodel | 壓縮比 |
|-------|------|-------------|-------------|
| SCRFD det_10g | ~28 MB (ONNX) | **5.2 MB** | 5.4× |
| MobileFaceNet | ~4.0 MB (PT) | **1.5 MB** | 2.7× |

### KV260 推論延遲 *(僅 DPU, 50 次平均)*

| 模型 | 延遲 | FPS |
|-------|---------|-----|
| SCRFD det_10g (640×640) | 31.8 ms | 31.4 |
| MobileFaceNet (112×112) | 1.3 ms | 758 |
| 完整 pipeline (SCRFD + MFN) | ~33 ms | ~30 |

---

## 踩坑與分析

### MLFW 精度掉 −2.65%
MLFW 含大量遮擋與口罩情境。128 維 MobileFaceNet embedding 對「戴口罩人臉辨識」的表達能力有限 ——
這是模型架構本身的限制，不是量化造成的問題。FP32 baseline 在 MLFW 本來就只有 73.15%。
INT8 掉點(−2.65%)在小模型 PTQ 的正常範圍內。

### FixedReshape (SCRFD PTQ 會 crash)
ONNX→PyTorch 轉換器(onnx2torch)把 Reshape op 表示成「用動態 shape tensor 當第二個輸入」的形式。
Vitis AI 的 NNDCT quantizer 追蹤這個呼叫時會產生 `aten::unbind`，XIR compiler 無法對應這個 op。
修法: 在 float 推論階段用 forward hook 抓出實際 shape，
再把每個 `OnnxReshape` module 換成把 shape 寫死、忽略第二個參數的 `FixedReshape`。

### 校正後 fix_point=None
校正完後，`quant_info.json` 裡有些 tensor 的 `fix_point` 是 `null`
(未被校正到，通常是 batch-norm bias tensor 在校正過程中沒被觸發到)。
不處理的話，這些會在 test 階段輸出 NaN。
修法: 把所有 null 的 fix_point 設成 6 再載入量化模型。

---

## 模型檔案

放在 `models/`:

| 檔案 | 大小 |
|------|------|
| `scrfd_det10g.xmodel` | 5.2 MB |
| `mobilefacenet.xmodel` | 1.5 MB |

---

## 量化腳本

`quantize/` 內是實際在 Vitis AI Docker 裡跑過的 PTQ 腳本：

| 腳本 | 說明 |
|--------|-------------|
| `quantize_scrfd.py` | SCRFD det_10g 校正 + INT8 匯出 |
| `compile_scrfd.sh` | `vai_c_xir` 編譯 → `scrfd_det10g.xmodel` |
| `quantize_mfn.py` | MobileFaceNet 校正 + INT8 匯出 |
| `compile_mfn.sh` | `vai_c_xir` 編譯 → `mobilefacenet.xmodel` |

在 Vitis AI Docker (`xilinx/vitis-ai-cpu:latest`) 內執行:

```bash
docker run -v $(pwd):/workspace xilinx/vitis-ai-cpu:latest bash
# 容器內:
conda activate vitis-ai-pytorch
python quantize/quantize_scrfd.py    # → quantize_result/scrfd_det10g_int.xmodel
bash quantize/compile_scrfd.sh       # → compiled_model/scrfd_det10g.xmodel
```

---

## 環境設置 (在 KV260 上)

`vart` 跟 `xir` 是 KV260 PetaLinux image 內建的，不需要 pip install。
只有 OpenCV 需要另外裝(若尚未安裝):

```bash
pip install opencv-python-headless
```

每次重開機後，跑之前先載入 DPU overlay:

```bash
xmutil loadapp kv260-benchmark-b4096
```

## 使用方式

```bash
# 註冊人員 (預期格式 <testset>/<name>/real/*.jpg)
python demo.py register /home/root/testset

# 辨識單張圖片
python demo.py identify /path/to/face.jpg

# 精度 smoke-test (每人先註冊前 5 張，測後 3 張)
python demo.py test /home/root/testset
```
