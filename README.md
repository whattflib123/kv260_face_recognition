# KV260 Face Recognition Demo

Real-time face detection + recognition pipeline fully deployed on AMD Kria KV260,
using INT8-quantized DPU models compiled with Vitis AI.

```
Input frame
    │
    ▼
┌─────────────────────────┐
│   SCRFD det_10g (DPU)   │  640×640 input → detect faces + 5 landmarks
└────────────┬────────────┘
             │ bbox + keypoints
             ▼
┌─────────────────────────┐
│   ArcFace Alignment     │  5-point affine warp → 112×112 face crop
└────────────┬────────────┘
             │ aligned face
             ▼
┌─────────────────────────┐
│  MobileFaceNet (DPU)    │  128-dim L2-normalized embedding
└────────────┬────────────┘
             │ embedding vector
             ▼
┌─────────────────────────┐
│   Cosine Similarity DB  │  match against registered faces
└─────────────────────────┘
```

---

## Hardware

| Item | Spec |
|------|------|
| Board | AMD Kria KV260 Vision AI Starter Kit |
| SoC | Zynq UltraScale+ MPSoC (ARM Cortex-A53) |
| DPU | DPUCZDX8G\_ISA1\_B4096 |
| DPU compute | ~1.96 TOPS @ 300 MHz |
| Vitis AI | 3.0 (Docker: `xilinx/vitis-ai-cpu:latest`) |

---

## Quantization Pipeline

Both models follow the same **Post-Training Quantization (PTQ)** flow inside the Vitis AI Docker:

```
FP32 model (ONNX / PyTorch)
    │
    ▼  torch_quantizer("calib")
Calibration — forward pass on ~200 representative images
    │         (collects activation statistics for fix_point calculation)
    ▼  torch_quantizer("test") + export_quant_config()
INT8 quantized model (NNDCT simulation)
    │
    ▼  export_xmodel()
XIR graph (.xmodel) — hardware-mapped compute graph
    │
    ▼  vai_c_xir --arch DPUCZDX8G/KV260/arch.json
Compiled .xmodel — DPU binary ready for KV260
```

### Calibration details

| Model | Calibration images | Source |
|-------|-------------------|--------|
| SCRFD det_10g | 300 face images | Diverse lighting/pose subset |
| MobileFaceNet | 300 face crops | Aligned 112×112, varied identities |

---

## Accuracy: FP32 vs INT8

| Model | Dataset | FP32 | INT8 (KV260) | Drop |
|-------|---------|------|--------------|------|
| MobileFaceNet | LFW (6k pairs) | 99.33% | 99.10% | −0.23% |
| MobileFaceNet | CFP-FP (7k pairs) | 84.81% | 83.40% | −1.41% |
| MobileFaceNet | MLFW (6k pairs) | 73.15% | 70.50% | −2.65% |
| SCRFD det_10g | CASIA-FaceV5 (2500 imgs) | 100% det | 100% det | 0% |

### Model size

| Model | FP32 | INT8 xmodel | Compression |
|-------|------|-------------|-------------|
| SCRFD det_10g | ~28 MB (ONNX) | **5.2 MB** | 5.4× |
| MobileFaceNet | ~4.0 MB (PT) | **1.5 MB** | 2.7× |

### Inference latency on KV260 *(DPU-only, 50-run average)*

| Model | Latency | FPS |
|-------|---------|-----|
| SCRFD det_10g (640×640) | 31.8 ms | 31.4 |
| MobileFaceNet (112×112) | 1.3 ms | 758 |
| Pipeline (SCRFD + MFN) | ~33 ms | ~30 |

---

## Pitfalls & Analysis

### MLFW accuracy drop (−2.65%)
MLFW contains heavy occlusions and masks. The 128-dim MobileFaceNet embedding has limited
capacity for masked-face discrimination — this is a model-architecture limitation, not a
quantization issue. FP32 baseline itself scores only 73.15% on MLFW. The INT8 drop (−2.65%)
is within normal PTQ range for a compact model.

### FixedReshape (SCRFD PTQ crash)
The ONNX→PyTorch converter (onnx2torch) represents Reshape ops with a dynamic shape tensor
as a second input. Vitis AI's NNDCT quantizer traces this call and produces `aten::unbind`,
which the XIR compiler cannot map. Fix: capture the concrete shape via a forward hook during
float inference, then replace every `OnnxReshape` module with a `FixedReshape` that hard-codes
the shape and ignores the second argument.

### Calibration fix_point=None
After calibration, some tensor entries in `quant_info.json` have `fix_point = null`
(uncalibrated, typically batch-norm bias tensors never activated during calib).
Left unfixed, these cause NaN outputs at test time. Fix: set any null fix_point to 6
before loading the quantized model.

---

## Model Files

Place in `models/`:

| File | Size |
|------|------|
| `scrfd_det10g.xmodel` | 5.2 MB |
| `mobilefacenet.xmodel` | 1.5 MB |

---

## Quantization Scripts

`quantize/` contains the actual PTQ scripts run inside the Vitis AI Docker:

| Script | Description |
|--------|-------------|
| `quantize_scrfd.py` | SCRFD det_10g calibration + INT8 export |
| `compile_scrfd.sh` | `vai_c_xir` compile → `scrfd_det10g.xmodel` |
| `quantize_mfn.py` | MobileFaceNet calibration + INT8 export |
| `compile_mfn.sh` | `vai_c_xir` compile → `mobilefacenet.xmodel` |

Run inside the Vitis AI Docker (`xilinx/vitis-ai-cpu:latest`):

```bash
docker run -v $(pwd):/workspace xilinx/vitis-ai-cpu:latest bash
# inside container:
conda activate vitis-ai-pytorch
python quantize/quantize_scrfd.py    # → quantize_result/scrfd_det10g_int.xmodel
bash quantize/compile_scrfd.sh       # → compiled_model/scrfd_det10g.xmodel
```

---

## Setup (on KV260)

`vart` and `xir` are part of the KV260 PetaLinux image — no pip install needed.
Only OpenCV is required if not already present:

```bash
pip install opencv-python-headless
```

Load the DPU overlay before running (required after every reboot):

```bash
xmutil loadapp kv260-benchmark-b4096
```

## Usage

```bash
# Register persons (expects <testset>/<name>/real/*.jpg)
python demo.py register /home/root/testset

# Identify one image
python demo.py identify /path/to/face.jpg

# Accuracy smoke-test (register first 5, test next 3 per person)
python demo.py test /home/root/testset
```
