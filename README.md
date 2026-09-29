# KV260 Face Demo

SCRFD (DPU) + MobileFaceNet (DPU)

## Model files needed (place in `models/`)

| File | Source |
|------|--------|
| `scrfd_det10g.xmodel` | `/home/root/models/scrfd_det10g.xmodel` |
| `mobilefacenet.xmodel` | `/home/root/models/mobilefacenet.xmodel` |

## Usage

```bash
# Register persons from testset
python demo.py register /home/root/testset

# Identify one image
python demo.py identify /path/to/face.jpg

# Accuracy smoke-test (register first 5, test next 3 per person)
python demo.py test /home/root/testset
```

## Accuracy summary

| Model | Metric | FP32 | INT8 (KV260) |
|-------|--------|------|--------------|
| MobileFaceNet | LFW | 99.33% | 99.10% |
| MobileFaceNet | CFP-FP | 84.81% | 83.40% |
| MobileFaceNet | MLFW | 73.15% | 70.50% |
| SCRFD det_10g | CASIA-V5 det rate | 100% | 100% |
