#!/usr/bin/env python3
"""
MobileFaceNet INT8 PTQ for KV260 (DPUCZDX8G) via pytorch_nndct.

Usage (inside Vitis-AI Docker):
    python3 quantize_mfn.py \
        --model  /workspace/mobilefacenet.pt \
        --calib  /workspace/calib_faces \
        --out    /workspace/quantize_result \
        --n-calib 300
"""
import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path

from mobilefacenet import MobileFaceNet

INPUT_H, INPUT_W = 112, 112


def preprocess(path: str) -> torch.Tensor:
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = (img.astype(np.float32) - 127.5) / 128.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,112,112)


def load_model(ckpt_path: str) -> torch.nn.Module:
    model = MobileFaceNet()
    state = torch.load(ckpt_path, map_location="cpu")
    # unwrap common wrapper keys
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="/workspace/mobilefacenet.pt")
    ap.add_argument("--calib",   default="/workspace/calib_faces")
    ap.add_argument("--out",     default="/workspace/quantize_result")
    ap.add_argument("--n-calib", type=int, default=300)
    args = ap.parse_args()

    from pytorch_nndct.apis import torch_quantizer

    print(f"[load] {args.model}")
    model = load_model(args.model)

    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)

    imgs = sorted(
        [str(p) for p in Path(args.calib).glob("*.jpg")]
        + [str(p) for p in Path(args.calib).glob("*.png")]
    )[:args.n_calib]
    print(f"[calib] {len(imgs)} images from {args.calib}")

    # ── Phase 1: calibration ─────────────────────────────────────────────────
    print("[1/2] Calibration...")
    quantizer = torch_quantizer("calib", model, (dummy,), output_dir=args.out)
    quant_model = quantizer.quant_model

    with torch.no_grad():
        for i, p in enumerate(imgs):
            t = preprocess(p)
            if t is None:
                continue
            quant_model(t)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(imgs)}")

    quantizer.export_quant_config()
    print(f"[1/2] Done → {args.out}/")

    # ── Phase 2: export xmodel ───────────────────────────────────────────────
    print("[2/2] Exporting xmodel...")
    quantizer = torch_quantizer("test", model, (dummy,), output_dir=args.out)
    quant_model = quantizer.quant_model

    with torch.no_grad():
        quant_model(dummy)

    quantizer.export_xmodel(output_dir=args.out, deploy_check=False)
    print(f"[2/2] Done → {args.out}/MobileFaceNet_int.xmodel")


if __name__ == "__main__":
    main()
