#!/usr/bin/env python3
"""
SCRFD det_10g INT8 PTQ for KV260 (DPUCZDX8G) via pytorch_nndct.

Usage (inside Vitis-AI Docker):
    python3 quantize_det.py \
        --model  /workspace/det_10g.onnx \
        --calib  /workspace/calib_faces \
        --out    /workspace/quantize_result_det \
        --n-calib 300
"""
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

INPUT_H, INPUT_W = 640, 640


def preprocess(path: str) -> torch.Tensor:
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = (img.astype(np.float32) - 127.5) / 128.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,640,640)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="/workspace/det_10g.onnx")
    ap.add_argument("--calib",   default="/workspace/calib_faces")
    ap.add_argument("--out",     default="/workspace/quantize_result_det")
    ap.add_argument("--n-calib", type=int, default=300)
    args = ap.parse_args()

    import onnx2torch
    from pytorch_nndct.apis import torch_quantizer

    print(f"[convert] {args.model}", flush=True)
    model = onnx2torch.convert(args.model)
    model.eval()

    # pytorch_nndct tries to set requires_grad on ALL nn.Parameters, including
    # integer constants (reshape shapes, slice indices) from ONNX → demote them
    # to buffers so nndct skips them.
    _demoted = 0
    for mod in model.modules():
        for pname, param in list(mod._parameters.items()):
            if param is not None and not param.is_floating_point():
                del mod._parameters[pname]
                mod.register_buffer(pname, param.data)
                _demoted += 1
    print(f"  conversion ok (demoted {_demoted} int params → buffers)", flush=True)

    # NNDCT traces nn.Buffer as value.node=None (external subgraph input), not a const
    # node, causing binary_op() to crash with "'NoneType' object has no attribute 'name'".
    # Fix: promote ALL non-BN float buffers to nn.Parameter so NNDCT builds const nodes.
    # BN running_mean/running_var stay as buffers (NNDCT handles BN ops internally).
    _BN_STATS = {'running_mean', 'running_var', 'num_batches_tracked'}
    _p_from_buf = 0
    for mod in model.modules():
        for bname in list(mod._buffers.keys()):
            if bname in _BN_STATS:
                continue
            buf = mod._buffers[bname]
            if buf is not None and buf.is_floating_point():
                # 0-d tensors → [1] to avoid XIR zero-dim assertion; others keep shape
                val = buf.unsqueeze(0) if buf.dim() == 0 else buf
                del mod._buffers[bname]
                mod.register_parameter(
                    bname, torch.nn.Parameter(val.clone().detach(), requires_grad=False))
                _p_from_buf += 1
    if _p_from_buf:
        print(f"  promoted {_p_from_buf} float buffer(s) → nn.Parameter", flush=True)

    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)

    # onnx2torch's OnnxReshape.forward() calls shape.tolist() which JIT-traces as
    # aten::unbind + aten::Int ops.  NNDCT warns but XIR crashes when it tries to
    # auto-define these unknown op types.  Fix: replace every OnnxReshape with a
    # FixedReshape that pre-bakes the shape tuple so JIT only sees reshape(const_tuple).
    class FixedReshape(torch.nn.Module):
        def __init__(self, shape):
            super().__init__()
            self._shape = shape
        def forward(self, x, *_):
            return x.reshape(self._shape)

    _shape_cache = {}
    _hooks = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == 'OnnxReshape':
            def _make_hook(n):
                def _hook(module, args, output):
                    if len(args) >= 2 and isinstance(args[1], torch.Tensor):
                        _shape_cache[n] = tuple(args[1].cpu().tolist())
                return _hook
            _hooks.append(mod.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        outs = model(dummy)

    for h in _hooks:
        h.remove()

    _replaced = 0
    for name, shape in _shape_cache.items():
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], FixedReshape(shape))
        _replaced += 1
    if _replaced:
        print(f"  replaced {_replaced} OnnxReshape → FixedReshape (no unbind/Int ops)", flush=True)

    n_out = len(outs) if isinstance(outs, (list, tuple)) else 1
    print(f"  outputs: {n_out}", flush=True)

    imgs = sorted(
        [str(p) for p in Path(args.calib).glob("*.jpg")]
        + [str(p) for p in Path(args.calib).glob("*.png")]
    )[:args.n_calib]
    print(f"[calib] {len(imgs)} images from {args.calib}", flush=True)

    # ── Phase 1: calibration ─────────────────────────────────────────────────
    print("[1/2] Calibration...", flush=True)
    quantizer = torch_quantizer("calib", model, (dummy,), output_dir=args.out)
    quant_model = quantizer.quant_model

    with torch.no_grad():
        for i, p in enumerate(imgs):
            t = preprocess(p)
            if t is None:
                continue
            quant_model(t)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(imgs)}", flush=True)

    # Some onnx2torch constants land in _QuantInfo['param'] but were never calibrated
    # (fix_point=None).  _check_calibration_completion() rejects None fix_points, and
    # removing them causes Phase 2 to treat the tensor as an external input (node=None),
    # crashing binary_op() in xop_creator.  Fix: assign fix_point=6 (range ±2, adequate
    # for scale factors ~0.85–1.1) so they pass the check AND stay in quant_info.json,
    # allowing Phase 2 to build proper Param const nodes for them.
    _q = quantizer.processor.quantizer
    _fixed = 0
    for pname in list(_q._QuantInfo['param'].keys()):
        bnfp = _q._QuantInfo['param'][pname]
        if bnfp is None or bnfp[1] is None:
            bit_width = (bnfp[0] if bnfp is not None and bnfp[0] is not None else 8)
            _q._QuantInfo['param'][pname] = [bit_width, 6]
            print(f"  [fixup] set fix_point=6 for uncalibrated param: {pname}", flush=True)
            _fixed += 1
    if _fixed:
        print(f"  [fixup] fixed {_fixed} uncalibrated param(s) with default fix_point", flush=True)

    quantizer.export_quant_config()
    print(f"[1/2] Done → {args.out}/", flush=True)

    # ── Phase 2: export xmodel ───────────────────────────────────────────────
    print("[2/2] Exporting xmodel...", flush=True)
    quantizer = torch_quantizer("test", model, (dummy,), output_dir=args.out)
    quant_model = quantizer.quant_model

    with torch.no_grad():
        quant_model(dummy)

    quantizer.export_xmodel(output_dir=args.out, deploy_check=False)
    print(f"[2/2] Done → {args.out}/", flush=True)


if __name__ == "__main__":
    main()
