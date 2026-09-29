"""MobileFaceNet DPU runner — 1 DPU subgraph, 128-dim embedding."""
import numpy as np
import cv2
import xir
import vart

# ArcFace 5-point alignment destination (96×112 crop)
_ARCFACE_DST = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041],
], dtype=np.float32)


def _fix_scale(tensor):
    try:
        return float(2 ** tensor.get_attr("fix_point"))
    except Exception:
        return float(tensor.get_attr("scale"))


def align_face(bgr: np.ndarray, kps) -> np.ndarray:
    """Affine-align face to 112×112 using 5 landmarks."""
    src = np.array(kps, dtype=np.float32)
    dst = _ARCFACE_DST * np.array([112/96, 1.0])
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = (0, 0, w, h)
        return cv2.resize(bgr[y1:y2, x1:x2], (112, 112))
    return cv2.warpAffine(bgr, M, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0)


class MFNExtractor:
    def __init__(self, xmodel_path: str):
        graph = xir.Graph.deserialize(xmodel_path)
        for sg in graph.get_root_subgraph().toposort_child_subgraph():
            if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                self._runner = vart.Runner.create_runner(sg, "run")
                break
        self._in_t  = self._runner.get_input_tensors()
        self._out_t = self._runner.get_output_tensors()
        self._in_scale  = _fix_scale(self._in_t[0])
        self._out_scale = 1.0 / _fix_scale(self._out_t[0])

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Input: 112×112 BGR. Output: L2-normalised 128-dim float32 embedding."""
        img = cv2.cvtColor(cv2.resize(face_bgr, (112, 112)), cv2.COLOR_BGR2RGB)
        f32 = (img.astype(np.float32) - 127.5) / 128.0
        i8  = (f32[np.newaxis] * self._in_scale).clip(-128, 127).astype(np.int8)
        out = np.zeros(tuple(self._out_t[0].dims), dtype=np.int8)
        self._runner.wait(self._runner.execute_async([i8], [out]))
        emb = out.astype(np.float32).flatten() * self._out_scale
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-6 else emb
