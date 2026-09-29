"""SCRFD det_10g DPU runner — 1 DPU subgraph."""
import numpy as np
import cv2
import xir
import vart

_STRIDES = [8, 16, 32]
_NUM_ANCHORS = 2


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _fix_scale(tensor):
    try:
        return float(2 ** tensor.get_attr("fix_point"))
    except Exception:
        return float(tensor.get_attr("scale"))


class SCRFDDetector:
    def __init__(self, xmodel_path: str, conf_thresh: float = 0.5, nms_thresh: float = 0.4):
        self._conf = conf_thresh
        self._nms  = nms_thresh
        graph = xir.Graph.deserialize(xmodel_path)
        for sg in graph.get_root_subgraph().toposort_child_subgraph():
            if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                self._runner = vart.Runner.create_runner(sg, "run")
                break
        self._in_t  = self._runner.get_input_tensors()
        self._out_t = self._runner.get_output_tensors()

    def detect(self, bgr: np.ndarray):
        """Returns list of dicts: {bbox:[x1,y1,x2,y2], score:float, kps:[[x,y]×5]}"""
        h, w = bgr.shape[:2]
        img = cv2.cvtColor(cv2.resize(bgr, (640, 640)), cv2.COLOR_BGR2RGB)
        inp = (img.astype(np.float32) - 127.5) / 128.0
        inp_i8 = (inp[np.newaxis] * _fix_scale(self._in_t[0])).clip(-128, 127).astype(np.int8)

        out_bufs = [np.zeros(tuple(t.dims), dtype=np.int8) for t in self._out_t]
        self._runner.wait(self._runner.execute_async([inp_i8], out_bufs))

        # group outputs by spatial size
        by_hw = {}
        for t, b in zip(self._out_t, out_bufs):
            f = b.astype(np.float32) / _fix_scale(t)
            key = (f.shape[1], f.shape[2])
            by_hw.setdefault(key, []).append(f)

        boxes, scores, kps_list = [], [], []
        for stride in _STRIDES:
            feat_h, feat_w = 640 // stride, 640 // stride
            group = by_hw.get((feat_h, feat_w), [])
            if len(group) < 3:
                continue
            sc_t  = next(g for g in group if g.shape[-1] == _NUM_ANCHORS)
            bx_t  = next(g for g in group if g.shape[-1] == _NUM_ANCHORS * 4)
            kp_t  = next(g for g in group if g.shape[-1] == _NUM_ANCHORS * 10)

            sc  = _sigmoid(sc_t[0])                # (H, W, 2)
            bx  = bx_t[0]                          # (H, W, 8)
            kp  = kp_t[0]                          # (H, W, 20)

            for a in range(_NUM_ANCHORS):
                mask = sc[:, :, a] >= self._conf
                ys, xs = np.where(mask)
                if len(ys) == 0:
                    continue

                cx = (xs + 0.5) * stride
                cy = (ys + 0.5) * stride
                off = a * 4
                x1 = cx - bx[ys, xs, off + 0] * stride
                y1 = cy - bx[ys, xs, off + 1] * stride
                x2 = cx + bx[ys, xs, off + 2] * stride
                y2 = cy + bx[ys, xs, off + 3] * stride

                koff = a * 10
                pk = kp[ys, xs, koff:koff + 10].reshape(-1, 5, 2)
                pk[:, :, 0] = cx[:, None] + pk[:, :, 0] * stride
                pk[:, :, 1] = cy[:, None] + pk[:, :, 1] * stride

                for i in range(len(ys)):
                    boxes.append([x1[i], y1[i], x2[i], y2[i]])
                    scores.append(float(sc[ys[i], xs[i], a]))
                    kps_list.append(pk[i].tolist())

        if not boxes:
            return []

        boxes  = np.array(boxes)
        scores = np.array(scores)
        keep   = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self._conf, self._nms
        )
        if len(keep) == 0:
            return []
        keep = keep.flatten()

        sx, sy = w / 640.0, h / 640.0
        results = []
        for i in keep:
            b = boxes[i]
            k = kps_list[i]
            results.append({
                "bbox":  [int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)],
                "score": float(scores[i]),
                "kps":   [[p[0]*sx, p[1]*sy] for p in k],
            })
        return results
