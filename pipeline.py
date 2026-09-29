"""Face pipeline: SCRFD (DPU) → MobileFaceNet (DPU)."""
import json
import numpy as np
from pathlib import Path
from typing import Optional

from scrfd import SCRFDDetector
from mfn   import MFNExtractor, align_face

_DEFAULT_MODELS = Path(__file__).parent / "models"
_SCRFD_XMODEL   = str(_DEFAULT_MODELS / "scrfd_det10g.xmodel")
_MFN_XMODEL     = str(_DEFAULT_MODELS / "mobilefacenet.xmodel")
_DB_PATH        = str(Path(__file__).parent / "face_db.json")

MATCH_THRESHOLD = 0.30


class FacePipeline:
    def __init__(
        self,
        scrfd_model: str   = _SCRFD_XMODEL,
        mfn_model:   str   = _MFN_XMODEL,
        db_path:     str   = _DB_PATH,
        match_thr:   float = MATCH_THRESHOLD,
    ):
        print("Loading SCRFD (DPU) ...")
        self._det = SCRFDDetector(scrfd_model)
        print("Loading MobileFaceNet (DPU) ...")
        self._emb = MFNExtractor(mfn_model)
        self._thr     = match_thr
        self._db_path = db_path
        self._db: dict[str, list] = {}
        if Path(db_path).exists():
            self.load_db(db_path)

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str, frames: list) -> int:
        """Register a person from BGR frames. Returns count added."""
        added = 0
        for bgr in frames:
            dets = self._det.detect(bgr)
            if not dets:
                continue
            det  = max(dets, key=lambda d: d["score"])
            face = align_face(bgr, det["kps"])
            emb  = self._emb.embed(face).tolist()
            self._db.setdefault(name, []).append(emb)
            added += 1
        return added

    # ── Identification ────────────────────────────────────────────────────────

    def identify(self, bgr) -> dict:
        """
        Returns:
          person : matched name or None
          score  : cosine similarity
          bbox   : [x1,y1,x2,y2] or None
        """
        dets = self._det.detect(bgr)
        if not dets:
            return {"person": None, "score": 0.0, "bbox": None}

        det  = max(dets, key=lambda d: d["score"])
        face = align_face(bgr, det["kps"])
        emb  = self._emb.embed(face)
        person, score = self._match(emb)
        return {"person": person, "score": score, "bbox": det["bbox"]}

    def _match(self, emb: np.ndarray):
        best_name, best_score = None, -1.0
        for name, embeddings in self._db.items():
            sims  = [float(np.dot(emb, np.array(e))) for e in embeddings]
            score = float(np.mean(sims))
            if score > best_score:
                best_score, best_name = score, name
        if best_score >= self._thr:
            return best_name, best_score
        return None, best_score

    # ── DB persistence ────────────────────────────────────────────────────────

    def save_db(self, path: Optional[str] = None):
        path = path or self._db_path
        with open(path, "w") as f:
            json.dump(self._db, f)
        print(f"DB saved → {path}  ({len(self._db)} people)")

    def load_db(self, path: Optional[str] = None):
        path = path or self._db_path
        with open(path) as f:
            self._db = json.load(f)
        print(f"DB loaded ← {path}  ({len(self._db)} people)")
