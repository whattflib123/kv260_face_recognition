#!/usr/bin/env python3
"""
KV260 Face Demo — SCRFD (DPU) + MobileFaceNet (DPU)

Usage:
    # Register persons from test-set directory:
    python demo.py register /home/root/testset

    # Identify one image:
    python demo.py identify /path/to/image.jpg

    # Smoke-test (register first 5, identify next 3 per person):
    python demo.py test /home/root/testset
"""
import sys, cv2
from pathlib import Path
from pipeline import FacePipeline

MODELS = Path(__file__).parent / "models"
DB     = Path(__file__).parent / "face_db.json"


def load_pipeline():
    return FacePipeline(
        scrfd_model = str(MODELS / "scrfd_det10g.xmodel"),
        mfn_model   = str(MODELS / "mobilefacenet.xmodel"),
        db_path     = str(DB),
    )


def cmd_register(testset_dir: str):
    """
    Expects:
        <testset>/<person_name>/real/*.jpg
    Registers up to 5 real images per person.
    """
    p    = load_pipeline()
    root = Path(testset_dir)
    total = 0
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        real_dir = person_dir / "real"
        if not real_dir.exists():
            real_dir = person_dir
        imgs   = sorted(real_dir.glob("*.jpg"))[:5]
        frames = [cv2.imread(str(i)) for i in imgs if cv2.imread(str(i)) is not None]
        n = p.register(person_dir.name, frames)
        print(f"  {person_dir.name}: {n}/{len(frames)} faces registered")
        total += n
    p.save_db()
    print(f"\nDone. {total} embeddings, {len(p._db)} persons.")


def cmd_identify(image_path: str):
    p   = load_pipeline()
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"Cannot read {image_path}")
        sys.exit(1)
    r = p.identify(bgr)
    print(f"Person : {r['person'] or '(unknown)'}")
    print(f"Score  : {r['score']:.4f}")
    print(f"BBox   : {r['bbox']}")


def cmd_test(testset_dir: str):
    root = Path(testset_dir)
    p    = load_pipeline()

    test_imgs = {}
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        real_dir = person_dir / "real"
        if not real_dir.exists():
            real_dir = person_dir
        imgs = sorted(real_dir.glob("*.jpg"))
        reg  = [cv2.imread(str(i)) for i in imgs[:5] if cv2.imread(str(i)) is not None]
        if not reg:
            continue
        p.register(person_dir.name, reg)
        test_imgs[person_dir.name] = [cv2.imread(str(i)) for i in imgs[5:8] if cv2.imread(str(i)) is not None]

    print(f"\nRegistered {len(p._db)} persons")

    correct = total = 0
    for name, frames in test_imgs.items():
        for bgr in frames:
            r  = p.identify(bgr)
            ok = r["person"] == name
            total += 1
            correct += int(ok)
            print(f"  [{'✓' if ok else '✗'}] GT={name}  pred={r['person'] or '(unknown)'}  score={r['score']:.3f}")

    if total:
        print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "register" and len(sys.argv) == 3:
        cmd_register(sys.argv[2])
    elif cmd == "identify" and len(sys.argv) == 3:
        cmd_identify(sys.argv[2])
    elif cmd == "test" and len(sys.argv) == 3:
        cmd_test(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
