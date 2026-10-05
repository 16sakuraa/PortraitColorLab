"""
Before/after sheets using the real SkinSmoother code path (blemish.py + the
exported ONNX model on the CPU), plus a full-resolution timing test.

  python demo.py runs/run1/retouch.onnx

Writes, next to the .onnx:
  compare_heldout.jpg   held-out faces: original | this model | pro retoucher
  compare_studio.jpg    a studio headshot from the DM Portrait help images
  (timing printed)      whole pipeline on a 6000x4000 photo, CPU only
"""

from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skin_smoother"))
import smoother  # noqa: E402
import blemish   # noqa: E402

PAIRS = os.path.join(HERE, "data", "pairs")
HELP_IMG = os.path.join(HERE, "..", "help", "image", "last017 copy.jpg")


def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def heldout_sheet(out_path: str, n_busy: int = 5, n_typical: int = 3) -> None:
    with open(os.path.join(PAIRS, "index.json")) as fh:
        idx = json.load(fh)
    by = {p["id"]: p for p in idx["pairs"]}
    val = idx["val"]
    busy = sorted(val, key=lambda i: -len(by[i]["cells"]))[:n_busy]
    typical = [i for i in val[::37] if i not in busy][:n_typical]
    rows = []
    for i in busy + typical:
        o = cv2.imread(os.path.join(PAIRS, "original", f"{i:05d}.webp"))
        r = cv2.imread(os.path.join(PAIRS, "retouched", f"{i:05d}.png"))
        res, n = blemish.remove_blemishes(o, amount=1.0)
        cy, cx = np.median(np.array(by[i]["cells"]), axis=0).astype(int)
        y0 = int(np.clip(cy - 256, 0, 1024 - 512))
        x0 = int(np.clip(cx - 256, 0, 1024 - 512))
        sl = (slice(y0, y0 + 512), slice(x0, x0 + 512))
        rows.append(np.hstack([_label(o[sl], f"#{i:05d} original"),
                               _label(res[sl], "this model" + ("" if n else " (NO FACE)")),
                               _label(r[sl], "professional retoucher")]))
    cv2.imwrite(out_path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {out_path}")


def studio_sheet(out_path: str) -> None:
    img = cv2.imdecode(np.fromfile(HELP_IMG, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("help image not found; skipping studio sheet")
        return
    panel = img[60:455, 0:395]          # the un-retouched "before" photo
    big = cv2.resize(panel, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    res, n = blemish.remove_blemishes(big, amount=1.0)
    cv2.imwrite(out_path, np.hstack([_label(big, "original (help image, 2x)"),
                                     _label(res, f"this model ({n} face)")]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {out_path}")


def timing() -> None:
    """A 6000x4000 'DSLR' frame with one large face (a held-out FFHQ face
    scaled 2.5x), run through face detection + blemish removal."""
    with open(os.path.join(PAIRS, "index.json")) as fh:
        i = json.load(fh)["val"][0]
    face = cv2.imread(os.path.join(PAIRS, "original", f"{i:05d}.webp"))
    face = cv2.resize(face, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    canvas = np.full((4000, 6000, 3), 128, np.uint8)
    canvas[700:700 + face.shape[0], 1700:1700 + face.shape[1]] = face
    for threads in (cv2.getNumberOfCPUs(), 4):
        cv2.setNumThreads(threads)
        blemish.remove_blemishes(canvas[::4, ::4].copy())  # warm-up
        t0 = time.perf_counter()
        faces = smoother._detect_faces(canvas)
        t1 = time.perf_counter()
        _, n = blemish.remove_blemishes(canvas, faces=faces)
        t2 = time.perf_counter()
        print(f"6000x4000, {threads:2d} threads: face detection {t1-t0:.2f} s + "
              f"blemish removal {t2-t1:.2f} s  ({n} face)")
    cv2.setNumThreads(cv2.getNumberOfCPUs())


if __name__ == "__main__":
    onnx = os.path.abspath(sys.argv[1])
    blemish.get_model_path = lambda: onnx   # test this model, not the installed one
    out_dir = os.path.dirname(onnx)
    heldout_sheet(os.path.join(out_dir, "compare_heldout.jpg"))
    studio_sheet(os.path.join(out_dir, "compare_studio.jpg"))
    timing()
