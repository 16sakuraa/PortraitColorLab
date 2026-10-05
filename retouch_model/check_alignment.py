"""
Check that SkinSmoother's face cropping reproduces the FFHQ framing.

FFHQ photos are already aligned crops, so running our YuNet-based alignment on
them should give back (almost) the whole 1024x1024 frame. The corner error
tells us how far real-photo crops will be from what the model was trained on;
training uses +/-25% scale jitter, so a few % is fine.

  python check_alignment.py [N]
"""

from __future__ import annotations
import os
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skin_smoother"))
import smoother  # noqa: E402
import blemish   # noqa: E402

ORIG = os.path.join(HERE, "data", "pairs", "original")


def main(n: int = 200) -> None:
    files = sorted(os.listdir(ORIG))[:n]
    target = np.array([[0, 0], [0, 1024], [1024, 1024], [1024, 0]], float)
    errs, scales, missed = [], [], 0
    for fn in files:
        img = cv2.imread(os.path.join(ORIG, fn), cv2.IMREAD_COLOR)
        faces = smoother._detect_faces(img)
        if faces is None or len(faces) == 0:
            missed += 1
            continue
        f = max(faces, key=lambda r: r[2] * r[3])   # main face
        q = blemish.ffhq_quad(f)
        errs.append(np.abs(q - target).mean())
        scales.append(np.hypot(*(q[3] - q[0])) / 1024)
    errs, scales = np.array(errs), np.array(scales)
    print(f"faces found {len(errs)}/{len(files)}  (missed {missed})")
    print(f"corner error: median {np.median(errs):.1f} px  p90 {np.percentile(errs, 90):.1f} px "
          f"(of 1024)")
    print(f"crop scale vs FFHQ: median {np.median(scales):.3f}  "
          f"p10 {np.percentile(scales, 10):.3f}  p90 {np.percentile(scales, 90):.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
