"""
Desktop vs browser parity check.

  python web/tools/parity.py ref        # run the desktop code on web/test/p_*.png
  (run the browser side: open /tools/parity.html from devserver.py; it saves
   to web/test/out/. Add ?engine=wasm to test the CPU fallback.)
  python web/tools/parity.py compare    # stage-by-stage comparison

Stages compared, all with Remove blemishes = 1, Strength = 0.6, Keep texture
= 0.35 (the app's defaults):
  mask    the skin mask (0..1)              max / mean difference
  healed  after blemish removal             PSNR, % of pixels that differ
  final   after smoothing                   PSNR, % of pixels that differ
PSNR above ~45 dB is invisible; "identical" means every pixel matches.
"""

from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np
import cv2

WEB = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.dirname(WEB)
TEST, REF, OUT = (os.path.join(WEB, "test", d) for d in ("", "ref", "out"))
for cand in (os.path.join(ROOT, "skin_smoother"), os.path.join(ROOT, "..", "skin_smoother")):
    if os.path.exists(os.path.join(cand, "smoother.py")):
        sys.path.insert(0, cand)
        break
import smoother  # noqa: E402
import blemish   # noqa: E402

PARAMS = {"blemish": 1.0, "strength": 0.6, "texture": 0.35}


def names():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(TEST, "p_*.png")))


def ref() -> None:
    os.makedirs(REF, exist_ok=True)
    smoother.set_opencl(False)               # the browser has no OpenCL path
    for n in names():
        img = cv2.imread(os.path.join(TEST, n + ".png"))
        faces = smoother._detect_faces(img)
        mask = smoother.build_skin_mask(img)
        corr, count = blemish.correction(img, faces)
        healed = blemish.apply(img, corr, PARAMS["blemish"])
        final = smoother.smooth_face(healed, PARAMS["strength"], PARAMS["texture"], mask)
        mask.astype(np.float32).tofile(os.path.join(REF, n + ".mask.f32"))
        cv2.imwrite(os.path.join(REF, n + ".healed.png"), healed)
        cv2.imwrite(os.path.join(REF, n + ".final.png"), final)
        print(f"{n}: {count} face(s) healed")
    with open(os.path.join(REF, "params.json"), "w") as fh:
        json.dump(PARAMS, fh)


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def compare() -> int:
    worst = float("inf")
    print(f"{'image':10s} {'mask max/mean diff':>20s} {'healed':>24s} {'final':>24s}")
    for n in names():
        img = cv2.imread(os.path.join(TEST, n + ".png"))
        h, w = img.shape[:2]
        try:
            m_ref = np.fromfile(os.path.join(REF, n + ".mask.f32"), np.float32).reshape(h, w)
            m_web = np.fromfile(os.path.join(OUT, n + ".mask.f32"), np.float32).reshape(h, w)
        except FileNotFoundError:
            print(f"{n:10s} (missing browser output)")
            continue
        md = np.abs(m_ref - m_web)
        cells = [f"{md.max():.3f} / {md.mean():.5f}"]
        for stage in ("healed", "final"):
            a = cv2.imread(os.path.join(REF, f"{n}.{stage}.png"))
            b = cv2.imread(os.path.join(OUT, f"{n}.{stage}.png"))
            p = _psnr(a, b)
            diff = np.any(a != b, axis=2).mean() * 100
            cells.append("identical" if p == float("inf") else f"{p:5.1f} dB, {diff:5.2f}% px")
            if stage == "final":
                worst = min(worst, p)
        print(f"{n:10s} {cells[0]:>20s} {cells[1]:>24s} {cells[2]:>24s}")
    print(f"worst final PSNR: {worst:.1f} dB")
    return 0


if __name__ == "__main__":
    {"ref": ref, "compare": compare}[sys.argv[1]]()
