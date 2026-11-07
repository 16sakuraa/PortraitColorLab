"""
Desktop vs browser parity check.

  python web/tools/parity.py ref        # run the desktop code on web/test/p_*.png
  (run the browser side: open /tools/parity.html from devserver.py; it saves
   to web/test/out/. Add ?engine=wasm to test the CPU fallback.)
  python web/tools/parity.py compare    # stage-by-stage comparison

Set BLEMISH_ENGINE=network for the "ref" step to check the blemish network
instead of the default classic detector; the browser side follows whatever
"ref" used.

Stages compared, all with Remove blemishes = 1, Strength = 0.6, Keep texture
= 0.6, Add pores = 0 (the app's defaults):
  mask       the skin mask (0..1)              max / mean difference
  face only  the mask with Face only           max / mean difference
  healed     after blemish removal             PSNR, % of pixels that differ
  final      after smoothing                   PSNR, % of pixels that differ
  pores      after smoothing with Add pores    PSNR, % of pixels that differ
  brush      after smoothing, with brush strokes  PSNR, % of pixels that differ
             (the mask's own difference is added to the worst mask difference)
PSNR above ~45 dB is invisible; "identical" means every pixel matches. A mean
mask difference above 0.001 is worth looking into.
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
import mask_edit  # noqa: E402

PARAMS = {"blemish": 1.0, "strength": 0.6, "texture": 0.6, "pores": 0.0}
PORES_STAGE = 1.0   # the pores stage's Add pores value: the strongest, so differences show
# The brush stage's strokes: some hard, some soft, crossing, across the middle of the photo.
BRUSH_STROKES = [
    {"id": 1, "mode": "protect", "size": 0.03, "soft": 0.3, "points": [0.3, 0.42, 0.5, 0.45, 0.7, 0.42]},
    {"id": 2, "mode": "skin", "size": 0.05, "soft": 0.5, "points": [0.2, 0.7, 0.45, 0.72, 0.6, 0.8, 0.8, 0.82]},
    {"id": 3, "mode": "protect", "size": 0.012, "soft": 0.0, "points": [0.5, 0.3, 0.52, 0.75]},
    {"id": 4, "mode": "skin", "size": 0.02, "soft": 0.2, "points": [0.1234, 0.1234]},
]


def names():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(TEST, "p_*.png")))


def ref() -> None:
    os.makedirs(REF, exist_ok=True)
    for n in names():
        img = cv2.imread(os.path.join(TEST, n + ".png"))
        faces = smoother._detect_faces(img)
        mask = smoother.build_skin_mask(img, faces=faces)
        smoother.build_skin_mask(img, faces=faces, face_only=True).astype(np.float32).tofile(
            os.path.join(REF, n + ".mask_face.f32"))
        corr, count = blemish.correction(img, faces, mask)
        healed = blemish.apply(img, corr, PARAMS["blemish"])
        final = smoother.smooth_face(healed, PARAMS["strength"], PARAMS["texture"], mask, faces, PARAMS["pores"])
        with_pores = smoother.smooth_face(healed, PARAMS["strength"], PARAMS["texture"], mask, faces, PORES_STAGE)
        region = mask_edit.region_of(mask_edit.layer_from(BRUSH_STROKES, img.shape[1], img.shape[0]), img.shape[1], img.shape[0])
        brushed_mask = mask_edit.apply_to_mask(mask, region)
        brushed = smoother.smooth_face(blemish.apply(img, mask_edit.apply_to_correction(corr, region), PARAMS["blemish"]),
                                       PARAMS["strength"], PARAMS["texture"], brushed_mask, faces, PARAMS["pores"])
        mask.astype(np.float32).tofile(os.path.join(REF, n + ".mask.f32"))
        brushed_mask.astype(np.float32).tofile(os.path.join(REF, n + ".mask_brush.f32"))
        cv2.imwrite(os.path.join(REF, n + ".healed.png"), healed)
        cv2.imwrite(os.path.join(REF, n + ".final.png"), final)
        cv2.imwrite(os.path.join(REF, n + ".pores.png"), with_pores)
        cv2.imwrite(os.path.join(REF, n + ".brush.png"), brushed)
        print(f"{n}: {blemish.count_text(count)}")
    with open(os.path.join(REF, "params.json"), "w") as fh:
        json.dump({**PARAMS, "pores_stage": PORES_STAGE, "brush_strokes": BRUSH_STROKES, "blemish_engine": blemish.ENGINE}, fh)


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def compare() -> int:
    worst = float("inf")
    worst_mask = 0.0
    print(f"{'image':10s} {'mask max/mean diff':>20s} {'face only':>20s} {'healed':>24s} {'final':>24s} {'pores':>24s} {'brush':>24s}")
    for n in names():
        img = cv2.imread(os.path.join(TEST, n + ".png"))
        h, w = img.shape[:2]
        cells = []
        try:
            for stage in ("mask", "mask_face", "mask_brush"):
                m_ref = np.fromfile(os.path.join(REF, f"{n}.{stage}.f32"), np.float32).reshape(h, w)
                m_web = np.fromfile(os.path.join(OUT, f"{n}.{stage}.f32"), np.float32).reshape(h, w)
                md = np.abs(m_ref - m_web)
                if stage != "mask_brush":
                    cells.append(f"{md.max():.3f} / {md.mean():.5f}")
                worst_mask = max(worst_mask, float(md.mean()))
        except FileNotFoundError:
            print(f"{n:10s} (missing browser output)")
            continue
        for stage in ("healed", "final", "pores", "brush"):
            a = cv2.imread(os.path.join(REF, f"{n}.{stage}.png"))
            b = cv2.imread(os.path.join(OUT, f"{n}.{stage}.png"))
            p = _psnr(a, b)
            diff = np.any(a != b, axis=2).mean() * 100
            cells.append("identical" if p == float("inf") else f"{p:5.1f} dB, {diff:5.2f}% px")
            if stage != "healed":
                worst = min(worst, p)
        print(f"{n:10s} {cells[0]:>20s} {cells[1]:>20s} {cells[2]:>24s} {cells[3]:>24s} {cells[4]:>24s} {cells[5]:>24s}")
    print(f"worst final or pores PSNR: {worst:.1f} dB; worst mean mask difference: {worst_mask:.5f}")
    return 0


if __name__ == "__main__":
    {"ref": ref, "compare": compare}[sys.argv[1]]()
