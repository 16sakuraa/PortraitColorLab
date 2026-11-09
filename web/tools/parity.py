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
  small      as pores, for a small print (coarser grain)  PSNR, % of pixels that differ
  shine      after smoothing with Reduce shine  PSNR, % of pixels that differ
  hard       after smoothing for hard lighting   PSNR, % of pixels that differ
  brush      after smoothing, with brush strokes  PSNR, % of pixels that differ
             (the mask's own difference is added to the worst mask difference)
  heal       after smoothing, with heal, spot, light, Smooth again and Undo smoothing among
             Restore and Add skin              PSNR, % of pixels that differ
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
import heal      # noqa: E402
import mask_edit  # noqa: E402

PARAMS = {"blemish": 1.0, "strength": 0.6, "texture": 0.6, "pores": 0.0}
PORES_STAGE = 1.0   # the pores stage's Add pores value: the strongest, so differences show
GRAIN_STAGE = smoother.PRINT_GRAIN["small"]   # the small stage's grain, with Add pores at PORES_STAGE
SHINE_STAGE = 1.0   # the shine stage's Reduce shine value: the strongest, so differences show
LIGHT_STAGE = smoother.LIGHTING["hard"]   # the hard stage's lighting
# The brush stage's strokes: some hard, some soft, crossing, across the middle of the photo.
BRUSH_STROKES = [
    {"id": 1, "mode": "protect", "size": 0.03, "soft": 0.3, "points": [0.3, 0.42, 0.5, 0.45, 0.7, 0.42]},
    {"id": 2, "mode": "skin", "size": 0.05, "soft": 0.5, "points": [0.2, 0.7, 0.45, 0.72, 0.6, 0.8, 0.8, 0.82]},
    {"id": 3, "mode": "protect", "size": 0.012, "soft": 0.0, "points": [0.5, 0.3, 0.52, 0.75]},
    {"id": 4, "mode": "skin", "size": 0.02, "soft": 0.2, "points": [0.1234, 0.1234]},
    # Part-strength Restore and Add skin, crossing the strokes above.
    {"id": 5, "mode": "protect", "size": 0.04, "soft": 0.4, "amount": 0.4, "points": [0.25, 0.55, 0.75, 0.5]},
    {"id": 6, "mode": "skin", "size": 0.03, "soft": 0.2, "amount": 0.6, "points": [0.4, 0.2, 0.45, 0.9]},
]
# The heal stage's strokes: a heal over an earlier Restore (it shows), a 40% Restore over
# a heal (60% of it stays), a large heal (worked out on a coarser grid on p_dslr), one
# whose source runs off the top of the photo, spots, and Undo smoothing over the first
# Restore and heal (the fills come back, the heal stays), with a Restore over that.
HEAL_STROKES = [
    {"id": 11, "mode": "protect", "size": 0.04, "soft": 0.3, "points": [0.4, 0.55, 0.6, 0.55]},
    {"id": 12, "mode": "heal", "size": 0.03, "soft": 0.3, "offset": [0.08, -0.1], "points": [0.45, 0.5, 0.5, 0.56, 0.55, 0.6]},
    {"id": 13, "mode": "heal", "size": 0.06, "soft": 0.5, "amount": 0.8, "offset": [-0.12, 0.03], "points": [0.3, 0.4, 0.35, 0.45]},
    {"id": 14, "mode": "protect", "size": 0.05, "soft": 0.4, "amount": 0.4, "points": [0.28, 0.42, 0.38, 0.42]},
    {"id": 15, "mode": "heal", "size": 0.04, "soft": 0.2, "offset": [0.0, -0.3], "points": [0.7, 0.15, 0.72, 0.2]},
    {"id": 16, "mode": "spot", "size": 0.02, "soft": 0.3, "points": [0.62, 0.38]},
    {"id": 17, "mode": "spot", "size": 0.015, "soft": 0.0, "amount": 0.7, "points": [0.42, 0.62, 0.43, 0.63]},
    {"id": 18, "mode": "skin", "size": 0.03, "soft": 0.2, "points": [0.5, 0.3, 0.55, 0.32]},
    {"id": 19, "mode": "unsmooth", "size": 0.05, "soft": 0.3, "points": [0.42, 0.5, 0.58, 0.58]},
    {"id": 20, "mode": "protect", "size": 0.02, "soft": 0.2, "points": [0.5, 0.52, 0.52, 0.56]},
    {"id": 21, "mode": "unsmooth", "size": 0.04, "soft": 0.4, "amount": 0.5, "points": [0.6, 0.35, 0.66, 0.4]},
    # Light brush: one with a 40% Restore and an Undo smoothing over it, a large one, and
    # one whose colour is picked at the photo's corner.
    {"id": 22, "mode": "light", "size": 0.05, "soft": 0.5, "amount": 0.3, "pick": [0.55, 0.45],
     "points": [0.4, 0.62, 0.5, 0.66, 0.6, 0.62]},
    {"id": 23, "mode": "protect", "size": 0.03, "soft": 0.3, "amount": 0.4, "points": [0.48, 0.6, 0.52, 0.68]},
    {"id": 24, "mode": "unsmooth", "size": 0.03, "soft": 0.3, "points": [0.42, 0.6, 0.44, 0.66]},
    {"id": 25, "mode": "light", "size": 0.08, "soft": 0.3, "amount": 0.2, "pick": [0.3, 0.3], "points": [0.25, 0.35, 0.3, 0.4]},
    {"id": 26, "mode": "light", "size": 0.02, "soft": 0.0, "amount": 0.5, "pick": [0.999, 0.001], "points": [0.65, 0.3]},
    # Smooth again: a full one, a 60% one, and a Restore over the first.
    {"id": 27, "mode": "resmooth", "size": 0.06, "soft": 0.4, "points": [0.35, 0.5, 0.45, 0.55]},
    {"id": 28, "mode": "resmooth", "size": 0.03, "soft": 0.3, "amount": 0.6, "points": [0.6, 0.6, 0.65, 0.65]},
    {"id": 29, "mode": "protect", "size": 0.02, "soft": 0.3, "points": [0.4, 0.52, 0.42, 0.54]},
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
        small = smoother.smooth_face(healed, PARAMS["strength"], PARAMS["texture"], mask, faces, PORES_STAGE,
                                     GRAIN_STAGE)
        shiny = smoother.smooth_face(healed, PARAMS["strength"], PARAMS["texture"], mask, faces, PARAMS["pores"],
                                     shine=SHINE_STAGE)
        hard = smoother.smooth_face(healed, PARAMS["strength"], PARAMS["texture"], mask, faces, PARAMS["pores"],
                                    light=LIGHT_STAGE)
        region = mask_edit.region_of(mask_edit.layer_from(BRUSH_STROKES, img.shape[1], img.shape[0]), img.shape[1], img.shape[0])
        brushed_mask = mask_edit.apply_to_mask(mask, region)
        brushed = smoother.smooth_face(blemish.apply(img, mask_edit.apply_to_correction(corr, region), PARAMS["blemish"]),
                                       PARAMS["strength"], PARAMS["texture"], brushed_mask, faces, PARAMS["pores"])
        region, fills, again = mask_edit.edits(HEAL_STROKES, img.shape[1], img.shape[0])
        filled = heal.apply(blemish.apply(img, mask_edit.apply_to_correction(corr, fills), PARAMS["blemish"]), HEAL_STROKES)
        heal_mask = mask_edit.apply_to_mask(mask, region)
        healed_brush = smoother.smooth_face(filled, PARAMS["strength"], PARAMS["texture"], heal_mask, faces, PARAMS["pores"])
        if again is not None:
            healed_brush = smoother.smooth_again(filled, healed_brush, again, heal_mask, faces, PARAMS["texture"],
                                                 PARAMS["pores"])
        healed_brush = heal.apply_light(healed_brush, HEAL_STROKES, smoother.detail_sigma(img.shape, faces))
        cv2.imwrite(os.path.join(REF, n + ".heal.png"), healed_brush)
        mask.astype(np.float32).tofile(os.path.join(REF, n + ".mask.f32"))
        brushed_mask.astype(np.float32).tofile(os.path.join(REF, n + ".mask_brush.f32"))
        cv2.imwrite(os.path.join(REF, n + ".healed.png"), healed)
        cv2.imwrite(os.path.join(REF, n + ".final.png"), final)
        cv2.imwrite(os.path.join(REF, n + ".pores.png"), with_pores)
        cv2.imwrite(os.path.join(REF, n + ".small.png"), small)
        cv2.imwrite(os.path.join(REF, n + ".shine.png"), shiny)
        cv2.imwrite(os.path.join(REF, n + ".hard.png"), hard)
        cv2.imwrite(os.path.join(REF, n + ".brush.png"), brushed)
        print(f"{n}: {blemish.count_text(count)}")
    with open(os.path.join(REF, "params.json"), "w") as fh:
        json.dump({**PARAMS, "pores_stage": PORES_STAGE, "grain_stage": GRAIN_STAGE, "shine_stage": SHINE_STAGE,
                   "light_stage": LIGHT_STAGE,
                   "brush_strokes": BRUSH_STROKES, "heal_strokes": HEAL_STROKES,
                   "blemish_engine": blemish.ENGINE}, fh)


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def compare() -> int:
    worst = float("inf")
    worst_mask = 0.0
    print(f"{'image':10s} {'mask max/mean diff':>20s} {'face only':>20s} {'healed':>24s} {'final':>24s} {'pores':>24s} "
          f"{'small':>24s} {'shine':>24s} {'hard':>24s} {'brush':>24s} {'heal':>24s}")
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
        for stage in ("healed", "final", "pores", "small", "shine", "hard", "brush", "heal"):
            a = cv2.imread(os.path.join(REF, f"{n}.{stage}.png"))
            b = cv2.imread(os.path.join(OUT, f"{n}.{stage}.png"))
            p = _psnr(a, b)
            diff = np.any(a != b, axis=2).mean() * 100
            cells.append("identical" if p == float("inf") else f"{p:5.1f} dB, {diff:5.2f}% px")
            if stage != "healed":
                worst = min(worst, p)
        print(f"{n:10s} {cells[0]:>20s} {cells[1]:>20s} {cells[2]:>24s} {cells[3]:>24s} {cells[4]:>24s} "
              f"{cells[5]:>24s} {cells[6]:>24s} {cells[7]:>24s} {cells[8]:>24s} {cells[9]:>24s}")
    print(f"worst smoothing-stage PSNR: {worst:.1f} dB; worst mean mask difference: {worst_mask:.5f}")
    return 0


if __name__ == "__main__":
    {"ref": ref, "compare": compare}[sys.argv[1]]()
