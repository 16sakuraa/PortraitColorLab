"""
How the blemish button behaves outside the single-headshot case.

  python test_scenarios.py

  A. photos with no face (calendar templates, studio backdrops in ../Image)
  B. a tight synthetic group photo: 6 held-out faces shoulder to shoulder
  C. one face at full-body / half-body sizes inside a 6000x4000 frame
  D. FFHQ photos that really contain two people

Sheets are written to runs/scenarios/.
"""

from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skin_smoother"))
import smoother  # noqa: E402
import blemish   # noqa: E402

PAIRS = os.path.join(HERE, "data", "pairs")
OUT = os.path.join(HERE, "runs", "scenarios")


def _read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def _orig(i):
    return cv2.imread(os.path.join(PAIRS, "original", f"{i:05d}.webp"))


def _change_map(before, after):
    """Visual map of what the button changed (x4, grey = untouched)."""
    d = after.astype(np.int16) - before
    return np.clip(128 + 4 * d, 0, 255).astype(np.uint8)


def _fit(img, w):
    return cv2.resize(img, (w, round(img.shape[0] * w / img.shape[1])),
                      interpolation=cv2.INTER_AREA)


def no_face():
    print("A. photos with no face")
    # Any folder of photos without faces; here, DM Portrait's backdrop images
    # next to the repo (not part of it).
    for p in sorted(glob.glob(os.path.join(HERE, "..", "..", "Image", "*.jpg"))):
        img = _read(p)
        faces = smoother._detect_faces(img)
        out, n = blemish.remove_blemishes(img)
        diff = np.abs(out.astype(np.int16) - img).max()
        name = os.path.basename(p).encode("ascii", "replace").decode()
        print(f"   {name:32s} detected {0 if faces is None else len(faces)}, "
              f"processed {n}, max pixel change {diff}")


def group():
    print("B. tight group photo (6 faces shoulder to shoulder)")
    with open(os.path.join(PAIRS, "index.json")) as fh:
        val = json.load(fh)["val"]
    tiles = []
    for i in val[10:16]:
        # Middle of an FFHQ frame = the face, cropped tight so neighbours touch.
        tiles.append(cv2.resize(_orig(i)[200:900, 250:774], (420, 560),
                                interpolation=cv2.INTER_AREA))
    row = np.hstack(tiles)
    canvas = np.full((900, row.shape[1] + 200, 3), 200, np.uint8)
    canvas[170:170 + row.shape[0], 100:100 + row.shape[1]] = row
    faces = smoother._detect_faces(canvas)
    quads = blemish.select_faces(faces) if faces is not None else []
    out, n = blemish.remove_blemishes(canvas)
    print(f"   detected {0 if faces is None else len(faces)}, selected {len(quads)}, "
          f"processed {n} (expected 6)")
    vis = canvas.copy()
    for q in quads:
        cv2.polylines(vis, [q.astype(np.int32)], True, (0, 0, 255), 2)
    sheet = np.vstack([vis, out, _change_map(canvas, out)])
    cv2.imwrite(os.path.join(OUT, "group.jpg"), _fit(sheet, 1600), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return canvas, out


def small_faces():
    print("C. one face at smaller sizes in a 6000x4000 frame")
    with open(os.path.join(PAIRS, "index.json")) as fh:
        i = json.load(fh)["val"][3]
    face = _orig(i)
    ref, _ = blemish.remove_blemishes(face)
    ref_change = np.abs(ref.astype(np.float32) - face).mean()
    rows = []
    for size in (1024, 512, 256, 160):
        f = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
        canvas = np.full((4000, 6000, 3), 150, np.uint8)
        canvas[1500:1500 + size, 2500:2500 + size] = f
        out, n = blemish.remove_blemishes(canvas)
        crop_o = canvas[1500:1500 + size, 2500:2500 + size]
        crop_r = out[1500:1500 + size, 2500:2500 + size]
        ch = np.abs(crop_r.astype(np.float32) - crop_o).mean()
        print(f"   FFHQ frame {size:4d}px (face ~{size//2}px wide): processed {n}, "
              f"avg change {ch:.2f} (full-size reference {ref_change:.2f})")
        z = lambda a: cv2.resize(a, (400, 400), interpolation=cv2.INTER_CUBIC)
        rows.append(np.hstack([z(crop_o), z(crop_r), z(_change_map(crop_o, crop_r))]))
    cv2.imwrite(os.path.join(OUT, "small_faces.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def two_people():
    print("D. FFHQ photos with two people")
    for i in (44, 47):
        img = _orig(i)
        faces = smoother._detect_faces(img)
        quads = blemish.select_faces(faces) if faces is not None else []
        out, n = blemish.remove_blemishes(img)
        print(f"   #{i:05d}: detected {0 if faces is None else len(faces)}, processed {n}")
        vis = img.copy()
        for q in quads:
            cv2.polylines(vis, [q.astype(np.int32)], True, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(OUT, f"two_people_{i:05d}.jpg"),
                    _fit(np.hstack([vis, out, _change_map(img, out)]), 1500),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    no_face()
    group()
    small_faces()
    two_people()
