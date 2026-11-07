"""
Side-by-side sheets for judging a retouch by eye (not committed: sheets go to
web/test/sheets/).

  python web/tools/contact_sheet.py compare [photo ...]
      Original, baseline, current and (if present) other reference versions of
      each test photo: the whole face, then a 100% crop of the cheek.
      Reads web/test/<photo>.png, web/test/baseline/, web/test/ref/ and any
      web/test/reference/<photo>.* file.

  python web/tools/contact_sheet.py grid PHOTO [strength=0.6] [texture=0.6] [pores=0] KEY=A,B,C [KEY2=D,E]
      Runs the desktop retouch with each value of a smoother.TUNE setting or
      slider: KEY across, KEY2 down. Every cell is the same 100% cheek crop.

  python web/tools/contact_sheet.py engines [photo ...]
      The two blemish engines side by side. Top: the whole face with smoothing
      at the app's defaults. Bottom: the original with what each engine changes
      outlined in green.

  python web/tools/contact_sheet.py masks NAME [photo ...]
      What the skin mask covers: the photo, the older method, the segmentation
      model, and the model with Face only. Anything not treated is dimmed.
      A photo is a name in web/test/ or a path. Saved as masks_NAME.png.
"""

from __future__ import annotations
import glob
import itertools
import os
import sys

import numpy as np
import cv2

WEB = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST = os.path.join(WEB, "test")
SHEETS = os.path.join(TEST, "sheets")
sys.path.insert(0, os.path.join(os.path.dirname(WEB), "skin_smoother"))
import smoother  # noqa: E402
import blemish   # noqa: E402

PHOTOS = ["p_user", "p_02709", "p_00044", "p_00078", "p_group", "p_dslr"]
FACE_H = 560     # height the whole-face crop is shown at
CHEEK = 280      # side of the 100% cheek crop
MASK_H = 300     # height of a photo on a mask sheet
LABEL_H = 26


def _largest_face(faces) -> np.ndarray:
    return max(faces, key=lambda f: float(f[2] * f[3]))


def _face_box(face, img) -> tuple[int, int, int, int]:
    x, y, w, h = (float(v) for v in face[:4])
    h_img, w_img = img.shape[:2]
    x0, x1 = int(max(0, x - 0.25 * w)), int(min(w_img, x + 1.25 * w))
    y0, y1 = int(max(0, y - 0.3 * h)), int(min(h_img, y + 1.25 * h))
    return x0, y0, x1, y1


def _cheek_box(face, img, size: int = CHEEK) -> tuple[int, int, int, int]:
    """A square on the cheek: below one eye, level with the nose."""
    ex, ey, nx, ny = (float(face[k]) for k in (4, 5, 8, 9))
    cx, cy = (ex + (ex + nx) / 2) / 2, (ey + 2 * ny) / 3
    h_img, w_img = img.shape[:2]
    size = min(size, h_img, w_img)
    x0 = int(np.clip(cx - size / 2, 0, w_img - size))
    y0 = int(np.clip(cy - size / 2, 0, h_img - size))
    return x0, y0, x0 + size, y0 + size


def _crop(img, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def _labelled(cell: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((LABEL_H, cell.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, cell])


def _save(sheet: np.ndarray, name: str) -> None:
    os.makedirs(SHEETS, exist_ok=True)
    path = os.path.join(SHEETS, name)
    cv2.imwrite(path, sheet)
    print(path)


def _versions(name: str) -> list[tuple[str, np.ndarray]]:
    found = [("original", os.path.join(TEST, name + ".png")),
             ("baseline", os.path.join(TEST, "baseline", name + ".final.png")),
             ("current", os.path.join(TEST, "ref", name + ".final.png"))]
    found += [(os.path.basename(p).split(".")[0] + " ref", p)
              for p in sorted(glob.glob(os.path.join(TEST, "reference", name + ".*")))]
    return [(label, img) for label, img in ((l, cv2.imread(p)) for l, p in found) if img is not None]


def compare(names: list[str]) -> None:
    for name in names or PHOTOS:
        versions = _versions(name)
        if not versions:
            print(f"{name}: no images found")
            continue
        faces = smoother._detect_faces(versions[0][1])
        if faces is None or len(faces) == 0:
            print(f"{name}: no face found")
            continue
        face = _largest_face(faces)
        top, bottom = [], []
        for label, img in versions:
            whole = _crop(img, _face_box(face, img))
            scale = FACE_H / whole.shape[0]
            whole = cv2.resize(whole, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            top.append(_labelled(whole, label))
            cheek = _crop(img, _cheek_box(face, img))
            pad = np.zeros((cheek.shape[0], max(0, whole.shape[1] - cheek.shape[1]), 3), np.uint8)
            bottom.append(_labelled(np.hstack([cheek, pad]), label + " 100%"))
        _save(np.vstack([np.hstack(top), np.hstack(bottom)]), f"compare_{name}.png")


def _outlined(img: np.ndarray, corr) -> np.ndarray:
    """A copy of img with the noticeable parts of a correction outlined."""
    out = img.copy()
    if corr is not None:
        x0, y0, c = corr
        changed = (np.abs(c).max(axis=2) > 4).astype(np.uint8)
        contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(out, [contour + np.array([x0, y0])], -1, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def engines(names: list[str]) -> None:
    saved = blemish.ENGINE
    try:
        for name in names or PHOTOS:
            img = cv2.imread(os.path.join(TEST, name + ".png"))
            if img is None:
                print(f"{name}: no image found")
                continue
            faces = smoother._detect_faces(img)
            if faces is None or len(faces) == 0:
                print(f"{name}: no face found")
                continue
            mask = smoother.build_skin_mask(img)
            face = _largest_face(faces)
            whole_box = _face_box(face, img)
            scale = FACE_H / (whole_box[3] - whole_box[1])
            size = lambda a: cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

            top = [_labelled(size(_crop(img, whole_box)), "original")]
            bottom = []
            for engine, title in (("network", "network"), ("classic", "non-AI")):
                blemish.ENGINE = engine
                corr, count = blemish.correction(img, faces, mask)
                healed = blemish.apply(img, corr, 1.0)
                final = smoother.smooth_face(healed, 0.6, 0.6, mask, faces)
                top.append(_labelled(size(_crop(final, whole_box)), f"{title}: {blemish.count_text(count)}"))
                bottom.append(_labelled(size(_crop(_outlined(img, corr), whole_box)), f"{title}: what it changes"))
            pad = np.zeros_like(top[0])
            _save(np.vstack([np.hstack(top), np.hstack([pad] + bottom)]), f"engines_{name}.png")
    finally:
        blemish.ENGINE = saved


def masks(name: str, photos: list[str]) -> None:
    import skin_segment
    rows = []
    for photo in photos or PHOTOS:
        path = photo if os.path.exists(photo) else os.path.join(TEST, photo + ".png")
        img = cv2.imread(path)
        if img is None:
            print(f"{photo}: no image found")
            continue
        faces = smoother._detect_faces(img)
        original = skin_segment.segment
        try:
            skin_segment.segment = lambda *_: None
            older = smoother.build_skin_mask(img, faces=faces)
        finally:
            skin_segment.segment = original
        scale = MASK_H / img.shape[0]
        size = (max(1, round(img.shape[1] * scale)), MASK_H)

        def shown(mask, label):
            kept = img.astype(np.float32) * (0.25 + 0.75 * mask[:, :, None])
            return _labelled(cv2.resize(kept.astype(np.uint8), size, interpolation=cv2.INTER_AREA), label)

        rows.append(np.hstack([
            _labelled(cv2.resize(img, size, interpolation=cv2.INTER_AREA), os.path.basename(path)),
            shown(older, "older method"),
            shown(smoother.build_skin_mask(img, faces=faces), "model"),
            shown(smoother.build_skin_mask(img, faces=faces, face_only=True), "model, Face only"),
        ]))
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)]) for r in rows]
    _save(np.vstack(rows), f"masks_{name}.png")


def _parse(args: list[str]) -> tuple[dict, list[tuple[str, list[float]]]]:
    fixed = {"strength": 0.6, "texture": 0.6, "pores": 0.0}
    axes = []
    for arg in args:
        key, _, vals = arg.partition("=")
        values = [float(v) for v in vals.split(",")]
        if key in fixed and len(values) == 1:
            fixed[key] = values[0]
        elif key in fixed or key in smoother.TUNE:
            axes.append((key, values))
        else:
            raise SystemExit(f"unknown setting {key}; choose from {sorted(smoother.TUNE)}")
    if not axes or len(axes) > 2:
        raise SystemExit("give one or two settings to vary, like luma_radius=0.01,0.02")
    return fixed, axes


def grid(photo: str, args: list[str]) -> None:
    fixed, axes = _parse(args)
    img = cv2.imread(os.path.join(TEST, photo + ".png"))
    faces = smoother._detect_faces(img)
    mask = smoother.build_skin_mask(img)
    corr, _ = blemish.correction(img, faces, mask)
    healed = blemish.apply(img, corr, 1.0)
    box = _cheek_box(_largest_face(faces), img)
    across = axes[0]
    down = axes[1] if len(axes) > 1 else (None, [None])
    saved = dict(smoother.TUNE)
    rows = []
    try:
        for dv in down[1]:
            row = []
            for av in across[1]:
                smoother.TUNE.update(saved)
                sliders = dict(fixed)
                pairs = [(across[0], av)] + ([(down[0], dv)] if dv is not None else [])
                for key, value in pairs:
                    (sliders if key in sliders else smoother.TUNE)[key] = value
                out = smoother.smooth_face(healed, sliders["strength"], sliders["texture"], mask, faces, sliders["pores"])
                label = "  ".join(f"{k}={v:g}" for k, v in pairs)
                row.append(_labelled(_crop(out, box), label))
            rows.append(np.hstack(row))
    finally:
        smoother.TUNE.update(saved)
    keys = "_".join(k for k, _ in axes)
    _save(np.vstack(rows), f"grid_{photo}_{keys}.png")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("compare", "grid", "engines", "masks"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "compare":
        compare(sys.argv[2:])
    elif sys.argv[1] == "engines":
        engines(sys.argv[2:])
    elif sys.argv[1] == "masks":
        masks(sys.argv[2], sys.argv[3:])
    else:
        grid(sys.argv[2], sys.argv[3:])
