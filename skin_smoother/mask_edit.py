"""
Manual edits to the skin mask, painted with a brush.

A stroke is resolution-independent, so one set of strokes gives the same edit on a
preview and on the full-size photo:

    {"mode": "protect" or "skin",
     "size": brush diameter, as a share of the photo's long side,
     "soft": 0..1, how soft its edge is,
     "points": [x0, y0, x1, y1, ...], each 0..1 of the photo's width or height}

Strokes are painted in order into a layer running from -1 (protect) to 1 (skin), so a
later stroke wins where they cross. The layer is then applied to the mask:

    protect  never smooth here, and don't fill any spots here
    skin     smooth here, even where the automatic mask left it out

Painting skin doesn't look for spots there: spot detection stays automatic.
"""

from __future__ import annotations

import numpy as np
import cv2

LAYER_MAX = 2048      # longest side of the layer; edits are soft, so more adds nothing
MIN_SOFT_SIGMA = 0.7  # px; keeps even a hard brush's edge from stepping
PRESENT = 1e-3        # a layer value this small counts as no edit


def new_layer(w: int, h: int) -> np.ndarray:
    """An empty layer for a photo of this size: float32, at most LAYER_MAX on its long side."""
    s = min(1.0, LAYER_MAX / max(w, h))
    return np.zeros((max(1, round(h * s)), max(1, round(w * s))), np.float32)


def paint_stroke(layer: np.ndarray, stroke: dict) -> None:
    """Add one stroke to a layer, in place."""
    lh, lw = layer.shape
    coords = stroke["points"]
    if len(coords) < 2:
        return
    points = [(round(coords[i] * lw), round(coords[i + 1] * lh)) for i in range(0, len(coords) - 1, 2)]
    diameter = max(1.0, float(stroke["size"]) * max(lw, lh))
    thick = max(1, round(diameter))
    sigma = max(MIN_SOFT_SIGMA, float(stroke.get("soft", 0.0)) * diameter * 0.5)

    # Only the part of the layer this stroke can reach, with room for the blur.
    margin = int(np.ceil(thick / 2.0 + 4.0 * sigma)) + 2
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, x1 = max(0, min(xs) - margin), min(lw, max(xs) + margin + 1)
    y0, y1 = max(0, min(ys) - margin), min(lh, max(ys) + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return
    stamp = np.zeros((y1 - y0, x1 - x0), np.uint8)
    moved = [(x - x0, y - y0) for x, y in points]
    cv2.circle(stamp, moved[0], max(1, thick // 2), 255, -1, cv2.LINE_8)
    for a, b in zip(moved, moved[1:]):
        cv2.line(stamp, a, b, 255, thick, cv2.LINE_8)
    soft = cv2.GaussianBlur(stamp.astype(np.float32) / 255.0, (0, 0), sigma)
    target = 1.0 if stroke["mode"] == "skin" else -1.0
    layer[y0:y1, x0:x1] = layer[y0:y1, x0:x1] * (1.0 - soft) + target * soft


def layer_from(strokes, w: int, h: int) -> np.ndarray:
    layer = new_layer(w, h)
    for stroke in strokes:
        paint_stroke(layer, stroke)
    return layer


def region_of(layer: np.ndarray, w: int, h: int):
    """The edited part of a layer at this photo size: (x0, y0, float32 HxW in -1..1), or None."""
    lh, lw = layer.shape
    ys, xs = np.nonzero(np.abs(layer) > PRESENT)
    if len(ys) == 0:
        return None
    sx, sy = w / lw, h / lh
    a0, a1 = max(0, int(xs.min()) - 1), min(lw, int(xs.max()) + 2)
    b0, b1 = max(0, int(ys.min()) - 1), min(lh, int(ys.max()) + 2)
    x0, x1 = int(a0 * sx), min(w, int(np.ceil(a1 * sx)))
    y0, y1 = int(b0 * sy), min(h, int(np.ceil(b1 * sy)))
    edit = cv2.resize(layer[b0:b1, a0:a1], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    return x0, y0, edit


def apply_to_mask(mask: np.ndarray, region) -> np.ndarray:
    """A copy of a full-image mask with an edit applied."""
    if region is None:
        return mask
    x0, y0, edit = region
    out = mask.copy()
    here = out[y0:y0 + edit.shape[0], x0:x0 + edit.shape[1]]
    here[:] = np.where(edit > 0, here + edit * (1.0 - here), here * (1.0 + edit))
    return out


def apply_to_correction(corr, region):
    """A blemish correction (x0, y0, HxWx3) with protected areas taken out of it."""
    if corr is None or region is None:
        return corr
    cx, cy, values = corr
    x0, y0, edit = region
    ax0, ay0 = max(cx, x0), max(cy, y0)
    ax1, ay1 = min(cx + values.shape[1], x0 + edit.shape[1]), min(cy + values.shape[0], y0 + edit.shape[0])
    if ax1 <= ax0 or ay1 <= ay0:
        return corr
    out = values.copy()
    keep = 1.0 + np.minimum(edit[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0], 0.0)
    out[ay0 - cy:ay1 - cy, ax0 - cx:ax1 - cx] *= keep[:, :, None]
    return cx, cy, out
