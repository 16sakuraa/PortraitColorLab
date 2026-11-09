"""
Manual edits to the skin mask, painted with a brush.

A stroke is resolution-independent, so one set of strokes gives the same edit on a
preview and on the full-size photo:

    {"mode": "protect", "unsmooth" or "skin",
     "size": brush diameter, as a share of the photo's long side,
     "soft": 0..1, how soft its edge is,
     "amount": 0..1, how far to go (optional, 1 if missing),
     "points": [x0, y0, x1, y1, ...], each 0..1 of the photo's width or height}

Strokes are painted in order into a layer running from -1 (protect) to 1 (skin), each
towards its own amount, so a later stroke wins where they cross: a 40% stroke over a
100% one leaves 40%. The layer is then applied to the mask:

    protect   shown as Restore: bring back this share of the original. At 1, never
              smooth here and don't fill any spots here
    unsmooth  shown as Undo smoothing: like Restore on the mask, but the spot fills
              (and heal.py's heals) stay
    skin      smooth here, even where the automatic mask left it out

Painting skin doesn't look for spots there: spot detection stays automatic.

The spot fills follow a second layer, the same as the first until an unsmooth stroke,
which paints it back towards no edit: so Undo smoothing over a Restore brings the fills
back, and a Restore over it takes them out again.

    resmooth  shown as Smooth again: smooth here once more with a finer grain
              (smoother.smooth_again), even where the mask left the skin out

Smooth again follows a third layer, from 0 to 1, painted towards each resmooth stroke's
amount; Restore and Undo smoothing strokes after it paint it back to 0.

The heal, spot and light modes change the photo, not the mask (heal.py), so the layers
skip them.
"""

from __future__ import annotations

import numpy as np
import cv2

LAYER_MAX = 2048      # longest side of the layer; edits are soft, so more adds nothing
MIN_SOFT_SIGMA = 0.7  # px; keeps even a hard brush's edge from stepping
PRESENT = 1e-3        # a layer value this small counts as no edit
LIGHT = "light"                  # the light brush (heal.py)
IMAGE_MODES = ("heal", "spot", LIGHT)   # strokes heal.py paints on the photo
UNSMOOTH = "unsmooth"            # Undo smoothing: off the mask, but the fixes stay
RESMOOTH = "resmooth"            # Smooth again: a finer smoothing, on its own layer


def new_layer(w: int, h: int) -> np.ndarray:
    """An empty layer for a photo of this size: float32, at most LAYER_MAX on its long side."""
    s = min(1.0, LAYER_MAX / max(w, h))
    return np.zeros((max(1, round(h * s)), max(1, round(w * s))), np.float32)


def draw_footprint(points, thick: int, w: int, h: int) -> np.ndarray:
    """A stroke's hard footprint (uint8, 255 inside) on a w x h grid, through these
    pixel points with this thickness."""
    stamp = np.zeros((h, w), np.uint8)
    cv2.circle(stamp, points[0], max(1, thick // 2), 255, -1, cv2.LINE_8)
    for a, b in zip(points, points[1:]):
        cv2.line(stamp, a, b, 255, thick, cv2.LINE_8)
    return stamp


def _stamp(shape, stroke: dict):
    """Where a stroke lands on a layer of this shape, and how strongly: (y0, y1, x0, x1,
    float32 0..1 over that box), or None."""
    lh, lw = shape
    coords = stroke["points"]
    if len(coords) < 2:
        return None
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
        return None
    stamp = draw_footprint([(x - x0, y - y0) for x, y in points], thick, x1 - x0, y1 - y0)
    return y0, y1, x0, x1, cv2.GaussianBlur(stamp.astype(np.float32) / 255.0, (0, 0), sigma)


def paint_stroke(layer: np.ndarray, stroke: dict, fills: np.ndarray | None = None,
                 again: np.ndarray | None = None) -> None:
    """Add one stroke to a layer, in place, and to the spot fills' and Smooth again's
    layers if there are those."""
    mode = stroke.get("mode")
    if mode in IMAGE_MODES or (mode == RESMOOTH and again is None):
        return
    placed = _stamp(layer.shape, stroke)
    if placed is None:
        return
    y0, y1, x0, x1, soft = placed
    if mode == RESMOOTH:
        target = float(stroke.get("amount", 1.0))
        again[y0:y1, x0:x1] = again[y0:y1, x0:x1] * (1.0 - soft) + target * soft
        return
    target = (1.0 if stroke["mode"] == "skin" else -1.0) * float(stroke.get("amount", 1.0))
    layer[y0:y1, x0:x1] = layer[y0:y1, x0:x1] * (1.0 - soft) + target * soft
    if fills is not None:
        keep = 0.0 if stroke["mode"] == UNSMOOTH else target
        fills[y0:y1, x0:x1] = fills[y0:y1, x0:x1] * (1.0 - soft) + keep * soft
    if again is not None and mode != "skin":      # Restore and Undo smoothing take it back
        again[y0:y1, x0:x1] = again[y0:y1, x0:x1] * (1.0 - soft) + 0.0 * soft


def _layers(strokes, w: int, h: int):
    """The layers for the mask, the spot fills and Smooth again. The fills' is None while
    it's the same as the mask's (until the first Undo smoothing stroke); Smooth again's
    is None until its first stroke."""
    layer, fills, again = new_layer(w, h), None, None
    for stroke in strokes:
        mode = stroke.get("mode")
        if mode == UNSMOOTH and fills is None:
            fills = layer.copy()
        if mode == RESMOOTH and again is None:
            again = np.zeros_like(layer)
        paint_stroke(layer, stroke, fills, again)
    return layer, fills, again


def layers_from(strokes, w: int, h: int):
    """The layer for the mask, and the one for the spot fills: None while it's the same,
    which it is until the first Undo smoothing stroke."""
    return _layers(strokes, w, h)[:2]


def layer_from(strokes, w: int, h: int) -> np.ndarray:
    return layers_from(strokes, w, h)[0]


def regions(strokes, w: int, h: int):
    """(the edit for apply_to_mask, the edit for apply_to_correction) at this photo
    size, as region_of gives them: the same object twice without Undo smoothing."""
    return edits(strokes, w, h)[:2]


def edits(strokes, w: int, h: int):
    """regions, plus the edit for Smooth again (smoother.smooth_again; 0..1), or None."""
    layer, fills, again = _layers(strokes, w, h)
    region = region_of(layer, w, h)
    return region, region if fills is None else region_of(fills, w, h), None if again is None else region_of(again, w, h)


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
