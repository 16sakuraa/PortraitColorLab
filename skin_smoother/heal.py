"""
The healing, spot and light brushes: strokes that change the photo itself. Heal and
spot strokes are painted after the spot fills and before smoothing (apply), so what
they leave is smoothed like the rest of the skin; light strokes are painted on the
smoothed result (apply_light).

Strokes have the form in mask_edit.py, with three more modes:

    {"mode": "heal", ..., "offset": [dx, dy]}
        Copy the photo from dx, dy away (shares of its width and height) and match it
        to the light and shading here: the detail comes from the source, the tone from
        the skin around the stroke. This is seamless cloning (Perez, Gangnet & Blake,
        2003; the healing brush of Georgiev, 2004), except that the smooth difference
        added to the copy is found by normalized convolution (Knutsson & Westin, 1993)
        of how the photo differs from the source just outside the stroke, instead of
        by solving Poisson's equation. So a dark scar or a cut under the brush has no
        say in the tone, and the copy's edge meets the skin around it.
    {"mode": "spot", ...}
        Fill the stroke from the skin around it, keeping some of its own fine
        detail: like the automatic spot fills (blemish_classic.py), at the brush's size.
    {"mode": "light", ..., "pick": [x, y]}
        Paint the tone found at the picked point (shares of width and height) over the
        stroke, keeping the skin's grain: painting on the tone layer of a frequency
        separation, split at the size smoothing keeps grain at. For blotches, seams
        between light and shadow, lines and dark circles, at a low strength.

Strokes apply in order, each reading the photo as the ones before it left it, so a
later one wins where they cross. A Restore stroke after one of them paints the photo
as it was before any of them back, at its strength: a 40% Restore over a heal leaves
60% of the heal. (Two 40% Restore strokes over it leave 36%, unlike on the mask.)
Restore strokes before them don't limit them, and Add skin and Undo smoothing don't
touch them.
"""

from __future__ import annotations

import math

import numpy as np
import cv2

import smoother
from mask_edit import IMAGE_MODES, LIGHT, MIN_SOFT_SIGMA, RESMOOTH, UNSMOOTH, draw_footprint

HEAL_MODES = ("heal", "spot")          # painted before smoothing
KEEPS = IMAGE_MODES + ("skin", UNSMOOTH, RESMOOTH)   # strokes that don't take these back; the rest are Restore

GRID_DIAMETER = 128.0  # px; a larger brush is worked out on a coarser grid, then scaled up
MEMBRANE_SPAN = 1.0    # how far around the stroke the tone is matched from, in brush radii
SPOT_SPAN = 1.0        # how far around a spot it's filled from, in brush radii
SPOT_SPLIT = 0.08      # a spot keeps detail finer than this, as a share of the brush size
SPOT_DETAIL = 0.5      # and this much of it, no stronger than the grain around it
PRIOR = 0.01           # weight of the average of all the skin around a stroke, for where nothing is near


def apply(img: np.ndarray, strokes) -> np.ndarray:
    """img (uint8 BGR) with the heal and spot strokes painted on: a new array, or img
    itself when there are none, so nothing changes for photos without them."""
    strokes = list(strokes or [])
    first = next((i for i, s in enumerate(strokes) if s.get("mode") in HEAL_MODES), None)
    if first is None:
        return img
    out = img.copy()
    for stroke in strokes[first:]:
        if len(stroke["points"]) < 2:
            continue
        mode = stroke.get("mode")
        if mode == "heal":
            _heal(out, stroke)
        elif mode == "spot":
            _spot(out, stroke)
        elif mode not in KEEPS:    # Restore: back towards the photo before the strokes
            _restore(out, img, stroke)
    return out


def apply_light(img: np.ndarray, strokes, split: float) -> np.ndarray:
    """img (uint8 BGR, the smoothed photo) with the light brush's strokes painted on: a
    new array, or img itself when there are none. split: the size (px) smoothing splits
    tone from grain at (smoother.detail_sigma); the brush paints tone only."""
    strokes = list(strokes or [])
    first = next((i for i, s in enumerate(strokes) if s.get("mode") == LIGHT), None)
    if first is None:
        return img
    out = img.copy()
    for stroke in strokes[first:]:
        if len(stroke["points"]) < 2:
            continue
        mode = stroke.get("mode")
        if mode == LIGHT:
            _light(out, stroke, split)
        elif mode not in KEEPS:
            _restore(out, img, stroke)
    return out


class _Place:
    """Where a stroke lands at this photo size: a box in its pixels, and a grid over
    the box (the same pixels for a brush up to GRID_DIAMETER) holding its footprint
    (uint8, 255 inside), and how much it paints on the box's pixels (alpha, 0..1).

    A brush's soft edge is centred on its outline, as on the mask. With cover, the
    whole outline is painted fully and the edge fades outside it instead, so a repair
    reaches everything the brush's circle covers. extra: px of room the work around
    the stroke needs, besides its soft edge and span (in brush radii)."""

    def __init__(self, stroke: dict, w: int, h: int, span: float, cover: bool = False, extra: float = 0.0):
        c = stroke["points"]
        points = [(round(c[i] * w), round(c[i + 1] * h)) for i in range(0, len(c) - 1, 2)]
        self.diameter = diameter = max(1.0, float(stroke["size"]) * max(w, h))
        thick = max(1, round(diameter))
        sigma = max(MIN_SOFT_SIGMA, float(stroke.get("soft", 0.0)) * diameter * 0.5)
        reach = 3.0 * max(sigma, span * diameter * 0.5, extra)
        margin = int(math.ceil(thick / 2.0 + reach)) + 2
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        self.x0, self.x1 = max(0, min(xs) - margin), min(w, max(xs) + margin + 1)
        self.y0, self.y1 = max(0, min(ys) - margin), min(h, max(ys) + margin + 1)
        self.empty = self.x1 <= self.x0 or self.y1 <= self.y0
        if self.empty:
            return
        bw, bh = self.x1 - self.x0, self.y1 - self.y0
        self.scale = s = min(1.0, GRID_DIAMETER / diameter)
        self.gw, self.gh = (max(1, round(bw * s)), max(1, round(bh * s))) if s < 1.0 else (bw, bh)
        fx, fy = self.gw / bw, self.gh / bh
        on_grid = [(round((x - self.x0 + 0.5) * fx - 0.5), round((y - self.y0 + 0.5) * fy - 0.5)) for x, y in points]
        self.footprint = draw_footprint(on_grid, max(1, round(diameter * s)), self.gw, self.gh)
        stamp = smoother._soft_blur(self.footprint.astype(np.float32) / 255.0, max(MIN_SOFT_SIGMA, sigma * s))
        if cover:
            stamp = np.minimum(stamp * np.float32(2.0), np.float32(1.0))
        self.alpha = self.up(stamp) * np.float32(float(stroke.get("amount", 1.0)))

    def box(self, a: np.ndarray) -> np.ndarray:
        return a[self.y0:self.y1, self.x0:self.x1]

    def down(self, a: np.ndarray) -> np.ndarray:
        """Box pixels (float32) onto the grid."""
        return a if self.scale >= 1.0 else cv2.resize(a, (self.gw, self.gh), interpolation=cv2.INTER_AREA)

    def up(self, a: np.ndarray) -> np.ndarray:
        """Grid values (float32) onto the box pixels."""
        if self.scale >= 1.0:
            return a
        return cv2.resize(a, (self.x1 - self.x0, self.y1 - self.y0), interpolation=cv2.INTER_LINEAR)

    def around(self, values: np.ndarray, span: float) -> np.ndarray:
        """A smooth fill of grid values (HxWx3) from just outside the footprint
        (normalized convolution), on the box pixels. Deep inside a broad stroke, where
        nothing outside is near enough, it's the average of everything around it."""
        outside = 1.0 - self.footprint.astype(np.float32) / 255.0
        weighted = values * outside[:, :, None]
        total = max(1.0, float(outside.sum(dtype=np.float64)))
        average = (weighted.reshape(-1, 3).sum(axis=0, dtype=np.float64) / total).astype(np.float32)
        sigma = span * self.diameter * 0.5 * self.scale
        num = smoother._soft_blur(weighted, sigma) + np.float32(PRIOR) * average
        den = smoother._soft_blur(outside, sigma) + np.float32(PRIOR)
        return self.up(num / den[:, :, None])

    def blend(self, out: np.ndarray, target: np.ndarray) -> None:
        """Paint target (box pixels, float32 HxWx3) into out at the stamp, rounding once."""
        here = self.box(out)
        now = here.astype(np.float32)
        mixed = now + self.alpha[:, :, None] * (target - now) + np.float32(0.5)
        here[:] = np.clip(mixed, 0, 255).astype(np.uint8)


def _source(out: np.ndarray, place: _Place, dx: int, dy: int) -> np.ndarray:
    """The box's pixels from dx, dy away (float32), repeating the photo's edge beyond it."""
    h, w = out.shape[:2]
    ys = np.clip(np.arange(place.y0, place.y1) + dy, 0, h - 1)
    xs = np.clip(np.arange(place.x0, place.x1) + dx, 0, w - 1)
    return out[ys][:, xs].astype(np.float32)


def _heal(out: np.ndarray, stroke: dict) -> None:
    h, w = out.shape[:2]
    offset = stroke.get("offset") or [0.0, 0.0]
    dx, dy = round(float(offset[0]) * w), round(float(offset[1]) * h)
    if dx == 0 and dy == 0:
        return                                 # copying from right here changes nothing
    place = _Place(stroke, w, h, MEMBRANE_SPAN, cover=True)
    if place.empty:
        return
    source = _source(out, place, dx, dy)
    here = place.box(out).astype(np.float32)
    difference = place.down(here) - place.down(source)
    place.blend(out, source + place.around(difference, MEMBRANE_SPAN))


def _spot(out: np.ndarray, stroke: dict) -> None:
    h, w = out.shape[:2]
    place = _Place(stroke, w, h, SPOT_SPAN, cover=True)
    if place.empty:
        return
    here = place.box(out).astype(np.float32)
    detail = here - smoother._soft_blur(here, max(MIN_SOFT_SIGMA, SPOT_SPLIT * place.diameter))
    # No stronger than the grain of the skin around it: a spot's own rim or white head
    # is detail too, and would stay as a faint ring or dot.
    level = np.sqrt(np.maximum(place.around(place.down(detail * detail), SPOT_SPAN), np.float32(0.0)))
    kept = np.clip(detail, -level, level)
    place.blend(out, place.around(place.down(here), SPOT_SPAN) + np.float32(SPOT_DETAIL) * kept)


def _tone_at(img: np.ndarray, pick, split: float) -> np.ndarray:
    """The photo's tone at a point (shares of width and height): the average of a
    square about the size of the grain, so the grain itself doesn't count (float32 BGR)."""
    h, w = img.shape[:2]
    x = min(w - 1, max(0, round(float(pick[0]) * w)))
    y = min(h - 1, max(0, round(float(pick[1]) * h)))
    r = max(1, round(split))
    patch = img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    total = patch.reshape(-1, 3).sum(axis=0, dtype=np.float64)
    return (total / (patch.shape[0] * patch.shape[1])).astype(np.float32)


def _light(out: np.ndarray, stroke: dict, split: float) -> None:
    pick = stroke.get("pick")
    if not pick:
        return
    h, w = out.shape[:2]
    colour = _tone_at(out, pick, split)
    place = _Place(stroke, w, h, 0.0, extra=split)
    if place.empty:
        return
    here = place.box(out).astype(np.float32)
    low = smoother._soft_blur(here, split)
    place.blend(out, here + (colour - low))


def _restore(out: np.ndarray, before: np.ndarray, stroke: dict) -> None:
    h, w = out.shape[:2]
    place = _Place(stroke, w, h, 0.0)
    if place.empty:
        return
    place.blend(out, place.box(before).astype(np.float32))
