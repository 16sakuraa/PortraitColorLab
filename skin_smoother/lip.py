"""
A thin moustache, taken off the upper lip: clear() replaces it instead of fading it.

A moustache is hair lying over skin, so nothing under it can be lightened back into a
bare lip: lifted, it stays a paler moustache. Here the skin is put there anew, the way
a retoucher heals it:

  tone     What a bare upper lip looks like there. Its shape comes from a small table
           (TABLE: how bright the skin between lip and nose is, place by place, against
           its own surroundings). Its level and side-to-side slope are fitted to this
           face from above: hair only ever darkens, so the fit rests on the clear skin
           and passes over the hair. Its colour is the cheeks', shaded. Beside the hair
           the new tone is matched to the clear skin next to it (as the healing brush
           matches a copy to its surroundings), so the new skin has no outline.
  grain    The cheeks' own grain, each half of the lip from its own side, brightness
           only, and no stronger than a cheek's usual grain: a glint or a spot on the
           cheek is not skin texture.
  where    Wherever the lip is darker than that tone, a little past the hair. Clear
           skin within reach keeps its own tone, and nothing is ever made darker.

It is made for a portrait taken straight on with the mouth closed, as for an ID photo,
and acts on no other face: a smile or a turned head throws shadows on the upper lip
that are not hair, and moves the nose over where the lip would be (_Frame.posed).

Whether there is anything to remove is read from the strip just above the lip. A lip
that is no darker there than a bare one is left as it is, so the step can stay on for a
whole set of such portraits. The measure of "no darker" was set on one studio's
portraits; under other light a shadowed lip may be taken for a faint moustache. A full,
dense moustache is left alone too, unless it is asked for (full, below): most who wear
one keep it. Stubble or a beard around the mouth is not touched, so a face with both
loses the moustache and keeps the beard: leave the step off for it.

A full moustache (clear(..., full=True)) is more than a thin one made darker, so it has
its own way through the same engine (_clear_full):

  where    The hair is followed from its dense core, through whatever is dark and rough
           like hair, to its thin ends: past the mouth's corners and down beside them,
           and up to the nose. A fixed zone would cut it. Between its two halves the
           hair is often fair, no darker than a bare lip; there its roughness alone
           says it is hair.
  tone     Most of the lip is under hair, so the table's shape is matched, colour
           channel by channel, to the clear skin around the whole moustache and carried
           inwards from there, wide where no clear skin is near. The philtrum is
           too narrow for the table: it is put in by itself (PHILTRUM: the groove down
           the lip's own middle, deepest just above the lip, and the light either side
           of it under the nose), somewhat stronger than the average bare lip's. It is
           an average philtrum, not this face's own, which the hair hides.
  beside   Skin is replaced in the moustache and wherever strands are beside it; the
           rest of the area it was grown by is clear skin and stays as the photo has it.
  grain    Pieces of each cheek's own patch, laid side by side and blended, each from a
           different part of the patch: a copy moved this far would run into the nose
           or a fold.
  lip      The hair lies on the lip's edge and over it, so the edge that is left is the
           one drawn here: a smooth curve through the mesh's points, at the photo's own
           size. Hair lying over the upper lip takes the lip's own colour, carried along
           the lip from where it is clear.

A moustache that runs into a beard is cut where the bounds end, and the skin put in its
place is matched to stubbled skin: it shows. That face is not what this is for.

It doesn't tell a mole from hair: a mark right above the lip, inside a moustache or its
shadow, goes with it. The brush's Restore brings it back (keep).

The work is done on a copy of the mouth at one fixed size (GRID_EYES between the eyes)
and only the grain is handled at the photo's own size, so a preview and the full-size
photo come out alike.

The step comes before smoothing, after the spot fills and before the healing brush's
strokes, so what it leaves is smoothed like the rest of the skin.
"""

from __future__ import annotations

import math

import numpy as np
import cv2

import smoother

# Face Mesh points.
UPPER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)      # the upper lip's outer edge, right corner to left
LOWER = (291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61)     # the lower lip's, back again
MOUTH_R, MOUTH_L, LIP_TOP, NOSE_BASE = 61, 291, 0, 2
LIP_INNER_TOP, LIP_INNER_FOOT = 13, 14                          # where the lips meet: apart when the mouth is open
LIP_INNER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)   # where the upper lip ends at the mouth's line, corner to corner
CHEEK_R, CHEEK_L = 205, 425
EYE_R, EYE_L = (33, 133), (263, 362)

GRID_EYES = 160.0   # px between the eyes on the copy the tone is worked out on
MIN_EYES = 24.0     # a face smaller than this (px between the eyes) is left alone

# The brightness of a bare upper lip, place by place, as a share of the brightness
# around it. Rows: from the lip's edge (first) up to the height of the nose's base
# (last), in eighths. Columns: from the middle of the lip outwards, in fifths of the
# way to the mouth's corner (so the sixth column is the corner, the last 1.6 corners
# out). The average of retouched portraits, both sides mirrored together. Only its
# shape matters: its level is fitted to each face.
TABLE = np.array([
    (0.85, 0.88, 0.92, 0.93, 0.88, 0.79, 0.86, 0.89, 0.91),
    (0.85, 0.88, 0.92, 0.93, 0.88, 0.79, 0.86, 0.89, 0.91),
    (0.86, 0.89, 0.93, 0.94, 0.90, 0.83, 0.87, 0.90, 0.91),
    (0.87, 0.90, 0.94, 0.96, 0.94, 0.90, 0.91, 0.92, 0.92),
    (0.90, 0.91, 0.95, 0.96, 0.96, 0.94, 0.94, 0.93, 0.94),
    (0.92, 0.92, 0.96, 0.97, 0.98, 0.97, 0.96, 0.95, 0.95),
    (0.92, 0.93, 0.95, 0.97, 0.99, 0.98, 0.96, 0.96, 0.97),
    (0.91, 0.92, 0.93, 0.94, 0.98, 0.98, 0.97, 0.98, 0.98),
    (0.87, 0.89, 0.87, 0.88, 0.94, 0.95, 0.97, 0.99, 0.99),
], np.float32)
TABLE_U, TABLE_V = 0.2, 0.125      # the table's steps: across (1 = the mouth's corner), and up (1 = the nose's base)

# A face it acts on, in distances between the eyes: straight on, mouth closed, not smiling
# broadly. The limits sit just outside what 81 ID portraits show (turn within 0.05, the
# lips at most 0.13 apart with all but two under 0.06, the mouth at most 0.99 wide, nose to
# lip at least 0.18); a set of candid photos lies outside them (turn 0.08 and more).
POSE = {
    "turn": 0.07,        # the nose's base is no further than this to one side of the middle of the eyes,
    "open": 0.09,        # the lips are no further apart than this,
    "smile": 1.0,        # the mouth is no wider than this,
    "short": 0.16,       # and the lip's top edge is at least this far under the nose's base
}

# Places are given in the lip's own frame: u across (-1 and 1 at the mouth's corners),
# v from the upper lip's edge (0) up to the height of the nose's base (1). Sizes are
# shares of the distance between the eyes.
TUNE = {
    "tone": 0.025,       # tone/grain split
    "lip_edge": 0.008,   # the lips' outline is softened by this
    "cheek": 0.16,       # the cheeks are read in discs of this radius
    "wide": 1.75,        # the zone reaches this far across,
    "top": 0.82,         # and this far up towards the nose's base over the mouth,
    "end_top": 0.35,     # narrowing to this height at its ends: a mark on the cheek beside the mouth is not its business,
    "droop": 0.25,       # where it also drops this far below the lip's edge, for a moustache's ends
    "taper": 0.75,       # and fades out from this share of the way to its ends
    "edge": 0.03,        # soft edge outside the zone
    "fit_wide": 2.3,     # the bare lip's level is fitted to the face over this width,
    "fit_lo": 0.03, "fit_hi": 0.95,   # and this height,
    "below": 0.06,       # counting skin darker than the fit this little: hair only darkens
    "fit_rounds": 12,
    "fit_most": 6000,    # points the fit looks at
    "drop": 0.04,        # a finished upper lip sits this far under the fitted level,
    "red": 0.05,         # and in its shade red falls this much less than green and blue: skin in shade isn't grey
    "dark_lo": 0.02,     # darker than a bare lip by this share: starting to be hair,
    "dark_hi": 0.07,     # and by this: hair
    "grow": 0.025,       # the replaced area grows past the hair by this
    "meet": 0.12,        # the new tone is matched to the clear skin within about this far,
    "meet_lo": 0.85, "meet_hi": 1.05, # darker by no more than the first, lighter by no more than the second,
    "meet_prior": 0.02,  # and not at all where none is near
    "strip_wide": 1.1, "strip_lo": 0.03, "strip_hi": 0.45,   # the strip above the lip that says whether there is a moustache:
    "there_lo": 0.064,   # on average this much darker than a bare lip is not yet one,
    "there_hi": 0.072,   # and this much is one: a narrow step, so a moustache goes whole or stays whole, never half;
    "dense_lo": 0.33,    # its darkest tenth this dark is a full moustache, which is kept
    "dense_hi": 0.39,
    "grain_most": 5.0,   # borrowed grain is held to this many times the cheeks' quietest quarter of grain (about 1.6 of its spread)
    "side": 0.6, "side_up": 0.45,     # each half of the lip borrows from the cheek that lies this way from it (u, v)
    "seam": 0.25,        # the two halves' grain blends over this much either side of the middle (u)
}


# A full moustache (_clear_full). Places in the lip's frame, sizes in eye distances, as in TUNE.
FULL = {
    "core": 0.25,          # this dark against a bare lip, inside the strip: the moustache's core
    "core_wide": 1.15, "core_lo": 0.02, "core_hi": 0.65,
    "hair_lo": 0.07,       # this dark and rough: hair,
    "hair_hi": 0.18,       # this dark: hair whatever its grain
    "rough": 1.6,          # rough: fine detail this many times the cheeks',
    "rough_at": 0.03,      # measured over this radius
    "mid_rough": 1.7, "mid_wide": 0.6,    # between the halves (u), this rough is hair even when it is not dark
    "tail_dark": 0.035, "tail_rough": 1.3, "tail_reach": 0.10,   # fainter hair within this reach of the moustache goes with it
    "bridge": 0.012,       # gaps this small between strands are closed
    "wide": 2.0,           # the hair is followed this far across,
    "top": 0.97,           # this far up (the nostrils are above),
    "drop": 1.1,           # and this far down beside the mouth,
    "drop_from": 0.85, "drop_to": 1.15,   # which opens between these (u): never under the lips
    "bound_edge": 0.04,    # the bounds' soft edge
    "grow": 0.035,         # the replaced area grows past the hair by this,
    "soft": 0.02,          # with this soft an edge
    "clear_off": 0.05,     # clear skin starts this far from the hair,
    "clear_lo": -1.3, "clear_hi": 1.05,   # and is looked for between these heights (v)
    "fill": (0.5, 0.25, 0.12, 0.06),      # its tone is carried inwards at these radii, the widest first,
    "fill_lo": 0.02, "fill_hi": 0.2,      # each finer one taking over where this share of what it sees is clear skin
    "span": (0.80, 1.20),  # the carried tone against the table's, per channel, is held to this
    "shape_soft": 0.05,    # the table's shape is rounded by this
    "tile": 0.12,          # the grain is laid in pieces of the cheek this large,
    "tile_wander": 0.07,   # each taken up to this far from the middle of the cheek's patch
    "lip_soft": 0.012,     # the lip's edge is this soft
    "lip_band": 0.6,       # hair is looked for this far into the upper lip (share of its thickness),
    "lip_dark": (0.06, 0.14),             # where it is darker than the lip's own tone by this,
    "lip_strand": 0.006,   # judged at this radius,
    "lip_reach": 0.04,     # and this near to where the skin above was replaced
    "lip_fill": (0.16, 0.08, 0.04),       # the lip's tone is carried along the lip at these radii,
    "lip_keep": (0.80, 0.92),             # from the lip that is this bright against its brighter half (the rest may be hair)
    "lip_grain": 0.7,      # the new lip gets this share of the skin's grain, so it is not flat
    # The philtrum (PHILTRUM, below): put in at this many times the average bare lip's, since an average of
    # faces is flatter than any one of them (a quarter of bare lips are deeper than 1.35 times it), out to
    # where the table takes over again.
    "philtrum": 1.6,
    "philtrum_light": 3.0,  # its lights (what is brighter than the lip beside it: under the nose, either side of the groove) at this many times
    "philtrum_to": (0.45, 0.6),
    # Clear skin is left as it is: inside the replaced area, skin is replaced only where there is hair.
    "strand_at": 0.03,     # A strand is darker than the skin around it within this radius,
    "strand": 3.0,         # by this many times the cheeks' own grain,
    "strand_span": (0.04, 0.08),          # held between these shares.
    "own_soft": 0.012,     # The moustache itself has this soft an edge.
    "own_at": 0.02,        # Outside it the photo stays as it is
    "own": (0.02, 0.08),   # where less than this share of pixels within that radius is hair.
}
# The philtrum of a bare upper lip: brightness by place, in percent against the lip's level beside it at the
# same height. Rows: from the lip's edge (first) up to the nose's base (the eleventh) and just past it, in
# tenths. Columns: from the lip's own middle outwards in steps of 0.05 (u), to 0.6. The groove is deepest
# just above the lip and fades towards the nose; under the nose the skin either side of it is lighter
# (the light under the nostrils). The average of retouched portraits with a bare lip, both sides mirrored
# together: the table's steps (TABLE_U) are four times too wide for it.
PHILTRUM = np.array([
    (-6.3, -5.8, -4.9, -3.5, -2.1, -0.5, +0.6, -0.4, -0.7, -0.5, +0.1, +0.4, -0.0),
    (-6.3, -5.8, -4.9, -3.5, -2.1, -0.5, +0.6, -0.4, -0.7, -0.5, +0.1, +0.4, -0.0),
    (-9.2, -8.4, -6.7, -5.1, -3.3, -1.8, -0.9, -0.6, -0.6, -0.3, -0.0, +0.2, +0.2),
    (-8.8, -7.8, -6.2, -4.5, -3.4, -2.5, -1.4, -0.8, -0.8, -0.5, -0.0, +0.2, +0.4),
    (-7.0, -6.1, -4.7, -3.5, -3.0, -2.1, -1.2, -0.9, -0.7, -0.5, -0.1, +0.3, +0.3),
    (-5.4, -4.5, -3.3, -2.5, -2.1, -1.4, -0.7, -0.7, -0.6, -0.3, -0.1, +0.1, +0.3),
    (-4.1, -3.5, -2.5, -1.8, -1.5, -0.9, -0.5, -0.3, -0.3, -0.3, -0.1, +0.1, +0.4),
    (-3.6, -2.9, -1.7, -1.4, -1.2, -0.6, -0.3, -0.3, -0.4, -0.3, -0.1, +0.0, +0.4),
    (-2.7, -2.1, -1.0, -0.6, -0.4, +0.0, +0.2, +0.1, -0.2, -0.4, -0.2, +0.1, +0.5),
    (-1.8, -1.4, -0.5, +0.2, +0.6, +1.1, +1.2, +0.9, +0.3, -0.2, -0.3, -0.0, +0.5),
    (-1.2, -0.8, +0.3, +1.5, +2.6, +3.4, +3.5, +2.7, +1.3, +0.1, -0.4, -0.3, +0.6),
    (+0.6, +1.3, +2.6, +3.7, +5.0, +6.4, +6.3, +4.4, +1.7, -0.4, -1.0, -0.4, +1.7),
], np.float32)
PHILTRUM_U, PHILTRUM_V = 0.05, 0.1
# Where in the cheek's patch each piece of grain comes from (shares of tile_wander, across and down).
WANDER = ((0.0, 0.0), (0.62, 0.31), (-0.45, 0.58), (0.21, -0.71), (-0.68, -0.29), (0.55, -0.52), (-0.18, 0.83),
          (0.85, -0.05), (-0.32, -0.80), (-0.90, 0.20), (0.30, 0.75))


def _luma(a: np.ndarray) -> np.ndarray:
    return a @ smoother._LUMA


def _percentile(values: np.ndarray, q: float) -> float:
    """The q-th percentile (0..100), between the two nearest values (as numpy's default)."""
    s = np.sort(np.asarray(values, np.float64).ravel())
    if s.size == 0:
        return 0.0
    pos = (s.size - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(s.size - 1, lo + 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def _interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp)


class _Frame:
    """One face's lip frame, in photo pixels."""

    def __init__(self, pts: np.ndarray):
        p = np.asarray(pts, np.float64)[:, :2]
        self.eyes = float(np.hypot(*(p[list(EYE_L)].mean(axis=0) - p[list(EYE_R)].mean(axis=0))))
        self.centre = (p[MOUTH_R] + p[MOUTH_L]) / 2.0
        self.half = (p[MOUTH_L] - p[MOUTH_R]) / 2.0
        self.half2 = float(self.half @ self.half)
        self.nose = p[NOSE_BASE]
        self.mid = tuple(float((p[i] - self.centre) @ self.half) / max(self.half2, 1e-9) for i in (LIP_TOP, NOSE_BASE))    # across (u): the lip's top point, the nose's base
        self.up = p[NOSE_BASE] - p[LIP_TOP]
        edge = p[list(UPPER)]
        across = (edge - self.centre) @ self.half / max(self.half2, 1e-9)
        order = np.argsort(across, kind="stable")
        self.edge_u, self.edge_x, self.edge_y = across[order], edge[order, 0], edge[order, 1]
        self.lips = p[list(UPPER + LOWER)]
        self.inner = p[list(LIP_INNER)]
        self.cheeks = (p[CHEEK_R], p[CHEEK_L])
        self.ok = self.eyes >= MIN_EYES and self.half2 > 1.0 and float(self.up @ self.up) > 1.0
        self.posed = False
        if self.ok:
            eye_r, eye_l = p[list(EYE_R)].mean(axis=0), p[list(EYE_L)].mean(axis=0)
            axis = eye_l - eye_r
            turn = float((self.nose - (eye_r + eye_l) / 2.0) @ axis) / float(axis @ axis)
            apart = float(np.hypot(*(p[LIP_INNER_TOP] - p[LIP_INNER_FOOT]))) / self.eyes
            self.posed = (abs(turn) <= POSE["turn"] and apart <= POSE["open"]
                          and 2.0 * math.sqrt(self.half2) / self.eyes <= POSE["smile"]
                          and math.sqrt(float(self.up @ self.up)) / self.eyes >= POSE["short"])

    def at(self, u: float, v: float) -> np.ndarray:
        """The photo's place for a place in the frame."""
        inside = min(1.0, max(-1.0, u))
        ex = float(_interp(np.array([inside]), self.edge_u, self.edge_x)[0]) + (u - inside) * self.half[0]
        ey = float(_interp(np.array([inside]), self.edge_u, self.edge_y)[0]) + (u - inside) * self.half[1]
        nx, ny = self.nose[0] + u * self.half[0], self.nose[1] + u * self.half[1]
        return np.array([ex + v * (nx - ex), ey + v * (ny - ey)])

    def uv(self, xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The frame's u and v at photo places (arrays)."""
        u = ((xx - self.centre[0]) * self.half[0] + (yy - self.centre[1]) * self.half[1]) / self.half2
        inside = np.clip(u, -1.0, 1.0)
        ex = _interp(inside, self.edge_u, self.edge_x) + (u - inside) * self.half[0]
        ey = _interp(inside, self.edge_u, self.edge_y) + (u - inside) * self.half[1]
        ux, uy = self.nose[0] + u * self.half[0] - ex, self.nose[1] + u * self.half[1] - ey
        v = ((xx - ex) * ux + (yy - ey) * uy) / (ux * ux + uy * uy + 1e-6)
        return u.astype(np.float32), v.astype(np.float32)


def _table_at(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """TABLE at places in the frame, between its entries and held at its edges."""
    rows, cols = TABLE.shape
    a = np.clip(np.abs(u) / TABLE_U, 0.0, cols - 1.0)
    b = np.clip(v / TABLE_V, 0.0, rows - 1.0)
    a0, b0 = np.minimum(np.floor(a).astype(np.int64), cols - 2), np.minimum(np.floor(b).astype(np.int64), rows - 2)
    fa, fb = (a - a0).astype(np.float32), (b - b0).astype(np.float32)
    top = TABLE[b0, a0] * (1.0 - fa) + TABLE[b0, a0 + 1] * fa
    low = TABLE[b0 + 1, a0] * (1.0 - fa) + TABLE[b0 + 1, a0 + 1] * fa
    return (top * (1.0 - fb) + low * fb).astype(np.float32)


def _shaded(level: np.ndarray, cheek: np.ndarray) -> np.ndarray:
    """The bare lip's tone: the cheeks' colour, shaded to the lip's brightness."""
    t = TUNE
    e = t["red"]
    k = np.array([1.0 + e, 1.0 + e, 1.0 - e * (0.701 / 0.299)], np.float32)     # B, G, R: average 1 by brightness
    shade = level * np.float32((1.0 - t["drop"]) / max(float(cheek @ smoother._LUMA), 1e-3))
    return cheek * np.power(np.maximum(shade, np.float32(1e-3))[:, :, None], k)


def _fit(seen: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    """seen ~ a + b u, from above: points under the line count little (TUNE below)."""
    t = TUNE
    seen, u = seen.astype(np.float64), u.astype(np.float64)
    w = np.ones(len(seen))
    a = b = 0.0
    for _ in range(t["fit_rounds"]):
        sw, su, suu = float(w.sum()), float((w * u).sum()), float((w * u * u).sum())
        sy, suy = float((w * seen).sum()), float((w * u * seen).sum())
        det = sw * suu - su * su
        if abs(det) < 1e-9:
            a, b = sy / max(sw, 1e-9), 0.0
        else:
            a, b = (sy * suu - su * suy) / det, (sw * suy - su * sy) / det
        w = np.where(seen - (a + b * u) < 0.0, t["below"], 1.0)
    return a, b


def _moved(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """a read from dx, dy away, repeating its edge beyond it."""
    h, w = a.shape[:2]
    ys = np.clip(np.arange(h) + dy, 0, h - 1)
    xs = np.clip(np.arange(w) + dx, 0, w - 1)
    return a[ys][:, xs]


def _inside(shape: tuple[int, int], dx: int, dy: int) -> np.ndarray:
    """1 where dx, dy away is still inside an array of this shape."""
    h, w = shape
    ys, xs = np.arange(h) + dy, np.arange(w) + dx
    return (((ys >= 0) & (ys < h))[:, None] & ((xs >= 0) & (xs < w))[None, :]).astype(np.float32)


def _clear_face(out: np.ndarray, pts, mask: np.ndarray, amount: float, keep, force: bool, full: bool = False) -> bool:
    """One face, changed in place in out. Returns whether anything changed."""
    t = TUNE
    f = _Frame(pts)
    if not f.ok or not (f.posed or force):
        return False
    if full and force:
        return _clear_full(out, f, mask, amount, keep)
    h, w = out.shape[:2]
    d = f.eyes
    # The box: the fitted width, lip to nose, and both cheeks, with room for the widest blur.
    corners = [f.at(su * t["fit_wide"], v) for su in (-1.0, 1.0) for v in (-0.8, 1.2)]
    xs = [c[0] for c in corners] + [c[0] + s * t["cheek"] * d for c in f.cheeks for s in (-1.0, 1.0)]
    ys = [c[1] for c in corners] + [c[1] + s * t["cheek"] * d for c in f.cheeks for s in (-1.0, 1.0)]
    pad = 3.0 * t["meet"] * d + 2.0
    x0, x1 = max(0, int(math.floor(min(xs) - pad))), min(w, int(math.ceil(max(xs) + pad)) + 1)
    y0, y1 = max(0, int(math.floor(min(ys) - pad))), min(h, int(math.ceil(max(ys) + pad)) + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    bw, bh = x1 - x0, y1 - y0
    scale = min(1.0, GRID_EYES / d)
    gw, gh = (max(8, round(bw * scale)), max(8, round(bh * scale))) if scale < 1.0 else (bw, bh)
    fx, fy = bw / gw, bh / gh                     # photo pixels per grid pixel
    eyes = d / ((fx + fy) / 2.0)                  # the distance between the eyes, on the grid

    box = out[y0:y1, x0:x1].astype(np.float32) / np.float32(255.0)
    box_mask = np.ascontiguousarray(mask[y0:y1, x0:x1], np.float32)
    small = box if scale >= 1.0 else cv2.resize(box, (gw, gh), interpolation=cv2.INTER_AREA)
    skin = box_mask if scale >= 1.0 else cv2.resize(box_mask, (gw, gh), interpolation=cv2.INTER_AREA)

    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float64)
    px, py = x0 + (gx + 0.5) * fx - 0.5, y0 + (gy + 0.5) * fy - 0.5      # the photo's place for each grid pixel
    u, v = f.uv(px, py)

    lips = np.zeros((gh, gw), np.uint8)
    poly = np.stack([(f.lips[:, 0] - x0 + 0.5) / fx - 0.5, (f.lips[:, 1] - y0 + 0.5) / fy - 0.5], axis=1)
    cv2.fillPoly(lips, [np.round(poly).astype(np.int32)], 255)
    valid = 1.0 - smoother._soft_blur(lips.astype(np.float32) / np.float32(255.0), max(0.6, t["lip_edge"] * eyes))

    sigma = t["tone"] * eyes
    tone = smoother._soft_blur(small * valid[:, :, None], sigma) / (smoother._soft_blur(valid, sigma)[:, :, None] + np.float32(1e-4))
    tone_y = _luma(tone)

    # The cheeks: their colour, for the new skin's.
    discs = [((px - c[0]) ** 2 + (py - c[1]) ** 2 < (t["cheek"] * d) ** 2) & (skin > 0.9) for c in f.cheeks]
    if min(int(m.sum()) for m in discs) < 30:
        return False
    cheek = (np.median(tone[discs[0]], axis=0) + np.median(tone[discs[1]], axis=0)).astype(np.float32) / np.float32(2.0)

    # The bare lip's level on this face: the table's shape, fitted from above.
    fit = (np.abs(u) < t["fit_wide"]) & (v > t["fit_lo"]) & (v < t["fit_hi"]) & (valid > 0.98) & (skin > 0.5)
    fys, fxs = np.nonzero(fit)
    if len(fys) < 200:
        return False
    step = max(1, len(fys) // t["fit_most"])
    fys, fxs = fys[::step], fxs[::step]
    shape_ = _table_at(u, v)
    a, b = _fit(np.log(np.maximum(tone_y[fys, fxs], 1e-3) / np.maximum(shape_[fys, fxs], 1e-3)), u[fys, fxs])
    level = shape_ * np.exp(np.float32(a) + np.float32(b) * u)
    target = _shaded(level, cheek)
    target_y = _luma(target)
    dark = (target_y - tone_y) / (target_y + np.float32(1e-4))

    # The zone: full to the lip's edge and past the moustache's ends, fading towards the nose and at the ends.
    past = np.clip((np.abs(u) - 1.0) / (t["wide"] - 1.0), 0.0, 1.0)
    zone = ((np.abs(u) < t["wide"]) & (v > -t["droop"] * past)
            & (v < t["top"] + (t["end_top"] - t["top"]) * past)).astype(np.float32)
    zone = np.minimum(smoother._soft_blur(zone, t["edge"] * eyes) * np.float32(2.0), np.float32(1.0)) * valid * (skin > 0.02)
    zone = zone * (1.0 - smoother._smoothstep(past, t["taper"], 1.0))

    # Is there a moustache, and is it one to keep? The strip just above the lip says.
    there = 1.0
    if not force:
        strip = (valid > 0.98) & (skin > 0.5) & (np.abs(u) < t["strip_wide"]) & (v > t["strip_lo"]) & (v < t["strip_hi"])
        if int(strip.sum()) < 30:
            return False
        values = dark[strip]
        there = float(smoother._smoothstep(np.float32(values.mean(dtype=np.float64)), t["there_lo"], t["there_hi"]))
        dense = float(smoother._smoothstep(np.float32(_percentile(values, 90.0)), t["dense_lo"], t["dense_hi"]))
        if full and there > 0.0 and dense > 0.0:
            return _clear_full(out, f, mask, amount, keep)        # a full one, and asked for: it goes whole, its own way
        there *= 1.0 - dense
        if there <= 0.0:
            return False

    # The new tone meets the clear skin beside the hair.
    hairy = smoother._smoothstep(dark, t["dark_lo"], t["dark_hi"])
    clear_skin = (1.0 - hairy) * valid * (skin > 0.5)
    reach = t["meet"] * eyes
    off = np.clip(tone_y / (target_y + np.float32(1e-4)), t["meet_lo"], t["meet_hi"])
    meet = (smoother._soft_blur(off * clear_skin, reach) + np.float32(t["meet_prior"])) \
        / (smoother._soft_blur(clear_skin, reach) + np.float32(t["meet_prior"]))
    final = target * meet[:, :, None]
    final_y = _luma(final)

    hair = np.minimum(smoother._soft_blur(hairy * zone, t["grow"] * eyes) * np.float32(2.0), np.float32(1.0)) * zone
    hair = hair * np.float32(there * min(1.0, float(amount)))
    if not float(hair.max()) > 1e-3:
        return False
    # Only ever towards lighter: where the skin is already as light as a bare lip, its tone stays.
    lift = smoother._smoothstep((final_y - tone_y) / (final_y + np.float32(1e-4)), -t["dark_lo"], t["dark_lo"])
    change = (lift[:, :, None] * (final - tone)).astype(np.float32)

    def up(a_: np.ndarray) -> np.ndarray:
        return a_ if scale >= 1.0 else cv2.resize(a_, (bw, bh), interpolation=cv2.INTER_LINEAR)

    # At the photo's own size: the grain.
    paint, ok = up(hair), up(valid)
    if keep is not None:
        kx, ky, edit = keep
        ax0, ay0 = max(x0, kx), max(y0, ky)
        ax1, ay1 = min(x1, kx + edit.shape[1]), min(y1, ky + edit.shape[0])
        if ax1 > ax0 and ay1 > ay0:
            paint = paint.copy()
            paint[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] *= 1.0 + np.minimum(edit[ay0 - ky:ay1 - ky, ax0 - kx:ax1 - kx], 0.0)
    grain = (box - up(tone)) * ok[:, :, None]
    grain_y = _luma(grain)
    usable = (box_mask > 0.9).astype(np.float32) * ok
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float64)
    near = np.zeros((bh, bw), bool)
    for c in f.cheeks:
        near |= (xx + x0 - c[0]) ** 2 + (yy + y0 - c[1]) ** 2 < (t["cheek"] * d) ** 2
    quiet = np.abs(grain_y[near & (box_mask > 0.9)])
    limit = t["grain_most"] * _percentile(quiet, 25.0)
    side_of = smoother._smoothstep(up(u), -t["seam"], t["seam"])           # 0: the right cheek's half, 1: the left's
    borrowed = np.zeros((bh, bw), np.float32)
    have = np.zeros((bh, bw), np.float32)
    for c, side, share in ((f.cheeks[0], -1.0, 1.0 - side_of), (f.cheeks[1], 1.0, side_of)):
        here = f.centre + side * t["side"] * f.half + t["side_up"] * f.up
        dx, dy = int(round(float(c[0] - here[0]))), int(round(float(c[1] - here[1])))
        borrowed += share * _moved(grain_y, dx, dy)
        have += share * _moved(usable, dx, dy) * _inside((bh, bw), dx, dy)
    borrowed = borrowed * have                         # no skin to borrow from: left smooth
    borrowed = borrowed * np.minimum(np.float32(1.0), np.float32(limit) / (np.abs(borrowed) + np.float32(1e-6)))

    new = box + paint[:, :, None] * (up(change) + borrowed[:, :, None] - grain)
    result = np.clip(new * 255.0 + 0.5, 0, 255).astype(np.uint8)
    here = out[y0:y1, x0:x1]
    if np.array_equal(result, here):
        return False
    here[:] = result
    return True


def _spline(points: np.ndarray, steps: int = 8) -> np.ndarray:
    """A smooth open curve through the points (Catmull-Rom), both ends included."""
    p = np.asarray(points, np.float64)
    q = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(q) - 2):
        p0, p1, p2, p3 = q[i - 1], q[i], q[i + 1], q[i + 2]
        for j in range(steps):
            s = j / steps
            out.append(0.5 * ((2.0 * p1) + (p2 - p0) * s + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * s * s
                              + (3.0 * p1 - p0 - 3.0 * p2 + p3) * s * s * s))
    out.append(p[-1])
    return np.array(out)


def _fill_in(data: np.ndarray, weight: np.ndarray, radii, unit: float, lo: float, hi: float) -> np.ndarray:
    """data (h, w, c), known where weight is 1, carried everywhere: the widest radius
    first, each finer one taking over where enough of what it sees is known."""
    out = None
    for r in sorted(radii, reverse=True):
        seen = smoother._soft_blur(weight, r * unit)
        here = smoother._soft_blur(data * weight[:, :, None], r * unit) / (seen[:, :, None] + np.float32(1e-6))
        if out is None:
            out = here
        else:
            c = smoother._smoothstep(seen, lo, hi)[:, :, None]
            out = out * (1.0 - c) + here * c
    return out


def _grown(seed: np.ndarray, within: np.ndarray) -> np.ndarray:
    """seed spread through within (both bool), neighbour to neighbour, until it stops."""
    k = np.ones((3, 3), np.uint8)
    inside = within.astype(np.uint8)
    cur = seed.astype(np.uint8) & inside
    for _ in range(seed.shape[0] + seed.shape[1]):
        nxt = cv2.dilate(cur, k) & inside
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur.astype(bool)


def _round_kernel(radius: float) -> np.ndarray:
    r = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _pieces(grain_y: np.ndarray, usable: np.ndarray, x0: int, y0: int, cheeks, heres, shares, size: float, wander: float,
            limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Grain for new skin, at the photo's own size: pieces of each cheek's own patch, laid side by side and
    blended into each other, each piece from a different part of the patch. grain_y, usable: the box's grain
    (brightness) and where it is skin to borrow from; x0, y0: the box's corner in the photo. For each cheek
    (its middle, in the photo): heres, where the middle of its first piece lies, and shares, how much of
    each pixel's grain is that cheek's. size: a piece's; wander: how far from the cheek's middle a piece may
    be taken; limit: the most a pixel's grain may be. Returns the grain and how much of it had skin behind it."""
    bh, bw = grain_y.shape
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float64)
    wander_x, wander_y = (np.array([w_[i] for w_ in WANDER]) for i in (0, 1))
    borrowed = np.zeros((bh, bw), np.float32)
    have = np.zeros((bh, bw), np.float32)
    for c, here, share in zip(cheeks, heres, shares):
        rx, ry = (xx + x0 - here[0]) / size, (yy + y0 - here[1]) / size
        total = np.zeros((bh, bw), np.float64)
        got = np.zeros((bh, bw), np.float64)
        squares = np.zeros((bh, bw), np.float64)
        for layer, (ox, oy) in enumerate(((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))):     # four layers of pieces, each half a piece along
            ax, ay = rx + ox, ry + oy
            nx, ny = np.floor(ax + 0.5), np.floor(ay + 0.5)                 # which piece,
            qx, qy = ax - nx, ay - ny                                       # and where inside it, -0.5..0.5
            weight = (np.cos(np.pi * qx) * np.cos(np.pi * qy)) ** 2         # the four layers' weights add up to 1
            pick = np.mod(nx * 3.0 + ny * 5.0 + layer * 2.0, len(WANDER)).astype(np.int64)
            sx = np.round(c[0] - x0 + qx * size + wander_x[pick] * wander).astype(np.int64)
            sy = np.round(c[1] - y0 + qy * size + wander_y[pick] * wander).astype(np.int64)
            inside = (sx >= 0) & (sx < bw) & (sy >= 0) & (sy < bh)
            sx, sy = np.clip(sx, 0, bw - 1), np.clip(sy, 0, bh - 1)
            total += weight * grain_y[sy, sx]
            got += weight * usable[sy, sx] * inside
            squares += weight * weight
        borrowed += share * (total / np.sqrt(np.maximum(squares, 1e-6))).astype(np.float32)   # blended, and as strong as one piece
        have += share * got.astype(np.float32)
    borrowed = borrowed * have                         # no skin to borrow from: left smooth
    borrowed = borrowed * np.minimum(np.float32(1.0), np.float32(limit) / (np.abs(borrowed) + np.float32(1e-6)))
    return borrowed, have


def _clear_full(out: np.ndarray, f: _Frame, mask: np.ndarray, amount: float, keep, seen: dict | None = None) -> bool:
    """A full moustache on one face, changed in place in out (see the top of the file).
    Returns whether anything changed. seen, if given, receives what was found on the
    way (the box, the hair, the replaced area...), for looking at it."""
    t, hv = TUNE, FULL
    h, w = out.shape[:2]
    d = f.eyes
    # The box: as the thin path's, and further down beside the mouth for the moustache's ends.
    corners = [f.at(su * t["fit_wide"], v) for su in (-1.0, 1.0) for v in (hv["clear_lo"] - 0.1, 1.2)]
    xs = [c[0] for c in corners] + [c[0] + s * t["cheek"] * d for c in f.cheeks for s in (-1.0, 1.0)]
    ys = [c[1] for c in corners] + [c[1] + s * t["cheek"] * d for c in f.cheeks for s in (-1.0, 1.0)]
    pad = 3.0 * t["meet"] * d + 2.0
    x0, x1 = max(0, int(math.floor(min(xs) - pad))), min(w, int(math.ceil(max(xs) + pad)) + 1)
    y0, y1 = max(0, int(math.floor(min(ys) - pad))), min(h, int(math.ceil(max(ys) + pad)) + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    bw, bh = x1 - x0, y1 - y0
    scale = min(1.0, GRID_EYES / d)
    gw, gh = (max(8, round(bw * scale)), max(8, round(bh * scale))) if scale < 1.0 else (bw, bh)
    fx, fy = bw / gw, bh / gh
    eyes = d / ((fx + fy) / 2.0)

    box = out[y0:y1, x0:x1].astype(np.float32) / np.float32(255.0)
    box_mask = np.ascontiguousarray(mask[y0:y1, x0:x1], np.float32)
    small = box if scale >= 1.0 else cv2.resize(box, (gw, gh), interpolation=cv2.INTER_AREA)
    skin = box_mask if scale >= 1.0 else cv2.resize(box_mask, (gw, gh), interpolation=cv2.INTER_AREA)

    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float64)
    px, py = x0 + (gx + 0.5) * fx - 0.5, y0 + (gy + 0.5) * fy - 0.5
    u, v = f.uv(px, py)
    au = np.abs(u)

    # The lips, outlined smoothly: here on the copy for the tone, further down at the photo's size for the edge.
    upper, lower = _spline(f.lips[:len(UPPER)]), _spline(f.lips[len(UPPER):])
    outline = np.vstack([upper, lower[1:-1]])
    lips = np.zeros((gh, gw), np.uint8)
    poly = np.stack([(outline[:, 0] - x0 + 0.5) / fx - 0.5, (outline[:, 1] - y0 + 0.5) / fy - 0.5], axis=1)
    cv2.fillPoly(lips, [np.round(poly).astype(np.int32)], 255)
    valid = 1.0 - smoother._soft_blur(lips.astype(np.float32) / np.float32(255.0), max(0.6, t["lip_edge"] * eyes))

    sigma = t["tone"] * eyes
    tone = smoother._soft_blur(small * valid[:, :, None], sigma) / (smoother._soft_blur(valid, sigma)[:, :, None] + np.float32(1e-4))
    tone_y = _luma(tone)

    discs = [((px - c[0]) ** 2 + (py - c[1]) ** 2 < (t["cheek"] * d) ** 2) & (skin > 0.9) for c in f.cheeks]
    if min(int(m.sum()) for m in discs) < 30:
        return False
    cheek = (np.median(tone[discs[0]], axis=0) + np.median(tone[discs[1]], axis=0)).astype(np.float32) / np.float32(2.0)

    # A first reading, as the thin path's (the table fitted from above), to find the hair by.
    fit = (au < t["fit_wide"]) & (v > t["fit_lo"]) & (v < t["fit_hi"]) & (valid > 0.98) & (skin > 0.5)
    fys, fxs = np.nonzero(fit)
    if len(fys) < 200:
        return False
    step = max(1, len(fys) // t["fit_most"])
    fys, fxs = fys[::step], fxs[::step]
    shape_ = _table_at(u, v)
    a, b = _fit(np.log(np.maximum(tone_y[fys, fxs], 1e-3) / np.maximum(shape_[fys, fxs], 1e-3)), u[fys, fxs])
    first_y = _luma(_shaded(shape_ * np.exp(np.float32(a) + np.float32(b) * u), cheek))
    dark = (first_y - tone_y) / (first_y + np.float32(1e-4))

    # How rough the skin is, against the cheeks': strands are rough, shade is not.
    fine = (_luma(small) - tone_y) * valid
    at = hv["rough_at"] * eyes
    rough = np.sqrt(np.maximum(smoother._soft_blur(fine * fine, at), 0.0) / (smoother._soft_blur(valid, at) + np.float32(1e-4)))
    rough = rough / np.float32(max(float(np.median(rough[discs[0] | discs[1]])), 1e-5))

    # Where: from the dense core, through hair, to its thin ends, within bounds.
    low = -hv["drop"] * smoother._smoothstep(au, hv["drop_from"], hv["drop_to"])
    bounds = (au < hv["wide"]) & (v < hv["top"]) & (v > low)
    on_skin = (valid > 0.5) & (skin > 0.02) & bounds
    core = (dark > hv["core"]) & (au < hv["core_wide"]) & (v > hv["core_lo"]) & (v < hv["core_hi"]) & on_skin
    hairy = (((dark > hv["hair_lo"]) & (rough > hv["rough"])) | (dark > hv["hair_hi"])) & on_skin
    bridge = _round_kernel(hv["bridge"] * eyes)
    hairy = cv2.morphologyEx(hairy.astype(np.uint8), cv2.MORPH_CLOSE, bridge).astype(bool) & on_skin
    between = (rough > hv["mid_rough"]) & (au < hv["mid_wide"]) & (v > 0.0) & on_skin
    region = _grown(core, hairy | between)
    if not region.any():
        return False
    tails = (dark > hv["tail_dark"]) & (rough > hv["tail_rough"]) & on_skin \
        & (cv2.dilate(region.astype(np.uint8), _round_kernel(hv["tail_reach"] * eyes)) > 0)
    tails = cv2.morphologyEx(tails.astype(np.uint8), cv2.MORPH_CLOSE, bridge).astype(bool) & on_skin
    region = _grown(region, region | tails)
    # The replaced area: the hair and a little past it, soft-edged. The skin mask has the lips cut out
    # with a hard edge, so the area runs on over them: the lip's own soft edge is applied further down.
    grow = _round_kernel(hv["grow"] * eyes)
    paint = np.minimum(smoother._soft_blur(cv2.dilate(region.astype(np.uint8), grow).astype(np.float32), hv["soft"] * eyes)
                       * np.float32(1.5), np.float32(1.0))
    paint = paint * np.minimum(smoother._soft_blur(bounds.astype(np.float32), hv["bound_edge"] * eyes) * np.float32(2.0), np.float32(1.0))
    paint = paint * ((skin > 0.02) | (cv2.dilate(lips, grow) > 0))
    paint = paint * np.float32(min(1.0, float(amount)))
    if not float(paint.max()) > 1e-3:
        return False

    # The tone: the table's shape in the cheeks' colour, matched channel by channel to the clear skin around.
    e = t["red"]
    k = np.array([1.0 + e, 1.0 + e, 1.0 - e * (0.701 / 0.299)], np.float32)     # B, G, R: as _shaded's
    # (the table's shape is rounded first: over a whole lip its creases would show, the one down the middle most)
    # (and the philtrum is put in by itself, about the lip's own middle: the table's broad dip there is
    # taken out first, by reading the table no nearer the middle than where the philtrum's map ends)
    middle = f.mid[0] + (f.mid[1] - f.mid[0]) * np.clip(v, 0.0, 1.0)
    aside = np.abs(u - middle)
    beside = middle + np.where(u < middle, -1.0, 1.0) * np.maximum(aside, np.float32(hv["philtrum_to"][0]))
    rounded = smoother._soft_blur(_table_at(beside.astype(np.float32), v), hv["shape_soft"] * eyes)
    rows_, cols_ = PHILTRUM.shape
    a_ = np.clip(aside / PHILTRUM_U, 0.0, cols_ - 1.0)
    b_ = np.clip(v / PHILTRUM_V, 0.0, rows_ - 1.0)
    a0, b0 = np.minimum(np.floor(a_).astype(np.int64), cols_ - 2), np.minimum(np.floor(b_).astype(np.int64), rows_ - 2)
    fa, fb = (a_ - a0).astype(np.float32), (b_ - b0).astype(np.float32)
    marks = (PHILTRUM[b0, a0] * (1.0 - fa) + PHILTRUM[b0, a0 + 1] * fa) * (1.0 - fb) \
        + (PHILTRUM[b0 + 1, a0] * (1.0 - fa) + PHILTRUM[b0 + 1, a0 + 1] * fa) * fb
    marks = marks * (1.0 - smoother._smoothstep(aside, hv["philtrum_to"][0], hv["philtrum_to"][1]))
    marks = np.where(marks > 0.0, np.float32(hv["philtrum_light"]), np.float32(hv["philtrum"])) * marks
    rounded = (rounded * (1.0 + np.float32(0.01) * marks)).astype(np.float32)
    prior = cheek * np.power(np.maximum(rounded, np.float32(1e-3))[:, :, None], k)
    near_hair = cv2.dilate((region | hairy | (dark > hv["hair_hi"])).astype(np.uint8), _round_kernel(hv["clear_off"] * eyes)) > 0
    clear_skin = ((valid > 0.98) & (skin > 0.5) & ~near_hair & (au < t["fit_wide"])
                  & (v > hv["clear_lo"]) & (v < hv["clear_hi"])).astype(np.float32)
    off = np.log(np.maximum(tone, np.float32(1e-3)) / np.maximum(prior, np.float32(1e-3)))
    off = np.clip(off, math.log(hv["span"][0]), math.log(hv["span"][1])).astype(np.float32)
    final = prior * np.exp(_fill_in(off, clear_skin, hv["fill"], eyes, hv["fill_lo"], hv["fill_hi"]))
    final_y = _luma(final)
    # Clear skin is left as it is. A strand is darker than the skin around it; skin is replaced in the
    # moustache itself and, beside it, wherever strands are. The rest of the area the moustache was grown
    # by keeps the photo's own tone and grain, so a crease beside the mouth or the nose stays.
    y_small = _luma(small)
    at = hv["strand_at"] * eyes
    around = smoother._soft_blur(y_small * valid, at) / (smoother._soft_blur(valid, at) + np.float32(1e-4))
    rel = (y_small - around) / (around + np.float32(1e-4))
    own = rel[discs[0] | discs[1]]
    limit = min(max(hv["strand"] * 1.4826 * float(np.median(np.abs(own - np.median(own)))), hv["strand_span"][0]), hv["strand_span"][1])
    strand = ((rel < -limit) | (dark > hv["hair_hi"])) & (valid > 0.5)
    hairs = np.maximum(np.minimum(smoother._soft_blur(region.astype(np.float32), hv["own_soft"] * eyes) * np.float32(2.0), np.float32(1.0)),
                       smoother._smoothstep(smoother._soft_blur(strand.astype(np.float32), hv["own_at"] * eyes), hv["own"][0], hv["own"][1]))
    paint = paint * hairs
    # Hair goes whatever its tone; the clear skin beside it only ever gets lighter.
    lift = smoother._smoothstep((final_y - tone_y) / (final_y + np.float32(1e-4)), -t["dark_lo"], t["dark_lo"])
    lift = np.maximum(lift, np.minimum(smoother._soft_blur(region.astype(np.float32), hv["soft"] * eyes) * np.float32(2.0), np.float32(1.0)))
    change = (lift[:, :, None] * (final - tone)).astype(np.float32)

    def up(a_: np.ndarray) -> np.ndarray:
        return a_ if scale >= 1.0 else cv2.resize(a_, (bw, bh), interpolation=cv2.INTER_LINEAR)

    def filled(curve_a: np.ndarray, curve_b: np.ndarray) -> np.ndarray:
        """1 between two curves that run the same way, at the photo's size."""
        m_ = np.zeros((bh, bw), np.uint8)
        cv2.fillPoly(m_, [np.round(np.vstack([curve_a, curve_b[::-1]]) - [x0, y0]).astype(np.int32)], 255)
        return m_.astype(np.float32) / np.float32(255.0)

    # At the photo's own size: the lip's edge.
    lips_f = np.zeros((bh, bw), np.uint8)
    cv2.fillPoly(lips_f, [np.round(outline - [x0, y0]).astype(np.int32)], 255)
    lips_soft = smoother._soft_blur(lips_f.astype(np.float32) / np.float32(255.0), max(0.6, hv["lip_soft"] * d))
    ok = 1.0 - lips_soft
    paint_f = up(paint)
    hold = None
    if keep is not None:
        kx, ky, edit = keep
        ax0, ay0 = max(x0, kx), max(y0, ky)
        ax1, ay1 = min(x1, kx + edit.shape[1]), min(y1, ky + edit.shape[0])
        if ax1 > ax0 and ay1 > ay0:
            hold = np.ones((bh, bw), np.float32)
            hold[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] += np.minimum(edit[ay0 - ky:ay1 - ky, ax0 - kx:ax1 - kx], 0.0)
            paint_f = paint_f * hold

    # The grain: pieces of each cheek's own patch, laid over its half of the lip side by side and
    # blended into each other, each piece from a different part of the patch. So it never comes
    # from a fold, the nose or the backdrop, and no piece of cheek shows twice in a row.
    grain = box - up(tone)
    grain_y = _luma(grain) * ok
    usable = (box_mask > 0.9).astype(np.float32) * ok
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float64)
    near = np.zeros((bh, bw), bool)
    for c in f.cheeks:
        near |= (xx + x0 - c[0]) ** 2 + (yy + y0 - c[1]) ** 2 < (t["cheek"] * d) ** 2
    limit = t["grain_most"] * _percentile(np.abs(grain_y[near & (box_mask > 0.9)]), 25.0)
    side_of = smoother._smoothstep(up(u), -t["seam"], t["seam"])           # 0: the right cheek's half, 1: the left's
    borrowed, have = _pieces(grain_y, usable, x0, y0, f.cheeks, [f.centre + side * t["side"] * f.half + t["side_up"] * f.up for side in (-1.0, 1.0)],
                             (1.0 - side_of, side_of), hv["tile"] * d, hv["tile_wander"] * d, limit)
    new_skin = box + paint_f[:, :, None] * (up(change) + borrowed[:, :, None] - grain)

    # The lip's side of the edge: hair lying over the upper lip takes the lip's own colour, carried
    # along the lip from where it is clear (as bright as the lip's brighter half: hair only darkens).
    p = f.inner
    inner = _spline(p)
    away = upper - inner
    away = away / (np.hypot(away[:, 0], away[:, 1])[:, None] + 1e-6)
    away[:3], away[-3:] = away[len(upper) // 2], away[len(upper) // 2]     # at its corners the lip has no thickness to point along
    far = upper + away * (4.0 * hv["lip_soft"] * d)
    box_y = _luma(box)
    whole = filled(far, inner) * (lips_soft > 0.5)
    within = filled(upper, inner) > 0.5
    bright = float(np.median(box_y[within & (box_y > np.median(box_y[within]))])) if within.any() else 0.0
    known = whole * smoother._smoothstep(box_y / np.float32(max(bright, 1e-3)), hv["lip_keep"][0], hv["lip_keep"][1])
    if not float(known.sum()) > 10.0:
        known = whole
    lip_tone = _fill_in(box, known, hv["lip_fill"], d, hv["fill_lo"], hv["fill_hi"])
    strand = max(0.6, hv["lip_strand"] * d)
    on_lip = smoother._smoothstep(1.0 - smoother._soft_blur(box_y, strand) / (_luma(lip_tone) + np.float32(1e-4)),
                                  hv["lip_dark"][0], hv["lip_dark"][1])
    band = smoother._soft_blur(filled(far, upper + hv["lip_band"] * (inner - upper)), max(0.6, 0.5 * hv["lip_soft"] * d))
    beside = np.minimum(smoother._soft_blur(paint_f, hv["lip_reach"] * d) * np.float32(3.0), np.float32(1.0))
    on_lip = np.minimum(smoother._soft_blur(on_lip * band * beside, strand) * np.float32(1.5), np.float32(1.0))
    on_lip = on_lip * np.float32(min(1.0, float(amount)))
    if hold is not None:
        on_lip = on_lip * hold
    new_lip = box + on_lip[:, :, None] * (lip_tone + np.float32(hv["lip_grain"]) * borrowed[:, :, None] - box)

    new = new_skin * ok[:, :, None] + new_lip * lips_soft[:, :, None]
    if seen is not None:
        seen.update(box=(x0, y0, x1, y1), frame=f, u=u, v=v, dark=dark, rough=rough, hairy=hairy, region=region,
                    paint=paint, paint_f=paint_f, clear_skin=clear_skin, final=final, prior=prior, tone=tone,
                    strand=strand, hairs=hairs,
                    have=have, lips_soft=lips_soft, lips_f=lips_f > 0, on_lip=on_lip, lip_tone=lip_tone, band=band)
    result = np.clip(new * 255.0 + 0.5, 0, 255).astype(np.uint8)
    here = out[y0:y1, x0:x1]
    if np.array_equal(result, here):
        return False
    here[:] = result
    return True


def clear(bgr: np.ndarray, points, mask: np.ndarray, amount: float, keep=None, force: bool = False,
          full: bool = False) -> np.ndarray:
    """bgr (uint8) with a thin moustache taken off each face's upper lip: a new array,
    or bgr itself when nothing changes. points: each face's landmarks or None
    (face_landmarks.landmarks). mask: the smoothing mask (where the skin is). amount:
    0..1, how much of it goes. keep: the brush's Restore edit (x0, y0, float32 -1..1),
    where the lip stays as it is. force: skip the questions whether the face is posed for
    it and whether there is a moustache (for the parity check, whose photos have neither).
    full: a full, dense moustache goes too (its own way, _clear_full); without it one is
    left alone. A thin moustache comes out the same with or without full."""
    if amount <= 0.0 or not points:
        return bgr
    out = bgr.copy()
    changed = False
    for pts in points:
        if pts is not None:
            changed = _clear_face(out, pts, mask, amount, keep, force, full) or changed
    return out if changed else bgr
