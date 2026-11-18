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
dense moustache is left alone too: taking half of one away helps nobody, and it reaches
past what this step replaces. Stubble or a beard around the mouth is not touched, so a
face with both loses the moustache and keeps the beard: leave the step off for it.

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
        self.up = p[NOSE_BASE] - p[LIP_TOP]
        edge = p[list(UPPER)]
        across = (edge - self.centre) @ self.half / max(self.half2, 1e-9)
        order = np.argsort(across, kind="stable")
        self.edge_u, self.edge_x, self.edge_y = across[order], edge[order, 0], edge[order, 1]
        self.lips = p[list(UPPER + LOWER)]
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


def _clear_face(out: np.ndarray, pts, mask: np.ndarray, amount: float, keep, force: bool) -> bool:
    """One face, changed in place in out. Returns whether anything changed."""
    t = TUNE
    f = _Frame(pts)
    if not f.ok or not (f.posed or force):
        return False
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
        there *= 1.0 - float(smoother._smoothstep(np.float32(_percentile(values, 90.0)), t["dense_lo"], t["dense_hi"]))
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


def clear(bgr: np.ndarray, points, mask: np.ndarray, amount: float, keep=None, force: bool = False) -> np.ndarray:
    """bgr (uint8) with a thin moustache taken off each face's upper lip: a new array,
    or bgr itself when nothing changes. points: each face's landmarks or None
    (face_landmarks.landmarks). mask: the smoothing mask (where the skin is). amount:
    0..1, how much of it goes. keep: the brush's Restore edit (x0, y0, float32 -1..1),
    where the lip stays as it is. force: skip the questions whether the face is posed for
    it and whether there is a moustache (for the parity check, whose photos have neither)."""
    if amount <= 0.0 or not points:
        return bgr
    out = bgr.copy()
    changed = False
    for pts in points:
        if pts is not None:
            changed = _clear_face(out, pts, mask, amount, keep, force) or changed
    return out if changed else bgr
