"""
A beard on the chin, taken off: clear() replaces it, as lip.clear(..., full=True) does a
full moustache, with the same engine (the tone put there anew, the cheeks' grain over it).

It is for hair on the chin itself, between the lower lip and a little below the jaw's
edge: a sparse goatee, straggly chin hair, a tuft under the lip. What a chin has that
an upper lip has not, and what is done about it:

  frame    Places are given across the chin (u: -1 and 1 under the mouth's corners) and
           down it (v: 0 at the lower lip's edge, 1 at the jaw's edge, on below it to the
           neck). The work is done on a copy of the chin straightened into this frame,
           so the lip's edge, the fold under it and the jaw's edge each run along a row.
  jaw      The mesh's outline lies a little off the photo's own jaw edge (0.03 eye
           distances on average over the studio's portraits). The frame is moved onto
           the edge as the photo has it: read beside the hair, a smooth curve under it.
  where    A strand is darker than what lies left and right of it on its row, so the
           jaw's edge, its shadow and the fold under the lip never count: they run
           along the rows, hair grows across them. Whether there is a beard at all is
           decided by strands that are line-shaped (a pore or a freckle is a dot) and
           by masses of hair; where they are dense is its core. From there the area is
           followed outwards through whatever is dark, within bounds: narrow under the
           lip (the mouth's corners are the moustache step's), wider on the chin, a
           little below the jaw for hair that hangs over the neck.
  tone     What a bare chin looks like there: a small table's shape (TABLE: the fold
           under the lip, the chin's highlight, the jaw's shadow, averaged over
           retouched portraits) in the cheeks' colour, matched colour channel by colour
           channel to the clear skin and carried inwards along the rows, never across
           them: under the hair the jaw's line and the shadow below it are the face's
           own, continued from left and right. Skin between sparse strands counts as
           clear unless it is darker than the skin around the beard says it should be:
           hair only darkens, and between dense strands the skin is in their shade.
  grain    The moustache's: pieces of each cheek's own patch (lip._pieces), a little
           quieter where the new tone is in shade.

It acts on a face straight on with its mouth closed (lip.POSE) whose chin has a usual
height, and only when asked: most who wear a beard keep it. A chin with no beard is
left as it is. Stubble or a beard across the jaw's sides is not what this is for: where
hair runs on past the bounds the face is left alone, since a clean patch cut out of
stubble shows. Short, even stubble is not always recognised as running on; then the
step may act in the middle of it.

What cannot be known is left plain: under the middle of the chin the shadow is often
deeper than beside it, and under a beard that cannot be seen, so the new chin's shadow
is as deep as it is beside the beard (shaped by the table). Single hairs well away
from the beard stay; the brush takes them. A mole inside the beard goes with it; the
brush's Restore brings it back (keep).

As the moustache step, the tone is worked out on a copy at one fixed size (lip.GRID_EYES
between the eyes) and only the grain and the lip's edge at the photo's own size, so a
preview and the full-size photo come out alike. The step comes right after lip.clear.
"""

from __future__ import annotations

import math

import numpy as np
import cv2

import smoother
import lip

# Face Mesh points.
LOWER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)      # the lower lip's outer edge, right corner to left
JAW = (136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365)    # the face's outline under the mouth, the same way
CHIN_POINT, LIP_FOOT = 152, 17

# The brightness of a bare chin, place by place, as a share of the cheeks'. Rows: from the
# lower lip's edge (first) down to the jaw's edge (the eleventh) and on to the neck, in
# tenths. Columns: from the middle outwards, in fifths of the way to under the mouth's
# corner (so the sixth column is under the corner). The average of retouched portraits
# with a bare chin, both sides mirrored together, each face's jaw edge found on the photo.
# Only its shape matters: its level is matched to each face.
TABLE = np.array([
    (0.74, 0.75, 0.78, 0.80, 0.78, 0.71, 0.79, 0.85, 0.88, 0.88),
    (0.74, 0.75, 0.78, 0.80, 0.78, 0.77, 0.80, 0.85, 0.87, 0.88),
    (0.83, 0.83, 0.83, 0.84, 0.84, 0.84, 0.84, 0.85, 0.86, 0.86),
    (0.89, 0.90, 0.89, 0.88, 0.88, 0.87, 0.86, 0.85, 0.86, 0.85),
    (0.92, 0.92, 0.92, 0.91, 0.89, 0.88, 0.86, 0.85, 0.85, 0.85),
    (0.92, 0.92, 0.94, 0.92, 0.90, 0.88, 0.86, 0.85, 0.84, 0.84),
    (0.89, 0.90, 0.92, 0.92, 0.90, 0.88, 0.86, 0.83, 0.83, 0.83),
    (0.85, 0.86, 0.87, 0.88, 0.88, 0.87, 0.85, 0.83, 0.82, 0.82),
    (0.82, 0.82, 0.83, 0.83, 0.85, 0.85, 0.83, 0.82, 0.81, 0.81),
    (0.76, 0.76, 0.77, 0.79, 0.81, 0.82, 0.80, 0.80, 0.80, 0.79),
    (0.67, 0.68, 0.71, 0.74, 0.76, 0.77, 0.76, 0.76, 0.77, 0.76),
    (0.65, 0.67, 0.69, 0.72, 0.74, 0.75, 0.75, 0.76, 0.76, 0.75),
    (0.72, 0.73, 0.73, 0.76, 0.77, 0.78, 0.76, 0.77, 0.77, 0.75),
    (0.78, 0.78, 0.79, 0.80, 0.80, 0.81, 0.80, 0.80, 0.79, 0.76),
    (0.81, 0.81, 0.82, 0.81, 0.81, 0.82, 0.82, 0.81, 0.80, 0.78),
    (0.83, 0.84, 0.84, 0.84, 0.84, 0.83, 0.83, 0.82, 0.80, 0.79),
    (0.84, 0.84, 0.83, 0.86, 0.87, 0.87, 0.84, 0.83, 0.81, 0.79),
], np.float32)
TABLE_U, TABLE_V = 0.2, 0.1        # the table's steps: across (1 = under the mouth's corner), and down (1 = the jaw's edge)

# Places in the chin's frame (u across, v down), sizes in distances between the eyes.
TUNE = {
    "tall": (0.36, 0.68),      # a chin it acts on: lower lip to chin point (80 ID portraits: 0.385 to 0.644)
    "span_u": 1.9, "span_v": (-0.25, 1.7),      # the straightened copy covers this much of the frame
    "pad": 0.3,                # room around it in the photo
    # the jaw's edge on the photo
    "edge_reach": 0.10,        # looked for this far either side of the mesh's outline,
    "edge_soft": (0.03, 0.012),    # on a copy smoothed this much along the row and across it;
    "edge_least": 0.012,       # a darkening of at least this share over two rows counts,
    "edge_out": 0.03,          # a column this far from the curve through the others does not,
    "edge_columns": 12, "edge_span": 1.0,       # and fewer columns than this, or over less width (u), move nothing
    "edge_even": 0.05,         # beside the hair the edge is taken as read, evened over this much (u)
    "edge_off": 0.03,          # columns this near the hair are not read
    # the hair
    "row": 0.03,               # a strand is darker than its row within this distance,
    "strand": 3.0,             # by this many times the cheeks' own grain,
    "strand_least": 0.04, "strand_most": 0.08,  # held between these shares;
    "long": 0.035,             # a line of them at least this long, running down or aslant, says hair: a pore or a freckle is a dot
    "mass_soft": 0.012, "mass_row": 0.25, "mass": 0.16,   # a mass of hair: darker than its row at this distance by this share, the table's shape taken out
    "dense_at": 0.035,         # lines and masses are counted within this radius,
    "line_gain": 3.0,          # a line this many times over (it is a pixel wide, a mass is solid);
    "core": 0.13,              # this share of them is the beard's core,
    "least": 0.02,             # and a core smaller than this (eye distances squared) is not a beard
    "hair": 0.06,              # this share of dark pixels is still hair, followed outwards from the core,
    "tail_reach": 0.10,        # and single strands within this reach of it go with it
    "bridge": 0.015,           # gaps this small are closed
    "grow": 0.035, "soft": 0.02,                # the replaced area grows past the hair by the first, with the second for its soft edge
    "top": 0.02, "foot": 1.35,                  # the bounds: from just under the lip to this far below the jaw,
    "wide_top": 0.8, "wide": 1.35, "widen": (0.15, 0.55),   # this wide under the lip, widening to this over these heights
    "bound_edge": 0.03,        # the bounds' soft edge
    "over_lip": -0.3,          # (the area may run on over the lip this far: the lip's own edge cuts it)
    "flank": 0.35, "flank_v": (0.3, 1.15),      # just outside the bounds, this far out and between these heights,
    "flank_most": 0.30,        # this share of hair: a beard across the jaw, left alone
    # the tone
    "fill": (0.5, 0.25, 0.12, 0.06, 0.03),      # carried along the row at these radii, the widest first,
    "across": 0.25, "across_most": 0.004,       # and across it at this share of them, never wider than this: the jaw's edge is a row
    "fill_lo": 0.08, "fill_hi": 0.35,           # each finer one taking over where this share of what it sees is clear skin
    "span": (0.6, 1.4),        # the carried tone against the table's, per channel, is held to this
    "clear_off": 1.0,          # clear skin starts this many copy pixels from a strand,
    "clear_dense": (0.08, 0.2),    # and where strands are this dense the skin between them is in their shade: not clear
    "first_off": 0.04,         # a first tone rests on skin this far from the beard;
    "above": (-0.05, -0.015),  # inside the beard, skin darker than it by this share is in the hair's shade, not clear
    "above_at": 0.008,         # (judged at this radius)
    # at the photo's size
    "tile": 0.12, "tile_wander": 0.07,          # the grain's pieces, as the moustache's
    "side": 0.6, "side_down": 0.5,              # each half of the chin borrows from its own cheek; the first piece lies here (u, v)
    "grain_shade": (0.8, 1.2), # the grain follows the new tone's brightness against the cheeks', within this
    "lip_soft": 0.012,         # the lip's edge is this soft
}


def _blur2(a: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """smoother._soft_blur with a different sigma along the row (sx) and across it (sy); the edge is repeated."""
    rx = max(0, round((math.sqrt(4.0 * sx * sx + 1.0) - 1.0) / 2.0))
    ry = max(0, round((math.sqrt(4.0 * sy * sy + 1.0) - 1.0) / 2.0))
    for _ in range(3):
        a = cv2.boxFilter(a, cv2.CV_32F, (2 * rx + 1, 2 * ry + 1), normalize=True, borderType=cv2.BORDER_REPLICATE)
    return a


def _row_mean(a: np.ndarray, on: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """a averaged over what is on (1 or 0) around each pixel."""
    return _blur2(a * on, sx, sy) / (_blur2(on, sx, sy) + np.float32(1e-4))


def _fill_in(data: np.ndarray, weight: np.ndarray, unit: float) -> np.ndarray:
    """lip._fill_in along the rows: data (h, w, c), known where weight is, carried everywhere."""
    t = TUNE
    out = None
    for r in sorted(t["fill"], reverse=True):
        sx, sy = r * unit, min(r * t["across"], t["across_most"]) * unit
        seen = _blur2(weight, sx, sy)
        here = _blur2(data * weight[:, :, None], sx, sy) / (seen[:, :, None] + np.float32(1e-6))
        if out is None:
            out = here
        else:
            c = smoother._smoothstep(seen, t["fill_lo"], t["fill_hi"])[:, :, None]
            out = out * (1.0 - c) + here * c
    return out


def _line_kernels(n: int) -> list:
    """Thin lines n long: straight down, and leaning 27 and 45 degrees either way."""
    n = n | 1
    mid = n // 2
    out = []
    for lean in (0.0, 0.5, -0.5, 1.0, -1.0):
        k = np.zeros((n, n), np.uint8)
        for i in range(n):
            k[i, int(round(mid + (i - mid) * lean))] = 1
        out.append(k)
    return out


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


class _Frame:
    """One face's chin frame, in photo pixels."""

    def __init__(self, pts: np.ndarray):
        p = np.asarray(pts, np.float64)[:, :2]
        self.mouth = lip._Frame(pts)
        self.eyes = self.mouth.eyes
        self.centre, self.half, self.half2 = self.mouth.centre, self.mouth.half, self.mouth.half2
        self.wide = math.sqrt(max(self.half2, 1e-9))                         # half the mouth's width, px
        self.down = np.array([-self.half[1], self.half[0]]) / self.wide      # one pixel towards the chin: away from the nose
        if float(self.down @ (self.centre - p[lip.NOSE_BASE])) < 0.0:
            self.down = -self.down
        self.top_u, self.top_h = self._curve(p[list(LOWER)])
        self.jaw_u, self.jaw_h = self._curve(p[list(JAW)])
        self.cheeks = self.mouth.cheeks
        self.lips = self.mouth.lips
        self.tall = float((p[CHIN_POINT] - p[LIP_FOOT]) @ self.down) / max(self.eyes, 1e-9)
        self.ok = bool(self.mouth.ok and 0.15 < self.tall < 1.5 and float(self.jaw(0.0) - self.top(0.0)) > 2.0)
        self.posed = bool(self.ok and self.mouth.posed and TUNE["tall"][0] <= self.tall <= TUNE["tall"][1])

    def _curve(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Points as a smooth curve (u, h): across, and how far below the mouth's line (px)."""
        u = (q - self.centre) @ self.half / max(self.half2, 1e-9)
        h = (q - self.centre) @ self.down
        order = np.argsort(u, kind="stable")
        s = lip._spline(np.stack([u[order], h[order]], axis=1))
        order = np.argsort(s[:, 0], kind="stable")
        return s[order, 0], s[order, 1]

    def top(self, u):
        """The lower lip's edge (px below the mouth's line); beyond the mouth's corners, as at them."""
        return np.interp(np.clip(u, -1.0, 1.0), self.top_u, self.top_h)

    def jaw(self, u):
        """The jaw's edge, the same way."""
        return np.interp(u, self.jaw_u, self.jaw_h)

    def at(self, u, v) -> tuple:
        """The photo's places (x, y) for places in the frame (arrays)."""
        h0, h1 = self.top(u), self.jaw(u)
        h = h0 + v * (h1 - h0)
        return self.centre[0] + u * self.half[0] + h * self.down[0], self.centre[1] + u * self.half[1] + h * self.down[1]

    def uv(self, xx: np.ndarray, yy: np.ndarray) -> tuple:
        """The frame's u and v at photo places (arrays)."""
        u = ((xx - self.centre[0]) * self.half[0] + (yy - self.centre[1]) * self.half[1]) / self.half2
        h = (xx - self.centre[0]) * self.down[0] + (yy - self.centre[1]) * self.down[1]
        h0, h1 = self.top(u), self.jaw(u)
        return u, (h - h0) / np.maximum(h1 - h0, 1e-3)


class _Work:
    """The chin's box of one photo, its copy at the fixed size, and that copy straightened into the frame."""

    def __init__(self, out: np.ndarray, f: _Frame, mask: np.ndarray):
        t, lt = TUNE, lip.TUNE
        self.f, self.ok = f, False
        h, w = out.shape[:2]
        d = f.eyes
        span, (v0, v1) = t["span_u"], t["span_v"]
        cu, cv_ = np.meshgrid(np.array([-span, span]), np.array([v0, v1]))
        cx, cy = f.at(cu.ravel(), cv_.ravel())
        xs = list(cx) + [c[0] + s * lt["cheek"] * d for c in f.cheeks for s in (-1.0, 1.0)]
        ys = list(cy) + [c[1] + s * lt["cheek"] * d for c in f.cheeks for s in (-1.0, 1.0)]
        pad = t["pad"] * d + 2.0
        x0, x1 = max(0, int(math.floor(min(xs) - pad))), min(w, int(math.ceil(max(xs) + pad)) + 1)
        y0, y1 = max(0, int(math.floor(min(ys) - pad))), min(h, int(math.ceil(max(ys) + pad)) + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        self.box_at = (x0, y0, x1, y1)
        bw, bh = x1 - x0, y1 - y0
        self.scale = min(1.0, lip.GRID_EYES / d)
        gw, gh = (max(8, round(bw * self.scale)), max(8, round(bh * self.scale))) if self.scale < 1.0 else (bw, bh)
        self.size, self.small_size = (bw, bh), (gw, gh)
        self.fx, self.fy = bw / gw, bh / gh                    # photo pixels per copy pixel
        self.eyes = d / ((self.fx + self.fy) / 2.0)            # the distance between the eyes, on the copy
        self.box = out[y0:y1, x0:x1].astype(np.float32) / np.float32(255.0)
        self.box_mask = np.ascontiguousarray(mask[y0:y1, x0:x1], np.float32)
        self.small = self.box if self.scale >= 1.0 else cv2.resize(self.box, (gw, gh), interpolation=cv2.INTER_AREA)
        self.skin = self.box_mask if self.scale >= 1.0 else cv2.resize(self.box_mask, (gw, gh), interpolation=cv2.INTER_AREA)
        gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float64)
        px, py = x0 + (gx + 0.5) * self.fx - 0.5, y0 + (gy + 0.5) * self.fy - 0.5      # the photo's place for each copy pixel
        # The lips, outlined smoothly and cut out of the tone (as lip._clear_full has them).
        upper, lower = lip._spline(f.lips[:len(lip.UPPER)]), lip._spline(f.lips[len(lip.UPPER):])
        self.outline = np.vstack([upper, lower[1:-1]])
        lips = np.zeros((gh, gw), np.uint8)
        poly = np.stack([(self.outline[:, 0] - x0 + 0.5) / self.fx - 0.5, (self.outline[:, 1] - y0 + 0.5) / self.fy - 0.5], axis=1)
        cv2.fillPoly(lips, [np.round(poly).astype(np.int32)], 255)
        self.valid = 1.0 - smoother._soft_blur(lips.astype(np.float32) / np.float32(255.0), max(0.6, lt["lip_edge"] * self.eyes))
        sigma = lt["tone"] * self.eyes
        self.tone = smoother._soft_blur(self.small * self.valid[:, :, None], sigma) \
            / (smoother._soft_blur(self.valid, sigma)[:, :, None] + np.float32(1e-4))
        self.discs = [((px - c[0]) ** 2 + (py - c[1]) ** 2 < (lt["cheek"] * d) ** 2) & (self.skin > 0.9) for c in f.cheeks]
        if min(int(m.sum()) for m in self.discs) < 30:
            return
        self.cheek = (np.median(self.tone[self.discs[0]], axis=0) + np.median(self.tone[self.discs[1]], axis=0)).astype(np.float32) / np.float32(2.0)
        self.ok = True
        self.straighten()

    def straighten(self) -> None:
        """Where each pixel of the straightened copy (columns u, rows v) lies on the copy, for the frame as it is now."""
        t, f = TUNE, self.f
        span, (v0, v1) = t["span_u"], t["span_v"]
        px = f.eyes / self.eyes                                 # photo pixels per copy pixel
        tall = float(f.jaw(0.0) - f.top(0.0))                   # the chin's height in the middle, px
        self.nu, self.nv = max(8, int(round(2.0 * span * f.wide / px))), max(8, int(round((v1 - v0) * tall / px)))
        self.du, self.dv = 2.0 * span / self.nu, (v1 - v0) / self.nv
        self.u1, self.v1 = -span + (np.arange(self.nu) + 0.5) * self.du, v0 + (np.arange(self.nv) + 0.5) * self.dv
        self.u, self.v = np.meshgrid(self.u1.astype(np.float32), self.v1.astype(np.float32))
        x, y = f.at(self.u.astype(np.float64), self.v.astype(np.float64))
        x0, y0 = self.box_at[:2]
        self.mx, self.my = ((x - x0 + 0.5) / self.fx - 0.5).astype(np.float32), ((y - y0 + 0.5) / self.fy - 0.5).astype(np.float32)
        gw, gh = self.small_size
        self.inside = ((self.mx > 0) & (self.mx < gw - 1) & (self.my > 0) & (self.my < gh - 1)).astype(np.float32)

    def rect(self, a: np.ndarray) -> np.ndarray:
        """A map of the copy, straightened."""
        return cv2.remap(np.ascontiguousarray(a, np.float32), self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def back(self, a: np.ndarray) -> np.ndarray:
        """A straightened map, at the photo's own size (the box); 0 outside it."""
        x0, y0, x1, y1 = self.box_at
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        u, v = self.f.uv(xx, yy)
        span, v0 = TUNE["span_u"], TUNE["span_v"][0]
        j, i = ((u + span) / self.du - 0.5).astype(np.float32), ((v - v0) / self.dv - 0.5).astype(np.float32)
        return cv2.remap(np.ascontiguousarray(a, np.float32), j, i, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _bounds(u: np.ndarray, v: np.ndarray, top: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Where the step may act, and how wide that is at each height."""
    t = TUNE
    wide = t["wide_top"] + (t["wide"] - t["wide_top"]) * smoother._smoothstep(v, t["widen"][0], t["widen"][1])
    return (np.abs(u) < wide) & (v > (t["top"] if top is None else top)) & (v < t["foot"]), wide


def _find(w: _Work) -> dict:
    """The beard on the straightened copy: its strands, masses, core, area ("region", None when there is no
    beard) and the replaced area ("paint"), with the maps they were found by."""
    t = TUNE
    e = w.eyes
    g = w.rect(w.small)
    skin = w.rect(w.skin) * w.inside
    y = lip._luma(g)
    on = (skin > 0.5).astype(np.float32) * (w.v > 0.0)
    below = np.minimum(on, w.rect(w.valid) > 0.98).astype(np.float32)         # skin, the lips and their soft edge left out
    row = _row_mean(y, on, t["row"] * e, 0.0)
    rel = (y - row) / (row + np.float32(1e-4))
    # The cheeks' own grain, read the same way.
    ys = lip._luma(w.small)
    rs = _blur2(ys, t["row"] * e, 0.0)
    cheek = ((ys - rs) / (rs + np.float32(1e-4)))[w.discs[0] | w.discs[1]]
    grain = 1.4826 * float(np.median(np.abs(cheek - np.median(cheek))))
    limit = min(max(t["strand"] * grain, t["strand_least"]), t["strand_most"])
    dark = ((rel < -limit) & (below > 0)).astype(np.uint8)                    # strands, and whatever else is dark in its row
    lines = np.zeros(dark.shape, np.uint8)                                    # the line-shaped ones: hair
    for k in _line_kernels(max(3, int(round(t["long"] * e)))):
        lines |= cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    soft = _row_mean(y, on, t["mass_soft"] * e, t["mass_soft"] * e) / np.maximum(_table_at(w.u, w.v), np.float32(1e-3))
    wide_row = _row_mean(soft, on, t["mass_row"] * e, 0.0)
    mass = ((soft - wide_row) / (wide_row + np.float32(1e-4)) < -t["mass"]) & (below > 0)
    at = t["dense_at"] * e
    sure = np.minimum(_blur2(mass.astype(np.float32), at, at) + np.float32(t["line_gain"]) * _blur2(lines.astype(np.float32), at, at), np.float32(1.0))
    strand = dark.astype(bool)
    dense = _blur2((strand | mass).astype(np.float32), at, at)
    bounds, wide = _bounds(w.u, w.v)
    inb = bounds & (skin > 0.5)
    core = (sure > t["core"]) & inb
    got = dict(g=g, y=y, skin=skin, below=below, strand=strand, lines=lines > 0, mass=mass, sure=sure, dense=dense, bounds=bounds,
               grain=grain, limit=limit, core_area=float(core.sum()) / (e * e), region=None, flank=0.0)
    if got["core_area"] < t["least"]:
        return got
    hairy = (dense > t["hair"]) & inb
    region = lip._grown(core, hairy)
    tails = (cv2.dilate(dark, lip._round_kernel(1)) > 0) & inb \
        & (cv2.dilate(region.astype(np.uint8), lip._round_kernel(t["tail_reach"] * e)) > 0)
    region = cv2.morphologyEx((region | tails).astype(np.uint8), cv2.MORPH_CLOSE, lip._round_kernel(t["bridge"] * e)).astype(bool) & inb
    # Hair just outside the side bounds: a beard across the jaw.
    au = np.abs(w.u)
    flank = (au >= wide) & (au < wide + t["flank"]) & (w.v > t["flank_v"][0]) & (w.v < t["flank_v"][1]) & (skin > 0.5)
    got["flank"] = float((dense[flank] > t["hair"]).mean()) if flank.any() else 0.0
    got["region"] = region
    # The replaced area: the hair and a little past it, soft-edged. It runs on over the lip (hair lies
    # against its edge); the lip's own soft edge is applied at the photo's size.
    paint = np.minimum(smoother._soft_blur(cv2.dilate(region.astype(np.uint8), lip._round_kernel(t["grow"] * e)).astype(np.float32),
                                           t["soft"] * e) * np.float32(1.5), np.float32(1.0))
    reach = _bounds(w.u, w.v, t["over_lip"])[0].astype(np.float32)
    paint = paint * np.minimum(smoother._soft_blur(reach, t["bound_edge"] * e) * np.float32(2.0), np.float32(1.0))
    got["paint"] = paint * ((skin > 0.02) | (w.v < 0.1))
    return got


def _snap(w: _Work, hair: np.ndarray | None = None) -> dict:
    """Moves the frame's jaw line onto the photo's own edge: in the columns clear of hair (hair: a map of the
    straightened copy, or None), the strongest darkening going down near v = 1; a smooth curve through
    those columns, carried under the hair; beside the hair, the edge as read. Returns what it found
    ("moved": whether the frame changed; the straightened copy is then laid out anew)."""
    t, f = TUNE, w.f
    e = w.eyes
    y = _blur2(lip._luma(w.rect(w.small)), t["edge_soft"][0] * e, t["edge_soft"][1] * e)
    fall = np.zeros_like(y)
    fall[1:-1] = (y[:-2] - y[2:]) / (y[2:] + y[:-2] + np.float32(1e-4))        # the share lost over two rows, going down
    tall = (f.jaw(w.u1) - f.top(w.u1)) / f.eyes                 # each column's height from lip to jaw, in eye distances
    skin = w.rect(w.skin) * w.inside
    us, offs, ws = [], [], []
    for j in np.nonzero(np.abs(w.u1) < t["wide"] + 0.15)[0]:
        rows = np.nonzero(np.abs(w.v1 - 1.0) < t["edge_reach"] / max(float(tall[j]), 1e-3))[0]
        if len(rows) < 3 or (hair is not None and hair[rows, j].any()) or float(skin[rows, j].min()) < 0.5:
            continue
        i = rows[int(np.argmax(fall[rows, j]))]
        if float(fall[i, j]) < t["edge_least"] or i in (rows[0], rows[-1]):
            continue
        us.append(float(w.u1[j]))
        offs.append(float((w.v1[i] - 1.0) * tall[j]))
        ws.append(float(fall[i, j]))
    found = dict(columns=len(us), moved=False)
    if len(us) < t["edge_columns"] or max(us) - min(us) < t["edge_span"]:
        return found
    us, offs, ws = np.array(us), np.array(offs), np.array(ws)
    keep = np.ones(len(us), bool)
    for _ in range(4):
        k = np.polyfit(us[keep], offs[keep], 2, w=ws[keep])
        keep = np.abs(offs - np.polyval(k, us)) < t["edge_out"]
        if int(keep.sum()) < t["edge_columns"]:
            return found
    k = np.polyfit(us[keep], offs[keep], 2, w=ws[keep])
    curve = np.polyval(k, np.clip(f.jaw_u, us[keep].min(), us[keep].max()))        # held past the last column read
    near = np.exp(-0.5 * ((f.jaw_u[:, None] - us[keep][None, :]) / t["edge_even"]) ** 2) * ws[keep][None, :]
    seen = near.sum(axis=1)
    local = (near * offs[keep][None, :]).sum(axis=1) / np.maximum(seen, 1e-9)
    trust = smoother._smoothstep(seen / max(float(np.median(ws[keep])), 1e-9), 1.0, 4.0)
    f.jaw_h = f.jaw_h + (curve + trust * (local - curve)) * f.eyes
    w.straighten()
    found.update(moved=True, kept=int(keep.sum()), middle=float(np.polyval(k, 0.0)),
                 left=float(np.std(offs[keep] - np.polyval(k, us[keep]))))
    return found


def _tone(w: _Work, hair: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bare chin's tone on the straightened copy, the table in the cheeks' colour it started from, and
    the clear pixels it was matched to."""
    t, lt = TUNE, lip.TUNE
    e = w.eyes
    red = lt["red"]
    k = np.array([1.0 + red, 1.0 + red, 1.0 - red * (0.701 / 0.299)], np.float32)      # B, G, R: as lip._shaded's
    prior = w.cheek * np.power(np.maximum(_table_at(w.u, w.v), np.float32(1e-3))[:, :, None], k)
    near = cv2.dilate((hair["strand"] | hair["mass"]).astype(np.uint8), lip._round_kernel(t["clear_off"])) > 0
    clear = ((hair["below"] > 0) & ~near).astype(np.float32)
    clear = clear * (1.0 - smoother._smoothstep(hair["dense"], t["clear_dense"][0], t["clear_dense"][1]))
    off = np.log(np.maximum(hair["g"], np.float32(1e-3)) / np.maximum(prior, np.float32(1e-3)))
    off = np.clip(off, math.log(t["span"][0]), math.log(t["span"][1])).astype(np.float32)
    # First from the skin well outside the beard; then the skin inside it counts where it is as light as that says.
    beard = cv2.dilate(hair["region"].astype(np.uint8), lip._round_kernel(t["first_off"] * e)) > 0
    first_y = lip._luma(prior * np.exp(_fill_in(off, clear * ~beard, e)))
    seen = _row_mean(hair["y"], clear, t["above_at"] * e, t["above_at"] * e)
    light = smoother._smoothstep((seen - first_y) / (first_y + np.float32(1e-4)), t["above"][0], t["above"][1])
    clear = clear * np.where(beard, light, np.float32(1.0))
    return prior * np.exp(_fill_in(off, clear, e)), prior, clear


def _clear_face(out: np.ndarray, pts, mask: np.ndarray, amount: float, keep, force: bool, seen: dict | None = None) -> bool:
    """One face, changed in place in out. Returns whether anything changed. seen, if given, receives
    what was found on the way, for looking at it."""
    t, lt = TUNE, lip.TUNE
    f = _Frame(pts)
    if not f.ok or not (f.posed or force):
        return False
    w = _Work(out, f, mask)
    if not w.ok:
        return False
    hair = _find(w)
    if seen is not None:
        seen.update(frame=f, work=w, hair=hair)
    if hair["region"] is None:
        return False
    beside = cv2.dilate((hair["dense"] > t["hair"]).astype(np.uint8), lip._round_kernel(t["edge_off"] * w.eyes)) > 0
    edge = _snap(w, beside)
    if edge["moved"]:
        hair = _find(w)                    # the same again, on the frame as moved
    if seen is not None:
        seen.update(edge=edge, hair=hair)
    if hair["region"] is None:
        return False
    if hair["flank"] > t["flank_most"] and not force:
        if seen is not None:
            seen["refused"] = "hair runs on past the bounds (%.2f)" % hair["flank"]
        return False
    paint = hair["paint"] * np.float32(min(1.0, float(amount)))
    if not float(paint.max()) > 1e-3:
        return False
    final, prior, clear = _tone(w, hair)
    tone_g = w.rect(w.tone)
    final_y, tone_y = lip._luma(final), lip._luma(tone_g)
    # Hair goes whatever its tone; the clear skin beside it only ever gets lighter.
    lift = smoother._smoothstep((final_y - tone_y) / (final_y + np.float32(1e-4)), -lt["dark_lo"], lt["dark_lo"])
    lift = np.maximum(lift, np.minimum(smoother._soft_blur(hair["region"].astype(np.float32), t["soft"] * w.eyes) * np.float32(2.0), np.float32(1.0)))
    change = (lift[:, :, None] * (final - tone_g)).astype(np.float32)

    # At the photo's own size: the lip's edge, the grain.
    x0, y0, x1, y1 = w.box_at
    bw, bh = w.size
    d = f.eyes
    lips_f = np.zeros((bh, bw), np.uint8)
    cv2.fillPoly(lips_f, [np.round(w.outline - [x0, y0]).astype(np.int32)], 255)
    lips_soft = smoother._soft_blur(lips_f.astype(np.float32) / np.float32(255.0), max(0.6, t["lip_soft"] * d))
    ok = 1.0 - lips_soft
    paint_f = w.back(paint)
    if keep is not None:
        kx, ky, edit = keep
        ax0, ay0 = max(x0, kx), max(y0, ky)
        ax1, ay1 = min(x1, kx + edit.shape[1]), min(y1, ky + edit.shape[0])
        if ax1 > ax0 and ay1 > ay0:
            hold = np.ones((bh, bw), np.float32)
            hold[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] += np.minimum(edit[ay0 - ky:ay1 - ky, ax0 - kx:ax1 - kx], 0.0)
            paint_f = paint_f * hold
    change_f = w.back(change)
    tone_f = w.tone if w.scale >= 1.0 else cv2.resize(w.tone, (bw, bh), interpolation=cv2.INTER_LINEAR)
    grain = w.box - tone_f
    grain_y = lip._luma(grain) * ok
    usable = (w.box_mask > 0.9).astype(np.float32) * ok
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float64)
    near = np.zeros((bh, bw), bool)
    for c in f.cheeks:
        near |= (xx + x0 - c[0]) ** 2 + (yy + y0 - c[1]) ** 2 < (lt["cheek"] * d) ** 2
    limit = lt["grain_most"] * lip._percentile(np.abs(grain_y[near & (w.box_mask > 0.9)]), 25.0)
    u_f, _ = f.uv(xx + x0, yy + y0)
    side_of = smoother._smoothstep(u_f.astype(np.float32), -lt["seam"], lt["seam"])       # 0: the right cheek's half, 1: the left's
    heres = [np.array([float(a[0]) for a in f.at(np.array([side * t["side"]]), np.array([t["side_down"]]))]) for side in (-1.0, 1.0)]
    borrowed, have = lip._pieces(grain_y, usable, x0, y0, f.cheeks, heres, (1.0 - side_of, side_of), t["tile"] * d, t["tile_wander"] * d, limit)
    shade = np.clip(lip._luma(tone_f + change_f) / np.float32(max(float(w.cheek @ smoother._LUMA), 1e-3)), t["grain_shade"][0], t["grain_shade"][1])
    borrowed = borrowed * shade
    new_skin = w.box + paint_f[:, :, None] * (change_f + borrowed[:, :, None] - grain)
    new = new_skin * ok[:, :, None] + w.box * lips_soft[:, :, None]
    if seen is not None:
        seen.update(box=w.box_at, final=final, prior=prior, clear=clear, paint=paint, paint_f=paint_f, have=have, lips_soft=lips_soft)
    result = np.clip(new * 255.0 + 0.5, 0, 255).astype(np.uint8)
    here = out[y0:y1, x0:x1]
    if np.array_equal(result, here):
        return False
    here[:] = result
    return True


def clear(bgr: np.ndarray, points, mask: np.ndarray, amount: float = 1.0, keep=None, force: bool = False) -> np.ndarray:
    """bgr (uint8) with the beard taken off each face's chin: a new array, or bgr itself
    when nothing changes. points: each face's landmarks or None (face_landmarks.landmarks).
    mask: the smoothing mask (where the skin is). amount: 0..1, how much of it goes. keep:
    the brush's Restore edit (x0, y0, float32 -1..1), where the chin stays as it is. force:
    skip the questions whether the face is posed for it and whether the hair runs on across
    the jaw (for the parity check, whose photos are not posed)."""
    if amount <= 0.0 or not points:
        return bgr
    out = bgr.copy()
    changed = False
    for pts in points:
        if pts is not None:
            changed = _clear_face(out, pts, mask, amount, keep, force) or changed
    return out if changed else bgr
