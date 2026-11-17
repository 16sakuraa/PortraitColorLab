"""
Two finishing steps of a studio retouch that aren't smoothing. Both are off unless
asked for, and are applied to the smoothed photo, before the light brush's strokes.

under_eyes
    Evens the skin under each eye, and lifts it a little towards the tone of the cheek
    just below it: bags, creases and the darkest part of a dark circle go, while the eye
    socket keeps its own shading (painting the cheek's tone straight over it would make
    the face flat). Only the tone layer of a frequency separation moves (as with the
    light brush, heal.py), so the skin's grain stays. Needs the face's landmarks
    (face_landmarks.py); a face without them is left as it is.

balance
    One brightness curve for all the skin in the photo, for a face well darker or
    brighter than the middle of the range. A face inside the range is left alone; one
    outside it moves part of the way back. The curve is a power curve applied to each
    colour channel, like a Curves adjustment on the skin: black and white stay where
    they are, and brightening takes a little colour out, as it does by hand. It changes
    how light or dark a complexion looks, not just exposure, so it is a matter of taste:
    the amounts here follow one studio's retouching of ID photos.

The face's brightness is measured once per photo (skin_luma), on the photo as it was
opened, so a preview and the full-size result get the same curve.
"""

from __future__ import annotations

import math

import numpy as np
import cv2

import face_landmarks
import smoother

BALANCE = {
    "centre": 0.575,   # the middle of the range, as skin_luma measures it (0..1, the file's own values)
    "slack": 0.05,     # faces within this of the centre are left alone
    "pull": 0.7,       # share of the way back to the range's edge that a face outside it is moved
    "limit": 0.12,     # and no further than this
}
EYES = {
    "feather": 0.05,   # soft edge of the under-eye zone, a fraction of the distance between the eyes
    "tone": 0.03,      # grain finer than this stays as it is, same units
    "even": 0.1,       # bags and creases up to about this size are evened out
    "below": 0.3,      # the cheek tone is read this far below, down the face
    "wide": 0.07,      # averaged over about this much skin
    "lift": 0.12,      # share of the way from the evened tone to the cheek's, at full amount
    "limit": 0.25,     # the tone moves no further than this (0..1)
}
# Looks: settings chosen together (sliders 0..1). Natural is how the app opens. Studio
# follows how a portrait studio retouches ID photos by hand: fitted to a set of its
# before/after pairs, it evens blotches and folds more (the wider light setting), keeps a
# small print's grain, and balances skin tone. Under-eyes stays off in it: with the wider
# light setting the skin under the eyes already comes out as it does by hand, and any more
# moved it further from the hand retouch; the slider is for heavy bags.
# The pages keep the same values (web/js/looks.js).
LOOKS = {
    "natural": {"strength": 0.6, "texture": 0.6, "lighting": "soft", "print": "large", "eyes": 0.0, "tone": 0.0},
    "studio": {"strength": 0.8, "texture": 0.75, "lighting": "hard", "print": "small", "eyes": 0.0, "tone": 1.0},
}
_LUMA_WEIGHTS = np.array([114, 587, 299], np.int32)    # B, G, R, in thousandths
MIN_LUMA, MAX_LUMA = 0.05, 0.95      # outside this a face's brightness isn't a measure of anything
SURE_SKIN = 0.9                      # the mask value that counts as skin well inside the mask


def skin_luma(bgr: np.ndarray, mask: np.ndarray, faces) -> float | None:
    """Median brightness (0..1, the file's own values) of the faces' skin: skin well
    inside the mask, inside the face boxes (anywhere in the mask without faces). None
    without such skin."""
    h, w = mask.shape[:2]
    sure = mask > SURE_SKIN
    if faces is not None and len(faces) > 0:
        inside = np.zeros((h, w), bool)
        for f in faces:
            x0, y0 = max(0, int(f[0])), max(0, int(f[1]))
            x1, y1 = min(w, int(f[0] + f[2])), min(h, int(f[1] + f[3]))
            if x1 > x0 and y1 > y0:
                inside[y0:y1, x0:x1] = True
        sure &= inside
    px = bgr[sure]
    if len(px) == 0:
        return None
    return float(np.median(px.astype(np.int32) @ _LUMA_WEIGHTS)) / 255000.0


def wanted_luma(luma: float, amount: float) -> float:
    """Where balance moves a face of this brightness to, at this amount (0..1)."""
    t = BALANCE
    off = luma - t["centre"]
    move = -t["pull"] * math.copysign(max(abs(off) - t["slack"], 0.0), off)
    return luma + float(np.clip(amount, 0.0, 1.0)) * max(-t["limit"], min(t["limit"], move))


def balance(bgr: np.ndarray, area: np.ndarray, luma: float | None, amount: float) -> np.ndarray:
    """bgr with the skin's brightness balanced: a new array, or bgr itself when nothing
    changes. area: all the skin (smoother.skin_layers), float32 0..1. luma: skin_luma of
    the photo as it was opened."""
    if luma is None or amount <= 0.0 or not MIN_LUMA < luma < MAX_LUMA:
        return bgr
    want = wanted_luma(luma, amount)
    if abs(want - luma) < 1e-6:
        return bgr
    ys, xs = np.nonzero(area > 1e-3)
    if len(ys) == 0:
        return bgr
    gamma = math.log(want) / math.log(luma)
    curve = ((np.arange(256, dtype=np.float64) / 255.0) ** gamma * 255.0).astype(np.float32)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    out = bgr.copy()
    here = out[y0:y1, x0:x1]
    now = here.astype(np.float32)
    mixed = now + area[y0:y1, x0:x1, None] * (curve[here] - now) + np.float32(0.5)
    here[:] = np.clip(mixed, 0, 255).astype(np.uint8)
    return out


def under_eyes(bgr: np.ndarray, points, mask: np.ndarray, amount: float) -> np.ndarray:
    """bgr with the skin under the eyes evened towards the cheek below: a new array, or
    bgr itself when nothing changes. points: each face's landmarks or None
    (smoother.skin_layers). mask: the smoothing mask, whose skin (no eyes, lashes or
    brows) is where tone is read from, and where the change shows."""
    if amount <= 0.0 or not points:
        return bgr
    amount = min(1.0, float(amount))
    t = EYES
    h, w = bgr.shape[:2]
    out = bgr
    for pts in points:
        if pts is None:
            continue
        zones, (dx, dy), eye_d = face_landmarks.under_eye_zones(pts, w, h)
        sx, sy = round(t["below"] * dx), round(t["below"] * dy)
        feather = max(smoother.SOFT_BLUR_MIN, t["feather"] * eye_d)
        fine = max(smoother.SOFT_BLUR_MIN, t["tone"] * eye_d)
        wide = max(smoother.SOFT_BLUR_MIN, t["wide"] * eye_d)
        even = max(smoother.SOFT_BLUR_MIN, t["even"] * eye_d)
        pad = math.ceil(3.0 * max(feather, wide, even)) + max(abs(sx), abs(sy)) + 2
        for zone, (zx, zy) in zones:
            x0, y0 = max(0, zx - pad), max(0, zy - pad)
            x1, y1 = min(w, zx + zone.shape[1] + pad), min(h, zy + zone.shape[0] + pad)
            inside = np.zeros((y1 - y0, x1 - x0), np.float32)
            inside[zy - y0:zy - y0 + zone.shape[0], zx - x0:zx - x0 + zone.shape[1]] = zone / np.float32(255.0)
            skin = mask[y0:y1, x0:x1].astype(np.float32)
            weight = smoother._soft_blur(inside, feather) * skin
            if not float(weight.max()) > 1e-3:
                continue
            box = out[y0:y1, x0:x1].astype(np.float32) / np.float32(255.0)
            lit = box * skin[:, :, None]
            tone = smoother._soft_blur(lit, fine) / (smoother._soft_blur(skin, fine)[:, :, None] + np.float32(1e-4))
            flat = smoother._soft_blur(lit, even) / (smoother._soft_blur(skin, even)[:, :, None] + np.float32(1e-4))
            seen = _moved(smoother._soft_blur(skin, wide), sx, sy)
            cheek = _moved(smoother._soft_blur(lit, wide), sx, sy) / (seen[:, :, None] + np.float32(1e-4))
            # Where there's little skin below (a photo's edge, glasses, hair), nothing to go by.
            sure = np.clip(seen * 2.0 - 0.5, 0.0, 1.0)
            target = flat + np.float32(t["lift"]) * sure[:, :, None] * (cheek - flat)
            change = np.clip(target - tone, -t["limit"], t["limit"]) * (np.float32(amount) * weight)[:, :, None]
            if out is bgr:
                out = bgr.copy()
            out[y0:y1, x0:x1] = np.clip((box + change) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def _moved(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """a read from dx, dy away: out[y, x] = a[y + dy, x + dx], zero where that's outside."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    ya, yb = max(0, -dy), min(h, h - dy)
    xa, xb = max(0, -dx), min(w, w - dx)
    if yb > ya and xb > xa:
        out[ya:yb, xa:xb] = a[ya + dy:yb + dy, xa + dx:xb + dx]
    return out


def finish(bgr: np.ndarray, mask: np.ndarray, area: np.ndarray | None, points, luma: float | None,
           eyes: float = 0.0, tone: float = 0.0) -> np.ndarray:
    """Both steps, under-eyes first: bgr itself with both off (the default)."""
    out = under_eyes(bgr, points, mask, eyes)
    return balance(out, mask if area is None else area, luma, tone)
