"""
Blemish removal with classic image processing (no neural network).

An alternative to the network in blemish.py, with the same output, so it needs
no model file and no training data.

  1. Find small dark or red spots: a morphological black-hat on the green channel
     (reddening absorbs green, so red spots show up as dark there too). The
     threshold comes from the photo's own skin (median plus a multiple of the
     median absolute deviation), so it adapts to exposure and skin tone.
  2. Keep only what looks like a spot: reject candidates by size, roundness and
     how solid they are, anything that merges into a larger shape at a lower
     threshold (hair, stubble, creases), and anything too dark to be a blemish
     (hair, moles). Redness absorbs green but not red, so something that's also
     dark in the red channel is pigment or hair: a mole, or stubble. Moles are
     left as they are.
  3. Fill each spot on the tone layer from the skin around it (normalized
     convolution; Knutsson & Westin, 1993), feather the edge, and reduce the fine
     detail inside it, which would otherwise leave a ring.

Sizes are fractions of face width. A face is about 15 cm wide, so 0.01 is 1.5 mm.
"""

from __future__ import annotations

import numpy as np
import cv2

import smoother

WORK_FACE_PX = 450     # 3 px per mm: even the smallest spot covers a few pixels
MIN_FACE_PX = 80       # smaller faces have no spots worth finding
MIN_SPOT_PX = 4        # fewest pixels that can be told apart from noise

TUNE = {
    "presmooth": 0.003,    # light blur before detecting, about 0.5 mm
    "window": 0.045,       # black-hat window: just above the largest spot, about 7 mm
    "edge_margin": 0.015,  # stay this far inside the mask, away from eyes, brows and hair
    "threshold": 7.0,      # how many median absolute deviations a spot must stand out by
    "min_contrast": 0.035, # and how dark it must be at least (0..1), about 9 of 255 levels
    "max_contrast": 0.25,  # darker than this is hair or a mole, not a blemish
    "max_red": 0.1,        # this much darker than its surroundings in red (a share) is a mole or stubble
    "min_size": 0.007,     # spot diameter, about 1 mm
    "max_size": 0.04,      # about 6 mm
    "max_aspect": 2.5,     # bounding box width to height, either way
    "min_fill": 0.4,       # share of its bounding box a spot fills
    "isolation": 0.6,      # at this share of the threshold, a spot must still be
    "island_size": 0.05,   # an island no longer than this (about 7.5 mm), or it's hair, stubble or a crease
    "grow": 0.008,         # widen a spot by about 1 mm to cover its soft edge
    "fill_sigma": 0.015,   # how far around a spot to take the fill from, about 2 mm
    "feather": 0.003,      # softness of the spot's edge
    "detail_cut": 0.8,     # share of the fine detail removed inside a spot
    "near_face": 0.5,      # spots are looked for this many face sizes around each face
}


def _odd(v: float, minimum: int = 3) -> int:
    return max(minimum, int(round(v))) | 1


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def find_spots(green: np.ndarray, red: np.ndarray, mask: np.ndarray, face_w: float) -> tuple[np.ndarray, int]:
    """Spots on one group of similarly sized faces: (uint8 0/1 map, how many).

    green, red : float32 0..1. mask : float32 0..1, zero outside this group.
    face_w : face width in these pixels.
    """
    t = TUNE
    empty = np.zeros(green.shape, np.uint8)
    # A light blur first: spots are larger than pixel grain, and it makes the
    # statistics below continuous. On raw 8-bit values the median moves in whole
    # steps, so one rounding difference could shift the threshold a lot.
    blur = max(0.7, face_w * t["presmooth"])
    window = _ellipse(_odd(face_w * t["window"]))
    soft_green = smoother._soft_blur(green, blur)
    bh = cv2.morphologyEx(soft_green, cv2.MORPH_BLACKHAT, window)
    soft_red = smoother._soft_blur(red, blur)
    bh_red = cv2.morphologyEx(soft_red, cv2.MORPH_BLACKHAT, window)
    red_dip = bh_red / (soft_red + bh_red + 1e-6)     # a share of what surrounds it, so it holds on any skin tone

    margin = max(1, int(round(face_w * t["edge_margin"])))
    core = cv2.erode((mask > 0.9).astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    values = bh[core > 0]
    if values.size < 100:
        return empty, 0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    threshold = max(t["min_contrast"], med + t["threshold"] * mad)

    # Shapes at a lower threshold show what a candidate is part of. A real spot
    # stays a small island; on hair, stubble or a crease it joins up into
    # something much larger.
    loose = ((bh > t["isolation"] * threshold) & (mask > 0.3)).astype(np.uint8)
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(loose, connectivity=8)
    too_big = np.maximum(stats2[:, cv2.CC_STAT_WIDTH], stats2[:, cv2.CC_STAT_HEIGHT]) > face_w * t["island_size"]

    candidates = ((bh > threshold) & (core > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    min_area = max(MIN_SPOT_PX, np.pi * (face_w * t["min_size"] / 2) ** 2)
    max_area = np.pi * (face_w * t["max_size"] / 2) ** 2
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not min_area <= area <= max_area:
            continue
        if max(w, h) > t["max_aspect"] * min(w, h) or area < t["min_fill"] * w * h:
            continue
        here = labels[y:y + h, x:x + w] == i
        if too_big[np.bincount(labels2[y:y + h, x:x + w][here]).argmax()]:
            continue
        if bh[y:y + h, x:x + w][here].max() > t["max_contrast"] or red_dip[y:y + h, x:x + w][here].max() > t["max_red"]:
            continue
        keep[i] = 1
    count = int(keep.sum())
    if count == 0:
        return empty, 0
    spots = keep[labels]
    return cv2.dilate(spots, _ellipse(_odd(face_w * t["grow"]))), count


def correction(bgr: np.ndarray, faces=None, mask: np.ndarray | None = None):
    """What to add to the photo to remove its spots, and how many there are.

    Same form as blemish.correction: ((x0, y0, float32 HxWx3 in 0..255 units) or
    None, count). Compute once per photo; blemish.apply is then cheap.
    """
    if faces is None:
        faces = smoother._detect_faces(bgr)
    if faces is None or len(faces) == 0:
        return None, 0
    if mask is None:
        mask = smoother.build_skin_mask(bgr)

    t = TUNE
    h, w = bgr.shape[:2]
    # Stay near the faces: the mask may cover arms and legs too, which would use
    # up the working pixels and leave the faces too small to find spots on.
    near = np.zeros((h, w), np.float32)
    for f in faces:
        reach = t["near_face"] * float(max(f[2], f[3]))
        near[int(max(0.0, float(f[1]) - reach)):int(min(float(h), float(f[1] + f[3]) + reach)),
             int(max(0.0, float(f[0]) - reach)):int(min(float(w), float(f[0] + f[2]) + reach))] = 1.0
    mask = mask * near
    ys, xs = np.nonzero(mask > 1e-3)
    if len(ys) == 0:
        return None, 0
    widths = smoother._face_widths(faces, (h * h + w * w) ** 0.5)
    pad = int(np.ceil(float(widths.max()) * t["fill_sigma"] * 4)) + 8
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
    roi = bgr[y0:y1, x0:x1]
    mask_r = mask[y0:y1, x0:x1].astype(np.float32)
    rh, rw = roi.shape[:2]

    scale = min(1.0, WORK_FACE_PX / float(widths.max()), (smoother.WORK_PIXELS / (rh * rw)) ** 0.5)
    if scale < 1.0:
        size = (max(1, round(rw * scale)), max(1, round(rh * scale)))
        work = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
        mask_w = cv2.resize(mask_r, size, interpolation=cv2.INTER_AREA)
    else:
        work, mask_w = roi, mask_r
    work = work.astype(np.float32) / 255.0
    green = work[:, :, 1]

    sigma = max(smoother.MIN_SPLIT_SIGMA, float(np.median(widths)) * smoother.TUNE["split"])
    low = smoother._soft_blur(work, sigma * scale) if sigma * scale >= 0.5 else work

    delta = np.zeros_like(work)
    soft_all = np.zeros(green.shape, np.float32)
    count = 0
    for width, weight in smoother._scale_regions(faces, widths, scale, x0, y0, green.shape):
        face_w = width * scale
        if face_w < MIN_FACE_PX:
            continue
        group_mask = mask_w if weight is None else mask_w * (weight > 0.5)
        spots, found = find_spots(green, work[:, :, 2], group_mask, face_w)
        if not found:
            continue
        count += found
        spots = spots.astype(np.float32)
        soft = np.clip(cv2.GaussianBlur(spots, (0, 0), max(0.7, face_w * t["feather"])) * 1.5, 0, 1)
        # Read only surrounding skin: not the spots, and not hair or eyes.
        valid = (1.0 - spots) * mask_w
        sigma_fill = max(1.5, face_w * t["fill_sigma"])
        num = smoother._soft_blur(low * valid[:, :, None], sigma_fill)
        den = smoother._soft_blur(valid, sigma_fill) + 1e-6
        delta += soft[:, :, None] * (num / den[:, :, None] - low)
        soft_all = np.maximum(soft_all, soft)
    if count == 0:
        return None, 0

    # Back to full resolution, cut to the area the spots cover.
    if scale < 1.0:
        delta = cv2.resize(delta, (rw, rh), interpolation=cv2.INTER_LINEAR)
        soft_all = cv2.resize(soft_all, (rw, rh), interpolation=cv2.INTER_LINEAR)
    rows = np.flatnonzero((soft_all > 0).any(axis=1))
    cols = np.flatnonzero((soft_all > 0).any(axis=0))
    by0, by1, bx0, bx1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    corr = delta[by0:by1, bx0:bx1] * 255.0
    soft_box = soft_all[by0:by1, bx0:bx1]
    roi_box = roi[by0:by1, bx0:bx1]
    for top in range(0, by1 - by0, smoother.STRIP_ROWS):
        bottom = min(by1 - by0, top + smoother.STRIP_ROWS)
        _, high = smoother._detail_strip(roi_box, top, bottom, sigma)
        corr[top:bottom] -= (t["detail_cut"] * 255.0 * soft_box[top:bottom])[:, :, None] * high
    return (x0 + bx0, y0 + by0, corr.astype(np.float32)), count
