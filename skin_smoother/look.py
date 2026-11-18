"""
Finishing steps of a studio retouch that aren't smoothing. All are off unless asked
for, and are applied to the smoothed photo, before the light brush's strokes.

matte
    The large-patch half of Reduce shine (smoother.smooth_face tames the small glints).
    Skin tone brighter than the face's own brighter skin is pulled part of the way back
    to that colour, channel by channel, the way a retoucher paints skin colour over a
    highlight in Darken mode: a glossy forehead or cheek goes towards matte and keeps
    the skin's colour instead of turning grey. Only the tone layer moves, so the grain
    stays. It doesn't tell shine from light: the brightest part of any face comes down
    by the same share, so on a face that isn't shiny a high amount flattens its
    shading a little. Each face is judged against its own skin, so a fair face beside
    a dark one isn't taken for shine.

tidy
    Stray hairs around the head, against a plain backdrop: just outside the person's
    outline (the segmentation model's), anything thin that differs from the backdrop
    around it is replaced by that backdrop, filled from the backdrop nearby the way a
    spot is filled from the skin nearby. A strand has to be coloured like the hair
    seen through the backdrop (somewhere between the backdrop's colour and the hair's
    at the top of the head), so a logo, a seam or a lamp behind the head stays. A busy
    background has no backdrop to tidy against, and is left alone. Strands shorter
    than the gap kept beside the outline stay, and so does the outline itself: it is
    not redrawn.

straighten
    A tilted head, turned upright by a share of its tilt (the eye line's, from the
    face's landmarks). Everything above the jaw turns as one piece about a point just
    under the chin, and the turn fades out down the neck, so the shoulders and clothes
    stay where they are: what a retoucher does by turning the head and blending the
    neck, far more often than turning the whole photo. Nothing is made up and the face
    keeps its shape; the photo is resampled where it turns. It is the last step, so
    the brush's strokes, which are places on the photo as it was, still land where
    they were painted; the pages leave it off while the brush is in use.

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

The face's brightness and its brighter skin's colour are measured once per photo
(skin_luma, bright_colours), on the photo as it was opened, so a preview and the
full-size result get the same change.
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
MATTE = {
    "tone": 0.012,     # grain finer than this share of face width stays as it is, about 2 mm
    "from": 0.7,       # the face's brighter skin: between these two shares of the way up
    "to": 0.8,         # its skin, ordered by brightness
    "cut": 0.6,        # share of the tone above that colour that is taken off, at full amount
    "span": 0.04,      # the step fades in over this much brightness (0..1) above that colour
}
MATTE_GRID = 250000    # with several faces, whose skin is whose is worked out on about this many pixels,
MATTE_BORDER = 0.05    # with a border between faces this wide, a share of the smallest face's width:
                       # narrow, or one face's colour would be held against the edge of the next face
TIDY = {
    "gap": 0.012,      # this close to the person's outline nothing is touched (shares of face width)
    "reach": 0.22,     # strands are looked for this far out from it
    "below": 0.15,     # and down to this far below the face's box: around the head, not the shoulders
    "fill": 0.03,      # the backdrop is read from about this far around
    "thin": 0.02,      # anything narrower than this is a strand; wider things stay
    "lo": 5.0,         # a strand differs from the backdrop around it by this much (0..255) ...
    "hi": 12.0,        # ... and counts fully from this much
    "busy_lo": 3.0,    # a backdrop that itself varies this much (0..255, the median) is busy:
    "busy_hi": 6.0,    # the step fades out, and is off from here
    "feather": 0.004,  # softness of a strand's edge
    "crown": 0.25,     # the hair's colour is read above this share of the way down each face's box,
    "rim": 0.03,       # from this far inside the person's outline
    "mix_lo": 0.03,    # a strand is the backdrop with at least this share of the hair's colour mixed in,
    "mix_hi": 1.25,     # at most this (1 = all hair; over 1 = darker or lighter than the hair itself),
    "stray": 0.4,      # and what the mix doesn't explain is under this share of its difference from
    "stray_floor": 6.0,  # the backdrop, plus this much (0..255)
    "other": 25.0,     # a thin thing of another colour has to differ from the backdrop by this much
                       # (0..255) to keep its surroundings from being tidied: a sharpening halo doesn't
}
TIDY_MIN_HAIR = 50     # fewer pixels of hair than this at the top of the heads: nothing to go by
LEVEL = {
    "pivot": 0.25,     # the head turns about a point this far below the chin (shares of the eye distance)
    "above": 0.15,     # the full turn applies from this far above that point upwards,
    "neck": 0.75,      # and has faded out this far below it
    "side": 1.6,       # sideways it is full out to this far from the face's middle,
    "side_end": 2.6,   # and has faded out by here
    "most": 8.0,       # a tilt beyond this many degrees is a pose, not a slip: only this much is undone
}
TIDY_FACE_PX = 600     # the search is done with faces at most this wide: thin strands still cover a few pixels
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


def _faces_skin(bgr: np.ndarray, mask: np.ndarray, faces) -> np.ndarray:
    """The pixels (N x 3, uint8) of the faces' skin: skin well inside the mask, inside the
    face boxes (anywhere in the mask without faces)."""
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
    return bgr[sure]


def skin_luma(bgr: np.ndarray, mask: np.ndarray, faces) -> float | None:
    """Median brightness (0..1, the file's own values) of the faces' skin (_faces_skin).
    None without such skin."""
    px = _faces_skin(bgr, mask, faces)
    if len(px) == 0:
        return None
    return float(np.median(px.astype(np.int32) @ _LUMA_WEIGHTS)) / 255000.0


def bright_colour(bgr: np.ndarray, mask: np.ndarray, faces) -> tuple[float, float, float] | None:
    """The colour (B, G, R, 0..1) of the faces' brighter skin: the average of the skin
    (_faces_skin) between MATTE's two shares of the way up it, ordered by brightness.
    matte pulls brighter skin tone back towards it. None without such skin."""
    px = _faces_skin(bgr, mask, faces)
    if len(px) == 0:
        return None
    y = px.astype(np.int32) @ _LUMA_WEIGHTS
    ordered = np.sort(y)
    lo = int(ordered[int(MATTE["from"] * (len(y) - 1))])
    hi = int(ordered[int(MATTE["to"] * (len(y) - 1))])
    band = px[(y >= lo) & (y <= hi)]
    total = band.sum(axis=0, dtype=np.int64)
    return tuple(float(t) / (len(band) * 255.0) for t in total)


def bright_colours(bgr: np.ndarray, mask: np.ndarray, faces) -> list:
    """bright_colour for each face on its own (None for a face without such skin), or
    for all the skin in the mask, as a list of one, without faces."""
    if faces is None or len(faces) == 0:
        return [bright_colour(bgr, mask, None)]
    return [bright_colour(bgr, mask, faces[i:i + 1]) for i in range(len(faces))]


def matte(bgr: np.ndarray, mask: np.ndarray, faces, colours, amount: float) -> np.ndarray:
    """bgr with large shiny patches on skin taken down: a new array, or bgr itself when
    nothing changes. mask: the smoothing mask, float32 0..1. colours: bright_colours of
    the photo as it was opened, one per face (each face's skin, and the body skin
    nearest it, is judged against its own), or one colour for everything."""
    if colours is None or amount <= 0.0:
        return bgr
    if len(colours) == 3 and not isinstance(colours[0], (tuple, list, np.ndarray, type(None))):
        colours = [colours]                       # one colour for everything
    if all(c is None for c in colours):
        return bgr
    amount = min(1.0, float(amount))
    t = MATTE
    ys, xs = np.nonzero(mask > 1e-3)
    if len(ys) == 0:
        return bgr
    h, w = bgr.shape[:2]
    widths = smoother._face_widths(faces, (h * h + w * w) ** 0.5)
    sigma = max(smoother.SOFT_BLUR_MIN, float(np.median(widths)) * t["tone"])
    pad = math.ceil(3.0 * sigma) + 2
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
    box = bgr[y0:y1, x0:x1].astype(np.float32) / np.float32(255.0)
    skin = mask[y0:y1, x0:x1].astype(np.float32)
    # The tone of skin only: hair and backdrop beside it have no say.
    tone = smoother._soft_blur(box * skin[:, :, None], sigma) / (smoother._soft_blur(skin, sigma)[:, :, None] + np.float32(1e-4))

    def above(colour):
        """How far the tone is above a colour, per channel, where the skin as a whole is
        brighter than it and not where one channel is: a blush is redder than it, and
        no brighter."""
        ref = np.array(colour, np.float32)
        over = np.maximum(tone - ref, np.float32(0.0))
        gate = smoother._smoothstep(tone @ smoother._LUMA - np.float32(float(ref @ smoother._LUMA)), 0.0, t["span"])
        return over * gate[:, :, None]

    bh, bw = box.shape[:2]
    regions = [(0.0, None)]
    if faces is not None and len(colours) == len(faces) > 1:
        grid = min(1.0, (MATTE_GRID / (bh * bw)) ** 0.5)
        shape = (max(1, round(bh * grid)), max(1, round(bw * grid)))
        regions = smoother._scale_regions(faces, widths, grid, x0, y0, shape, each=True, soft=MATTE_BORDER)
    if regions[0][1] is None:
        colour = next(c for c in colours if c is not None)
        excess = above(colour)
    else:
        # Each face's skin, and the body skin nearest it, against that face's own colour.
        excess = np.zeros_like(box)
        for (_, weight), colour in zip(regions, colours):
            if colour is not None:
                excess += above(colour) * cv2.resize(weight, (bw, bh), interpolation=cv2.INTER_LINEAR)[:, :, None]
    by = skin * np.float32(amount * t["cut"])
    out = bgr.copy()
    out[y0:y1, x0:x1] = np.clip((box - excess * by[:, :, None]) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


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


def person_of(seg):
    """What tidy needs of the segmentation model's result (skin_segment.Segments):
    (x0, y0, x1, y1, person scores at working size), or None without it."""
    return None if seg is None else (seg.x0, seg.y0, seg.x1, seg.y1, seg.person)


def _box(r: float) -> np.ndarray:
    side = 2 * max(1, int(round(r))) + 1
    return np.ones((side, side), np.uint8)


def tidy(bgr: np.ndarray, person, faces, amount: float, keep=None) -> np.ndarray:
    """bgr with stray hairs around the heads replaced by the backdrop: a new array, or bgr
    itself when nothing changes. person: person_of the photo's segmentation. keep: a
    brush edit (mask_edit region: x0, y0, float32 -1..1) whose Restore strokes leave
    the photo as it was."""
    if person is None or amount <= 0.0 or faces is None or len(faces) == 0:
        return bgr
    amount = min(1.0, float(amount))
    t = TIDY
    x0, y0, x1, y1, scores = person
    rw, rh = x1 - x0, y1 - y0
    if rw < 8 or rh < 8:
        return bgr
    face_w = float(np.median(np.asarray(faces, np.float64)[:, 2]))
    scale = min(1.0, TIDY_FACE_PX / face_w)
    ww, wh = (max(1, round(rw * scale)), max(1, round(rh * scale))) if scale < 1.0 else (rw, rh)
    roi = bgr[y0:y1, x0:x1]
    work = (cv2.resize(roi, (ww, wh), interpolation=cv2.INTER_AREA) if scale < 1.0 else roi).astype(np.float32)
    fw = face_w * scale
    body = (cv2.resize(scores, (ww, wh), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    near = cv2.dilate(body, _box(t["gap"] * fw))
    ring = cv2.dilate(body, _box(t["reach"] * fw)) & (1 - near)
    heads = np.zeros((wh, ww), np.uint8)            # around each head, not the shoulders
    for f in faces:
        hy1 = int(min(wh, max(0.0, (float(f[1]) + float(f[3]) * (1.0 + t["below"]) - y0) * scale)))
        hx0 = int(min(ww, max(0.0, (float(f[0]) - float(f[2]) - x0) * scale)))
        hx1 = int(min(ww, max(0.0, (float(f[0]) + 2.0 * float(f[2]) - x0) * scale)))
        heads[:hy1, hx0:hx1] = 1
    ring &= heads
    if int(ring.sum()) < 100:
        return bgr
    # The hair's colour: the person's rim at the top of each head.
    rim = body & (1 - cv2.erode(body, _box(t["rim"] * fw)))
    crown = np.zeros((wh, ww), np.uint8)
    for f in faces:
        cy1 = int(min(wh, max(0.0, (float(f[1]) + float(f[3]) * t["crown"] - y0) * scale)))
        hx0 = int(min(ww, max(0.0, (float(f[0]) - float(f[2]) - x0) * scale)))
        hx1 = int(min(ww, max(0.0, (float(f[0]) + 2.0 * float(f[2]) - x0) * scale)))
        crown[:cy1, hx0:hx1] = 1
    hair_px = work[(rim & crown) > 0]
    if len(hair_px) < TIDY_MIN_HAIR:
        return bgr
    hair = np.median(hair_px, axis=0).astype(np.float32)
    sigma = max(smoother.SOFT_BLUR_MIN, t["fill"] * fw)
    thin = _box(t["thin"] * fw)

    def backdrop(valid):
        """The backdrop's colour everywhere, from the pixels marked valid nearby."""
        return smoother._soft_blur(work * valid[:, :, None], sigma) / (smoother._soft_blur(valid, sigma)[:, :, None] + np.float32(1e-4))

    def strands(fill):
        """How far each pixel is from the backdrop, and how much of a thin thing it is
        (0..1): different from the backdrop, and too thin to survive an opening."""
        dist = np.abs(work - fill).max(axis=2)
        fine = dist - cv2.morphologyEx(dist, cv2.MORPH_OPEN, thin)
        return dist, smoother._smoothstep(np.minimum(dist, fine * np.float32(1.5)), t["lo"], t["hi"])

    def hairlike(fill):
        """(1 where a pixel is the backdrop with some of the hair's colour mixed in, 1
        where it is clearly some other colour: a logo, a seam, a lamp). Neither, for a
        pixel too close to the backdrop's colour to say."""
        diff = work - fill
        towards = hair - fill
        share = (diff * towards).sum(axis=2) / ((towards * towards).sum(axis=2) + np.float32(1e-3))
        rest = np.sqrt(((diff - share[:, :, None] * towards) ** 2).sum(axis=2))
        size = np.sqrt((diff * diff).sum(axis=2))
        off = (rest >= t["stray"] * size + t["stray_floor"]) | (share >= t["mix_hi"]) | (share <= -t["mix_lo"])
        return ((share > t["mix_lo"]) & ~off).astype(np.float32), off & (size >= t["other"])

    outside = (1 - near).astype(np.float32)
    _, first = strands(backdrop(outside))
    fill = backdrop(outside * (first < 0.2))         # the backdrop without the thin things in it
    dist, found = strands(fill)
    like, off = hairlike(fill)
    # The faint edge of a thin thing of another colour could pass for hair: nothing
    # right beside one counts.
    other = cv2.dilate(((found > 0.5) & off).astype(np.uint8), _box(t["thin"] * fw / 2.0))
    found = found * like * (1 - other) * ring
    # A busy background has no backdrop to tidy against.
    quiet = dist[(ring > 0) & (found < 0.2)]
    rough = float(np.median(quiet)) if quiet.size else t["busy_hi"]
    plain = 1.0 - float(smoother._smoothstep(np.float32(rough), t["busy_lo"], t["busy_hi"]))
    if plain <= 0.0 or not float(found.max()) > 0.0:
        return bgr
    soft = cv2.GaussianBlur(cv2.dilate(found, np.ones((3, 3), np.uint8)), (0, 0), max(0.7, t["feather"] * fw))
    soft = np.clip(soft * np.float32(1.3), 0.0, 1.0) * ring
    if scale < 1.0:
        soft = cv2.resize(soft, (rw, rh), interpolation=cv2.INTER_LINEAR)
        fill = cv2.resize(fill, (rw, rh), interpolation=cv2.INTER_LINEAR)
    if keep is not None:
        kx, ky, edit = keep
        ax0, ay0 = max(x0, kx), max(y0, ky)
        ax1, ay1 = min(x1, kx + edit.shape[1]), min(y1, ky + edit.shape[0])
        if ax1 > ax0 and ay1 > ay0:
            soft = soft.copy()
            soft[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] *= 1.0 + np.minimum(edit[ay0 - ky:ay1 - ky, ax0 - kx:ax1 - kx], 0.0)
    now = roi.astype(np.float32)
    out = bgr.copy()
    mixed = now + (np.float32(amount * plain) * soft)[:, :, None] * (fill - now) + np.float32(0.5)
    out[y0:y1, x0:x1] = np.clip(mixed, 0, 255).astype(np.uint8)
    return out


def head_tilt(pts: np.ndarray) -> float:
    """One face's tilt in degrees: the eye line's angle, positive when the eye on the
    right of the photo is lower."""
    eye_r, eye_l = face_landmarks._eye_centres(pts)
    return math.degrees(math.atan2(eye_l[1] - eye_r[1], eye_l[0] - eye_r[0]))


def straighten(bgr: np.ndarray, points, amount: float) -> np.ndarray:
    """bgr with each tilted head turned upright by this share of its tilt: a new array, or
    bgr itself when nothing changes. points: each face's landmarks or None."""
    if amount <= 0.0 or not points:
        return bgr
    amount = min(1.0, float(amount))
    t = LEVEL
    h, w = bgr.shape[:2]
    out = bgr
    for pts in points:
        if pts is None:
            continue
        eye_r, eye_l = face_landmarks._eye_centres(pts)
        d = float(np.hypot(eye_l[0] - eye_r[0], eye_l[1] - eye_r[1]))
        if not d > 4.0:
            continue
        tilt = max(-t["most"], min(t["most"], head_tilt(pts)))
        turn = math.radians(tilt) * amount                 # the output is the photo turned back by this
        if abs(turn) < 1e-4:
            continue
        ax, ay = (eye_l[0] - eye_r[0]) / d, (eye_l[1] - eye_r[1]) / d     # along the eyes
        dx, dy = -ay, ax                                                    # down the face ...
        mx, my = (eye_r[0] + eye_l[0]) / 2.0, (eye_r[1] + eye_l[1]) / 2.0
        if (pts[face_landmarks.CHIN, 0] - mx) * dx + (pts[face_landmarks.CHIN, 1] - my) * dy < 0:
            dx, dy = -dx, -dy                                               # ... on the chin's side
        px = float(pts[face_landmarks.CHIN, 0]) + t["pivot"] * d * dx
        py = float(pts[face_landmarks.CHIN, 1]) + t["pivot"] * d * dy
        # The part of the photo that can move: everything within reach of the pivot sideways,
        # from the top of the photo's corner of it down to where the neck's fade ends.
        reach = t["side_end"] * d
        corners = [(px + sx * reach * ax + u * dx, py + sx * reach * ay + u * dy)
                   for sx in (-1.0, 1.0) for u in (t["neck"] * d, -float(h + w))]
        x0 = max(0, int(math.floor(min(c[0] for c in corners))))
        x1 = min(w, int(math.ceil(max(c[0] for c in corners))) + 1)
        y0 = max(0, int(math.floor(min(c[1] for c in corners))))
        y1 = min(h, int(math.ceil(max(c[1] for c in corners))) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1, dtype=np.float32) - np.float32(px),
                             np.arange(y0, y1, dtype=np.float32) - np.float32(py))
        down = gx * np.float32(dx) + gy * np.float32(dy)          # how far below the pivot, down the face
        across = np.abs(gx * np.float32(ax) + gy * np.float32(ay))
        weight = (1.0 - smoother._smoothstep(down, -t["above"] * d, t["neck"] * d)) \
            * (1.0 - smoother._smoothstep(across, t["side"] * d, t["side_end"] * d))
        angle = np.float32(turn) * weight
        cos, sin = np.cos(angle), np.sin(angle)
        # Where each pixel of the result comes from: its place turned forward again about the pivot.
        map_x = (np.float32(px) + cos * gx - sin * gy).astype(np.float32)
        map_y = (np.float32(py) + sin * gx + cos * gy).astype(np.float32)
        turned = cv2.remap(out, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        if out is bgr:
            out = bgr.copy()
        moved = weight > 0
        out[y0:y1, x0:x1][moved] = turned[moved]
    return out


def finish(bgr: np.ndarray, mask: np.ndarray, area: np.ndarray | None, points, luma: float | None,
           eyes: float = 0.0, tone: float = 0.0, shine: float = 0.0, faces=None, bright=None,
           strays: float = 0.0, person=None, keep=None, level: float = 0.0) -> np.ndarray:
    """The steps in order: shiny patches (shine, with the faces and bright_colours), then
    under-eyes, then skin tone, then stray hairs (strays, with the faces, person_of and
    the brush's Restore edit), then straightening the heads (level). bgr itself with all
    of them off (the default)."""
    out = matte(bgr, mask, faces, bright, shine)
    out = under_eyes(out, points, mask, eyes)
    out = balance(out, mask if area is None else area, luma, tone)
    out = tidy(out, person, faces, strays, keep)
    return straighten(out, points, level)
