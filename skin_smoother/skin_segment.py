"""
Skin segmentation with a small neural network.

Google's MediaPipe multiclass selfie model labels every pixel as background, hair,
body skin, face skin, clothes or other (accessories). It runs on the CPU through
OpenCV, so no GPU or extra install is needed. It sees only 256x256 pixels, so:

  1. Faces are run as crops, so each one fills enough of the model's input. Faces
     close together share a crop. A pass over the whole photo adds body skin
     outside the crops, when people are large enough in the frame for that to mean
     anything.
  2. The results are blended at working resolution and snapped to the photo's own
     edges with a guided filter, because the model's masks aren't pixel-perfect.
  3. Body skin is kept only on a person with a detected face, and only near that
     face's skin colour, so skin-coloured walls and strangers stay out.

The model file is models/selfie_multiclass_256x256.onnx (next to the program, or
bundled inside the .exe). Without it, the skin mask falls back to its older method.
"""

from __future__ import annotations
import os
import sys
from typing import NamedTuple

import numpy as np
import cv2

import smoother

MODEL_NAME = "selfie_multiclass_256x256.onnx"
SIZE = 256           # the model's input, a side in pixels
BACKGROUND, HAIR, BODY, FACE, CLOTHES, OTHERS = range(6)
MAX_CROPS = 8        # each one is a model run; faces past this use the older method
MIN_FACE_SCORE = 0.6 # a face the model doesn't clearly see (average face-skin score) uses the older method

TUNE = {
    "crop_reach": 1.0,    # a crop reaches this many face sizes out from a face's centre
    "crop_faces": 3.5,    # a crop's side is at most this many times its smallest face: about 73 model pixels
    "whole_share": 0.6,   # a crop this share of the photo is no better than the whole photo
    "whole_face": 32.0,   # the whole-photo pass needs the largest face to span this many model pixels
    "crop_feather": 0.1,  # crop edges fade over this share of the crop's side
    "prob_lo": 0.2,       # score ranges that become "not skin" and "skin", so skin is fully 1
    "prob_hi": 0.8,
    "snap_radius": 0.02,  # edge-snapping reach, a fraction of face width
    "snap_edge": 0.03,    # contrast (0..1) the snapping treats as a real edge
    "snap_lo": 0.25,      # snapping leaves a faint tail past an edge: what's below this goes
    "snap_hi": 0.75,      # and what's above this is fully skin
    "colour_face": 0.8,   # face-skin score of the pixels that define a face's skin colour
    "colour_lo": 10.0,    # body skin this far (Lab a/b, 8-bit units) from a face's colour is kept
    "colour_hi": 20.0,    # and this far is dropped
}

_net = None
_net_path = None
_net_failed = False


class Segments(NamedTuple):
    """Class scores for a region of the photo, at working resolution."""
    x0: int              # the region, in photo pixels
    y0: int
    x1: int
    y1: int
    work: np.ndarray     # the region at working resolution, uint8 BGR
    face: np.ndarray     # float32 scores 0..1, same size as work
    body: np.ndarray
    person: np.ndarray   # anything but background
    covered: tuple       # indices of the faces this result can be trusted for


def get_model_path() -> str | None:
    """Same search order as the face model: next to the .exe/script first, then
    the copy bundled in the .exe."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(smoother._base_dir(), "models", MODEL_NAME)]
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(sys._MEIPASS, "models", MODEL_NAME))
    candidates.append(os.path.join(here, "models", MODEL_NAME))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _get_net():
    global _net, _net_path, _net_failed
    path = get_model_path()
    if path is None or _net_failed:
        return None
    if _net is None or _net_path != path:
        try:
            _net = cv2.dnn.readNetFromONNX(path)
            _net_path = path
        except cv2.error:        # an OpenCV too old for this model: use the older method
            _net_failed = True
            return None
    return _net


def have_model() -> bool:
    return _get_net() is not None


def _scores(net, part: np.ndarray) -> np.ndarray:
    """Class scores for part of a photo: float32 6 x SIZE x SIZE, adding up to 1."""
    small = cv2.resize(part, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    x = (small[:, :, ::-1].astype(np.float32) - 127.5) / 127.5   # RGB, -1..1
    net.setInput(x.transpose(2, 0, 1)[None])
    logits = net.forward()[0]
    e = np.exp(logits - logits.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def _crops(faces, w: int, h: int) -> list[tuple]:
    """Where to run the model: [(x0, y0, x1, y1, [face indices])], largest faces first.

    A face joins a crop it overlaps while the crop stays small enough for its
    smallest face to fill a useful part of the model's input.
    """
    t = TUNE
    clusters = []
    for i in sorted(range(len(faces)), key=lambda k: -float(faces[k][2])):
        f = faces[i]
        reach = t["crop_reach"] * float(max(f[2], f[3]))
        cx, cy = float(f[0]) + float(f[2]) / 2.0, float(f[1]) + float(f[3]) / 2.0
        x0, y0, x1, y1 = cx - reach, cy - reach, cx + reach, cy + reach
        for c in clusters:
            nx0, ny0, nx1, ny1 = min(c[0], x0), min(c[1], y0), max(c[2], x1), max(c[3], y1)
            smallest = min(c[4], float(f[2]))
            if x0 < c[2] and c[0] < x1 and y0 < c[3] and c[1] < y1 \
                    and max(nx1 - nx0, ny1 - ny0) <= t["crop_faces"] * smallest:
                c[:5] = nx0, ny0, nx1, ny1, smallest
                c[5].append(i)
                break
        else:
            clusters.append([x0, y0, x1, y1, float(f[2]), [i]])
    rects = []
    for x0, y0, x1, y1, _, members in clusters:
        half = max(x1 - x0, y1 - y0) / 2.0     # square, so a face keeps its shape
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rects.append((int(max(0.0, cx - half)), int(max(0.0, cy - half)),
                      int(min(float(w), cx + half)), int(min(float(h), cy + half)), members))
    return rects


def _ramp(n: int, span: float, open_start: bool, open_end: bool) -> np.ndarray:
    """1 in the middle, fading to 0 over span at each end that's marked open."""
    i = np.arange(n, dtype=np.float32)
    up = (i + 0.5) / span if open_start else np.ones(n, np.float32)
    down = (n - 0.5 - i) / span if open_end else np.ones(n, np.float32)
    return np.clip(np.minimum(up, down), 0.0, 1.0)


def segment(bgr: np.ndarray, faces) -> Segments | None:
    """Run the model on a photo's faces. None without a model or faces, or if this
    OpenCV can't run it: the mask then falls back to its older method."""
    global _net_failed
    net = _get_net()
    if net is None or faces is None or len(faces) == 0:
        return None
    try:
        return _segment(net, bgr, faces)
    except cv2.error:
        _net_failed = True
        return None


def _segment(net, bgr: np.ndarray, faces) -> Segments:
    t = TUNE
    h, w = bgr.shape[:2]
    rects = _crops(faces, w, h)
    widest = max(float(f[2]) for f in faces)
    whole = widest * SIZE / max(h, w) >= t["whole_face"]

    # Crops past the limit aren't run; a crop nearly as big as the photo adds nothing.
    covered = []
    runs = []
    for rect in rects[:MAX_CROPS]:
        covered += rect[4]
        if not (whole and (rect[2] - rect[0]) * (rect[3] - rect[1]) >= t["whole_share"] * h * w):
            runs.append(rect)

    # Work on the whole photo if it has a pass of its own, else only where crops are.
    if whole or not runs:
        rx0, ry0, rx1, ry1 = 0, 0, w, h
    else:
        rx0, ry0 = min(r[0] for r in runs), min(r[1] for r in runs)
        rx1, ry1 = max(r[2] for r in runs), max(r[3] for r in runs)
    rw, rh = rx1 - rx0, ry1 - ry0
    scale = min(1.0, (smoother.WORK_PIXELS / (rh * rw)) ** 0.5)
    ww, wh = max(1, round(rw * scale)), max(1, round(rh * scale))
    region = bgr[ry0:ry1, rx0:rx1]
    work = cv2.resize(region, (ww, wh), interpolation=cv2.INTER_AREA) if scale < 1.0 else region.copy()
    sx, sy = ww / rw, wh / rh

    def planes(scores, cols, rows):
        return [cv2.resize(scores[c], (cols, rows), interpolation=cv2.INTER_LINEAR) for c in (FACE, BODY, BACKGROUND)]

    if whole:
        s = planes(_scores(net, bgr), ww, wh)
    else:
        s = [np.zeros((wh, ww), np.float32), np.zeros((wh, ww), np.float32), np.ones((wh, ww), np.float32)]

    # Crops take over from the whole-photo pass where they reach, fading at their
    # edges. Where crops overlap they're averaged by weight.
    if runs:
        num = [np.zeros((wh, ww), np.float32) for _ in s]
        den = np.zeros((wh, ww), np.float32)
        cover = np.zeros((wh, ww), np.float32)
        for x0, y0, x1, y1, _ in runs:
            cx0, cx1 = int((x0 - rx0) * sx), min(ww, int(np.ceil((x1 - rx0) * sx)))
            cy0, cy1 = int((y0 - ry0) * sy), min(wh, int(np.ceil((y1 - ry0) * sy)))
            cols, rows = cx1 - cx0, cy1 - cy0
            span = max(1.0, t["crop_feather"] * max(cols, rows))
            # No fade at the photo's own edges: a face there would lose its side.
            weight = np.minimum(_ramp(rows, span, y0 > 0, y1 < h)[:, None],
                                _ramp(cols, span, x0 > 0, x1 < w)[None, :])
            for total, plane in zip(num, planes(_scores(net, bgr[y0:y1, x0:x1]), cols, rows)):
                total[cy0:cy1, cx0:cx1] += weight * plane
            den[cy0:cy1, cx0:cx1] += weight
            np.maximum(cover[cy0:cy1, cx0:cx1], weight, out=cover[cy0:cy1, cx0:cx1])
        for i in range(len(s)):
            s[i] = cover * (num[i] / np.maximum(den, 1e-6)) + (1.0 - cover) * s[i]
    face, body, background = s

    # Faces the model barely sees use the older method rather than a hole in the mask.
    trusted = []
    for i in covered:
        x0, y0, x1, y1 = _middle(faces[i], rx0, ry0, sx, sy, ww, wh)
        if x1 > x0 and y1 > y0 and float(face[y0:y1, x0:x1].mean()) >= MIN_FACE_SCORE:
            trusted.append(i)
    return Segments(rx0, ry0, rx1, ry1, work, face, body, 1.0 - background, tuple(sorted(trusted)))


def _middle(f, rx0: int, ry0: int, sx: float, sy: float, ww: int, wh: int) -> tuple[int, int, int, int]:
    """The central half of a face box, in working pixels."""
    x0 = int((float(f[0]) + 0.25 * float(f[2]) - rx0) * sx)
    y0 = int((float(f[1]) + 0.25 * float(f[3]) - ry0) * sy)
    x1 = int((float(f[0]) + 0.75 * float(f[2]) - rx0) * sx)
    y1 = int((float(f[1]) + 0.75 * float(f[3]) - ry0) * sy)
    return max(0, x0), max(0, y0), min(ww, x1), min(wh, y1)


def _body_keep(seg: Segments, faces) -> np.ndarray:
    """0..1 for each working pixel: could body skin here belong to a detected person?

    It has to be on a connected person shape that contains a detected face, and
    near that face's skin colour (a and b only, since shade varies with the light).
    """
    t = TUNE
    wh, ww = seg.face.shape
    rw, rh = seg.x1 - seg.x0, seg.y1 - seg.y0
    sx, sy = ww / rw, wh / rh
    n, labels = cv2.connectedComponents((seg.person > 0.5).astype(np.uint8), connectivity=8)
    lab = cv2.cvtColor(seg.work, cv2.COLOR_BGR2LAB).astype(np.float32)
    on_person = np.zeros(n, np.float32)
    nearest = np.full((wh, ww), 1e9, np.float32)
    for f in faces:
        x0, y0, x1, y1 = _middle(f, seg.x0, seg.y0, sx, sy, ww, wh)
        if x1 <= x0 or y1 <= y0:
            continue
        on_person[labels[y0:y1, x0:x1]] = 1.0
        sel = seg.face[y0:y1, x0:x1] > t["colour_face"]
        if int(sel.sum()) < 16:
            continue
        a0 = float(np.median(lab[y0:y1, x0:x1, 1][sel]))
        b0 = float(np.median(lab[y0:y1, x0:x1, 2][sel]))
        nearest = np.minimum(nearest, np.maximum(np.abs(lab[:, :, 1] - a0), np.abs(lab[:, :, 2] - b0)))
    on_person[0] = 0.0
    return on_person[labels] * (1.0 - smoother._smoothstep(nearest, t["colour_lo"], t["colour_hi"]))


def skin_area(seg: Segments, faces, face_only: bool = False):
    """The skin the model found: (x0, y0, float32 HxW in 0..1), or None if there's
    none. Sharp where the photo has an edge, in photo pixels, cut to the skin's box.
    """
    t = TUNE
    area = seg.face if face_only else seg.face + seg.body * _body_keep(seg, faces)
    area = smoother._smoothstep(area, t["prob_lo"], t["prob_hi"]).astype(np.float32)

    wh, ww = area.shape
    rw, rh = seg.x1 - seg.x0, seg.y1 - seg.y0
    width = float(np.median(np.asarray(faces, np.float64)[:, 2])) * ww / rw
    gray = cv2.cvtColor(seg.work.astype(np.float32) / 255.0, cv2.COLOR_BGR2GRAY)
    radius = max(2, round(width * t["snap_radius"]))
    area = smoother.guided_filter(gray, area, radius, t["snap_edge"] ** 2)
    area = smoother._smoothstep(area, t["snap_lo"], t["snap_hi"]).astype(np.float32)

    ys, xs = np.nonzero(area > 0)
    if len(ys) == 0:
        return None
    sx, sy = rw / ww, rh / wh
    a0, a1 = max(0, int(xs.min()) - 1), min(ww, int(xs.max()) + 2)
    b0, b1 = max(0, int(ys.min()) - 1), min(wh, int(ys.max()) + 2)
    x0, x1 = int(a0 * sx), min(rw, int(np.ceil(a1 * sx)))
    y0, y1 = int(b0 * sy), min(rh, int(np.ceil(b1 * sy)))
    full = cv2.resize(area[b0:b1, a0:a1], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    return seg.x0 + x0, seg.y0 + y0, full
