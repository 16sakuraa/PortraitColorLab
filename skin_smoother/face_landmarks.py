"""
Eye, brow and lip outlines from a face landmark network.

Google's MediaPipe Face Mesh V2 places 478 points on a face. It runs on the CPU
through OpenCV, so no GPU or extra install is needed, and takes milliseconds per face.

It expects an upright 256x256 crop with some margin around the face, so each face
takes two passes: the first from the face detector's box and eyes, the second from
the first pass's own points, which gives a tighter, straighter crop.

These outlines are what smoothing has to leave alone. The older method in smoother.py
(eye_mask, brow_mask, lip_mask) takes over, face by face, where the model doesn't
clearly see a face, and for every face if the model file is missing.

The model file is models/face_landmarks.onnx (next to the program, or bundled inside
the .exe).
"""

from __future__ import annotations
import os
import sys

import numpy as np
import cv2

import smoother

MODEL_NAME = "face_landmarks.onnx"
SIZE = 256           # the model's input, a side in pixels
POINTS = 478
MAX_FACES = 30       # two model runs each; smaller faces past this use the older method
MIN_FLAG = 0.0       # the model's "this is a face" score, before its sigmoid

# Point numbers around each feature, from Google's Face Mesh map. "Right" is the
# person's right, as with the face detector's eyes.
LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
EYE_R = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
EYE_L = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
BROW_R = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
BROW_L = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
EYE_CORNERS_R = (33, 133)
EYE_CORNERS_L = (263, 362)

TUNE = {
    "crop_side": 1.5,     # a crop's side, times the face's size: the model wants 25% margin each way
    "eye_margin": 0.05,   # how far past each outline to protect, a fraction of the distance between the eyes
    "brow_margin": 0.03,
    "lip_margin": 0.01,
}

_net = None
_net_path = None
_net_failed = False


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


def _run(net, bgr: np.ndarray, cx: float, cy: float, side: float, angle: float):
    """Run the model on a square around (cx, cy), turned by angle (radians).

    Returns (points in photo pixels, face flag), or None if the square misses the photo.
    """
    h, w = bgr.shape[:2]
    c, s = np.cos(angle), np.sin(angle)
    scale = side / SIZE                  # photo pixels per model pixel
    # Shrink a large crop first, so the turn doesn't alias.
    shrink = min(1.0, 1.0 / scale)
    reach = int(np.ceil(side * 0.75)) + 2
    x0, y0 = max(0, int(cx) - reach), max(0, int(cy) - reach)
    x1, y1 = min(w, int(cx) + reach), min(h, int(cy) + reach)
    if x1 <= x0 or y1 <= y0:
        return None
    part = bgr[y0:y1, x0:x1]
    if shrink < 1.0:
        part = cv2.resize(part, (max(1, round((x1 - x0) * shrink)), max(1, round((y1 - y0) * shrink))),
                          interpolation=cv2.INTER_AREA)
    sx, sy = part.shape[1] / (x1 - x0), part.shape[0] / (y1 - y0)
    # Model pixel -> photo pixel, and on into the (shrunk) part.
    a = np.array([[scale * c, -scale * s, cx - scale * (c * SIZE / 2 - s * SIZE / 2)],
                  [scale * s, scale * c, cy - scale * (s * SIZE / 2 + c * SIZE / 2)]], np.float64)
    to_part = np.array([[a[0, 0] * sx, a[0, 1] * sx, (a[0, 2] - x0) * sx],
                        [a[1, 0] * sy, a[1, 1] * sy, (a[1, 2] - y0) * sy]], np.float64)
    crop = cv2.warpAffine(part, to_part, (SIZE, SIZE), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT)
    x = (crop[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]   # RGB, 0..1
    net.setInput(x)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    points = next(o for o in outs if o.size == POINTS * 3).reshape(-1, 3)[:, :2].astype(np.float64)
    flag = float(next(o for o in outs if o.size == 1).ravel()[0])
    photo = np.stack([a[0, 0] * points[:, 0] + a[0, 1] * points[:, 1] + a[0, 2],
                      a[1, 0] * points[:, 0] + a[1, 1] * points[:, 1] + a[1, 2]], axis=1)
    return photo, flag


def _eye_centres(pts: np.ndarray) -> tuple:
    return pts[list(EYE_CORNERS_R)].mean(axis=0), pts[list(EYE_CORNERS_L)].mean(axis=0)


def _fit(net, bgr: np.ndarray, f):
    """One face's (points, flag) after both passes, or None."""
    side = TUNE["crop_side"]
    first = _run(net, bgr, float(f[0]) + float(f[2]) / 2, float(f[1]) + float(f[3]) / 2,
                 side * max(float(f[2]), float(f[3])), np.arctan2(float(f[7]) - float(f[5]), float(f[6]) - float(f[4])))
    if first is None:
        return None
    pts = first[0]
    eye_r, eye_l = _eye_centres(pts)
    angle = np.arctan2(eye_l[1] - eye_r[1], eye_l[0] - eye_r[0])
    # The points' box, measured upright, gives the second crop.
    c, s = np.cos(angle), np.sin(angle)
    ux = c * pts[:, 0] + s * pts[:, 1]
    uy = c * pts[:, 1] - s * pts[:, 0]
    mx, my = (ux.min() + ux.max()) / 2, (uy.min() + uy.max()) / 2
    return _run(net, bgr, c * mx - s * my, s * mx + c * my, side * max(ux.max() - ux.min(), uy.max() - uy.min()), angle)


def landmarks(bgr: np.ndarray, faces) -> list:
    """For each face, its points in photo pixels (float64, POINTS x 2), or None where
    the older method should be used."""
    global _net_failed
    found = [None] * len(faces)
    net = _get_net()
    if net is None:
        return found
    try:
        for i in sorted(range(len(faces)), key=lambda k: -float(faces[k][2]))[:MAX_FACES]:
            fit = _fit(net, bgr, faces[i])
            if fit is not None and fit[1] > MIN_FLAG:
                found[i] = fit[0]
    except cv2.error:            # an OpenCV that loads the model but can't run it
        _net_failed = True
        return [None] * len(faces)
    return found


def zones(pts: np.ndarray, w: int, h: int) -> list:
    """One face's eyes, brows and lips, as [(uint8 mask, (x0, y0))] in a w x h photo."""
    t = TUNE
    eye_r, eye_l = _eye_centres(pts)
    eye_d = float(np.hypot(eye_l[0] - eye_r[0], eye_l[1] - eye_r[1]))
    out = []
    for ids, margin in ((EYE_R, t["eye_margin"]), (EYE_L, t["eye_margin"]), (BROW_R, t["brow_margin"]),
                        (BROW_L, t["brow_margin"]), (LIPS, t["lip_margin"])):
        poly = np.round(pts[list(ids)]).astype(np.int32)
        size = max(3, int(round(margin * eye_d))) | 1
        pad = size // 2 + 1
        x0, y0 = max(0, int(poly[:, 0].min()) - pad), max(0, int(poly[:, 1].min()) - pad)
        x1, y1 = min(w, int(poly[:, 0].max()) + pad + 1), min(h, int(poly[:, 1].max()) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [poly - np.array([x0, y0], np.int32)], 255)
        out.append((cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))), (x0, y0)))
    return out
