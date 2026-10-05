"""
Learned blemish removal (pimples, spots, small marks).

A small neural network trained on professionally retouched faces (FFHQR) runs
on the CPU through OpenCV, so no GPU or extra install is needed.

How a photo is processed, per face:
  1. From YuNet's eye and mouth points, cut out the face the same way the
     training faces were framed (FFHQ alignment) at 1024x1024.
  2. The network predicts a *correction* for that crop (what to add to each
     pixel to heal blemishes; zero almost everywhere).
  3. The correction is warped back and scaled up onto the full-resolution
     photo. Only the correction is resized, never the photo, so the camera's
     own detail is kept everywhere the network did not heal.

The model file is models/blemish_ffhqr.onnx (next to the program, or bundled
inside the .exe). Without it, the feature simply reports itself unavailable.
"""

from __future__ import annotations
import os
import sys

import numpy as np
import cv2

import smoother

MODEL_NAME = "blemish_ffhqr.onnx"
SIZE = 1024          # the network's working resolution (FFHQ framing)
_net = None
_net_path = None


def get_model_path() -> str | None:
    """Same search order as the face model: next to the .exe/script first,
    then the copy bundled in the .exe."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(smoother._base_dir(), "models", MODEL_NAME)]
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(sys._MEIPASS, "models", MODEL_NAME))
    candidates.append(os.path.join(here, "models", MODEL_NAME))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def have_model() -> bool:
    return get_model_path() is not None


def _get_net():
    global _net, _net_path
    path = get_model_path()
    if path is None:
        return None
    if _net is None or _net_path != path:
        _net = cv2.dnn.readNetFromONNX(path)
        _net_path = path
    return _net


def ffhq_quad(face_row) -> np.ndarray:
    """FFHQ-style crop square (4 corners: TL, BL, BR, TR) from a YuNet face.

    Mirrors the alignment used to build FFHQ (eye centres + mouth corners).
    """
    e1 = np.array(face_row[4:6], dtype=np.float64)
    e2 = np.array(face_row[6:8], dtype=np.float64)
    m1 = np.array(face_row[10:12], dtype=np.float64)
    m2 = np.array(face_row[12:14], dtype=np.float64)
    eye_avg = (e1 + e2) / 2
    mouth_avg = (m1 + m2) / 2
    eye_to_mouth = mouth_avg - eye_avg
    eye_to_eye = e2 - e1
    # Make eye_to_eye run from image-left eye to image-right eye (relative to
    # the face's own orientation), whichever way the detector ordered them.
    if eye_to_eye[0] * eye_to_mouth[1] - eye_to_eye[1] * eye_to_mouth[0] < 0:
        eye_to_eye = -eye_to_eye
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    # YuNet's eye points sit slightly closer together than the dlib averages
    # FFHQ used; measured on 294 FFHQ faces our crop came out 4.4% small.
    x *= 1.046
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return np.stack([c - x - y, c - x + y, c + x + y, c + x - y])


def _edge_window(size: int, fade: int) -> np.ndarray:
    """1 in the middle, fading to 0 over `fade` px at the crop border, so a
    correction never ends in a hard line."""
    r = np.arange(size, dtype=np.float32)
    ramp = np.clip(np.minimum(r, size - 1 - r) / fade, 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)  # smoothstep
    return np.minimum(ramp[:, None], ramp[None, :])


def _crop_weights(size: int) -> np.ndarray:
    """Two weights per crop pixel, used when several faces' crops overlap:
      [0] blend : how much this crop is trusted there (high near the face it
                  was centred on, low towards its edges)
      [1] fade  : how much correction may be applied at all (soft crop edge)
    """
    fade = _edge_window(size, size // 24)
    r = np.arange(size, dtype=np.float32) - (size - 1) / 2
    rr = np.hypot(r[:, None], r[None, :]) / (size / 2)
    blend = fade * np.exp(-(rr / 0.6) ** 2)
    return np.dstack([blend, fade]).astype(np.float32)


_WEIGHTS = None


def _correction_for_face(bgr: np.ndarray, quad: np.ndarray):
    """Network correction for one face, mapped back onto the photo.

    Returns (x0, y0, corr, weights): corr is float32 HxWx3 in 0..255 units for
    bgr[y0:y0+H, x0:x0+W], weights is HxWx2 (see _crop_weights). None if the
    face is too small to be useful."""
    global _WEIGHTS
    net = _get_net()
    h, w = bgr.shape[:2]
    qsize = float(np.hypot(*(quad[3] - quad[0])))
    if qsize < MIN_QUAD:
        return None

    # Region of the photo the crop covers (clipped to the photo).
    x0 = int(max(0, np.floor(quad[:, 0].min())))
    y0 = int(max(0, np.floor(quad[:, 1].min())))
    x1 = int(min(w, np.ceil(quad[:, 0].max()) + 1))
    y1 = int(min(h, np.ceil(quad[:, 1].max()) + 1))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    src = bgr[y0:y1, x0:x1]
    q = quad - [x0, y0]

    # Shrink large faces with proper averaging first, so the crop isn't
    # aliased (DSLR faces are often 2-3x the network's size).
    f = 1.0
    if qsize > SIZE * 1.2:
        f = SIZE / qsize
        src = cv2.resize(src, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    dst_pts = np.float32([[0, 0], [0, SIZE], [SIZE, 0]])
    M_small = cv2.getAffineTransform(np.float32(q[[0, 1, 3]] * f), dst_pts)
    crop = cv2.warpAffine(src, M_small, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

    blob = crop.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    net.setInput(blob)
    out = net.forward()[0].transpose(1, 2, 0)
    corr = (np.clip(out, 0, 1) * 255.0 - crop.astype(np.float32))

    if _WEIGHTS is None:
        _WEIGHTS = _crop_weights(SIZE)

    # Map the correction (and its weights) back onto the full-res region.
    M_full = M_small.astype(np.float64).copy()
    M_full[:, :2] *= f
    back = lambda a: cv2.warpAffine(a, M_full, (x1 - x0, y1 - y0),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return x0, y0, back(corr), back(_WEIGHTS)


def _plausible(face_row) -> bool:
    """Does the landmark layout look like a face?

    Ears, hair, dark corners and blurry background get detected as faces now
    and then; running the network there would edit random texture. The score
    can't separate them (real large faces often score only 0.6-0.75), but the
    layout can. Measured on 294 real FFHQ faces (1st-99th percentile):
      eye spacing / box width    0.41-0.56
      eyes-to-mouth / eye spacing 0.75-1.38
      eye line vs eyes-to-mouth  ~perpendicular (|cos| <= 0.04)
      nose position eyes->mouth  0.52-0.79
    The limits below leave room for turned and tilted heads (common in group
    photos). A non-face that still slips through costs little: on hair or
    jewellery the network changes almost nothing, and where its crop overlaps
    a real face the blending in correction() favours the real face's crop.
    """
    e1, e2, nose, m1, m2 = (np.array(face_row[4 + 2 * k:6 + 2 * k], float) for k in range(5))
    eye_avg, mouth_avg = (e1 + e2) / 2, (m1 + m2) / 2
    ee, em = e2 - e1, mouth_avg - eye_avg
    eye_d, em_d = float(np.hypot(*ee)), float(np.hypot(*em))
    if eye_d < 0.28 * float(face_row[2]) or em_d < 1e-6:
        return False
    cos = abs(float(ee @ em)) / (eye_d * em_d)
    t_nose = float((nose - eye_avg) @ em) / em_d ** 2
    return 0.6 <= em_d / eye_d <= 2.2 and cos <= 0.12 and 0.35 <= t_nose <= 1.0


MIN_QUAD = 96  # faces whose crop would be smaller than this are skipped


def select_faces(faces) -> list[np.ndarray]:
    """Crop squares (see ffhq_quad) of the faces worth processing.

    Largest first. A detection whose landmarks sit inside an already chosen
    face's box is the same face found twice (at another scale) and is
    skipped. Neighbours in a group photo are separate faces and each gets its
    own crop. (A duplicate that slips past this is harmless: overlapping
    corrections are blended, not added.)
    """
    chosen: list[tuple[np.ndarray, np.ndarray]] = []
    for f in sorted(faces, key=lambda r: -float(r[2] * r[3])):
        if not _plausible(f):
            continue
        quad = ffhq_quad(f)
        if np.hypot(*(quad[3] - quad[0])) < MIN_QUAD:
            continue
        cx, cy = float(np.mean(f[4:14:2])), float(np.mean(f[5:14:2]))
        if any(g[0] <= cx <= g[0] + g[2] and g[1] <= cy <= g[1] + g[3] for g, _ in chosen):
            continue
        chosen.append((f, quad))
    return [q for _, q in chosen]


def correction(bgr: np.ndarray, faces=None):
    """The network's correction for every face in the photo, and how many
    faces it covers. Compute once per photo; apply() is then cheap, so an
    amount slider can move freely.

    The correction only covers the faces' region: (x0, y0, array HxWx3, float32,
    0..255 units), or None. Where two faces' crops overlap (people close
    together), each pixel takes a blend favouring the crop whose face is
    nearest, so nothing is corrected twice.
    """
    if _get_net() is None:
        return None, 0
    if faces is None:
        faces = smoother._detect_faces(bgr)
    if faces is None or len(faces) == 0:
        return None, 0
    parts = [p for p in (_correction_for_face(bgr, q) for q in select_faces(faces))
             if p is not None]
    if not parts:
        return None, 0
    X0 = min(p[0] for p in parts)
    Y0 = min(p[1] for p in parts)
    X1 = max(p[0] + p[2].shape[1] for p in parts)
    Y1 = max(p[1] + p[2].shape[0] for p in parts)
    num = np.zeros((Y1 - Y0, X1 - X0, 3), np.float32)
    den = np.zeros((Y1 - Y0, X1 - X0), np.float32)
    fade = np.zeros((Y1 - Y0, X1 - X0), np.float32)
    for x0, y0, corr, wts in parts:
        sl = (slice(y0 - Y0, y0 - Y0 + corr.shape[0]), slice(x0 - X0, x0 - X0 + corr.shape[1]))
        num[sl] += corr * wts[:, :, :1]
        den[sl] += wts[:, :, 0]
        np.maximum(fade[sl], wts[:, :, 1], out=fade[sl])
    out = num * (fade / np.maximum(den, 1e-6))[:, :, None]
    return (X0, Y0, out), len(parts)


def apply(bgr: np.ndarray, corr, amount: float = 1.0,
          gate: np.ndarray | None = None) -> np.ndarray:
    """bgr + amount * corr (from correction()), optionally only where gate
    (full-image float mask 0..1) > 0."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if corr is None or amount <= 0:
        return bgr.copy()
    x0, y0, c = corr
    h, w = c.shape[:2]
    if gate is not None:
        c = c * gate[y0:y0 + h, x0:x0 + w, None]
    out = bgr.copy()
    region = bgr[y0:y0 + h, x0:x0 + w].astype(np.float32) + amount * c
    out[y0:y0 + h, x0:x0 + w] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
    return out


def remove_blemishes(bgr: np.ndarray, amount: float = 1.0, faces=None,
                     gate: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Heal blemishes on every detected face.

    amount : 0..1   how much of the network's correction to apply
    faces  : optional YuNet rows (from smoother._detect_faces) to reuse
    gate   : optional float mask in [0,1] (e.g. a widened skin mask); the
             correction is only applied where it is > 0
    Returns (result, faces_processed).
    """
    corr, n = correction(bgr, faces)
    return apply(bgr, corr, amount, gate), n
