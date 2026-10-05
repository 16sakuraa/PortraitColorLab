"""
Face-aware skin smoothing.

Pipeline (the DM Portrait-style method):
  1. Detect the face and build a feathered skin mask (face oval minus eyes,
     eyebrows, mouth), intersected with a skin-tone mask so hair/background
     are left alone.
  2. Edge-preserving smoothing (bilateral) flattens blotches while keeping
     real edges (nose, jawline) crisp.
  3. Add back a fraction of the original fine texture so skin keeps pores and
     does not look like plastic.
  4. Composite the smoothed skin back through the soft mask.

Face detection uses OpenCV's YuNet model when the model file is present next to
this script (models/face_detection_yunet.onnx). Without it, the tool falls back
to a skin-tone mask so it still works. Nothing from DM Portrait is used -- this
is standard image processing (OpenCV).
"""

from __future__ import annotations
import os
import sys
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
# Default download location used by get_model.py (source runs).
YUNET_PATH = os.path.join(_HERE, "models", "face_detection_yunet.onnx")


def _base_dir() -> str:
    """Folder next to the running program (the .exe when frozen, else source)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return _HERE


def get_model_path() -> str | None:
    """Locate the YuNet model. Search order:
      1. models/ next to the .exe or script (user can drop it in, no rebuild),
      2. the copy bundled inside the .exe (PyInstaller _MEIPASS).
    Returns the first path that exists, or None.
    """
    candidates = [os.path.join(_base_dir(), "models", "face_detection_yunet.onnx")]
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(sys._MEIPASS, "models",
                                        "face_detection_yunet.onnx"))
    candidates.append(YUNET_PATH)
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def have_face_model() -> bool:
    return get_model_path() is not None


# ---------- Optional GPU (OpenCL) acceleration ----------
# The heavy smoothing filters can run on the GPU through OpenCV's "Transparent
# API" (UMat). It is optional and always falls back to the CPU if OpenCL is
# unavailable or a kernel is missing, so results are identical either way.
_USE_OPENCL = False


def opencl_available() -> bool:
    try:
        return bool(cv2.ocl.haveOpenCL())
    except Exception:
        return False


def set_opencl(enabled: bool) -> bool:
    """Turn the OpenCL path on/off. Returns the state actually in effect
    (False if OpenCL isn't available on this machine)."""
    global _USE_OPENCL
    want = bool(enabled) and opencl_available()
    try:
        cv2.ocl.setUseOpenCL(want)
        want = bool(cv2.ocl.useOpenCL())
    except Exception:
        want = False
    _USE_OPENCL = want
    return want


def using_opencl() -> bool:
    return _USE_OPENCL


DETECT_MAX = 1280  # longest side YuNet sees; large faces are missed above this


def _detect_faces(bgr: np.ndarray):
    """Return YuNet face rows (N x 15) or None if the model is unavailable.

    Large photos are searched on a reduced copy: at DSLR resolution a
    headshot face is far bigger than YuNet's largest anchor and it misses it.
    Coordinates are scaled back to the input image.
    """
    model_path = get_model_path()
    if model_path is None:
        return None
    h, w = bgr.shape[:2]
    s = min(1.0, DETECT_MAX / max(h, w))
    img = bgr
    if s < 1.0:
        img = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA)
    dh, dw = img.shape[:2]
    try:
        det = cv2.FaceDetectorYN.create(
            model_path, "", (dw, dh),
            score_threshold=0.6, nms_threshold=0.3, top_k=50,
        )
        det.setInputSize((dw, dh))
        _, faces = det.detect(img)
    except Exception:
        return None
    if faces is not None and s < 1.0:
        faces = faces.copy()
        faces[:, :14] /= s
    return faces


def skin_tone_mask(bgr: np.ndarray) -> np.ndarray:
    """Rough skin-tone mask in YCrCb. Returns uint8 0/255."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    m = cv2.inRange(ycrcb, lower, upper)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return m


def _odd_kernel(size: float) -> np.ndarray:
    k = max(3, int(size)) | 1
    return np.ones((k, k), np.uint8)


def _runs(flags: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Contiguous True runs [start, end] in a 1-D array, merging runs
    separated by at most max_gap False entries."""
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return runs


def lip_mask(bgr: np.ndarray, face_row) -> tuple[np.ndarray, tuple, bool]:
    """Ellipse covering the lips (and teeth / open mouth) of one face.

    YuNet's two mouth-corner points are coarse: on studio headshots they often
    sit on the skin *above* the lip corners, so an ellipse around them blanks
    the skin between nose and lip and smoothing skips it. Here the ellipse is
    placed and sized from where the lips actually are: rows and columns under
    the nose that are redder than the face's skin (or tooth-white). Only the
    position and size come from colour; the shape stays a smooth ellipse, so
    reddish cheeks or chin can't drag it out of shape.

    Returns (mask uint8 0/255 for a window, (x0, y0) of the window, measured):
    measured is False when the lips weren't distinct enough to measure and a
    default placement was used.
    """
    h, w = bgr.shape[:2]
    e1, e2 = np.array(face_row[4:6], float), np.array(face_row[6:8], float)
    m1, m2 = np.array(face_row[10:12], float), np.array(face_row[12:14], float)
    eye_d = float(np.hypot(*(e2 - e1))) or float(face_row[2]) * 0.4
    mw = max(float(np.hypot(*(m2 - m1))), 0.6 * eye_d)   # mouth width estimate
    c = (m1 + m2) / 2
    angle = float(np.degrees(np.arctan2(m2[1] - m1[1], m2[0] - m1[0])))
    x0, x1 = int(max(0, c[0] - 0.85 * mw)), int(min(w, c[0] + 0.85 * mw))
    y0, y1 = int(max(0, c[1] - 0.3 * mw)), int(min(h, c[1] + 0.8 * mw))
    out = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.uint8)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return out, (x0, y0), False

    # Default: YuNet's corners tend to sit high, so centre a little lower.
    ecx, ecy = c[0] - x0, c[1] - y0 + 0.12 * mw
    half_w, half_h = 0.6 * mw, 0.28 * mw
    measured = False

    # Skin reference: skin-tone pixels in the face box, outside the window.
    fx0, fy0 = int(max(0, face_row[0])), int(max(0, face_row[1]))
    fx1, fy1 = int(min(w, face_row[0] + face_row[2])), int(min(h, face_row[1] + face_row[3]))
    box = bgr[fy0:fy1, fx0:fx1]
    sel = skin_tone_mask(box) > 0
    sel[max(0, y0 - fy0):max(0, y1 - fy0), max(0, x0 - fx0):max(0, x1 - fx0)] = False
    lab_box = cv2.cvtColor(box, cv2.COLOR_BGR2LAB).reshape(-1, 3)[sel.ravel()].astype(np.float32)
    if len(lab_box) < 50:
        lab_box = cv2.cvtColor(box, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    skin_L, skin_a = np.median(lab_box[:, 0]), np.median(lab_box[:, 1])
    mad_a = float(np.median(np.abs(lab_box[:, 1] - skin_a))) or 1.0

    lab = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    lips = A > skin_a + max(6.0, 3.0 * mad_a)
    teeth = (L > skin_L + 20) & (np.abs(A - 128) + np.abs(B - 128) < 28)
    ind = cv2.morphologyEx((lips | teeth).astype(np.uint8), cv2.MORPH_OPEN,
                           _odd_kernel(mw * 0.03)).astype(np.float32)

    # Rows: share of the centre columns (under the nose) that look like lip.
    cx = int(round(c[0] - x0))
    cl, cr = max(0, cx - int(0.3 * mw)), min(ind.shape[1], cx + int(0.3 * mw))
    rows = ind[:, cl:cr].mean(axis=1) if cr > cl else np.zeros(ind.shape[0])
    k = max(1, int(mw * 0.03))
    rows = np.convolve(rows, np.ones(k) / k, mode="same")
    # Lip-coloured bands near where the mouth is expected. Start from the
    # strongest; an open smile shows as several bands right next to each
    # other (upper lip, teeth, lower lip, split by the dark mouth), so
    # neighbours within a small gap are added. Reddish skin further away
    # (philtrum, chin) is not.
    band_runs = [r for r in _runs(rows > 0.3, max_gap=int(0.06 * mw))
                 if -0.4 * mw <= (r[0] + r[1]) / 2 - ecy <= 0.45 * mw]
    if band_runs:
        top, bot = max(band_runs, key=lambda r: rows[r[0]:r[1] + 1].sum())
        grew = True
        while grew:
            grew = False
            for r in band_runs:
                gap = max(r[0] - bot, top - r[1])
                if 0 < gap <= 0.15 * mw and rows[r[0]:r[1] + 1].max() >= 0.45 \
                        and max(bot, r[1]) - min(top, r[0]) <= 0.65 * mw:
                    top, bot, grew = min(top, r[0]), max(bot, r[1]), True
        if 0.1 * mw <= bot - top <= 0.9 * mw:
            ecy = (top + bot) / 2
            # Small margin over the lip edge; never thinner than a closed
            # mouth, in case only one lip of an open mouth was measured.
            half_h = max((bot - top) / 2 + 0.06 * mw, 0.2 * mw)
            # Columns within that band: how wide the lips are.
            cols = ind[top:bot + 1].mean(axis=0) > 0.3
            col_runs = _runs(cols, max_gap=int(0.1 * mw))
            if col_runs:
                lft, rgt = min(col_runs, key=lambda r: 0 if r[0] <= cx <= r[1]
                               else min(abs(r[0] - cx), abs(r[1] - cx)))
                if rgt - lft >= 0.4 * mw:
                    ecx = float(np.clip((lft + rgt) / 2, cx - 0.15 * mw, cx + 0.15 * mw))
                    half_w = float(np.clip((rgt - lft) / 2 + 0.06 * mw, 0.45 * mw, 0.8 * mw))
            measured = True

    cv2.ellipse(out, (int(ecx), int(ecy)), (int(half_w), int(half_h)), angle, 0, 360, 255, -1)
    return out, (x0, y0), measured


def _window(bgr, x0, y0, x1, y1):
    h, w = bgr.shape[:2]
    x0, y0 = int(max(0, x0)), int(max(0, y0))
    x1, y1 = int(min(w, x1)), int(min(h, y1))
    return x0, y0, x1, y1


def _fill_holes(blob: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = blob.copy()
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out


def eye_mask(bgr: np.ndarray, ex: float, ey: float, eye_d: float,
             angle: float) -> tuple[np.ndarray, tuple, bool]:
    """The eye itself (lashes, iris, eye-white, eyeliner) around one eye point.

    YuNet's eye point is coarse (often on the lower edge of the iris), and a
    circle around it also blanked the under-eye skin, the area a retoucher
    softens first. Here the eye is the blob of clearly-darker-than-skin or
    eye-white pixels that contains the eye point, so the skin right below the
    lower lashes is left to the smoothing. Returns (mask uint8 0/255 for a
    window, (x0, y0), measured); an ellipse is used if the eye can't be found.
    """
    x0, y0, x1, y1 = _window(bgr, ex - 0.55 * eye_d, ey - 0.38 * eye_d,
                             ex + 0.55 * eye_d, ey + 0.3 * eye_d)
    out = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.uint8)
    sx, sy = int(ex - x0), int(ey - y0)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return out, (x0, y0), False
    lab = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    skin_L = float(np.median(L))      # skin is most of the window
    # Lashes, iris, pupil, liner: the darkest ~12% of the window, and clearly
    # darker than skin. (Eye-socket shading and dark circles are darker than
    # skin too, but not this dark.)
    dark = L < min(skin_L - 35, float(np.percentile(L, 12)))
    white = (L > skin_L + 8) & (np.abs(A - 128) + np.abs(B - 128) < 20)
    cand = (dark | white).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, _odd_kernel(eye_d * 0.06))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _odd_kernel(eye_d * 0.02))
    # Keep it to the eye opening: YuNet's point is near the lower lash line,
    # the upper lash line about 0.15 eye-distances above it.
    bound = np.zeros_like(cand)
    cv2.ellipse(bound, (sx, int(sy - 0.06 * eye_d)), (int(0.5 * eye_d), int(0.19 * eye_d)),
                angle, 0, 360, 1, -1)
    cand &= bound

    n, labels = cv2.connectedComponents(cand)
    r = max(2, int(0.1 * eye_d))
    near = labels[max(0, sy - r):sy + r, max(0, sx - r):sx + r]
    near = near[near > 0]
    if n > 1 and near.size:
        blob = (labels == np.bincount(near).argmax()).astype(np.uint8)
        ys, xs = np.nonzero(blob)
        if 0.3 * eye_d <= np.ptp(xs) <= 1.0 * eye_d and np.ptp(ys) >= 0.07 * eye_d:
            blob = _fill_holes(blob)
            return cv2.dilate(blob, _odd_kernel(eye_d * 0.04)) * 255, (x0, y0), True

    cv2.ellipse(out, (sx, int(sy - 0.03 * eye_d)), (int(0.32 * eye_d), int(0.15 * eye_d)),
                angle, 0, 360, 255, -1)
    return out, (x0, y0), False


def brow_mask(bgr: np.ndarray, ex: float, ey: float, eye_d: float,
              angle: float) -> tuple[np.ndarray, tuple, bool]:
    """The eyebrow hair above one eye, found as the main dark band there.

    Smoothing must not blur brow hair, but a fixed ellipse also blanked the
    forehead skin above the brow. Returns (mask uint8 0/255 for a window,
    (x0, y0), measured); an ellipse is used if no brow is found (very light
    or shaved brows, a fringe).
    """
    x0, y0, x1, y1 = _window(bgr, ex - 0.6 * eye_d, ey - 0.75 * eye_d,
                             ex + 0.6 * eye_d, ey - 0.12 * eye_d)
    out = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.uint8)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return out, (x0, y0), False
    L = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    dark = (L < float(np.median(L)) - 18).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, _odd_kernel(eye_d * 0.02))
    # The brow is the strongest dark band in the columns right above the eye
    # (hair at the side of the window, thin forehead lines and the eyelid
    # crease don't make a strong band there).
    cx = int(ex - x0)
    cl, cr = max(0, cx - int(0.3 * eye_d)), min(dark.shape[1], cx + int(0.3 * eye_d))
    rows = dark[:, cl:cr].mean(axis=1) if cr > cl else np.zeros(dark.shape[0])
    bands = [r for r in _runs(rows > 0.25, max_gap=int(0.03 * eye_d))
             if 0.04 * eye_d <= r[1] - r[0] <= 0.3 * eye_d]
    if bands:
        top, bot = max(bands, key=lambda r: rows[r[0]:r[1] + 1].sum())
        pad = int(0.04 * eye_d)
        top, bot = max(0, top - pad), min(dark.shape[0] - 1, bot + pad)
        band = cv2.morphologyEx(dark[top:bot + 1], cv2.MORPH_CLOSE, _odd_kernel(eye_d * 0.05))
        # Horizontal extent: the run of brow columns that passes over the eye.
        col_runs = _runs(band.mean(axis=0) > 0.15, max_gap=int(0.05 * eye_d))
        over = [r for r in col_runs if r[0] <= cx + 0.2 * eye_d and r[1] >= cx - 0.2 * eye_d]
        if over:
            lft, rgt = min(r[0] for r in over), max(r[1] for r in over)
            if rgt - lft >= 0.3 * eye_d:
                out[top:bot + 1, lft:rgt + 1] = band[:, lft:rgt + 1]
                return cv2.dilate(out, _odd_kernel(eye_d * 0.03)) * 255, (x0, y0), True

    cv2.ellipse(out, (int(ex - x0), int(ey - y0 - 0.3 * eye_d)),
                (int(0.30 * eye_d), int(0.13 * eye_d)), angle, 0, 360, 255, -1)
    return out, (x0, y0), False


def _paint(area: np.ndarray, window_mask: np.ndarray, origin: tuple) -> None:
    """Set `area` to 255 where a window mask (placed at origin) is set."""
    x0, y0 = origin
    h, w = window_mask.shape
    region = area[y0:y0 + h, x0:x0 + w]
    region[window_mask[:region.shape[0], :region.shape[1]] > 0] = 255


def build_skin_mask(bgr: np.ndarray, feather: float = 1.0) -> np.ndarray:
    """Return a float32 mask in [0,1]: 1 where skin should be smoothed.

    Uses YuNet face geometry when available; otherwise falls back to skin tone.
    """
    h, w = bgr.shape[:2]
    faces = _detect_faces(bgr)
    face_area = np.zeros((h, w), dtype=np.uint8)
    protect = np.zeros((h, w), dtype=np.uint8)   # eyes, brows, lips: never smoothed
    found = False

    if faces is not None and len(faces) > 0:
        for f in faces:
            x, y, fw, fh = f[:4]
            rex, rey = f[4], f[5]      # right eye
            lex, ley = f[6], f[7]      # left eye
            eye_d = float(np.hypot(lex - rex, ley - rey)) or (fw * 0.4)

            # Face oval: ellipse over the box, extended a bit down to the chin
            # and in slightly at the sides.
            cx, cy = x + fw / 2.0, y + fh * 0.55
            ax, ay = fw * 0.52, fh * 0.72
            cv2.ellipse(face_area, (int(cx), int(cy)), (int(ax), int(ay)),
                        0, 0, 360, 255, -1)
            found = True

            # Protect the eyes, brows and lips themselves (found from how
            # they differ from skin), so the skin around them - under the
            # eyes, above the brows, between nose and lip - still gets
            # smoothed.
            angle = float(np.degrees(np.arctan2(ley - rey, lex - rex)))
            for (ex, ey) in ((rex, rey), (lex, ley)):
                m, origin, _ = eye_mask(bgr, ex, ey, eye_d, angle)
                _paint(protect, m, origin)
                m, origin, _ = brow_mask(bgr, ex, ey, eye_d, angle)
                _paint(protect, m, origin)
            m, origin, _ = lip_mask(bgr, f)
            _paint(protect, m, origin)

    if not found:
        face_area = skin_tone_mask(bgr)
    else:
        # Drop non-skin pixels (hair, glasses) that fall inside the oval.
        tone = cv2.dilate(skin_tone_mask(bgr), np.ones((9, 9), np.uint8))
        face_area = cv2.bitwise_and(face_area, tone)

    # Feather the face outline. Blur radius scales with image size.
    diag = (h * h + w * w) ** 0.5
    k = max(3, int(diag * 0.01 * max(feather, 0.01))) | 1
    soft = cv2.GaussianBlur(face_area.astype(np.float32) / 255.0, (k, k), 0)
    if found:
        # Eyes, brows and lips get a much narrower soft edge, placed just
        # outside them, so lashes and lip edges stay fully protected.
        ks = max(3, k // 4) | 1
        guard = cv2.dilate(protect, _odd_kernel(ks))
        soft *= 1.0 - cv2.GaussianBlur(guard.astype(np.float32) / 255.0, (ks, ks), 0)
    return np.clip(soft, 0.0, 1.0)


WORK_PIXELS = 1.5e6   # smoothing works on at most this many pixels


def _smooth_layer(bgr: np.ndarray, strength: float, diag: float | None = None,
                  scale: float = 1.0) -> np.ndarray:
    """Edge-preserving smoothing whose intensity scales with strength (0..1).

    diag  : diagonal of the full photo (sets the smoothing radius); defaults
            to this image's own.
    scale : how much `bgr` was shrunk from the full photo; the radius is
            shrunk with it, so the same area of skin is smoothed.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if diag is None:
        diag = (bgr.shape[0] ** 2 + bgr.shape[1] ** 2) ** 0.5
    d = max(5, int(diag * 0.006 * scale))
    sigma_color = 20 + strength * 90
    sigma_space = max(1.0, (15 + strength * 45) * scale)
    iterations = 1 + int(round(strength * 2))

    # GPU path: run the bilateral passes on a UMat. Any OpenCL failure falls
    # through to the CPU path below.
    if _USE_OPENCL:
        try:
            u = cv2.UMat(bgr)
            for _ in range(iterations):
                u = cv2.bilateralFilter(u, d, sigma_color, sigma_space)
            return u.get()
        except cv2.error:
            pass

    out = bgr
    for _ in range(iterations):
        out = cv2.bilateralFilter(out, d, sigma_color, sigma_space)
    return out


def smooth_face(
    bgr: np.ndarray,
    strength: float = 0.6,
    texture: float = 0.35,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Smooth facial skin.

    strength : 0..1  how much to smooth (0 = original, 1 = maximum).
    texture  : 0..1  how much original pore/detail to keep (higher = less plastic).
    mask     : optional precomputed float mask from build_skin_mask (for speed).
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return bgr.copy()

    if mask is None:
        mask = build_skin_mask(bgr)

    # Only the area the mask touches needs work (plus a border so the filter
    # sees real neighbours at the edge).
    h, w = bgr.shape[:2]
    diag = (h * h + w * w) ** 0.5
    ys, xs = np.nonzero(mask > 1e-3)
    if len(ys) == 0:
        return bgr.copy()
    pad = int(diag * 0.006) + 8
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    roi = bgr[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]

    # Smooth a reduced copy of large regions (a 24 MP photo would otherwise
    # take tens of seconds), then scale the smoothed base back up. The fine
    # detail the reduced copy can't hold is mostly what smoothing removes
    # anyway; what is kept comes back below at full resolution.
    scale = min(1.0, (WORK_PIXELS / (rh * rw)) ** 0.5)
    if scale < 1.0:
        small = cv2.resize(roi, (max(1, round(rw * scale)), max(1, round(rh * scale))),
                           interpolation=cv2.INTER_AREA)
        base = _smooth_layer(small, strength, diag, scale)
        base = cv2.resize(base, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        base = _smooth_layer(roi, strength, diag).astype(np.float32)

    # High-frequency texture from the original, added back at `texture` amount.
    blur = cv2.GaussianBlur(roi, (0, 0), 3).astype(np.float32)
    high = roi.astype(np.float32) - blur
    smoothed = base + texture * high

    alpha = (mask[y0:y1, x0:x1] * strength)[:, :, None]
    blended = roi.astype(np.float32) * (1.0 - alpha) + smoothed * alpha
    result = bgr.copy()
    result[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Face-aware skin smoothing.")
    ap.add_argument("input", help="input image path")
    ap.add_argument("output", nargs="?", help="output path (default: *_smooth.jpg)")
    ap.add_argument("-s", "--strength", type=float, default=0.6,
                    help="smoothing 0..1 (default 0.6)")
    ap.add_argument("-t", "--texture", type=float, default=0.35,
                    help="texture retention 0..1 (default 0.35)")
    ap.add_argument("--gpu", action="store_true",
                    help="use the GPU (OpenCL) path if available")
    ap.add_argument("-b", "--blemish", type=float, default=0.0,
                    help="remove blemishes first, 0..1 (needs models/blemish_ffhqr.onnx)")
    args = ap.parse_args()

    if args.gpu:
        eff = set_opencl(True)
        print(f"GPU (OpenCL): {'ON' if eff else 'requested but unavailable, using CPU'}")

    import photo_io
    img, info = photo_io.read(args.input)
    if img is None:
        print(f"Could not read image: {args.input}")
        return 1

    if args.blemish > 0:
        import blemish
        if blemish.have_model():
            img, n = blemish.remove_blemishes(img, amount=args.blemish)
            print(f"Blemishes healed on {n} face(s)")
        else:
            print("Blemish model not found (models/blemish_ffhqr.onnx); skipping")

    out = smooth_face(img, strength=args.strength, texture=args.texture)

    out_path = args.output
    if not out_path:
        stem, ext = os.path.splitext(args.input)
        out_path = f"{stem}_smooth{ext or '.jpg'}"
    # Keep the original's colour profile, EXIF, DPI and JPEG quality.
    if photo_io.write(out_path, out, info):
        print(f"Saved: {out_path}  (face model: {'yes' if have_face_model() else 'no, skin-tone fallback'})")
        return 0
    print("Failed to encode output.")
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
