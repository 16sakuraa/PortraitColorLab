"""
Face-aware skin smoothing.

Pipeline:
  1. Detect the faces and build a soft skin mask: a segmentation model finds face
     and body skin (skin_segment.py), minus the eyes, eyebrows and lips, which are
     found from how they differ from the skin. The mask can be edited by hand with
     a brush (mask_edit.py).
  2. Split the skin into a tone layer and a detail layer, using sizes relative
     to the face, so results don't depend on how the photo was cropped.
  3. Smooth the tone layer with guided filters (brightness over a small radius,
     colour over a wider one), reading only from skin so a coloured backdrop can't
     tint it, and keeping the face's large-scale shading.
  4. Treat the detail layer by how strong it is at each pixel: reduce pores
     gently, fine lines more, and leave real edges, lashes and stray hairs alone.
     Optionally add synthetic pores.
  5. Composite through the soft mask.

Each model file is looked for next to this script (models/). Without the
segmentation model, the mask falls back to a face oval from OpenCV's YuNet face
detector; without that too, to a skin-tone mask, so it still works. The smoothing
is standard image processing (OpenCV) built from published methods.
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


DETECT_MAX = 1280  # longest side YuNet sees; large faces are missed above this
SAME_FACE_OVERLAP = 0.3  # share of a smaller face's box inside a larger one's that makes them one face


def _plausible(face_row) -> bool:
    """Does the landmark layout look like a face?

    Ears, hair, dark corners and blurry background get detected as faces now
    and then. The score can't separate them (real large faces often score only
    0.6-0.75), but the layout can. Measured on 294 real FFHQ faces (1st-99th
    percentile):
      eye spacing / box width    0.41-0.56
      eyes-to-mouth / eye spacing 0.75-1.38
      eye line vs eyes-to-mouth  ~perpendicular (|cos| <= 0.04)
      nose position eyes->mouth  0.52-0.79
    The limits below leave room for turned and tilted heads (common in group
    photos).
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


def _separate_faces(faces):
    """Only one row per real face, in the same order.

    YuNet finds some large faces twice, at different scales, and its boxes can
    overlap too little for its own duplicate check. The extra row's landmarks are
    off, and would protect a patch of cheek as if it were an eye. A row whose
    landmarks sit inside a larger face's box, or whose box is largely inside it,
    is that face again. Largest first, so the better fit wins; a row that doesn't
    look like a face goes too, unless it's all there is. Neighbours in a group
    photo overlap far less, so they stay.
    """
    if faces is None or len(faces) < 2:
        return faces

    def same_face(f, g) -> bool:
        cx, cy = float(np.mean(f[4:14:2])), float(np.mean(f[5:14:2]))
        if g[0] <= cx <= g[0] + g[2] and g[1] <= cy <= g[1] + g[3]:
            return True
        across = min(f[0] + f[2], g[0] + g[2]) - max(f[0], g[0])
        down = min(f[1] + f[3], g[1] + g[3]) - max(f[1], g[1])
        return across > 0 and down > 0 and across * down >= SAME_FACE_OVERLAP * f[2] * f[3]

    order = sorted(range(len(faces)), key=lambda i: -float(faces[i][2] * faces[i][3]))
    kept: list[int] = []
    for i in order:
        if _plausible(faces[i]) and not any(same_face(faces[i], faces[k]) for k in kept):
            kept.append(i)
    return faces[sorted(kept or order[:1])]


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
    return _separate_faces(faces)


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
                           _odd_kernel(mw * 0.03))

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
    # (philtrum, chin) is not. The small margin settles exact ties, which rounding
    # would otherwise tip either way, the same way in every build.
    tie = 1e-9
    band_runs = [r for r in _runs(rows > 0.3 + tie, max_gap=int(0.06 * mw))
                 if -0.4 * mw <= (r[0] + r[1]) / 2 - ecy <= 0.45 * mw]
    if band_runs:
        top, bot = max(band_runs, key=lambda r: rows[r[0]:r[1] + 1].sum())
        grew = True
        while grew:
            grew = False
            for r in band_runs:
                gap = max(r[0] - bot, top - r[1])
                if 0 < gap <= 0.15 * mw and rows[r[0]:r[1] + 1].max() >= 0.45 - tie \
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


def build_skin_mask(bgr: np.ndarray, feather: float = 1.0, faces=None,
                    face_only: bool = False) -> np.ndarray:
    """Return a float32 mask in [0,1]: 1 where skin should be smoothed.

    Uses the segmentation model (skin_segment.py) for skin, with body skin left
    out if face_only. Faces it can't speak for, or every face if it's missing,
    use YuNet's face geometry; without a face the mask comes from skin tone.
    feather softens that older outline; the model's outline is already soft.
    """
    h, w = bgr.shape[:2]
    if faces is None:
        faces = _detect_faces(bgr)
    face_area = np.zeros((h, w), dtype=np.uint8)
    protect = np.zeros((h, w), dtype=np.uint8)   # eyes, brows, lips: never smoothed
    found = False

    model_area, covered = None, ()
    if faces is not None and len(faces) > 0:
        import skin_segment
        seg = skin_segment.segment(bgr, faces)
        if seg is not None:
            model_area = skin_segment.skin_area(seg, faces, face_only)
            covered = seg.covered if model_area is not None else ()

        for i, f in enumerate(faces):
            x, y, fw, fh = f[:4]
            rex, rey = f[4], f[5]      # right eye
            lex, ley = f[6], f[7]      # left eye
            eye_d = float(np.hypot(lex - rex, ley - rey)) or (fw * 0.4)

            # Face oval: ellipse over the box, extended a bit down to the chin
            # and in slightly at the sides.
            if i not in covered:
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

    # Feather the face outline. Blur radius scales with image size.
    diag = (h * h + w * w) ** 0.5
    k = max(3, int(diag * 0.01 * max(feather, 0.01))) | 1
    if found and len(covered) == len(faces):
        soft = np.zeros((h, w), np.float32)
    else:
        if not found:
            face_area = skin_tone_mask(bgr)
        else:
            # Drop non-skin pixels (hair, glasses) that fall inside the oval.
            tone = cv2.dilate(skin_tone_mask(bgr), np.ones((9, 9), np.uint8))
            face_area = cv2.bitwise_and(face_area, tone)
        soft = cv2.GaussianBlur(face_area.astype(np.float32) / 255.0, (k, k), 0)
    if model_area is not None:
        mx, my, area = model_area
        here = soft[my:my + area.shape[0], mx:mx + area.shape[1]]
        np.maximum(here, area, out=here)
    if found:
        # Eyes, brows and lips get a much narrower soft edge, placed just
        # outside them, so lashes and lip edges stay fully protected.
        ks = max(3, k // 4) | 1
        guard = cv2.dilate(protect, _odd_kernel(ks))
        soft *= 1.0 - cv2.GaussianBlur(guard.astype(np.float32) / 255.0, (ks, ks), 0)
    return np.clip(soft, 0.0, 1.0)


WORK_PIXELS = 1.5e6   # tone smoothing works on at most this many pixels
STRIP_ROWS = 256      # full-resolution work is done this many rows at a time

# Settings, derived from physical sizes on a roughly 15 cm wide face and from
# skin statistics measured on test photos.
#   Fractions of face width: split, luma_radius, chroma_radius.
#   luma_edge / chroma_edge: local spread (0..1) at about strength 0.5 that the
#       guided filter treats as a real edge rather than something to smooth.
#   Multiples of the photo's own median detail level: line_*, keep_*.
TUNE = {
    "split": 0.006,          # detail/tone split, about 1 mm: visible skin grain stays detail
    "split_edge": 0.08,      # contrast (0..1) above which something stays out of the detail layer
    "luma_radius": 0.015,    # about 2 mm: small marks and uneven tone
    "luma_edge": 0.045,
    "chroma_radius": 0.05,   # about 7.5 mm: redness and blotches
    "chroma_edge": 0.05,
    "contour": 1.0,          # how much large-scale tone change to undo (keeps shading)
    "contour_ratio": 1.5,    # "large-scale" starts at this many smoothing radii
    "line_lo": 2.5,          # brightness detail above pores: fine lines
    "line_hi": 5.0,
    "line_cut": 0.8,         # share of the kept texture removed from fine lines
    "keep_lo": 9.0,          # brightness detail this strong is a real edge: left alone
    "keep_hi": 16.0,
    "keep_colour_lo": 3.5,   # colour detail this strong (lashes, stray hair): left alone
    "keep_colour_hi": 7.0,
    "pore_size": 0.0012,     # noise blur: a fraction of face width, puts pits about 1.2 mm apart
    "pore_band": 2.0,        # larger structure is removed from this many times that size
    "pore_lean": 0.5,        # dark side this much deeper than average, light side this much less
    "pore_depth": 1.0,       # at full slider, the median pit matches the photo's own median detail
    "pore_min_level": 0.006, # softer skin than this (0..1) is treated as this, so the slider still works there
}
MIN_PORE_SIGMA = 0.6      # px; pores on faces too small to reach this get fainter instead
MIN_PORE_GAIN = 0.15      # faces whose pores would be fainter than this are skipped: under a third of a level
PORE_SEED = 0x9E3779B9
PORE_PATCH = 256          # the noise's spread is measured once, on a patch this wide
SOFT_BLUR_MIN = 1.4       # _soft_blur can't blur less than this sigma
NO_FACE_WIDTH = 0.25      # assumed face width, as a fraction of the image diagonal
GROUP_RATIO = 1.5         # faces within this size ratio share one smoothing pass
MIN_SPLIT_SIGMA = 0.6     # px; below this a blur no longer separates anything
SPLIT_RADIUS_PER_SIGMA = 1.2  # guided filter radius that blurs about as much as a Gaussian
MIN_DETAIL_LEVEL = 0.002  # 0..1; half an 8-bit step, so clean skin can't divide by ~0
LEVEL_BANDS = 8           # rows of samples used to measure the detail level
LEVEL_BAND_ROWS = 32
_LUMA = np.array([0.114, 0.587, 0.299], np.float32)   # B, G, R weights of brightness


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Guided filter (He, Sun, Tang, "Guided Image Filtering", ECCV 2010).

    An edge-preserving blur of src that follows the edges of guide. Both are
    float32 with the same channels (each channel is filtered on its own). Local
    variation well below sqrt(eps) is flattened; edges well above it are kept.
    Built only from box filters, so its cost doesn't depend on the radius, and
    the browser version can match it.
    """
    size = (2 * radius + 1, 2 * radius + 1)

    def box(a):
        return cv2.boxFilter(a, cv2.CV_32F, size, normalize=True, borderType=cv2.BORDER_DEFAULT)

    mean_g, mean_s = box(guide), box(src)
    var_g = box(guide * guide) - mean_g * mean_g
    cov = box(guide * src) - mean_g * mean_s
    a = cov / (var_g + eps)
    b = mean_s - a * mean_g
    return box(a) * guide + box(b)


def _soft_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Three box blurs: practically a Gaussian of this sigma, at a cost that
    doesn't grow with it (the wide blurs here would otherwise dominate the run time)."""
    radius = max(1, round((np.sqrt(4.0 * sigma * sigma + 1.0) - 1.0) / 2.0))
    size = (2 * radius + 1, 2 * radius + 1)
    for _ in range(3):
        a = cv2.boxFilter(a, cv2.CV_32F, size, normalize=True, borderType=cv2.BORDER_DEFAULT)
    return a


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _face_widths(faces, diag: float) -> np.ndarray:
    if faces is None or len(faces) == 0:
        return np.array([NO_FACE_WIDTH * diag], np.float64)
    return np.asarray(faces, np.float64)[:, 2]


def _scale_regions(faces, widths: np.ndarray, scale: float, x0: int, y0: int,
                   shape: tuple[int, int]) -> list[tuple[float, np.ndarray | None]]:
    """[(face width in full-resolution pixels, weight map or None)].

    Faces of similar size share one smoothing pass. Very different sizes get
    separate passes, blended by a soft map of which face each pixel is nearest
    to. Most photos have one group, which needs no map.
    """
    order = np.argsort(widths, kind="stable")
    groups: list[list[int]] = []
    for i in order:
        if groups and widths[i] <= GROUP_RATIO * widths[groups[-1][0]]:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    mean_width = [float(np.exp(np.mean(np.log(widths[g])))) for g in groups]
    if len(groups) == 1:
        return [(mean_width[0], None)]

    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    best = np.full(shape, np.inf, np.float32)
    nearest = np.zeros(shape, np.int32)
    f = np.asarray(faces, np.float64)
    for g, members in enumerate(groups):
        for i in members:
            cx = (f[i, 0] + f[i, 2] / 2 - x0) * scale
            cy = (f[i, 1] + f[i, 3] / 2 - y0) * scale
            d = np.hypot(xx - cx, yy - cy) / (f[i, 2] * scale)
            closer = d < best
            best[closer] = d[closer]
            nearest[closer] = g
    # Soft edges, but narrower than the smallest face so it fully gets its own pass.
    sigma = max(2.0, 0.25 * float(widths.min()) * scale)
    maps = [cv2.GaussianBlur((nearest == g).astype(np.float32), (0, 0), sigma)
            for g in range(len(groups))]
    total = np.sum(maps, axis=0) + 1e-6
    return [(mean_width[g], maps[g] / total) for g in range(len(groups))]


def _tone_change(roi: np.ndarray, mask: np.ndarray, regions_of, sigma: float, scale: float,
                 strength: float) -> np.ndarray:
    """What to add to each pixel's tone layer, on a reduced copy of the region
    (scaled back up by the caller): float32 BGR, working rows by full width.

    Brightness is smoothed over a small radius and colour over a wider one,
    guided by brightness so it can't cross real edges. Changing colour alone
    doesn't flatten a face, but changing brightness can, so the large-scale
    part of each change is removed again: small marks go, shading stays.
    """
    t = TUNE
    rh, rw = roi.shape[:2]
    if scale < 1.0:
        size = (max(1, round(rw * scale)), max(1, round(rh * scale)))
        work = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
        mask_w = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    else:
        work, mask_w = roi, mask
    work = work.astype(np.float32) / 255.0
    if sigma * scale >= 0.5:
        # Only from skin, or the backdrop's colour would mix into the skin at its edge.
        s = sigma * scale
        work = _soft_blur(work * mask_w[:, :, None], s) / (_soft_blur(mask_w, s)[:, :, None] + 1e-4)
    Y, Cr, Cb = cv2.split(cv2.cvtColor(work, cv2.COLOR_BGR2YCrCb))

    d_y = np.zeros_like(Y)
    d_cr = np.zeros_like(Y)
    d_cb = np.zeros_like(Y)
    eps_y = (t["luma_edge"] * (0.5 + strength)) ** 2
    eps_c = (t["chroma_edge"] * (0.5 + strength)) ** 2

    def mask_blur(sigma_m):
        if sigma_m not in mask_blurs:
            mask_blurs[sigma_m] = _soft_blur(mask_w, sigma_m) + 1e-4
        return mask_blurs[sigma_m]

    def change(plane, radius, eps):
        # Normalized convolution (Knutsson & Westin, 1993) reads only from skin,
        # so hair and background can't leak in. First for the filter's input: skin
        # and a coloured backdrop can be equally bright, and then the brightness
        # guide can't keep the backdrop's colour off the skin beside it. Everything
        # outside the skin takes the colour of the skin nearby.
        here = mask_w * plane + (1.0 - mask_w) * (_soft_blur(plane * mask_w, radius) / mask_blur(radius))
        d = guided_filter(Y, here, radius, eps) - here
        # Then for the large-scale part of the change.
        sigma_c = t["contour_ratio"] * radius
        return d - t["contour"] * _soft_blur(d * mask_w, sigma_c) / mask_blur(sigma_c)

    mask_blurs: dict[float, np.ndarray] = {}
    for width, weight in regions_of(Y.shape):
        w = 1.0 if weight is None else weight
        r_y = max(1, round(width * scale * t["luma_radius"]))
        r_c = max(2, round(width * scale * t["chroma_radius"]))
        d_y += w * change(Y, r_y, eps_y)
        d_cr += w * change(Cr, r_c, eps_c)
        d_cb += w * change(Cb, r_c, eps_c)

    # The YCrCb change expressed in B, G, R (OpenCV's float conversion).
    delta = np.dstack([d_y + 1.773 * d_cb, d_y - 0.714 * d_cr - 0.344 * d_cb, d_y + 1.403 * d_cr])
    if delta.shape[1] != rw:
        delta = cv2.resize(delta, (rw, delta.shape[0]), interpolation=cv2.INTER_LINEAR)
    return delta


def _detail_strip(roi: np.ndarray, y0: int, y1: int, sigma: float):
    """Rows y0..y1 of the region as float32, and their detail layer.

    The split is edge-preserving: with a plain blur, part of every strong edge
    would land in the detail layer, and reducing detail would then soften it.
    Filters with enough extra rows that strip edges can't show.
    """
    radius = max(1, round(SPLIT_RADIUS_PER_SIGMA * sigma))
    margin = 2 * radius + 1
    a, b = max(0, y0 - margin), min(roi.shape[0], y1 + margin)
    part = roi[a:b].astype(np.float32) / 255.0
    high = part - guided_filter(part, part, radius, TUNE["split_edge"] ** 2)
    return part[y0 - a:y1 - a], high[y0 - a:y1 - a]


def _detail_level(roi: np.ndarray, mask: np.ndarray, sigma: float) -> float:
    """Median brightness detail on skin well inside the mask, in 0..1 units.

    Skin detail strength varies with exposure, sharpness and resolution, so
    the detail bands are multiples of this rather than fixed values. Measured
    on evenly spaced rows, which is enough for a stable median.
    """
    rows = np.flatnonzero((mask > 0.9).any(axis=1))
    if len(rows) == 0:
        return MIN_DETAIL_LEVEL
    samples = []
    for i in range(LEVEL_BANDS):
        y0 = int(rows[(2 * i + 1) * len(rows) // (2 * LEVEL_BANDS)])
        y1 = min(roi.shape[0], y0 + LEVEL_BAND_ROWS)
        _, high = _detail_strip(roi, y0, y1, sigma)
        samples.append(np.abs(high @ _LUMA)[mask[y0:y1] > 0.9])
    samples = np.concatenate(samples)
    return max(MIN_DETAIL_LEVEL, float(np.median(samples))) if samples.size else MIN_DETAIL_LEVEL


def _position_noise(x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Noise in -1..1 from each pixel's position, so every run and both apps get
    the same values. Positions up to 65536 px are distinct and the hash (the
    finalizer of MurmurHash3) maps distinct inputs to distinct outputs, so the
    pattern never repeats within a photo.
    """
    xs = np.arange(x0, x0 + w, dtype=np.int64) & 0xFFFF
    ys = np.arange(y0, y0 + h, dtype=np.int64) & 0xFFFF
    v = ((ys[:, None] << 16) | xs[None, :]).astype(np.uint32) ^ PORE_SEED
    v ^= v >> 16
    v *= 0x85EBCA6B
    v ^= v >> 13
    v *= 0xC2B2AE35
    v ^= v >> 16
    return (v >> 8).astype(np.float32) / 8388608.0 - 1.0


def _pore_band(x0: int, y0: int, w: int, h: int, sigma: float) -> np.ndarray:
    """Noise with structure at one size only: finer grain is blurred away and
    anything larger is subtracted, so it can't read as blotches."""
    noise = _position_noise(x0, y0, w, h)
    fine = _soft_blur(noise, sigma) if sigma >= SOFT_BLUR_MIN else cv2.GaussianBlur(noise, (0, 0), sigma)
    return fine - _soft_blur(fine, TUNE["pore_band"] * sigma)


def _lean_dark(v: np.ndarray) -> np.ndarray:
    """Pores are pits: deepen the dark side and flatten the light side."""
    lean = TUNE["pore_lean"]
    return np.where(v < 0, v * (1.0 + lean), v * (1.0 - lean))


def _pore_groups(regions, rw: int, work_rows: int) -> list[tuple]:
    """(sigma, gain, spread, mean, weight) per group of similar face sizes.

    Faces too small to show pores get fainter ones, then none, so they don't
    pick up a grain that's coarse for their size. The fade is about how much
    real pores weaken when a photo is shrunk, so a preview shows roughly what the
    export will look like at that size. spread and mean are measured on a fixed
    patch, which keeps every strip on the same scale. weight is (work_rows, rw),
    or None for one group.
    """
    groups = []
    for width, weight in regions:
        size = width * TUNE["pore_size"]
        gain = min(1.0, size / MIN_PORE_SIGMA)
        if gain < MIN_PORE_GAIN:
            continue
        sigma = max(MIN_PORE_SIGMA, size)
        band = _pore_band(0, 0, PORE_PATCH, PORE_PATCH, sigma)
        spread = float(np.median(np.abs(band)))
        mean = float(np.mean(_lean_dark(band / spread), dtype=np.float64))
        if weight is not None:
            weight = cv2.resize(weight, (rw, work_rows), interpolation=cv2.INTER_LINEAR)
        groups.append((sigma, gain, spread, mean, weight))
    return groups


def _pore_strip(group: tuple, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """One group's pattern for a strip: median size 1, averaging zero, leaning
    dark. Made with enough extra pixels that strip edges can't show."""
    sigma, gain, spread, mean, _ = group
    reach = int(np.ceil((4.0 + 3.0 * TUNE["pore_band"]) * sigma)) + 4
    band = _pore_band(x0 - reach, y0 - reach, w + 2 * reach, h + 2 * reach, sigma)
    return gain * (_lean_dark(band[reach:reach + h, reach:reach + w] / spread) - mean)


def _near_edge(roi: np.ndarray, y0: int, y1: int, radius: int, lo: float, hi: float) -> np.ndarray:
    """0..1 for rows y0..y1: how strongly each pixel sits by a hard edge or a hair.

    The detail layer can't show this, because its edge-preserving split keeps
    strong edges out of it. A plain box blur doesn't, so what it changes does.
    """
    a, b = max(0, y0 - radius), min(roi.shape[0], y1 + radius)
    luma = (roi[a:b].astype(np.float32) / 255.0) @ _LUMA
    size = (2 * radius + 1, 2 * radius + 1)
    local = luma - cv2.boxFilter(luma, cv2.CV_32F, size, normalize=True, borderType=cv2.BORDER_DEFAULT)
    return _smoothstep(np.abs(local[y0 - a:y1 - a]), lo, hi)


def smooth_face(
    bgr: np.ndarray,
    strength: float = 0.6,
    texture: float = 0.6,
    mask: np.ndarray | None = None,
    faces=None,
    pores: float = 0.0,
) -> np.ndarray:
    """Smooth skin, keeping its shape and pores.

    strength : 0..1  how much to smooth (0 = original, 1 = maximum).
    texture  : 0..1  how much original pore/detail to keep (higher = less plastic).
    mask     : optional precomputed float mask from build_skin_mask (for speed).
    faces    : optional YuNet rows from _detect_faces, in this image's pixels.
    pores    : 0..1  how much synthetic pore texture to add (0 = none).

    The region is split into a tone layer and a detail layer, using sizes
    relative to the face. The tone layer gets _tone_change. The detail layer is
    treated by how strong it is at each pixel: pores are reduced according to
    texture, stronger detail such as fine lines is reduced more, and the
    strongest (real edges, lashes, stray hairs) is left alone.

    Synthetic pores are scaled to the detail measured on this photo and sized to
    the face. They change brightness only and follow the smoothing: through the
    mask, in proportion to strength, and not on anything left alone above.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return bgr.copy()
    texture = float(np.clip(texture, 0.0, 1.0))
    pores = float(np.clip(pores, 0.0, 1.0))

    if faces is None:
        faces = _detect_faces(bgr)
    if mask is None:
        mask = build_skin_mask(bgr)

    # Only the area the mask touches needs work, plus enough border that the
    # filters see real neighbours at its edge.
    t = TUNE
    h, w = bgr.shape[:2]
    diag = (h * h + w * w) ** 0.5
    ys, xs = np.nonzero(mask > 1e-3)
    if len(ys) == 0:
        return bgr.copy()
    widths = _face_widths(faces, diag)
    pad = int(np.ceil(float(widths.max()) * t["chroma_radius"])) + 8
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
    roi = bgr[y0:y1, x0:x1]
    mask_r = mask[y0:y1, x0:x1].astype(np.float32)
    rh, rw = roi.shape[:2]

    scale = min(1.0, (WORK_PIXELS / (rh * rw)) ** 0.5)
    sigma = max(MIN_SPLIT_SIGMA, float(np.median(widths)) * t["split"])

    def regions_of(shape):
        return _scale_regions(faces, widths, scale, x0, y0, shape)

    tone = _tone_change(roi, mask_r, regions_of, sigma, scale, strength)
    work_rows = tone.shape[0]
    level = _detail_level(roi, mask_r, sigma)
    pore_groups = []
    if pores > 0.0:
        work_cols = max(1, round(rw * scale)) if scale < 1.0 else rw
        pore_groups = _pore_groups(regions_of((work_rows, work_cols)), rw, work_rows)
    pore_amount = pores * t["pore_depth"] * max(level, t["pore_min_level"])
    edge_radius = max(1, round(SPLIT_RADIUS_PER_SIGMA * sigma))

    result = bgr.copy()
    for top in range(0, rh, STRIP_ROWS):
        bottom = min(rh, top + STRIP_ROWS)
        part, high = _detail_strip(roi, top, bottom, sigma)

        # Tone change for these rows: linear between working rows.
        pos = (np.arange(top, bottom) + 0.5) * work_rows / rh - 0.5
        below = np.floor(pos)
        frac = (pos - below).astype(np.float32)[:, None, None]
        i0 = np.clip(below.astype(np.int64), 0, work_rows - 1)
        i1 = np.minimum(i0 + 1, work_rows - 1)
        change = tone[i0] * (1.0 - frac) + tone[i1] * frac

        luma = high @ _LUMA
        amount = np.abs(luma)
        colour = np.abs(high - luma[:, :, None]).max(axis=2)
        keep = np.maximum(
            _smoothstep(amount, t["keep_lo"] * level, t["keep_hi"] * level),
            _smoothstep(colour, t["keep_colour_lo"] * level, t["keep_colour_hi"] * level))
        line = _smoothstep(amount, t["line_lo"] * level, t["line_hi"] * level)
        removed = 1.0 - texture * (1.0 - t["line_cut"] * line)

        # A real edge is left alone completely: the tone change was worked out
        # on a blurred copy, so applying it there would soften the edge.
        amount_here = (mask_r[top:bottom] * strength * (1.0 - keep))[:, :, None]
        detail = change - removed[:, :, None] * high
        if pore_groups:
            room = 1.0 - _near_edge(roi, top, bottom, edge_radius, t["keep_lo"] * level, t["keep_hi"] * level)
            for group in pore_groups:
                pattern = _pore_strip(group, x0, y0 + top, rw, bottom - top)
                weight = group[4]
                if weight is not None:
                    pattern = pattern * (weight[i0] * (1.0 - frac[:, :, 0]) + weight[i1] * frac[:, :, 0])
                detail = detail + (pore_amount * pattern * room)[:, :, None]
        out = part + amount_here * detail
        result[y0 + top:y0 + bottom, x0:x1] = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return result


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Face-aware skin smoothing.")
    ap.add_argument("input", help="input image path")
    ap.add_argument("output", nargs="?", help="output path (default: *_smooth.jpg)")
    ap.add_argument("-s", "--strength", type=float, default=0.6,
                    help="smoothing 0..1 (default 0.6)")
    ap.add_argument("-t", "--texture", type=float, default=0.6,
                    help="texture retention 0..1 (default 0.6)")
    ap.add_argument("-p", "--pores", type=float, default=0.0,
                    help="add synthetic pore texture 0..1 (default 0, none)")
    ap.add_argument("--face-only", action="store_true",
                    help="treat the face only, not the neck or other body skin")
    ap.add_argument("-b", "--blemish", type=float, default=0.0,
                    help="remove blemishes first, 0..1")
    ap.add_argument("--blemish-engine", choices=("classic", "network"), default=None,
                    help="how to find blemishes: classic image processing (the default, no model) "
                         "or the network (needs models/blemish_ffhqr.onnx, non-commercial licence)")
    args = ap.parse_args()

    import photo_io
    img, info = photo_io.read(args.input)
    if img is None:
        print(f"Could not read image: {args.input}")
        return 1

    faces = _detect_faces(img)
    mask = build_skin_mask(img, faces=faces, face_only=args.face_only)
    healed = img
    if args.blemish > 0:
        import blemish
        if args.blemish_engine:
            blemish.ENGINE = args.blemish_engine
        if blemish.have_model():
            corr, n = blemish.correction(img, faces, mask)
            healed = blemish.apply(img, corr, args.blemish)
            print(blemish.count_text(n))
        else:
            print("Blemish model not found (models/blemish_ffhqr.onnx); skipping")

    out = smooth_face(healed, strength=args.strength, texture=args.texture, mask=mask,
                      faces=faces, pores=args.pores)

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
