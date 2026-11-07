// Eye, brow and lip outlines from a face landmark network - a port of
// skin_smoother/face_landmarks.py.
//
// Google's MediaPipe Face Mesh V2 places 478 points on a face. It expects an upright
// 256x256 crop with some margin around the face, so each face takes two passes: the
// first from the face detector's box and eyes, the second from the first pass's own
// points, which gives a tighter, straighter crop.
//
// landmarks() runs the model once per photo. zones() is cheap, so Face only can switch
// without running the model again. A face without points uses the older method in
// skinmask.js (eyeMask, browMask, lipMask).
//
// Allocating a Mat can move the WebAssembly heap, so views like mat.data are read
// after the allocations they follow.

import { pyRound } from './skinmask.js';

const SIZE = 256, POINTS = 478, MAX_FACES = 30, MIN_FLAG = 0;

// Point numbers around each feature, from Google's Face Mesh map. "Right" is the
// person's right, as with the face detector's eyes.
const LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
const EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
const BROW_R = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
const BROW_L = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];

// Same settings as face_landmarks.TUNE; see the notes there.
export const TUNE = {
  crop_side: 1.5,
  eye_margin: 0.05,
  brow_margin: 0.03,
  lip_margin: 0.01,
};

// Runs the model on a square around (cx, cy), turned by angle (radians). Returns
// { pts: x0, y0, x1, y1, ... in photo pixels, flag }, or null if it misses the photo.
async function run(cv, ort, session, bgr, cx, cy, side, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const scale = side / SIZE;                 // photo pixels per model pixel
  const shrink = Math.min(1, 1 / scale);     // shrink a large crop first, so the turn doesn't alias
  const reach = Math.ceil(side * 0.75) + 2;
  const x0 = Math.max(0, Math.trunc(cx) - reach), y0 = Math.max(0, Math.trunc(cy) - reach);
  const x1 = Math.min(bgr.cols, Math.trunc(cx) + reach), y1 = Math.min(bgr.rows, Math.trunc(cy) + reach);
  if (x1 <= x0 || y1 <= y0) return null;
  let part = bgr.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
  if (shrink < 1) {
    const small = new cv.Mat();
    cv.resize(part, small, new cv.Size(Math.max(1, pyRound((x1 - x0) * shrink)), Math.max(1, pyRound((y1 - y0) * shrink))), 0, 0, cv.INTER_AREA);
    part.delete();
    part = small;
  }
  const sx = part.cols / (x1 - x0), sy = part.rows / (y1 - y0);
  // Model pixel -> photo pixel, and on into the (shrunk) part.
  const a00 = scale * c, a01 = -scale * s, a02 = cx - scale * (c * SIZE / 2 - s * SIZE / 2);
  const a10 = scale * s, a11 = scale * c, a12 = cy - scale * (s * SIZE / 2 + c * SIZE / 2);
  const toPart = cv.matFromArray(2, 3, cv.CV_64F, [a00 * sx, a01 * sx, (a02 - x0) * sx, a10 * sy, a11 * sy, (a12 - y0) * sy]);
  const crop = new cv.Mat();
  cv.warpAffine(part, crop, toPart, new cv.Size(SIZE, SIZE), cv.INTER_LINEAR | cv.WARP_INVERSE_MAP, cv.BORDER_CONSTANT, new cv.Scalar());
  part.delete(); toPart.delete();
  const plane = SIZE * SIZE, input = new Float32Array(3 * plane), d = crop.data;
  for (let i = 0, j = 0; i < plane; i++, j += 3) {     // RGB, 0..1
    input[i] = d[j + 2] / 255;
    input[plane + i] = d[j + 1] / 255;
    input[2 * plane + i] = d[j] / 255;
  }
  crop.delete();
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) });
  const tensors = session.outputNames.map((n) => out[n]);
  const raw = tensors.find((t) => t.data.length === POINTS * 3).data;
  const flag = tensors.find((t) => t.data.length === 1).data[0];
  const pts = new Float64Array(2 * POINTS);
  for (let i = 0; i < POINTS; i++) {
    pts[2 * i] = a00 * raw[3 * i] + a01 * raw[3 * i + 1] + a02;
    pts[2 * i + 1] = a10 * raw[3 * i] + a11 * raw[3 * i + 1] + a12;
  }
  for (const t of tensors) t.dispose?.();
  return { pts, flag };
}

const eyeCentres = (pts) => [
  (pts[66] + pts[266]) / 2, (pts[67] + pts[267]) / 2,    // points 33 and 133
  (pts[526] + pts[724]) / 2, (pts[527] + pts[725]) / 2,  // points 263 and 362
];

// One face's { pts, flag } after both passes, or null.
async function fit(cv, ort, session, bgr, f) {
  const first = await run(cv, ort, session, bgr, f[0] + f[2] / 2, f[1] + f[3] / 2,
    TUNE.crop_side * Math.max(f[2], f[3]), Math.atan2(f[7] - f[5], f[6] - f[4]));
  if (!first) return null;
  const [rx, ry, lx, ly] = eyeCentres(first.pts);
  const angle = Math.atan2(ly - ry, lx - rx);
  // The points' box, measured upright, gives the second crop.
  const c = Math.cos(angle), s = Math.sin(angle);
  let uxMin = Infinity, uxMax = -Infinity, uyMin = Infinity, uyMax = -Infinity;
  for (let i = 0; i < POINTS; i++) {
    const x = first.pts[2 * i], y = first.pts[2 * i + 1];
    const ux = c * x + s * y, uy = c * y - s * x;
    uxMin = Math.min(uxMin, ux); uxMax = Math.max(uxMax, ux);
    uyMin = Math.min(uyMin, uy); uyMax = Math.max(uyMax, uy);
  }
  const mx = (uxMin + uxMax) / 2, my = (uyMin + uyMax) / 2;
  return run(cv, ort, session, bgr, c * mx - s * my, s * mx + c * my, TUNE.crop_side * Math.max(uxMax - uxMin, uyMax - uyMin), angle);
}

// For each face, its points in photo pixels (x0, y0, x1, y1, ...), or null where the
// older method should be used. Faces past MAX_FACES (the smallest) use that method too.
export async function landmarks(cv, ort, session, bgr, faces) {
  const found = faces.map(() => null);
  if (!session) return found;
  const order = faces.map((_, i) => i).sort((a, b) => faces[b][2] - faces[a][2]).slice(0, MAX_FACES);
  try {
    for (const i of order) {
      const result = await fit(cv, ort, session, bgr, faces[i]);
      if (result && result.flag > MIN_FLAG) found[i] = result.pts;
    }
  } catch (e) {
    console.warn('Face landmarks failed; using the older eye and lip finder.', e);
    return faces.map(() => null);
  }
  return found;
}

// One face's eyes, brows and lips, as [{ mask, x0, y0 }] in a w x h photo. The caller
// deletes the masks.
export function zones(cv, pts, w, h) {
  const t = TUNE, [rx, ry, lx, ly] = eyeCentres(pts);
  const eyeD = Math.hypot(lx - rx, ly - ry);
  const out = [];
  for (const [ids, margin] of [[EYE_R, t.eye_margin], [EYE_L, t.eye_margin], [BROW_R, t.brow_margin], [BROW_L, t.brow_margin], [LIPS, t.lip_margin]]) {
    const xs = ids.map((id) => pyRound(pts[2 * id])), ys = ids.map((id) => pyRound(pts[2 * id + 1]));
    const size = Math.max(3, pyRound(margin * eyeD)) | 1;
    const pad = (size >> 1) + 1;
    const x0 = Math.max(0, Math.min(...xs) - pad), y0 = Math.max(0, Math.min(...ys) - pad);
    const x1 = Math.min(w, Math.max(...xs) + pad + 1), y1 = Math.min(h, Math.max(...ys) + pad + 1);
    if (x1 <= x0 || y1 <= y0) continue;
    const filled = cv.Mat.zeros(y1 - y0, x1 - x0, cv.CV_8U);
    const poly = cv.matFromArray(ids.length, 1, cv.CV_32SC2, ids.flatMap((_, i) => [xs[i] - x0, ys[i] - y0]));
    const polys = new cv.MatVector();
    polys.push_back(poly);
    cv.fillPoly(filled, polys, new cv.Scalar(255));
    const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size, size));
    const mask = new cv.Mat();
    cv.dilate(filled, mask, kernel);
    filled.delete(); poly.delete(); polys.delete(); kernel.delete();
    out.push({ mask, x0, y0 });
  }
  return out;
}
