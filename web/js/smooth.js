// Face-skin smoothing - a line-for-line port of smoother.smooth_face.
//
// Edge-preserving (bilateral) smoothing of the masked skin, with a share of
// the original fine texture added back. Large regions are smoothed on a
// reduced copy (at most WORK_PIXELS) and scaled back up; the fine texture
// comes back at full resolution.

import { pyRound } from './skinmask.js';
import { copyMat } from './cv.js';

export const WORK_PIXELS = 1.5e6;

function smoothLayer(cv, img, strength, diag, scale = 1.0) {
  strength = Math.min(Math.max(strength, 0), 1);
  const d = Math.max(5, Math.trunc(diag * 0.006 * scale));
  const sigmaColor = 20 + strength * 90;
  const sigmaSpace = Math.max(1.0, (15 + strength * 45) * scale);
  const iterations = 1 + pyRound(strength * 2);
  let out = copyMat(cv, img);
  for (let i = 0; i < iterations; i++) {
    const next = new cv.Mat();
    cv.bilateralFilter(out, next, d, sigmaColor, sigmaSpace, cv.BORDER_DEFAULT);
    out.delete();
    out = next;
  }
  return out;
}

// bgr: cv.Mat (CV_8UC3); mask: { x0, y0, w, h, data } from buildSkinMask.
export function smoothFace(cv, bgr, strength, texture, mask) {
  strength = Math.min(Math.max(strength, 0), 1);
  if (strength <= 0) return copyMat(cv, bgr);
  const H = bgr.rows, W = bgr.cols, diag = Math.hypot(H, W);

  // Bounding box of the mask (> 1e-3), in photo coordinates.
  let minX = Infinity, maxX = -1, minY = Infinity, maxY = -1;
  for (let y = 0; y < mask.h; y++) {
    for (let x = 0; x < mask.w; x++) {
      if (mask.data[y * mask.w + x] > 1e-3) {
        if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return copyMat(cv, bgr);
  const pad = Math.trunc(diag * 0.006) + 8;
  const y0 = Math.max(0, minY + mask.y0 - pad), y1 = Math.min(H, maxY + mask.y0 + 1 + pad);
  const x0 = Math.max(0, minX + mask.x0 - pad), x1 = Math.min(W, maxX + mask.x0 + 1 + pad);
  const view = bgr.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
  const roi = copyMat(cv, view);
  view.delete();
  const rh = roi.rows, rw = roi.cols;

  const scale = Math.min(1.0, Math.sqrt(WORK_PIXELS / (rh * rw)));
  let base;
  if (scale < 1.0) {
    const small = new cv.Mat();
    cv.resize(roi, small, new cv.Size(Math.max(1, pyRound(rw * scale)), Math.max(1, pyRound(rh * scale))), 0, 0, cv.INTER_AREA);
    const sm = smoothLayer(cv, small, strength, diag, scale);
    small.delete();
    base = new cv.Mat();
    cv.resize(sm, base, new cv.Size(rw, rh), 0, 0, cv.INTER_LINEAR);
    sm.delete();
  } else {
    base = smoothLayer(cv, roi, strength, diag);
  }
  const blur = new cv.Mat();
  cv.GaussianBlur(roi, blur, new cv.Size(0, 0), 3);

  // smoothed = base + texture * (roi - blur); blend through mask * strength.
  const out = copyMat(cv, bgr);
  const o = out.data, r = roi.data, b = base.data, g = blur.data, md = mask.data;
  const tex = Math.fround(texture), str = Math.fround(strength);   // NumPy casts scalars to float32
  for (let y = 0; y < rh; y++) {
    const py = y + y0, my = py - mask.y0;
    for (let x = 0; x < rw; x++) {
      const px = x + x0, mx = px - mask.x0;
      const m = my >= 0 && my < mask.h && mx >= 0 && mx < mask.w ? md[my * mask.w + mx] : 0;
      const a = Math.fround(m * str), ia = Math.fround(1.0 - a);
      const i = 3 * (y * rw + x), j = 3 * (py * W + px);
      for (let k = 0; k < 3; k++) {
        const high = Math.fround(r[i + k] - g[i + k]);
        const sm = Math.fround(b[i + k] + Math.fround(tex * high));
        const v = Math.fround(Math.fround(r[i + k] * ia) + Math.fround(sm * a));
        o[j + k] = v <= 0 ? 0 : v >= 255 ? 255 : Math.trunc(v);
      }
    }
  }
  roi.delete(); base.delete(); blur.delete();
  return out;
}
