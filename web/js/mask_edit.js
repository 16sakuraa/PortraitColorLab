// Manual edits to the skin mask - a port of skin_smoother/mask_edit.py.
//
// A stroke is resolution-independent, so one set of strokes gives the same edit on a
// preview and on the full-size photo:
//   { mode: 'protect', 'unsmooth' or 'skin', size: brush diameter as a share of the
//     long side, soft: 0..1, amount: 0..1 (optional, 1 if missing),
//     points: [x0, y0, x1, y1, ...] each 0..1 of the width or height }
// Strokes are painted in order into a layer from -1 (protect) to 1 (skin), each towards
// its own amount, so a later stroke wins where they cross. Protect, shown as Restore,
// brings back that share of the original: at 1, never smooth here and don't fill any
// spots here. Unsmooth, shown as Undo smoothing, is Restore on the mask, but the spot
// fills (and heal.js's heals) stay. Skin means smooth here even where the automatic
// mask left it out.
// The spot fills follow a second layer (layer.fills), the same as the first until an
// unsmooth stroke, which paints it back towards no edit.
// Resmooth, shown as Smooth again, smooths once more with finer grain (smoothAgain in
// smooth.js). It follows a third layer (layer.again, 0..1), painted towards each
// resmooth stroke's amount; Restore and Undo smoothing strokes after it paint it to 0.
// The heal, spot and light modes change the photo, not the mask (heal.js), so the
// layers skip them.

import { pyRound } from './skinmask.js';

const LAYER_MAX = 2048, PRESENT = 1e-3;
export const MIN_SOFT_SIGMA = 0.7;
export const LIGHT = 'light';                  // the light brush (heal.js)
export const IMAGE_MODES = ['heal', 'spot', LIGHT];   // strokes heal.js paints on the photo
export const UNSMOOTH = 'unsmooth';            // Undo smoothing: off the mask, but the fixes stay
export const RESMOOTH = 'resmooth';            // Smooth again: a finer smoothing, on its own layer

// A stroke's hard footprint (a new CV_8U Mat, 255 inside) on a w x h grid, through
// these pixel points with this thickness.
export function drawFootprint(cv, points, thick, w, h) {
  const stamp = cv.Mat.zeros(h, w, cv.CV_8U), white = new cv.Scalar(255);
  const at = (p) => new cv.Point(p[0], p[1]);
  cv.circle(stamp, at(points[0]), Math.max(1, thick >> 1), white, -1, cv.LINE_8);
  for (let i = 1; i < points.length; i++) cv.line(stamp, at(points[i - 1]), at(points[i]), white, thick, cv.LINE_8);
  return stamp;
}

// An empty layer for a photo of this size. strokes records what's painted into it;
// fills is the spot fills' layer, null while it's the same as data; again is Smooth
// again's, null until its first stroke.
export function newLayer(w, h) {
  const s = Math.min(1, LAYER_MAX / Math.max(w, h));
  const lw = Math.max(1, pyRound(w * s)), lh = Math.max(1, pyRound(h * s));
  return { w: lw, h: lh, data: new Float32Array(lw * lh), fills: null, again: null, strokes: [] };
}

// Add one stroke to a layer (and to its fills and Smooth again layers, if it has them), in place.
export function paintStroke(cv, layer, stroke) {
  const { w: lw, h: lh } = layer, c = stroke.points, mode = stroke.mode;
  if (IMAGE_MODES.includes(mode) || (mode === RESMOOTH && !layer.again) || c.length < 2) return;
  const points = [];
  for (let i = 0; i + 1 < c.length; i += 2) points.push([pyRound(c[i] * lw), pyRound(c[i + 1] * lh)]);
  const diameter = Math.max(1, stroke.size * Math.max(lw, lh));
  const thick = Math.max(1, pyRound(diameter));
  const sigma = Math.max(MIN_SOFT_SIGMA, (stroke.soft || 0) * diameter * 0.5);

  // Only the part of the layer this stroke can reach, with room for the blur.
  const margin = Math.ceil(thick / 2 + 4 * sigma) + 2;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [px, py] of points) {
    if (px < minX) minX = px; if (px > maxX) maxX = px; if (py < minY) minY = py; if (py > maxY) maxY = py;
  }
  const x0 = Math.max(0, minX - margin), x1 = Math.min(lw, maxX + margin + 1);
  const y0 = Math.max(0, minY - margin), y1 = Math.min(lh, maxY + margin + 1);
  if (x1 <= x0 || y1 <= y0) return;
  const bw = x1 - x0, bh = y1 - y0;
  const stamp = drawFootprint(cv, points.map(([px, py]) => [px - x0, py - y0]), thick, bw, bh);
  const asFloat = new cv.Mat(), blurred = new cv.Mat();
  stamp.convertTo(asFloat, cv.CV_32F, 1 / 255);
  stamp.delete();
  cv.GaussianBlur(asFloat, blurred, new cv.Size(0, 0), sigma);
  asFloat.delete();
  const soft = blurred.data32F, data = layer.data, fills = layer.fills, again = layer.again;
  const paint = (into, toward) => {
    for (let y = 0, p = 0; y < bh; y++) {
      let q = (y + y0) * lw + x0;
      for (let x = 0; x < bw; x++, p++, q++) into[q] = into[q] * (1 - soft[p]) + toward * soft[p];
    }
  };
  if (mode === RESMOOTH) {
    paint(again, stroke.amount ?? 1);
    blurred.delete();
    return;
  }
  const target = (stroke.mode === 'skin' ? 1 : -1) * (stroke.amount ?? 1);
  for (let y = 0, p = 0; y < bh; y++) {
    let q = (y + y0) * lw + x0;
    for (let x = 0; x < bw; x++, p++, q++) data[q] = data[q] * (1 - soft[p]) + target * soft[p];
  }
  if (fills) {
    const keep = stroke.mode === UNSMOOTH ? 0 : target;
    for (let y = 0, p = 0; y < bh; y++) {
      let q = (y + y0) * lw + x0;
      for (let x = 0; x < bw; x++, p++, q++) fills[q] = fills[q] * (1 - soft[p]) + keep * soft[p];
    }
  }
  if (again && mode !== 'skin') paint(again, 0);   // Restore and Undo smoothing take it back
  blurred.delete();
}

// A layer holding exactly these strokes. Strokes usually only get added, so a previous
// layer is reused when it's still a prefix of them; each stroke has an id for this.
export function layerFor(cv, w, h, strokes, previous = null) {
  let layer = previous;
  const fits = layer && layer.strokes.length <= strokes.length
    && layer.strokes.every((id, i) => id === strokes[i].id);
  if (!fits) layer = newLayer(w, h);
  for (const stroke of strokes.slice(layer.strokes.length)) {
    if (stroke.mode === UNSMOOTH && !layer.fills) layer.fills = new Float32Array(layer.data);
    if (stroke.mode === RESMOOTH && !layer.again) layer.again = new Float32Array(layer.data.length);
    paintStroke(cv, layer, stroke);
    layer.strokes.push(stroke.id);
  }
  return layer;
}

// The edited part of a layer at this photo size: { x0, y0, w, h, data } in -1..1, or null.
export function regionOf(cv, layer, w, h) {
  const { w: lw, h: lh, data } = layer;
  let a0 = lw, a1 = 0, b0 = lh, b1 = 0;
  for (let y = 0, i = 0; y < lh; y++) {
    for (let x = 0; x < lw; x++, i++) {
      if (Math.abs(data[i]) > PRESENT) {
        if (x < a0) a0 = x; if (x >= a1) a1 = x + 1; if (y < b0) b0 = y; if (y >= b1) b1 = y + 1;
      }
    }
  }
  if (a1 === 0) return null;
  a0 = Math.max(0, a0 - 1); a1 = Math.min(lw, a1 + 1);
  b0 = Math.max(0, b0 - 1); b1 = Math.min(lh, b1 + 1);
  const sx = w / lw, sy = h / lh;
  const x0 = Math.trunc(a0 * sx), x1 = Math.min(w, Math.ceil(a1 * sx));
  const y0 = Math.trunc(b0 * sy), y1 = Math.min(h, Math.ceil(b1 * sy));
  const cols = a1 - a0, rows = b1 - b0, part = new Float32Array(cols * rows);
  for (let y = 0; y < rows; y++) part.set(data.subarray((y + b0) * lw + a0, (y + b0) * lw + a1), y * cols);
  const src = cv.matFromArray(rows, cols, cv.CV_32F, part), big = new cv.Mat();
  cv.resize(src, big, new cv.Size(x1 - x0, y1 - y0), 0, 0, cv.INTER_LINEAR);
  const out = { x0, y0, w: x1 - x0, h: y1 - y0, data: new Float32Array(big.data32F) };
  src.delete(); big.delete();
  return out;
}

// A new mask ({ x0, y0, w, h, data }) with an edit applied, grown to include it.
export function applyToMask(mask, region) {
  if (!region) return mask;
  const x0 = Math.min(mask.x0, region.x0), y0 = Math.min(mask.y0, region.y0);
  const x1 = Math.max(mask.x0 + mask.w, region.x0 + region.w), y1 = Math.max(mask.y0 + mask.h, region.y0 + region.h);
  const w = x1 - x0, h = y1 - y0, data = new Float32Array(w * h);
  for (let y = 0; y < mask.h; y++) {
    data.set(mask.data.subarray(y * mask.w, (y + 1) * mask.w), (y + mask.y0 - y0) * w + mask.x0 - x0);
  }
  for (let y = 0, p = 0; y < region.h; y++) {
    let q = (y + region.y0 - y0) * w + region.x0 - x0;
    for (let x = 0; x < region.w; x++, p++, q++) {
      const e = region.data[p], m = data[q];
      data[q] = e > 0 ? m + e * (1 - m) : m * (1 + e);
    }
  }
  return { x0, y0, w, h, data };
}

// A blemish correction ({ x0, y0, w, h, data }) with protected areas taken out of it.
export function applyToCorrection(corr, region) {
  if (!corr || !region) return corr;
  const ax0 = Math.max(corr.x0, region.x0), ay0 = Math.max(corr.y0, region.y0);
  const ax1 = Math.min(corr.x0 + corr.w, region.x0 + region.w), ay1 = Math.min(corr.y0 + corr.h, region.y0 + region.h);
  if (ax1 <= ax0 || ay1 <= ay0) return corr;
  const data = new Float32Array(corr.data);
  for (let y = ay0; y < ay1; y++) {
    for (let x = ax0; x < ax1; x++) {
      const e = region.data[(y - region.y0) * region.w + x - region.x0];
      if (e >= 0) continue;
      const j = 3 * ((y - corr.y0) * corr.w + x - corr.x0), keep = 1 + e;
      data[j] *= keep; data[j + 1] *= keep; data[j + 2] *= keep;
    }
  }
  return { ...corr, data };
}
