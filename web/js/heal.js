// The healing, spot and light brushes - a port of skin_smoother/heal.py. Strokes that
// change the photo itself: heal and spot after the spot fills and before smoothing
// (apply), light on the smoothed result (applyLight).
//
// Strokes have the form in mask_edit.js, with three more modes:
//   { mode: 'heal', ..., offset: [dx, dy] }  copy the photo from dx, dy away (shares
//     of its width and height) and match it to the light and shading here: seamless
//     cloning (Perez, Gangnet & Blake, 2003; Georgiev, 2004), with the smooth
//     difference added to the copy found by normalized convolution (Knutsson &
//     Westin, 1993) of how the photo differs from the source just outside the stroke.
//   { mode: 'spot', ... }  fill the stroke from the skin around it, keeping some of
//     its own fine detail, no stronger than the grain around it.
//   { mode: 'light', ..., pick: [x, y] }  paint the tone found at the picked point over
//     the stroke, keeping the skin's grain: painting on the tone layer of a frequency
//     separation, split at the size smoothing keeps grain at.
// Strokes apply in order, each reading the photo as the ones before it left it. A
// Restore stroke after them paints the photo before any of them back, at its strength;
// Add skin and Undo smoothing don't touch them.

import { copyMat } from './cv.js';
import { pyRound } from './skinmask.js';
import { softBlur } from './smooth.js';
import { IMAGE_MODES, LIGHT, MIN_SOFT_SIGMA, RESMOOTH, UNSMOOTH, drawFootprint } from './mask_edit.js';

const GRID_DIAMETER = 128, MEMBRANE_SPAN = 1, SPOT_SPAN = 1, SPOT_SPLIT = 0.08, SPOT_DETAIL = 0.5, PRIOR = 0.01;
const HEAL_MODES = ['heal', 'spot'];                          // painted before smoothing
const KEEPS = [...IMAGE_MODES, 'skin', UNSMOOTH, RESMOOTH];   // strokes that don't take these back; the rest are Restore
const f = Math.fround;

// Runs paint(img, stroke) for each stroke from the first one first() picks, and Restore
// for the strokes after it: a new Mat, or bgr itself when there are none.
function paintAll(cv, bgr, strokes, first, paint) {
  const list = strokes || [];
  const from = list.findIndex(first);
  if (from < 0) return bgr;
  const out = copyMat(cv, bgr);
  // Fresh views each time: OpenCV's memory can grow and move between calls.
  const img = { w: bgr.cols, h: bgr.rows, get data() { return out.data; }, get before() { return bgr.data; } };
  for (const stroke of list.slice(from)) {
    if (stroke.points.length < 2) continue;
    if (!paint(img, stroke) && !KEEPS.includes(stroke.mode)) restore(cv, img, stroke);   // back towards the photo before the strokes
  }
  return out;
}

// bgr (CV_8UC3) with the heal and spot strokes painted on: a new Mat, or bgr itself
// when there are none, so nothing changes for photos without them.
export function apply(cv, bgr, strokes) {
  return paintAll(cv, bgr, strokes, (s) => HEAL_MODES.includes(s.mode), (img, stroke) => {
    if (stroke.mode === 'heal') heal(cv, img, stroke);
    else if (stroke.mode === 'spot') spot(cv, img, stroke);
    else return false;
    return true;
  });
}

// bgr (the smoothed photo) with the light brush's strokes painted on: a new Mat, or bgr
// itself when there are none. split: the size (px) smoothing splits tone from grain at
// (detailSigma in smooth.js); the brush paints tone only.
export function applyLight(cv, bgr, strokes, split) {
  return paintAll(cv, bgr, strokes, (s) => s.mode === LIGHT, (img, stroke) => {
    if (stroke.mode !== LIGHT) return false;
    light(cv, img, stroke, split);
    return true;
  });
}

// ---------- float arrays through OpenCV ----------
function viaMat(cv, data, w, h, channels, op) {
  const src = cv.matFromArray(h, w, channels === 3 ? cv.CV_32FC3 : cv.CV_32F, data);
  const dst = op(src);
  const out = new Float32Array(dst.data32F);
  src.delete(); dst.delete();
  return out;
}
const resized = (cv, data, w, h, channels, toW, toH, how) => viaMat(cv, data, w, h, channels, (m) => {
  const dst = new cv.Mat();
  cv.resize(m, dst, new cv.Size(toW, toH), 0, 0, how);
  return dst;
});
const blurred = (cv, data, w, h, channels, sigma) => viaMat(cv, data, w, h, channels, (m) => softBlur(cv, m, sigma));

// Where a stroke lands at this photo size: a box in its pixels, and a grid over the
// box (the same pixels for a brush up to GRID_DIAMETER) holding its footprint, and how
// much it paints on the box's pixels. With cover, the whole outline is painted fully
// and the soft edge fades outside it, so a repair reaches everything the circle covers.
// extra: px of room the work around the stroke needs, besides its soft edge and span.
function place(cv, stroke, w, h, span, cover = false, extra = 0) {
  const c = stroke.points, points = [];
  for (let i = 0; i + 1 < c.length; i += 2) points.push([pyRound(c[i] * w), pyRound(c[i + 1] * h)]);
  const diameter = Math.max(1, stroke.size * Math.max(w, h));
  const thick = Math.max(1, pyRound(diameter));
  const sigma = Math.max(MIN_SOFT_SIGMA, (stroke.soft || 0) * diameter * 0.5);
  const reach = 3 * Math.max(sigma, span * diameter * 0.5, extra);
  const margin = Math.ceil(thick / 2 + reach) + 2;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [px, py] of points) {
    if (px < minX) minX = px; if (px > maxX) maxX = px; if (py < minY) minY = py; if (py > maxY) maxY = py;
  }
  const x0 = Math.max(0, minX - margin), x1 = Math.min(w, maxX + margin + 1);
  const y0 = Math.max(0, minY - margin), y1 = Math.min(h, maxY + margin + 1);
  if (x1 <= x0 || y1 <= y0) return null;
  const bw = x1 - x0, bh = y1 - y0, scale = Math.min(1, GRID_DIAMETER / diameter);
  const gw = scale < 1 ? Math.max(1, pyRound(bw * scale)) : bw, gh = scale < 1 ? Math.max(1, pyRound(bh * scale)) : bh;
  const fx = gw / bw, fy = gh / bh;
  const onGrid = points.map(([x, y]) => [pyRound((x - x0 + 0.5) * fx - 0.5), pyRound((y - y0 + 0.5) * fy - 0.5)]);
  const mat = drawFootprint(cv, onGrid, Math.max(1, pyRound(diameter * scale)), gw, gh);
  const footprint = new Uint8Array(mat.data);
  mat.delete();
  const p = { x0, y0, bw, bh, gw, gh, scale, diameter, footprint };
  const flat = new Float32Array(gw * gh);
  for (let i = 0; i < flat.length; i++) flat[i] = footprint[i] / 255;
  const stamp = blurred(cv, flat, gw, gh, 1, Math.max(MIN_SOFT_SIGMA, sigma * scale));
  if (cover) for (let i = 0; i < stamp.length; i++) stamp[i] = Math.min(stamp[i] * 2, 1);
  const alpha = up(cv, p, stamp, 1), amount = f(stroke.amount ?? 1);
  for (let i = 0; i < alpha.length; i++) alpha[i] = f(alpha[i] * amount);
  p.alpha = alpha;
  return p;
}

// Box pixels (float32) onto the grid, and grid values back onto the box pixels.
const down = (cv, p, a, channels) => (p.scale >= 1 ? a : resized(cv, a, p.bw, p.bh, channels, p.gw, p.gh, cv.INTER_AREA));
const up = (cv, p, a, channels) => (p.scale >= 1 ? a : resized(cv, a, p.gw, p.gh, channels, p.bw, p.bh, cv.INTER_LINEAR));

// A smooth fill of grid values (3 channels) from just outside the footprint
// (normalized convolution), on the box pixels. Deep inside a broad stroke, where
// nothing outside is near enough, it's the average of everything around it.
function around(cv, p, values, span) {
  const n = p.gw * p.gh, outside = new Float32Array(n), weighted = new Float32Array(3 * n);
  let total = 0;
  const sums = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const o = outside[i] = f(1 - p.footprint[i] / 255);
    total += o;
    for (let k = 0; k < 3; k++) sums[k] += weighted[3 * i + k] = f(values[3 * i + k] * o);
  }
  const prior = f(PRIOR), lift = sums.map((s) => f(prior * f(s / Math.max(1, total))));
  const sigma = span * p.diameter * 0.5 * p.scale;
  const num = blurred(cv, weighted, p.gw, p.gh, 3, sigma), den = blurred(cv, outside, p.gw, p.gh, 1, sigma);
  for (let i = 0; i < n; i++) {
    const d = f(den[i] + prior);
    for (let k = 0; k < 3; k++) num[3 * i + k] = f(f(num[3 * i + k] + lift[k]) / d);
  }
  return up(cv, p, num, 3);
}

// The box's pixels (float32): from the photo, or from dx, dy away, repeating its edge.
function boxOf(img, p, data = img.data, dx = 0, dy = 0) {
  const out = new Float32Array(3 * p.bw * p.bh);
  for (let y = 0, j = 0; y < p.bh; y++) {
    const sy = Math.min(Math.max(y + p.y0 + dy, 0), img.h - 1);
    for (let x = 0; x < p.bw; x++, j += 3) {
      const i = 3 * (sy * img.w + Math.min(Math.max(x + p.x0 + dx, 0), img.w - 1));
      out[j] = data[i]; out[j + 1] = data[i + 1]; out[j + 2] = data[i + 2];
    }
  }
  return out;
}

// Paint target (box pixels) into the photo at the stamp, rounding once.
function blend(img, p, target) {
  const d = img.data, half = f(0.5);
  for (let y = 0, j = 0; y < p.bh; y++) {
    for (let x = 0; x < p.bw; x++) {
      const a = p.alpha[y * p.bw + x];
      for (let k = 0, i = 3 * ((y + p.y0) * img.w + x + p.x0); k < 3; k++, i++, j++) {
        const now = d[i], v = f(f(now + f(a * f(target[j] - now))) + half);
        d[i] = v <= 0 ? 0 : v >= 255 ? 255 : Math.trunc(v);
      }
    }
  }
}

function heal(cv, img, stroke) {
  const offset = stroke.offset || [0, 0];
  const dx = pyRound(offset[0] * img.w), dy = pyRound(offset[1] * img.h);
  if (dx === 0 && dy === 0) return;            // copying from right here changes nothing
  const p = place(cv, stroke, img.w, img.h, MEMBRANE_SPAN, true);
  if (!p) return;
  const source = boxOf(img, p, img.data, dx, dy), here = boxOf(img, p);
  const a = down(cv, p, here, 3), b = down(cv, p, source, 3), difference = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) difference[i] = f(a[i] - b[i]);
  const membrane = around(cv, p, difference, MEMBRANE_SPAN);
  for (let i = 0; i < source.length; i++) source[i] = f(source[i] + membrane[i]);
  blend(img, p, source);
}

function spot(cv, img, stroke) {
  const p = place(cv, stroke, img.w, img.h, SPOT_SPAN, true);
  if (!p) return;
  const here = boxOf(img, p);
  const low = blurred(cv, here, p.bw, p.bh, 3, Math.max(MIN_SOFT_SIGMA, SPOT_SPLIT * p.diameter));
  const detail = new Float32Array(here.length), squared = new Float32Array(here.length);
  for (let i = 0; i < here.length; i++) { detail[i] = f(here[i] - low[i]); squared[i] = f(detail[i] * detail[i]); }
  // No stronger than the grain of the skin around it: a spot's own rim or white head
  // is detail too, and would stay as a faint ring or dot.
  const level = around(cv, p, down(cv, p, squared, 3), SPOT_SPAN);
  const fill = around(cv, p, down(cv, p, here, 3), SPOT_SPAN), keep = f(SPOT_DETAIL);
  for (let i = 0; i < fill.length; i++) {
    const l = f(Math.sqrt(Math.max(level[i], 0)));
    fill[i] = f(fill[i] + f(keep * Math.min(Math.max(detail[i], -l), l)));
  }
  blend(img, p, fill);
}

// The photo's tone at a point (shares of width and height): the average of a square
// about the size of the grain, so the grain itself doesn't count (float32 BGR).
function toneAt(img, pick, split) {
  const x = Math.min(img.w - 1, Math.max(0, pyRound(pick[0] * img.w)));
  const y = Math.min(img.h - 1, Math.max(0, pyRound(pick[1] * img.h)));
  const r = Math.max(1, pyRound(split)), d = img.data, sum = [0, 0, 0];
  const xa = Math.max(0, x - r), xb = Math.min(img.w, x + r + 1), ya = Math.max(0, y - r), yb = Math.min(img.h, y + r + 1);
  for (let yy = ya; yy < yb; yy++) {
    for (let xx = xa, i = 3 * (yy * img.w + xa); xx < xb; xx++, i += 3) { sum[0] += d[i]; sum[1] += d[i + 1]; sum[2] += d[i + 2]; }
  }
  const n = (xb - xa) * (yb - ya);
  return sum.map((s) => f(s / n));
}

function light(cv, img, stroke, split) {
  if (!stroke.pick) return;
  const colour = toneAt(img, stroke.pick, split);
  const p = place(cv, stroke, img.w, img.h, 0, false, split);
  if (!p) return;
  const here = boxOf(img, p), low = blurred(cv, here, p.bw, p.bh, 3, split);
  for (let i = 0, k = 0; i < here.length; i++, k = k === 2 ? 0 : k + 1) here[i] = f(here[i] + f(colour[k] - low[i]));
  blend(img, p, here);
}

function restore(cv, img, stroke) {
  const p = place(cv, stroke, img.w, img.h, 0);
  if (p) blend(img, p, boxOf(img, p, img.before));
}
