// A beard on the chin, taken off: clear() replaces it, as lip.clear(..., full) does a full
// moustache. The same step as the desktop app's chin.py, piece by piece; see the notes
// there. In short: the chin is straightened into its own frame (u across, -1 and 1 under
// the mouth's corners; v down, 0 at the lower lip's edge, 1 at the jaw's edge), so the
// lip's edge, the fold under it and the jaw's edge each run along a row. A strand is
// darker than its row; line-shaped strands and masses of hair say whether there is a
// beard, and the area is followed outwards from where they are dense. The frame is moved
// onto the photo's own jaw edge, read beside the hair. The tone of a bare chin (TABLE's
// shape in the cheeks' colour) is matched to the clear skin and carried inwards along the
// rows only, so the jaw's line stays the face's own; the grain is the cheeks', in pieces
// (lip.grainPieces). Only on a face straight on with its mouth closed whose chin has a
// usual height, and only when asked; a bare chin and a beard across the jaw are left as
// they are.
// The tone is worked out on a copy at one fixed size (lip.GRID_EYES between the eyes) and
// only the grain and the lip's edge at the photo's own size, so a preview and the
// full-size photo come out alike. The step comes right after lip.clear.

import { copyMat } from './cv.js';
import { pyRound } from './skinmask.js';
import { medianOf } from './smooth.js';
import * as lip from './lip.js';

// Face Mesh points.
const LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291];     // the lower lip's outer edge, right corner to left
const JAW = [136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365];   // the face's outline under the mouth, the same way
const CHIN_POINT = 152, LIP_FOOT = 17, NOSE_BASE = 2;
const LUMA_B = 0.114, LUMA_G = 0.587, LUMA_R = 0.299;
const f = Math.fround;

// Same table as chin.TABLE: the brightness of a bare chin, place by place. Rows: from the
// lower lip's edge down to the jaw's edge (the eleventh) and on to the neck, in tenths.
// Columns: from the middle outwards, in fifths of the way to under the mouth's corner.
export const TABLE = [
  [0.74, 0.75, 0.78, 0.80, 0.78, 0.71, 0.79, 0.85, 0.88, 0.88],
  [0.74, 0.75, 0.78, 0.80, 0.78, 0.77, 0.80, 0.85, 0.87, 0.88],
  [0.83, 0.83, 0.83, 0.84, 0.84, 0.84, 0.84, 0.85, 0.86, 0.86],
  [0.89, 0.90, 0.89, 0.88, 0.88, 0.87, 0.86, 0.85, 0.86, 0.85],
  [0.92, 0.92, 0.92, 0.91, 0.89, 0.88, 0.86, 0.85, 0.85, 0.85],
  [0.92, 0.92, 0.94, 0.92, 0.90, 0.88, 0.86, 0.85, 0.84, 0.84],
  [0.89, 0.90, 0.92, 0.92, 0.90, 0.88, 0.86, 0.83, 0.83, 0.83],
  [0.85, 0.86, 0.87, 0.88, 0.88, 0.87, 0.85, 0.83, 0.82, 0.82],
  [0.82, 0.82, 0.83, 0.83, 0.85, 0.85, 0.83, 0.82, 0.81, 0.81],
  [0.76, 0.76, 0.77, 0.79, 0.81, 0.82, 0.80, 0.80, 0.80, 0.79],
  [0.67, 0.68, 0.71, 0.74, 0.76, 0.77, 0.76, 0.76, 0.77, 0.76],
  [0.65, 0.67, 0.69, 0.72, 0.74, 0.75, 0.75, 0.76, 0.76, 0.75],
  [0.72, 0.73, 0.73, 0.76, 0.77, 0.78, 0.76, 0.77, 0.77, 0.75],
  [0.78, 0.78, 0.79, 0.80, 0.80, 0.81, 0.80, 0.80, 0.79, 0.76],
  [0.81, 0.81, 0.82, 0.81, 0.81, 0.82, 0.82, 0.81, 0.80, 0.78],
  [0.83, 0.84, 0.84, 0.84, 0.84, 0.83, 0.83, 0.82, 0.80, 0.79],
  [0.84, 0.84, 0.83, 0.86, 0.87, 0.87, 0.84, 0.83, 0.81, 0.79],
].map((row) => row.map(f));
const TABLE_U = 0.2, TABLE_V = 0.1;

// Same settings as chin.TUNE; see the notes there.
export const TUNE = {
  tall: [0.36, 0.68], span_u: 1.9, span_v: [-0.25, 1.7], pad: 0.3,
  edge_reach: 0.10, edge_soft: [0.03, 0.012], edge_least: 0.012, edge_out: 0.03, edge_columns: 12, edge_span: 1.0,
  edge_even: 0.05, edge_off: 0.03,
  row: 0.03, strand: 3.0, strand_least: 0.04, strand_most: 0.08, long: 0.035,
  mass_soft: 0.012, mass_row: 0.25, mass: 0.16, dense_at: 0.035, line_gain: 3.0, core: 0.13, least: 0.02,
  hair: 0.06, tail_reach: 0.10, bridge: 0.015, grow: 0.035, soft: 0.02,
  top: 0.02, foot: 1.35, wide_top: 0.8, wide: 1.35, widen: [0.15, 0.55], bound_edge: 0.03, over_lip: -0.3,
  flank: 0.35, flank_v: [0.3, 1.15], flank_most: 0.30,
  fill: [0.5, 0.25, 0.12, 0.06, 0.03], across: 0.25, across_most: 0.004, fill_lo: 0.08, fill_hi: 0.35,
  span: [0.6, 1.4], clear_off: 1.0, clear_dense: [0.08, 0.2], first_off: 0.04, above: [-0.05, -0.015], above_at: 0.008,
  tile: 0.12, tile_wander: 0.07, side: 0.6, side_down: 0.5, grain_shade: [0.8, 1.2], lip_soft: 0.012,
};

const luma = (a, j) => f(f(f(a[j] * f(LUMA_B)) + f(a[j + 1] * f(LUMA_G))) + f(a[j + 2] * f(LUMA_R)));

// A blur with a different size along the row (sx) and across it (sy); the edge is repeated (chin._blur2).
export function blur2(cv, data, w, h, channels, sx, sy) {
  const rx = Math.max(0, pyRound((Math.sqrt(4.0 * sx * sx + 1.0) - 1.0) / 2.0)), ry = Math.max(0, pyRound((Math.sqrt(4.0 * sy * sy + 1.0) - 1.0) / 2.0));
  const mat = cv.matFromArray(h, w, channels === 3 ? cv.CV_32FC3 : cv.CV_32F, data);
  const size = new cv.Size(2 * rx + 1, 2 * ry + 1), anchor = new cv.Point(-1, -1);
  for (let k = 0; k < 3; k++) cv.boxFilter(mat, mat, cv.CV_32F, size, anchor, true, cv.BORDER_REPLICATE);
  const out = new Float32Array(mat.data32F);
  mat.delete();
  return out;
}

// a averaged over what is on (1 or 0) around each pixel (one channel).
function rowMean(cv, a, on, w, h, sx, sy) {
  const n = w * h, lit = new Float32Array(n), tiny = f(1e-4);
  for (let i = 0; i < n; i++) lit[i] = f(a[i] * on[i]);
  const sum = blur2(cv, lit, w, h, 1, sx, sy), share = blur2(cv, on, w, h, 1, sx, sy);
  for (let i = 0; i < n; i++) sum[i] = f(sum[i] / f(share[i] + tiny));
  return sum;
}

// data (three values a pixel), known where weight is, carried everywhere along the rows:
// the widest radius first, each finer one taking over where enough of what it sees is known.
function fillIn(cv, data, weight, w, h, unit) {
  const t = TUNE, n = w * h, tiny = f(1e-6), weighted = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let ch = 0; ch < 3; ch++) weighted[3 * i + ch] = f(data[3 * i + ch] * weight[i]);
  let out = null;
  for (const r of [...t.fill].sort((a, b) => b - a)) {
    const sx = r * unit, sy = Math.min(r * t.across, t.across_most) * unit;
    const seen = blur2(cv, weight, w, h, 1, sx, sy), here = blur2(cv, weighted, w, h, 3, sx, sy);
    for (let i = 0; i < n; i++) {
      const den = f(seen[i] + tiny);
      for (let ch = 0; ch < 3; ch++) here[3 * i + ch] = f(here[3 * i + ch] / den);
    }
    if (!out) { out = here; continue; }
    for (let i = 0; i < n; i++) {
      const c = lip.step32(seen[i], t.fill_lo, t.fill_hi), rest = f(1 - c);
      for (let ch = 0; ch < 3; ch++) out[3 * i + ch] = f(f(out[3 * i + ch] * rest) + f(here[3 * i + ch] * c));
    }
  }
  return out;
}

// Thin lines n long, as kernels: straight down, and leaning 27 and 45 degrees either way.
export function lineKernels(cv, length) {
  const n = length | 1, mid = n >> 1;
  return [0.0, 0.5, -0.5, 1.0, -1.0].map((lean) => {
    const k = new Uint8Array(n * n);
    for (let i = 0; i < n; i++) k[i * n + pyRound(mid + (i - mid) * lean)] = 1;
    return cv.matFromArray(n, n, cv.CV_8U, k);
  });
}

// TABLE at a place in the frame, between its entries and held at its edges.
export function tableAt(u, v) {
  const rows = TABLE.length, cols = TABLE[0].length;
  const a = Math.min(Math.max(f(Math.abs(u) / f(TABLE_U)), 0), cols - 1), b = Math.min(Math.max(f(v / f(TABLE_V)), 0), rows - 1);
  const a0 = Math.min(Math.floor(a), cols - 2), b0 = Math.min(Math.floor(b), rows - 2), fa = f(a - a0), fb = f(b - b0);
  const top = f(f(TABLE[b0][a0] * f(1 - fa)) + f(TABLE[b0][a0 + 1] * fa));
  const low = f(f(TABLE[b0 + 1][a0] * f(1 - fa)) + f(TABLE[b0 + 1][a0 + 1] * fa));
  return f(f(top * f(1 - fb)) + f(low * fb));
}

// A curve through points (xp increasing): at(x) is its height there, held at its ends, as
// numpy's interp works it out. It remembers where it last looked: neighbours ask next.
function curveOf(xp, fp) {
  const n = xp.length;
  let j = 0;
  return {
    xp, fp,
    at(x) {
      if (x > xp[n - 1]) return fp[n - 1];
      if (x < xp[0]) return fp[0];
      if (!(xp[j] <= x && (j === n - 1 || x < xp[j + 1]))) {
        let lo = 0, hi = n;                       // the last point at or before x
        while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xp[mid] <= x) lo = mid; else hi = mid; }
        j = lo;
      }
      if (j === n - 1 || xp[j] === x) return fp[j];
      const span = xp[j + 1] - xp[j];
      return span > 0 ? (fp[j + 1] - fp[j]) / span * (x - xp[j]) + fp[j] : fp[j];
    },
  };
}

// One face's chin frame, in photo pixels (chin._Frame).
export function frameOf(pts) {
  const at = (i) => [pts[2 * i], pts[2 * i + 1]];
  const mouth = lip.frameOf(pts), { eyes, centre, half } = mouth;
  const half2 = half[0] * half[0] + half[1] * half[1], wide = Math.sqrt(Math.max(half2, 1e-9));   // half the mouth's width, px
  let down = [-half[1] / wide, half[0] / wide];                 // one pixel towards the chin: away from the nose
  const nose = at(NOSE_BASE);
  if (down[0] * (centre[0] - nose[0]) + down[1] * (centre[1] - nose[1]) < 0) down = [-down[0], -down[1]];
  // Points as a smooth curve: across (u), and how far below the mouth's line (px).
  const curve = (ids) => {
    const q = ids.map((i) => {
      const p = at(i);
      return [((p[0] - centre[0]) * half[0] + (p[1] - centre[1]) * half[1]) / Math.max(half2, 1e-9), (p[0] - centre[0]) * down[0] + (p[1] - centre[1]) * down[1]];
    });
    const order = q.map((_, i) => i).sort((a, b) => q[a][0] - q[b][0] || a - b);
    const s = lip.spline(order.map((i) => q[i]));
    const again = s.map((_, i) => i).sort((a, b) => s[a][0] - s[b][0] || a - b);
    return curveOf(Float64Array.from(again, (i) => s[i][0]), Float64Array.from(again, (i) => s[i][1]));
  };
  const lipEdge = curve(LOWER);
  const fr = {
    mouth, eyes, centre, half, half2, wide, down, cheeks: mouth.cheeks, lips: mouth.lips,
    jawLine: curve(JAW),
    // The lower lip's edge (px below the mouth's line); beyond the mouth's corners, as at them.
    top: (u) => lipEdge.at(Math.min(1, Math.max(-1, u))),
    // The jaw's edge, the same way.
    jaw: (u) => fr.jawLine.at(u),
    // The photo's place for a place in the frame.
    place(u, v) {
      const h0 = fr.top(u), h1 = fr.jaw(u), h = h0 + v * (h1 - h0);
      return [centre[0] + u * half[0] + h * down[0], centre[1] + u * half[1] + h * down[1]];
    },
    // The frame's u and v at a place in the photo.
    uv(x, y) {
      const u = ((x - centre[0]) * half[0] + (y - centre[1]) * half[1]) / half2;
      const h = (x - centre[0]) * down[0] + (y - centre[1]) * down[1], h0 = fr.top(u), h1 = fr.jaw(u);
      return [u, (h - h0) / Math.max(h1 - h0, 1e-3)];
    },
  };
  const point = at(CHIN_POINT), foot = at(LIP_FOOT);
  fr.tall = ((point[0] - foot[0]) * down[0] + (point[1] - foot[1]) * down[1]) / Math.max(eyes, 1e-9);
  fr.ok = Boolean(mouth.ok && fr.tall > 0.15 && fr.tall < 1.5 && fr.jaw(0) - fr.top(0) > 2.0);
  fr.posed = Boolean(fr.ok && mouth.posed && fr.tall >= TUNE.tall[0] && fr.tall <= TUNE.tall[1]);
  return fr;
}

// The chin's box of one photo, its copy at the fixed size, and how that copy is straightened
// into the frame (chin._Work). null when there is nothing to work on.
function workOf(cv, out, fr, mask) {
  const t = TUNE, lt = lip.TUNE, W = out.cols, H = out.rows, d = fr.eyes;
  const span = t.span_u, [v0, v1] = t.span_v, xs = [], ys = [];
  for (const v of [v0, v1]) for (const u of [-span, span]) { const p = fr.place(u, v); xs.push(p[0]); ys.push(p[1]); }
  for (const c of fr.cheeks) for (const s of [-1, 1]) { xs.push(c[0] + s * lt.cheek * d); ys.push(c[1] + s * lt.cheek * d); }
  const pad = t.pad * d + 2.0;
  const x0 = Math.max(0, Math.floor(Math.min(...xs) - pad)), x1 = Math.min(W, Math.ceil(Math.max(...xs) + pad) + 1);
  const y0 = Math.max(0, Math.floor(Math.min(...ys) - pad)), y1 = Math.min(H, Math.ceil(Math.max(...ys) + pad) + 1);
  if (x1 - x0 < 8 || y1 - y0 < 8) return null;
  const bw = x1 - x0, bh = y1 - y0, bn = bw * bh, scale = Math.min(1, lip.GRID_EYES / d);
  const gw = scale < 1 ? Math.max(8, pyRound(bw * scale)) : bw, gh = scale < 1 ? Math.max(8, pyRound(bh * scale)) : bh, gn = gw * gh;
  const fx = bw / gw, fy = bh / gh, eyes = d / ((fx + fy) / 2);     // photo pixels per copy pixel; the distance between the eyes, on the copy

  const k255 = f(255), tiny = f(1e-4);
  const box = new Float32Array(3 * bn), boxMask = new Float32Array(bn);
  {
    const od = out.data;        // read here and now: making a Mat further down can move the heap this view looks at
    for (let y = 0; y < bh; y++) {
      for (let x = 0, i = y * bw, p = 3 * ((y + y0) * W + x0); x < bw; x++, i++, p += 3) {
        box[3 * i] = f(od[p] / k255); box[3 * i + 1] = f(od[p + 1] / k255); box[3 * i + 2] = f(od[p + 2] / k255);
      }
    }
  }
  for (let y = Math.max(y0, mask.y0); y < Math.min(y1, mask.y0 + mask.h); y++) {
    for (let x = Math.max(x0, mask.x0); x < Math.min(x1, mask.x0 + mask.w); x++) {
      boxMask[(y - y0) * bw + x - x0] = mask.data[(y - mask.y0) * mask.w + x - mask.x0];
    }
  }
  const small = scale < 1 ? lip.resized(cv, box, bw, bh, 3, gw, gh, cv.INTER_AREA) : box;
  const skin = scale < 1 ? lip.resized(cv, boxMask, bw, bh, 1, gw, gh, cv.INTER_AREA) : boxMask;

  // The lips, outlined smoothly and cut out of the tone (as lip.clearFull has them).
  const upper = lip.spline(fr.lips.slice(0, lip.UPPER.length)), lower = lip.spline(fr.lips.slice(lip.UPPER.length));
  const outline = [...upper, ...lower.slice(1, -1)];
  const lips = lip.polygon(cv, outline.map((p) => [(p[0] - x0 + 0.5) / fx - 0.5, (p[1] - y0 + 0.5) / fy - 0.5]), gw, gh);
  const valid = lip.blurred(cv, Float32Array.from(lips, (b) => f(b / k255)), gw, gh, 1, Math.max(0.6, lt.lip_edge * eyes));
  for (let i = 0; i < gn; i++) valid[i] = f(1 - valid[i]);
  const sigma = lt.tone * eyes, lit = new Float32Array(3 * gn);
  for (let i = 0; i < gn; i++) { lit[3 * i] = f(small[3 * i] * valid[i]); lit[3 * i + 1] = f(small[3 * i + 1] * valid[i]); lit[3 * i + 2] = f(small[3 * i + 2] * valid[i]); }
  const tone = lip.blurred(cv, lit, gw, gh, 3, sigma), toneDen = lip.blurred(cv, valid, gw, gh, 1, sigma);
  for (let i = 0; i < gn; i++) {
    const den = f(toneDen[i] + tiny);
    tone[3 * i] = f(tone[3 * i] / den); tone[3 * i + 1] = f(tone[3 * i + 1] / den); tone[3 * i + 2] = f(tone[3 * i + 2] / den);
  }
  // The cheeks: their colour, for the new skin's.
  const r2 = (lt.cheek * d) ** 2, onCheek = new Uint8Array(gn), wellSkin = f(0.9), medians = [];
  for (const c of fr.cheeks) {
    const found = [[], [], []];
    for (let y = 0, i = 0; y < gh; y++) {
      const py = y0 + (y + 0.5) * fy - 0.5;
      for (let x = 0; x < gw; x++, i++) {
        const px = x0 + (x + 0.5) * fx - 0.5;       // the photo's place for each copy pixel
        if ((px - c[0]) ** 2 + (py - c[1]) ** 2 < r2 && skin[i] > wellSkin) {
          found[0].push(tone[3 * i]); found[1].push(tone[3 * i + 1]); found[2].push(tone[3 * i + 2]);
          onCheek[i] = 1;
        }
      }
    }
    if (found[0].length < 30) return null;
    medians.push(found.map((values) => f(medianOf(values))));
  }
  const cheek = [0, 1, 2].map((ch) => f(f(medians[0][ch] + medians[1][ch]) / f(2)));
  const w = {
    fr, box: [x0, y0, x1, y1], bw, bh, gw, gh, fx, fy, scale, eyes, pixels: box, boxMask, small, skin, outline, valid, tone, onCheek, cheek,
  };
  straighten(w);
  return w;
}

// Where each pixel of the straightened copy (columns u, rows v) lies on the copy, for the frame as it is now.
function straighten(w) {
  const t = TUNE, fr = w.fr, span = t.span_u, [v0, v1] = t.span_v, [x0, y0] = w.box;
  const px = fr.eyes / w.eyes;                                  // photo pixels per copy pixel
  const tall = fr.jaw(0.0) - fr.top(0.0);                       // the chin's height in the middle, px
  const nu = Math.max(8, pyRound(2.0 * span * fr.wide / px)), nv = Math.max(8, pyRound((v1 - v0) * tall / px)), n = nu * nv;
  const du = 2.0 * span / nu, dv = (v1 - v0) / nv;
  const u1 = Float64Array.from({ length: nu }, (_, k) => -span + (k + 0.5) * du), v1s = Float64Array.from({ length: nv }, (_, k) => v0 + (k + 0.5) * dv);
  const u = new Float32Array(n), v = new Float32Array(n), mx = new Float32Array(n), my = new Float32Array(n), inside = new Float32Array(n);
  const tops = Float64Array.from(u1, (a) => fr.top(f(a))), jaws = Float64Array.from(u1, (a) => fr.jaw(f(a)));
  for (let r = 0, i = 0; r < nv; r++) {
    const vr = f(v1s[r]);
    for (let c = 0; c < nu; c++, i++) {
      const uc = f(u1[c]), h = tops[c] + vr * (jaws[c] - tops[c]);
      const x = fr.centre[0] + uc * fr.half[0] + h * fr.down[0], y = fr.centre[1] + uc * fr.half[1] + h * fr.down[1];
      u[i] = uc; v[i] = vr;
      mx[i] = f((x - x0 + 0.5) / w.fx - 0.5); my[i] = f((y - y0 + 0.5) / w.fy - 0.5);
      inside[i] = mx[i] > 0 && mx[i] < w.gw - 1 && my[i] > 0 && my[i] < w.gh - 1 ? 1 : 0;
    }
  }
  Object.assign(w, { nu, nv, du, dv, u1, v1: v1s, u, v, mx, my, inside, backMaps: null });
}

// data (w by h) read at the places the maps give (mw by mh), between its pixels; beyond its
// edge, the edge (held) or nothing.
function remapped(cv, data, w, h, channels, mapX, mapY, mw, mh, held) {
  const src = cv.matFromArray(h, w, channels === 3 ? cv.CV_32FC3 : cv.CV_32F, data), dst = new cv.Mat();
  const xs = cv.matFromArray(mh, mw, cv.CV_32F, mapX), ys = cv.matFromArray(mh, mw, cv.CV_32F, mapY);
  cv.remap(src, dst, xs, ys, cv.INTER_LINEAR, held ? cv.BORDER_REPLICATE : cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
  const out = new Float32Array(dst.data32F);
  src.delete(); dst.delete(); xs.delete(); ys.delete();
  return out;
}

// A map of the copy, straightened.
const rect = (cv, w, a, channels) => remapped(cv, a, w.gw, w.gh, channels, w.mx, w.my, w.nu, w.nv, true);

// A straightened map, at the photo's own size (the box); 0 outside it.
function back(cv, w, a, channels) {
  if (!w.backMaps) {
    const fr = w.fr, [x0, y0] = w.box, span = TUNE.span_u, v0 = TUNE.span_v[0], bn = w.bw * w.bh;
    const j = new Float32Array(bn), i = new Float32Array(bn), across = new Float32Array(bn);
    for (let y = 0, k = 0; y < w.bh; y++) {
      for (let x = 0; x < w.bw; x++, k++) {
        const [u, v] = fr.uv(x + x0, y + y0);
        across[k] = u;
        j[k] = f((u + span) / w.du - 0.5); i[k] = f((v - v0) / w.dv - 0.5);
      }
    }
    w.backMaps = { j, i, across };
  }
  return remapped(cv, a, w.nu, w.nv, channels, w.backMaps.j, w.backMaps.i, w.bw, w.bh, false);
}

// Where the step may act, and how wide that is at each height (chin._bounds).
function boundsOf(w, top = TUNE.top) {
  const t = TUNE, n = w.nu * w.nv, within = new Uint8Array(n), wide = new Float32Array(n);
  const narrow = f(t.wide_top), more = f(t.wide - t.wide_top), from = f(top), to = f(t.foot);
  for (let i = 0; i < n; i++) {
    wide[i] = f(narrow + f(more * lip.step32(w.v[i], t.widen[0], t.widen[1])));
    within[i] = Math.abs(w.u[i]) < wide[i] && w.v[i] > from && w.v[i] < to ? 1 : 0;
  }
  return { within, wide };
}

// The beard on the straightened copy: its strands, masses, core, area (region: null when
// there is no beard) and the replaced area (paint), with the maps they were found by (chin._find).
function find(cv, w) {
  const t = TUNE, e = w.eyes, { nu, nv } = w, n = nu * nv, tiny = f(1e-4), least = f(1e-3);
  const g = rect(cv, w, w.small, 3), skin = rect(cv, w, w.skin, 1), lipless = rect(cv, w, w.valid, 1);
  const y = new Float32Array(n), on = new Float32Array(n), below = new Float32Array(n), clearEdge = f(0.98);
  for (let i = 0; i < n; i++) {
    skin[i] = f(skin[i] * w.inside[i]);
    y[i] = luma(g, 3 * i);
    on[i] = skin[i] > 0.5 && w.v[i] > 0 ? 1 : 0;
    below[i] = on[i] && lipless[i] > clearEdge ? 1 : 0;          // skin, the lips and their soft edge left out
  }
  const row = rowMean(cv, y, on, nu, nv, t.row * e, 0.0);
  // The cheeks' own grain, read the same way.
  const gn = w.gw * w.gh, ys = new Float32Array(gn), own = [];
  for (let i = 0; i < gn; i++) ys[i] = luma(w.small, 3 * i);
  const rs = blur2(cv, ys, w.gw, w.gh, 1, t.row * e, 0.0);
  for (let i = 0; i < gn; i++) if (w.onCheek[i]) own.push(f(f(ys[i] - rs[i]) / f(rs[i] + tiny)));
  const usual = f(medianOf(own)), grain = 1.4826 * f(medianOf(own.map((r) => Math.abs(f(r - usual)))));
  const limit = Math.min(Math.max(t.strand * grain, t.strand_least), t.strand_most), darker = f(-limit);
  const dark = new Uint8Array(n);                               // strands, and whatever else is dark in its row
  for (let i = 0; i < n; i++) dark[i] = f(f(y[i] - row[i]) / f(row[i] + tiny)) < darker && below[i] > 0 ? 1 : 0;
  const lines = new Uint8Array(n);                              // the line-shaped ones: hair
  for (const kernel of lineKernels(cv, Math.max(3, pyRound(t.long * e)))) {
    const src = cv.matFromArray(nv, nu, cv.CV_8U, dark), dst = new cv.Mat();
    cv.morphologyEx(src, dst, cv.MORPH_OPEN, kernel);
    const opened = dst.data;
    for (let i = 0; i < n; i++) if (opened[i]) lines[i] = 1;
    src.delete(); dst.delete(); kernel.delete();
  }
  const soft = rowMean(cv, y, on, nu, nv, t.mass_soft * e, t.mass_soft * e);
  for (let i = 0; i < n; i++) soft[i] = f(soft[i] / Math.max(tableAt(w.u[i], w.v[i]), least));
  const wideRow = rowMean(cv, soft, on, nu, nv, t.mass_row * e, 0.0);
  const mass = new Uint8Array(n), either = new Float32Array(n), massive = f(-t.mass);
  for (let i = 0; i < n; i++) {
    mass[i] = f(f(soft[i] - wideRow[i]) / f(wideRow[i] + tiny)) < massive && below[i] > 0 ? 1 : 0;
    either[i] = dark[i] || mass[i] ? 1 : 0;
  }
  const at = t.dense_at * e;
  const sure = blur2(cv, Float32Array.from(mass), nu, nv, 1, at, at), lined = blur2(cv, Float32Array.from(lines), nu, nv, 1, at, at);
  const gain = f(t.line_gain);
  for (let i = 0; i < n; i++) sure[i] = Math.min(f(sure[i] + f(gain * lined[i])), 1);
  const dense = blur2(cv, either, nu, nv, 1, at, at);
  const { within: bounds, wide } = boundsOf(w);
  const inb = new Uint8Array(n), core = new Uint8Array(n), hairy = new Uint8Array(n), coreAt = f(t.core), hairAt = f(t.hair);
  let coreCount = 0;
  for (let i = 0; i < n; i++) {
    inb[i] = bounds[i] && skin[i] > 0.5 ? 1 : 0;
    core[i] = sure[i] > coreAt && inb[i] ? 1 : 0;
    hairy[i] = dense[i] > hairAt && inb[i] ? 1 : 0;
    coreCount += core[i];
  }
  const got = {
    g, y, skin, below, strand: dark, lines, mass, sure, dense, bounds, grain, limit, coreArea: coreCount / (e * e), region: null, flank: 0.0,
  };
  if (got.coreArea < t.least) return got;
  let region = lip.grown(core, hairy, nu, nv);
  {
    const one = lip.roundKernel(cv, 1), reach = lip.roundKernel(cv, t.tail_reach * e), bridge = lip.roundKernel(cv, t.bridge * e);
    const beside = lip.morphed(cv, dark, nu, nv, one), near = lip.morphed(cv, region, nu, nv, reach);
    for (let i = 0; i < n; i++) region[i] = region[i] || (beside[i] && inb[i] && near[i]) ? 1 : 0;
    region = lip.morphed(cv, region, nu, nv, bridge, true);
    for (let i = 0; i < n; i++) region[i] = region[i] && inb[i] ? 1 : 0;
    one.delete(); reach.delete(); bridge.delete();
  }
  // Hair just outside the side bounds: a beard across the jaw.
  {
    const past = f(t.flank), from = f(t.flank_v[0]), to = f(t.flank_v[1]);
    let beside = 0, hair = 0;
    for (let i = 0; i < n; i++) {
      const au = Math.abs(w.u[i]);
      if (au >= wide[i] && au < f(wide[i] + past) && w.v[i] > from && w.v[i] < to && skin[i] > 0.5) { beside++; if (dense[i] > hairAt) hair++; }
    }
    got.flank = beside ? hair / beside : 0.0;
  }
  got.region = region;
  // The replaced area: the hair and a little past it, soft-edged. It runs on over the lip (hair lies
  // against its edge); the lip's own soft edge is applied at the photo's size.
  const grow = lip.roundKernel(cv, t.grow * e), wider = lip.morphed(cv, region, nu, nv, grow);
  grow.delete();
  const paint = lip.blurred(cv, Float32Array.from(wider), nu, nv, 1, t.soft * e);
  const reach = lip.blurred(cv, Float32Array.from(boundsOf(w, t.over_lip).within), nu, nv, 1, t.bound_edge * e);
  const someSkin = f(0.02), lipHigh = f(0.1);
  for (let i = 0; i < n; i++) {
    paint[i] = f(f(Math.min(f(paint[i] * f(1.5)), 1) * Math.min(f(reach[i] * 2), 1)) * (skin[i] > someSkin || w.v[i] < lipHigh ? 1 : 0));
  }
  got.paint = paint;
  return got;
}

// Sums as numpy adds a row up: in eights, pair by pair, so the last digits agree.
function total(a, from = 0, n = a.length) {
  if (n < 8) { let s = 0; for (let i = 0; i < n; i++) s += a[from + i]; return s; }
  if (n <= 128) {
    const r = [0, 1, 2, 3, 4, 5, 6, 7].map((k) => a[from + k]);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let k = 0; k < 8; k++) r[k] += a[from + i + k];
    let s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) s += a[from + i];
    return s;
  }
  let half = n >> 1;
  half -= half % 8;
  return total(a, from, half) + total(a, from + half, n - half);
}

// y ~ k[0] x x + k[1] x + k[2], each point's distance counted w times over (numpy's polyfit
// with w): its columns scaled to the same size, then solved by reflections.
export function parabola(x, y, w) {
  const n = x.length, a = [], b = new Float64Array(n), scale = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    a.push([x[i] * x[i] * w[i], x[i] * w[i], w[i]]);
    b[i] = y[i] * w[i];
    for (let c = 0; c < 3; c++) scale[c] += a[i][c] * a[i][c];
  }
  for (let c = 0; c < 3; c++) scale[c] = Math.sqrt(scale[c]) || 1;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) a[i][c] /= scale[c];
  for (let c = 0; c < 3; c++) {
    let norm = 0;
    for (let i = c; i < n; i++) norm += a[i][c] * a[i][c];
    norm = Math.sqrt(norm);
    if (!(norm > 0)) continue;
    const alpha = a[c][c] > 0 ? -norm : norm, v = new Float64Array(n);
    for (let i = c; i < n; i++) v[i] = a[i][c];
    v[c] -= alpha;
    let vv = 0;
    for (let i = c; i < n; i++) vv += v[i] * v[i];
    if (!(vv > 0)) continue;
    for (let k = c; k < 3; k++) {
      let dot = 0;
      for (let i = c; i < n; i++) dot += v[i] * a[i][k];
      const by = 2 * dot / vv;
      for (let i = c; i < n; i++) a[i][k] -= by * v[i];
    }
    let dot = 0;
    for (let i = c; i < n; i++) dot += v[i] * b[i];
    const by = 2 * dot / vv;
    for (let i = c; i < n; i++) b[i] -= by * v[i];
  }
  const k = [0, 0, 0];
  for (let c = 2; c >= 0; c--) {
    let s = b[c];
    for (let m = c + 1; m < 3; m++) s -= a[c][m] * k[m];
    k[c] = a[c][c] !== 0 ? s / a[c][c] : 0;
  }
  return k.map((value, c) => value / scale[c]);
}
const parabolaAt = (k, x) => (k[0] * x + k[1]) * x + k[2];

// Moves the frame's jaw line onto the photo's own edge (chin._snap): in the columns clear of hair
// (beside: a map of the straightened copy), the strongest darkening going down near v = 1; a smooth
// curve through those columns, carried under the hair; beside the hair, the edge as read. Returns
// what it found (moved: whether the frame changed; the straightened copy is then laid out anew).
function snap(cv, w, hair, beside) {
  const t = TUNE, fr = w.fr, e = w.eyes, { nu, nv } = w, tiny = f(1e-4);
  const y = blur2(cv, hair.y, nu, nv, 1, t.edge_soft[0] * e, t.edge_soft[1] * e);
  const us = [], offs = [], ws = [];
  for (let j = 0; j < nu; j++) {
    if (!(Math.abs(w.u1[j]) < t.wide + 0.15)) continue;
    const tall = (fr.jaw(w.u1[j]) - fr.top(w.u1[j])) / fr.eyes;         // this column's height from lip to jaw, in eye distances
    const reach = t.edge_reach / Math.max(tall, 1e-3), rows = [];
    for (let r = 0; r < nv; r++) if (Math.abs(w.v1[r] - 1.0) < reach) rows.push(r);
    if (rows.length < 3 || rows.some((r) => beside[r * nu + j] || hair.skin[r * nu + j] < 0.5)) continue;
    // The share lost over two rows, going down: where it is largest.
    let most = -Infinity, at = -1;
    for (const r of rows) {
      const above = y[(r - 1) * nu + j], under = y[(r + 1) * nu + j];
      const fall = r > 0 && r < nv - 1 ? f(f(above - under) / f(f(under + above) + tiny)) : 0;
      if (fall > most) { most = fall; at = r; }
    }
    if (most < t.edge_least || at === rows[0] || at === rows[rows.length - 1]) continue;
    us.push(w.u1[j]); offs.push((w.v1[at] - 1.0) * tall); ws.push(most);
  }
  const found = { columns: us.length, moved: false };
  if (us.length < t.edge_columns || Math.max(...us) - Math.min(...us) < t.edge_span) return found;
  let keep = us.map(() => true), k = null;
  const kept = (a) => a.filter((_, i) => keep[i]);
  for (let round = 0; round < 4; round++) {
    k = parabola(kept(us), kept(offs), kept(ws));
    keep = us.map((u, i) => Math.abs(offs[i] - parabolaAt(k, u)) < t.edge_out);
    if (keep.filter(Boolean).length < t.edge_columns) return found;
  }
  const ku = kept(us), ko = kept(offs), kw = kept(ws);
  k = parabola(ku, ko, kw);
  const first = Math.min(...ku), last = Math.max(...ku), usual = Math.max(medianOf(kw), 1e-9);
  const jawU = fr.jawLine.xp, jawH = Float64Array.from(fr.jawLine.fp), near = new Float64Array(ku.length), pull = new Float64Array(ku.length);
  for (let m = 0; m < jawU.length; m++) {
    const curve = parabolaAt(k, Math.min(Math.max(jawU[m], first), last));        // held past the last column read
    for (let c = 0; c < ku.length; c++) {
      const q = (jawU[m] - ku[c]) / t.edge_even;
      near[c] = Math.exp(-0.5 * (q * q)) * kw[c];
      pull[c] = near[c] * ko[c];
    }
    const seen = total(near), local = total(pull) / Math.max(seen, 1e-9);
    let trust = (seen / usual - 1.0) / (4.0 - 1.0);
    trust = trust < 0 ? 0 : trust > 1 ? 1 : trust;
    trust = trust * trust * (3.0 - 2.0 * trust);
    jawH[m] += (curve + trust * (local - curve)) * fr.eyes;
  }
  fr.jawLine = curveOf(jawU, jawH);
  straighten(w);
  return Object.assign(found, { moved: true, kept: ku.length, k, us, offs, ws });
}

// The bare chin's tone on the straightened copy, the table in the cheeks' colour it started
// from, and the clear pixels it was matched to (chin._tone).
function toneOf(cv, w, hair) {
  const t = TUNE, lt = lip.TUNE, e = w.eyes, { nu, nv } = w, n = nu * nv, tiny = f(1e-4), least = f(1e-3);
  const red = lt.red, powers = [1 + red, 1 + red, 1 - red * (0.701 / 0.299)].map(f);     // B, G, R: as lip's
  const prior = new Float32Array(3 * n), off = new Float32Array(3 * n), spanLo = f(Math.log(t.span[0])), spanHi = f(Math.log(t.span[1]));
  const either = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const base = Math.max(tableAt(w.u[i], w.v[i]), least);
    for (let ch = 0; ch < 3; ch++) {
      const j = 3 * i + ch;
      prior[j] = f(w.cheek[ch] * f(Math.pow(base, powers[ch])));
      off[j] = Math.min(Math.max(f(Math.log(f(Math.max(hair.g[j], least) / Math.max(prior[j], least)))), spanLo), spanHi);
    }
    either[i] = hair.strand[i] || hair.mass[i] ? 1 : 0;
  }
  const one = lip.roundKernel(cv, t.clear_off), away = lip.roundKernel(cv, t.first_off * e);
  const near = lip.morphed(cv, either, nu, nv, one), beard = lip.morphed(cv, hair.region, nu, nv, away);
  one.delete(); away.delete();
  const clear = new Float32Array(n), outside = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    clear[i] = f((hair.below[i] > 0 && !near[i] ? 1 : 0) * f(1 - lip.step32(hair.dense[i], t.clear_dense[0], t.clear_dense[1])));
    outside[i] = beard[i] ? 0 : clear[i];
  }
  // First from the skin well outside the beard; then the skin inside it counts where it is as light as that says.
  const first = fillIn(cv, off, outside, nu, nv, e);
  const seen = rowMean(cv, hair.y, clear, nu, nv, t.above_at * e, t.above_at * e), lit = new Float32Array(3);
  for (let i = 0; i < n; i++) {
    if (!beard[i]) continue;
    for (let ch = 0; ch < 3; ch++) lit[ch] = f(prior[3 * i + ch] * f(Math.exp(first[3 * i + ch])));
    const firstY = luma(lit, 0);
    clear[i] = f(clear[i] * lip.step32(f(f(seen[i] - firstY) / f(firstY + tiny)), t.above[0], t.above[1]));
  }
  const final = fillIn(cv, off, clear, nu, nv, e);
  for (let j = 0; j < 3 * n; j++) final[j] = f(prior[j] * f(Math.exp(final[j])));
  return { final, prior, clear };
}

// One face, changed in place in the Mat out. Returns whether anything changed. seen, if
// given, receives what was found on the way, for looking at it.
export function clearFace(cv, out, pts, mask, amount, keep, force, seen = null) {
  const t = TUNE, lt = lip.TUNE, fr = frameOf(pts);
  if (!fr.ok || !(fr.posed || force)) return false;
  const w = workOf(cv, out, fr, mask);
  if (!w) return false;
  let hair = find(cv, w);
  if (seen) Object.assign(seen, { box: w.box, grid: [w.gw, w.gh], firstSize: [w.nu, w.nv], firstMx: w.mx, firstMy: w.my, first: hair });
  if (!hair.region) return false;
  let edge;
  {
    const n = w.nu * w.nv, hairAt = f(t.hair), off = lip.roundKernel(cv, t.edge_off * w.eyes);
    const beside = lip.morphed(cv, Uint8Array.from({ length: n }, (_, i) => (hair.dense[i] > hairAt ? 1 : 0)), w.nu, w.nv, off);
    off.delete();
    edge = snap(cv, w, hair, beside);
  }
  if (edge.moved) hair = find(cv, w);                           // the same again, on the frame as moved
  if (seen) Object.assign(seen, { edge, hair, size: [w.nu, w.nv], mx: w.mx, my: w.my, jawH: fr.jawLine.fp });
  if (!hair.region) return false;
  if (hair.flank > t.flank_most && !force) return false;        // hair runs on past the bounds: a beard across the jaw
  const { nu, nv } = w, n = nu * nv, by = f(Math.min(1, amount)), paint = new Float32Array(n);
  let most = 0;
  for (let i = 0; i < n; i++) { paint[i] = f(hair.paint[i] * by); if (paint[i] > most) most = paint[i]; }
  if (!(most > 1e-3)) return false;
  const { final, prior, clear: clearSkin } = toneOf(cv, w, hair);
  const tiny = f(1e-4), change = new Float32Array(3 * n);
  {
    const toneG = rect(cv, w, w.tone, 3), hairSoft = lip.blurred(cv, Float32Array.from(hair.region), nu, nv, 1, t.soft * w.eyes);
    for (let i = 0; i < n; i++) {
      const finalY = luma(final, 3 * i), toneY = luma(toneG, 3 * i);
      // Hair goes whatever its tone; the clear skin beside it only ever gets lighter.
      const lift = Math.max(lip.step32(f(f(finalY - toneY) / f(finalY + tiny)), -lt.dark_lo, lt.dark_lo), Math.min(f(hairSoft[i] * 2), 1));
      for (let ch = 0; ch < 3; ch++) change[3 * i + ch] = f(lift * f(final[3 * i + ch] - toneG[3 * i + ch]));
    }
  }

  // At the photo's own size: the lip's edge, the grain.
  const [x0, y0, x1, y1] = w.box, { bw, bh } = w, bn = bw * bh, d = fr.eyes, k255 = f(255), box = w.pixels, W = out.cols;
  const lipsFull = lip.polygon(cv, w.outline.map((p) => [p[0] - x0, p[1] - y0]), bw, bh);
  const lipsSoft = lip.blurred(cv, Float32Array.from(lipsFull, (b) => f(b / k255)), bw, bh, 1, Math.max(0.6, t.lip_soft * d));
  const ok = new Float32Array(bn);
  for (let i = 0; i < bn; i++) ok[i] = f(1 - lipsSoft[i]);
  const paintFull = back(cv, w, paint, 1);
  if (keep) {
    const ax0 = Math.max(x0, keep.x0), ay0 = Math.max(y0, keep.y0), ax1 = Math.min(x1, keep.x0 + keep.w), ay1 = Math.min(y1, keep.y0 + keep.h);
    for (let y = ay0; y < ay1; y++) {
      for (let x = ax0; x < ax1; x++) {
        const i = (y - y0) * bw + x - x0;
        paintFull[i] = f(paintFull[i] * f(1 + Math.min(keep.data[(y - keep.y0) * keep.w + x - keep.x0], 0)));
      }
    }
  }
  const changeFull = back(cv, w, change, 3);
  const toneFull = w.scale < 1 ? lip.resized(cv, w.tone, w.gw, w.gh, 3, bw, bh, cv.INTER_LINEAR) : w.tone;
  const grain = new Float32Array(3 * bn), grainY = new Float32Array(bn), usable = new Float32Array(bn), quiet = [];
  const r2 = (lt.cheek * d) ** 2, wellSkin = f(0.9);
  for (let y = 0, i = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++, i++) {
      for (let ch = 0; ch < 3; ch++) grain[3 * i + ch] = f(box[3 * i + ch] - toneFull[3 * i + ch]);
      grainY[i] = f(luma(grain, 3 * i) * ok[i]);
      usable[i] = w.boxMask[i] > wellSkin ? ok[i] : 0;
      if (w.boxMask[i] > wellSkin && fr.cheeks.some((c) => (x + x0 - c[0]) ** 2 + (y + y0 - c[1]) ** 2 < r2)) quiet.push(Math.abs(grainY[i]));
    }
  }
  const limit = f(lt.grain_most * lip.percentile(quiet, 25));
  const sideOf = Float32Array.from(w.backMaps.across, (u) => lip.step32(f(u), -lt.seam, lt.seam));      // 0: the right cheek's half, 1: the left's
  const { borrowed, have } = lip.grainPieces(grainY, usable, bw, bh, x0, y0, fr.cheeks,
    [-1, 1].map((side) => fr.place(side * t.side, t.side_down)), sideOf, t.tile * d, t.tile_wander * d, limit);
  if (seen) Object.assign(seen, { final, prior, clear: clearSkin, paint, paintFull, changeFull, have, lipsSoft });

  const cheekY = f(f(f(w.cheek[0] * f(LUMA_B)) + f(w.cheek[1] * f(LUMA_G))) + f(w.cheek[2] * f(LUMA_R)));
  const shadeBy = f(Math.max(cheekY, 1e-3)), shadeLo = f(t.grain_shade[0]), shadeHi = f(t.grain_shade[1]), half = f(0.5), lit = new Float32Array(3);
  let changed = false;
  // The photo's bytes, looked up again: the Mats made since the top may have moved the heap, and a
  // view taken before that no longer reaches it (what is written through it is lost).
  const od = out.data;
  for (let y = 0, i = 0; y < bh; y++) {
    for (let x = 0, p = 3 * ((y + y0) * W + x0); x < bw; x++, i++, p += 3) {
      for (let ch = 0; ch < 3; ch++) lit[ch] = f(toneFull[3 * i + ch] + changeFull[3 * i + ch]);
      // The grain follows the new tone's brightness against the cheeks'.
      const g = f(borrowed[i] * Math.min(Math.max(f(luma(lit, 0) / shadeBy), shadeLo), shadeHi));
      for (let ch = 0; ch < 3; ch++) {
        const j = 3 * i + ch;
        const newSkin = f(box[j] + f(paintFull[i] * f(f(changeFull[j] + g) - grain[j])));
        const value = f(f(f(f(newSkin * ok[i]) + f(box[j] * lipsSoft[i])) * k255) + half);
        const byte = value < 0 ? 0 : value > 255 ? 255 : Math.trunc(value);
        if (byte !== od[p + ch]) { od[p + ch] = byte; changed = true; }
      }
    }
  }
  return changed;
}

// bgr (CV_8UC3) with the beard taken off each face's chin: a new Mat, or bgr itself when
// nothing changes. points: each face's landmarks or null. mask: the smoothing mask ({ x0,
// y0, w, h, data }). amount: 0..1, how much of it goes. keep: the brush's Restore edit
// ({ x0, y0, w, h, data } in -1..1), where the chin stays as it is. force: skip the
// questions whether the face is posed for it and whether the hair runs on across the jaw
// (the parity check, whose photos are not posed).
export function clear(cv, bgr, points, mask, amount = 1, keep = null, force = false) {
  if (!(amount > 0) || !points || !points.length) return bgr;
  const out = copyMat(cv, bgr);
  let changed = false;
  for (const pts of points) {
    if (pts && clearFace(cv, out, pts, mask, amount, keep, force)) changed = true;
  }
  if (changed) return out;
  out.delete();
  return bgr;
}
