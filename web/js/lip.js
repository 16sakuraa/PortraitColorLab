// A thin moustache, taken off the upper lip: clear() replaces it instead of fading it.
// The same step as the desktop app's lip.py; see the notes there. In short: the tone of a
// bare upper lip (TABLE's shape, fitted to this face from above, in the cheeks' colour,
// matched to the clear skin beside the hair) and the cheeks' own grain are put wherever
// the lip is darker than that tone. Only on a face straight on with its mouth closed; a
// bare lip and a full moustache are left as they are.
// A full, dense moustache goes too when it is asked for (clear's full), its own way
// (clearFull, the desktop's _clear_full): the hair is followed from its dense core to its
// thin ends, the tone is matched to the clear skin around the whole moustache, the grain
// is laid in blended pieces of each cheek, and the lip's edge is drawn as a smooth curve
// at the photo's own size, with the lip's own colour put over hair lying on the lip.
// The tone is worked out on a copy of the mouth at one fixed size, and only the grain
// (and a full moustache's lip edge) at the photo's own size, so a preview and the
// full-size photo come out alike.

import { copyMat } from './cv.js';
import { pyRound } from './skinmask.js';
import { softBlur, smoothstep, medianOf } from './smooth.js';

// Face Mesh points.
const UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291];
const LOWER = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61];
const MOUTH_R = 61, MOUTH_L = 291, LIP_TOP = 0, NOSE_BASE = 2, CHEEK_R = 205, CHEEK_L = 425;
const LIP_INNER_TOP = 13, LIP_INNER_FOOT = 14;      // where the lips meet: apart when the mouth is open
const LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308];   // where the upper lip ends at the mouth's line, corner to corner
// Same limits as lip.POSE: a face the step acts on is straight on, mouth closed, not smiling broadly.
export const POSE = { turn: 0.07, open: 0.09, smile: 1.0, short: 0.16 };
const EYE_R = [33, 133], EYE_L = [263, 362];
const GRID_EYES = 160, MIN_EYES = 24;
const LUMA_B = 0.114, LUMA_G = 0.587, LUMA_R = 0.299;
const f = Math.fround;

// Same table as lip.TABLE: the brightness of a bare upper lip, place by place. Rows: from
// the lip's edge up to the height of the nose's base, in eighths. Columns: from the middle
// of the lip outwards, in fifths of the way to the mouth's corner.
export const TABLE = [
  [0.85, 0.88, 0.92, 0.93, 0.88, 0.79, 0.86, 0.89, 0.91],
  [0.85, 0.88, 0.92, 0.93, 0.88, 0.79, 0.86, 0.89, 0.91],
  [0.86, 0.89, 0.93, 0.94, 0.90, 0.83, 0.87, 0.90, 0.91],
  [0.87, 0.90, 0.94, 0.96, 0.94, 0.90, 0.91, 0.92, 0.92],
  [0.90, 0.91, 0.95, 0.96, 0.96, 0.94, 0.94, 0.93, 0.94],
  [0.92, 0.92, 0.96, 0.97, 0.98, 0.97, 0.96, 0.95, 0.95],
  [0.92, 0.93, 0.95, 0.97, 0.99, 0.98, 0.96, 0.96, 0.97],
  [0.91, 0.92, 0.93, 0.94, 0.98, 0.98, 0.97, 0.98, 0.98],
  [0.87, 0.89, 0.87, 0.88, 0.94, 0.95, 0.97, 0.99, 0.99],
].map((row) => row.map(f));
const TABLE_U = 0.2, TABLE_V = 0.125;

// Same settings as lip.TUNE.
export const TUNE = {
  tone: 0.025, lip_edge: 0.008, cheek: 0.16, wide: 1.75, top: 0.82, end_top: 0.35, droop: 0.25, taper: 0.75, edge: 0.03,
  fit_wide: 2.3, fit_lo: 0.03, fit_hi: 0.95, below: 0.06, fit_rounds: 12, fit_most: 6000, drop: 0.04, red: 0.05,
  dark_lo: 0.02, dark_hi: 0.07, grow: 0.025, meet: 0.12, meet_lo: 0.85, meet_hi: 1.05, meet_prior: 0.02,
  strip_wide: 1.1, strip_lo: 0.03, strip_hi: 0.45, there_lo: 0.064, there_hi: 0.072, dense_lo: 0.33, dense_hi: 0.39,
  grain_most: 5.0, side: 0.6, side_up: 0.45, seam: 0.25,
};

// Same settings as lip.FULL, for a full moustache (clearFull); see the notes there.
export const FULL = {
  core: 0.25, core_wide: 1.15, core_lo: 0.02, core_hi: 0.65,
  hair_lo: 0.07, hair_hi: 0.18, rough: 1.6, rough_at: 0.03, mid_rough: 1.7, mid_wide: 0.6,
  tail_dark: 0.035, tail_rough: 1.3, tail_reach: 0.10, bridge: 0.012,
  wide: 2.0, top: 0.97, drop: 1.1, drop_from: 0.85, drop_to: 1.15, bound_edge: 0.04,
  grow: 0.035, soft: 0.02, clear_off: 0.05, clear_lo: -1.3, clear_hi: 1.05,
  fill: [0.5, 0.25, 0.12, 0.06], fill_lo: 0.02, fill_hi: 0.2, span: [0.80, 1.20], shape_soft: 0.05,
  tile: 0.12, tile_wander: 0.07,
  lip_soft: 0.012, lip_band: 0.6, lip_dark: [0.06, 0.14], lip_strand: 0.006, lip_reach: 0.04,
  lip_fill: [0.16, 0.08, 0.04], lip_keep: [0.80, 0.92], lip_grain: 0.7,
};
// Same as lip.WANDER: where in the cheek's patch each piece of grain comes from.
const WANDER = [[0.0, 0.0], [0.62, 0.31], [-0.45, 0.58], [0.21, -0.71], [-0.68, -0.29], [0.55, -0.52], [-0.18, 0.83],
  [0.85, -0.05], [-0.32, -0.80], [-0.90, 0.20], [0.30, 0.75]];
// The settings as numpy compares them with float32 values.
const as32 = (table) => Object.fromEntries(Object.entries(table).filter(([, value]) => typeof value === 'number').map(([name, value]) => [name, f(value)]));
const TUNE32 = as32(TUNE), FULL32 = as32(FULL);

function blurred(cv, data, w, h, channels, sigma) {
  const src = cv.matFromArray(h, w, channels === 3 ? cv.CV_32FC3 : cv.CV_32F, data);
  const dst = softBlur(cv, src, sigma);
  const out = new Float32Array(dst.data32F);
  src.delete(); dst.delete();
  return out;
}

function resized(cv, data, w, h, channels, tw, th, how) {
  const src = cv.matFromArray(h, w, channels === 3 ? cv.CV_32FC3 : cv.CV_32F, data), dst = new cv.Mat();
  cv.resize(src, dst, new cv.Size(tw, th), 0, 0, how);
  const out = new Float32Array(dst.data32F);
  src.delete(); dst.delete();
  return out;
}

// The q-th percentile (0..100) of values, between the two nearest (as numpy's default).
export function percentile(values, q) {
  const s = Float64Array.from(values).sort();
  if (!s.length) return 0;
  const pos = (s.length - 1) * q / 100, lo = Math.floor(pos), hi = Math.min(s.length - 1, lo + 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

// fp at x, between the points xp (increasing), held at the ends (numpy's interp).
function interp(x, xp, fp) {
  const n = xp.length;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  let k = 1;
  while (xp[k] < x) k++;
  const span = xp[k] - xp[k - 1];
  return span > 0 ? fp[k - 1] + (fp[k] - fp[k - 1]) * (x - xp[k - 1]) / span : fp[k];
}

// One face's lip frame, in photo pixels: u across (-1 and 1 at the mouth's corners), v
// from the upper lip's edge (0) up to the height of the nose's base (1).
export function frameOf(pts) {
  const at = (i) => [pts[2 * i], pts[2 * i + 1]];
  const mean = (ids) => [(pts[2 * ids[0]] + pts[2 * ids[1]]) / 2, (pts[2 * ids[0] + 1] + pts[2 * ids[1] + 1]) / 2];
  const er = mean(EYE_R), el = mean(EYE_L), r = at(MOUTH_R), l = at(MOUTH_L), nose = at(NOSE_BASE), top = at(LIP_TOP);
  const eyes = Math.hypot(el[0] - er[0], el[1] - er[1]);
  const centre = [(r[0] + l[0]) / 2, (r[1] + l[1]) / 2], half = [(l[0] - r[0]) / 2, (l[1] - r[1]) / 2];
  const half2 = half[0] * half[0] + half[1] * half[1], up = [nose[0] - top[0], nose[1] - top[1]];
  const edge = UPPER.map((i) => { const p = at(i); return { u: ((p[0] - centre[0]) * half[0] + (p[1] - centre[1]) * half[1]) / Math.max(half2, 1e-9), x: p[0], y: p[1] }; });
  const order = edge.map((_, i) => i).sort((a, b) => edge[a].u - edge[b].u || a - b);
  const edgeU = order.map((i) => edge[i].u), edgeX = order.map((i) => edge[i].x), edgeY = order.map((i) => edge[i].y);
  const uv = (x, y) => {
    const u = ((x - centre[0]) * half[0] + (y - centre[1]) * half[1]) / half2, inside = Math.min(1, Math.max(-1, u));
    const ex = interp(inside, edgeU, edgeX) + (u - inside) * half[0], ey = interp(inside, edgeU, edgeY) + (u - inside) * half[1];
    const ux = nose[0] + u * half[0] - ex, uy = nose[1] + u * half[1] - ey;
    return [u, ((x - ex) * ux + (y - ey) * uy) / (ux * ux + uy * uy + 1e-6)];
  };
  const place = (u, v) => {
    const inside = Math.min(1, Math.max(-1, u));
    const ex = interp(inside, edgeU, edgeX) + (u - inside) * half[0], ey = interp(inside, edgeU, edgeY) + (u - inside) * half[1];
    const nx = nose[0] + u * half[0], ny = nose[1] + u * half[1];
    return [ex + v * (nx - ex), ey + v * (ny - ey)];
  };
  const ok = eyes >= MIN_EYES && half2 > 1 && up[0] * up[0] + up[1] * up[1] > 1;
  let posed = false;
  if (ok) {
    const ax = el[0] - er[0], ay = el[1] - er[1];
    const turn = ((nose[0] - (er[0] + el[0]) / 2) * ax + (nose[1] - (er[1] + el[1]) / 2) * ay) / (ax * ax + ay * ay);
    const a = at(LIP_INNER_TOP), b = at(LIP_INNER_FOOT), apart = Math.hypot(a[0] - b[0], a[1] - b[1]) / eyes;
    posed = Math.abs(turn) <= POSE.turn && apart <= POSE.open && 2 * Math.sqrt(half2) / eyes <= POSE.smile
      && Math.hypot(up[0], up[1]) / eyes >= POSE.short;
  }
  return {
    eyes, centre, half, up, uv, place, lips: [...UPPER, ...LOWER].map(at), inner: LIP_INNER.map(at),
    cheeks: [at(CHEEK_R), at(CHEEK_L)], ok, posed,
  };
}

// TABLE at a place in the frame, between its entries and held at its edges.
export function tableAt(u, v) {
  const rows = TABLE.length, cols = TABLE[0].length;
  const a = Math.min(Math.max(Math.abs(u) / TABLE_U, 0), cols - 1), b = Math.min(Math.max(v / TABLE_V, 0), rows - 1);
  const a0 = Math.min(Math.floor(a), cols - 2), b0 = Math.min(Math.floor(b), rows - 2), fa = f(a - a0), fb = f(b - b0);
  const top = f(f(TABLE[b0][a0] * f(1 - fa)) + f(TABLE[b0][a0 + 1] * fa));
  const low = f(f(TABLE[b0 + 1][a0] * f(1 - fa)) + f(TABLE[b0 + 1][a0 + 1] * fa));
  return f(f(top * f(1 - fb)) + f(low * fb));
}

// seen ~ a + b u, from above: points under the line count little (TUNE.below).
export function fitFromAbove(seen, u) {
  const n = seen.length, w = new Float64Array(n).fill(1);
  let a = 0, b = 0;
  for (let round = 0; round < TUNE.fit_rounds; round++) {
    let sw = 0, su = 0, suu = 0, sy = 0, suy = 0;
    for (let i = 0; i < n; i++) { sw += w[i]; su += w[i] * u[i]; suu += w[i] * u[i] * u[i]; sy += w[i] * seen[i]; suy += w[i] * u[i] * seen[i]; }
    const det = sw * suu - su * su;
    if (Math.abs(det) < 1e-9) { a = sy / Math.max(sw, 1e-9); b = 0; } else { a = (sy * suu - su * suy) / det; b = (sw * suy - su * sy) / det; }
    for (let i = 0; i < n; i++) w[i] = seen[i] - (a + b * u[i]) < 0 ? TUNE.below : 1;
  }
  return [a, b];
}

// a read from dx, dy away, repeating its edge beyond it (one channel).
function movedHeld(a, w, h, dx, dy) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(h - 1, Math.max(0, y + dy)) * w;
    for (let x = 0; x < w; x++) out[y * w + x] = a[sy + Math.min(w - 1, Math.max(0, x + dx))];
  }
  return out;
}

// One face, changed in place in the Mat out. Returns whether anything changed.
function clearFace(cv, out, pts, mask, amount, keep, force, full = false) {
  const t = TUNE, fr = frameOf(pts);
  if (!fr.ok || !(fr.posed || force)) return false;
  if (full && force) return clearFull(cv, out, fr, mask, amount, keep);
  const W = out.cols, H = out.rows, d = fr.eyes;
  // The box: the fitted width, lip to nose, and both cheeks, with room for the widest blur.
  const xs = [], ys = [];
  for (const su of [-1, 1]) for (const v of [-0.8, 1.2]) { const p = fr.place(su * t.fit_wide, v); xs.push(p[0]); ys.push(p[1]); }
  for (const c of fr.cheeks) for (const s of [-1, 1]) { xs.push(c[0] + s * t.cheek * d); ys.push(c[1] + s * t.cheek * d); }
  const pad = 3 * t.meet * d + 2;
  const x0 = Math.max(0, Math.floor(Math.min(...xs) - pad)), x1 = Math.min(W, Math.ceil(Math.max(...xs) + pad) + 1);
  const y0 = Math.max(0, Math.floor(Math.min(...ys) - pad)), y1 = Math.min(H, Math.ceil(Math.max(...ys) + pad) + 1);
  if (x1 - x0 < 8 || y1 - y0 < 8) return false;
  const bw = x1 - x0, bh = y1 - y0, bn = bw * bh, scale = Math.min(1, GRID_EYES / d);
  const gw = scale < 1 ? Math.max(8, pyRound(bw * scale)) : bw, gh = scale < 1 ? Math.max(8, pyRound(bh * scale)) : bh, gn = gw * gh;
  const fx = bw / gw, fy = bh / gh, eyes = d / ((fx + fy) / 2);

  const k255 = f(255);
  let od = out.data;          // for reading here and now: making a Mat further down can move the heap this view looks at
  const box = new Float32Array(3 * bn), boxMask = new Float32Array(bn);
  for (let y = 0; y < bh; y++) {
    for (let x = 0, i = y * bw, p = 3 * ((y + y0) * W + x0); x < bw; x++, i++, p += 3) {
      box[3 * i] = f(od[p] / k255); box[3 * i + 1] = f(od[p + 1] / k255); box[3 * i + 2] = f(od[p + 2] / k255);
    }
  }
  for (let y = Math.max(y0, mask.y0); y < Math.min(y1, mask.y0 + mask.h); y++) {
    for (let x = Math.max(x0, mask.x0); x < Math.min(x1, mask.x0 + mask.w); x++) {
      boxMask[(y - y0) * bw + x - x0] = mask.data[(y - mask.y0) * mask.w + x - mask.x0];
    }
  }
  const small = scale < 1 ? resized(cv, box, bw, bh, 3, gw, gh, cv.INTER_AREA) : box;
  const skin = scale < 1 ? resized(cv, boxMask, bw, bh, 1, gw, gh, cv.INTER_AREA) : boxMask;

  // The photo's place for each grid pixel, and its place in the lip's frame.
  const u = new Float32Array(gn), v = new Float32Array(gn), px = new Float64Array(gn), py = new Float64Array(gn);
  for (let y = 0, i = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++, i++) {
      px[i] = x0 + (x + 0.5) * fx - 0.5; py[i] = y0 + (y + 0.5) * fy - 0.5;
      const uv = fr.uv(px[i], py[i]);
      u[i] = uv[0]; v[i] = uv[1];
    }
  }

  const lipsMat = cv.Mat.zeros(gh, gw, cv.CV_8U);
  {
    const poly = fr.lips.flatMap((p) => [pyRound((p[0] - x0 + 0.5) / fx - 0.5), pyRound((p[1] - y0 + 0.5) / fy - 0.5)]);
    const m = cv.matFromArray(fr.lips.length, 1, cv.CV_32SC2, poly), polys = new cv.MatVector();
    polys.push_back(m);
    cv.fillPoly(lipsMat, polys, new cv.Scalar(255));
    m.delete(); polys.delete();
  }
  const lips = new Float32Array(gn);
  for (let i = 0; i < gn; i++) lips[i] = f(lipsMat.data[i] / k255);
  lipsMat.delete();
  const lipsSoft = blurred(cv, lips, gw, gh, 1, Math.max(0.6, t.lip_edge * eyes));
  const valid = new Float32Array(gn);
  for (let i = 0; i < gn; i++) valid[i] = f(1 - lipsSoft[i]);

  const sigma = t.tone * eyes, lit = new Float32Array(3 * gn);
  for (let i = 0; i < gn; i++) { lit[3 * i] = f(small[3 * i] * valid[i]); lit[3 * i + 1] = f(small[3 * i + 1] * valid[i]); lit[3 * i + 2] = f(small[3 * i + 2] * valid[i]); }
  const toneSum = blurred(cv, lit, gw, gh, 3, sigma), toneDen = blurred(cv, valid, gw, gh, 1, sigma);
  const tone = new Float32Array(3 * gn), toneY = new Float32Array(gn), tiny = f(1e-4);
  const luma = (a, j) => f(f(f(a[j] * f(LUMA_B)) + f(a[j + 1] * f(LUMA_G))) + f(a[j + 2] * f(LUMA_R)));
  for (let i = 0; i < gn; i++) {
    const den = f(toneDen[i] + tiny);
    tone[3 * i] = f(toneSum[3 * i] / den); tone[3 * i + 1] = f(toneSum[3 * i + 1] / den); tone[3 * i + 2] = f(toneSum[3 * i + 2] / den);
    toneY[i] = luma(tone, 3 * i);
  }

  // The cheeks: their colour, for the new skin's.
  const cheek = [0, 0, 0], r2 = (t.cheek * d) ** 2;
  for (const c of fr.cheeks) {
    const found = [[], [], []];
    for (let i = 0; i < gn; i++) {
      if ((px[i] - c[0]) ** 2 + (py[i] - c[1]) ** 2 < r2 && skin[i] > 0.9) { found[0].push(tone[3 * i]); found[1].push(tone[3 * i + 1]); found[2].push(tone[3 * i + 2]); }
    }
    if (found[0].length < 30) return false;
    for (let ch = 0; ch < 3; ch++) cheek[ch] += medianOf(found[ch]);
  }
  for (let ch = 0; ch < 3; ch++) cheek[ch] = f(f(cheek[ch]) / f(2));

  // The bare lip's level on this face: the table's shape, fitted from above.
  const shape = new Float32Array(gn), at = [];
  for (let i = 0; i < gn; i++) {
    shape[i] = tableAt(u[i], v[i]);
    if (Math.abs(u[i]) < t.fit_wide && v[i] > t.fit_lo && v[i] < t.fit_hi && valid[i] > 0.98 && skin[i] > 0.5) at.push(i);
  }
  if (at.length < 200) return false;
  const step = Math.max(1, Math.floor(at.length / t.fit_most)), seen = [], seenU = [];
  for (let k = 0; k < at.length; k += step) {
    const i = at[k];
    seen.push(f(Math.log(f(Math.max(toneY[i], f(1e-3)) / Math.max(shape[i], f(1e-3))))));
    seenU.push(u[i]);
  }
  const [a, b] = fitFromAbove(seen, seenU);
  const e = t.red, powers = [1 + e, 1 + e, 1 - e * (0.701 / 0.299)].map(f);
  const cheekY = f(f(f(cheek[0] * f(LUMA_B)) + f(cheek[1] * f(LUMA_G))) + f(cheek[2] * f(LUMA_R)));
  const shadeBy = f((1 - t.drop) / Math.max(cheekY, 1e-3));
  const target = new Float32Array(3 * gn), targetY = new Float32Array(gn), dark = new Float32Array(gn);
  for (let i = 0; i < gn; i++) {
    const level = f(shape[i] * f(Math.exp(f(f(a) + f(f(b) * u[i])))));
    const shade = Math.max(f(level * shadeBy), f(1e-3));
    for (let ch = 0; ch < 3; ch++) target[3 * i + ch] = f(cheek[ch] * f(Math.pow(shade, powers[ch])));
    targetY[i] = luma(target, 3 * i);
    dark[i] = f(f(targetY[i] - toneY[i]) / f(targetY[i] + tiny));
  }

  // The zone: full to the lip's edge and past the moustache's ends, fading towards the nose and at the ends.
  const zoneHard = new Float32Array(gn), past = new Float32Array(gn);
  for (let i = 0; i < gn; i++) {
    past[i] = Math.min(Math.max(f(f(Math.abs(u[i]) - 1) / f(t.wide - 1)), 0), 1);
    zoneHard[i] = Math.abs(u[i]) < t.wide && v[i] > -t.droop * past[i] && v[i] < t.top + (t.end_top - t.top) * past[i] ? 1 : 0;
  }
  const zone = blurred(cv, zoneHard, gw, gh, 1, t.edge * eyes);
  for (let i = 0; i < gn; i++) {
    zone[i] = f(f(f(Math.min(f(zone[i] * 2), 1) * valid[i]) * (skin[i] > 0.02 ? 1 : 0)) * f(1 - smoothstep(past[i], t.taper, 1)));
  }

  // Is there a moustache, and is it one to keep? The strip just above the lip says.
  let there = 1;
  if (!force) {
    const values = [];
    let sum = 0;
    for (let i = 0; i < gn; i++) {
      if (valid[i] > 0.98 && skin[i] > 0.5 && Math.abs(u[i]) < t.strip_wide && v[i] > t.strip_lo && v[i] < t.strip_hi) { values.push(dark[i]); sum += dark[i]; }
    }
    if (values.length < 30) return false;
    const dense = smoothstep(f(percentile(values, 90)), t.dense_lo, t.dense_hi);
    there = smoothstep(f(sum / values.length), t.there_lo, t.there_hi);
    if (full && there > 0 && dense > 0) return clearFull(cv, out, fr, mask, amount, keep);   // a full one, and asked for: it goes whole, its own way
    there = there * (1 - dense);
    if (!(there > 0)) return false;
  }

  // The new tone meets the clear skin beside the hair.
  const hairy = new Float32Array(gn), clearSkin = new Float32Array(gn), offSkin = new Float32Array(gn), seed = new Float32Array(gn);
  for (let i = 0; i < gn; i++) {
    hairy[i] = f(smoothstep(dark[i], t.dark_lo, t.dark_hi));
    clearSkin[i] = f(f(f(1 - hairy[i]) * valid[i]) * (skin[i] > 0.5 ? 1 : 0));
    const off = Math.min(Math.max(f(toneY[i] / f(targetY[i] + tiny)), f(t.meet_lo)), f(t.meet_hi));
    offSkin[i] = f(off * clearSkin[i]);
    seed[i] = f(hairy[i] * zone[i]);
  }
  const reach = t.meet * eyes, prior = f(t.meet_prior);
  const meetNum = blurred(cv, offSkin, gw, gh, 1, reach), meetDen = blurred(cv, clearSkin, gw, gh, 1, reach);
  const grown = blurred(cv, seed, gw, gh, 1, t.grow * eyes);
  const hair = new Float32Array(gn), change = new Float32Array(3 * gn), by = f(there * Math.min(1, amount));
  let most = 0;
  for (let i = 0; i < gn; i++) {
    const meet = f(f(meetNum[i] + prior) / f(meetDen[i] + prior));
    const fin = [f(target[3 * i] * meet), f(target[3 * i + 1] * meet), f(target[3 * i + 2] * meet)];
    const finY = f(f(f(fin[0] * f(LUMA_B)) + f(fin[1] * f(LUMA_G))) + f(fin[2] * f(LUMA_R)));
    hair[i] = f(f(f(Math.min(f(grown[i] * 2), 1) * zone[i])) * by);
    if (hair[i] > most) most = hair[i];
    // Only ever towards lighter: where the skin is already as light as a bare lip, its tone stays.
    const lift = f(smoothstep(f(f(finY - toneY[i]) / f(finY + tiny)), -t.dark_lo, t.dark_lo));
    for (let ch = 0; ch < 3; ch++) change[3 * i + ch] = f(lift * f(fin[ch] - tone[3 * i + ch]));
  }
  if (!(most > 1e-3)) return false;

  // At the photo's own size: the grain.
  const up = (a_, channels) => (scale < 1 ? resized(cv, a_, gw, gh, channels, bw, bh, cv.INTER_LINEAR) : a_);
  let paint = up(hair, 1);
  const ok = up(valid, 1), toneUp = up(tone, 3), changeUp = up(change, 3), uUp = up(u, 1);
  if (keep) {
    const ax0 = Math.max(x0, keep.x0), ay0 = Math.max(y0, keep.y0), ax1 = Math.min(x1, keep.x0 + keep.w), ay1 = Math.min(y1, keep.y0 + keep.h);
    if (ax1 > ax0 && ay1 > ay0) {
      paint = new Float32Array(paint);
      for (let y = ay0; y < ay1; y++) {
        for (let x = ax0; x < ax1; x++) {
          const i = (y - y0) * bw + x - x0;
          paint[i] = f(paint[i] * f(1 + Math.min(keep.data[(y - keep.y0) * keep.w + x - keep.x0], 0)));
        }
      }
    }
  }
  const grain = new Float32Array(3 * bn), grainY = new Float32Array(bn), usable = new Float32Array(bn), quiet = [];
  for (let y = 0, i = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++, i++) {
      for (let ch = 0; ch < 3; ch++) grain[3 * i + ch] = f(f(box[3 * i + ch] - toneUp[3 * i + ch]) * ok[i]);
      grainY[i] = luma(grain, 3 * i);
      usable[i] = boxMask[i] > 0.9 ? ok[i] : 0;
      if (boxMask[i] > 0.9 && fr.cheeks.some((c) => (x + x0 - c[0]) ** 2 + (y + y0 - c[1]) ** 2 < r2)) quiet.push(Math.abs(grainY[i]));
    }
  }
  const limit = f(t.grain_most * percentile(quiet, 25));
  const borrowed = new Float32Array(bn), have = new Float32Array(bn);
  fr.cheeks.forEach((c, k) => {
    const side = k === 0 ? -1 : 1;
    const hx = fr.centre[0] + side * t.side * fr.half[0] + t.side_up * fr.up[0], hy = fr.centre[1] + side * t.side * fr.half[1] + t.side_up * fr.up[1];
    const dx = pyRound(c[0] - hx), dy = pyRound(c[1] - hy);
    const g = movedHeld(grainY, bw, bh, dx, dy), s = movedHeld(usable, bw, bh, dx, dy);
    for (let y = 0, i = 0; y < bh; y++) {
      const inY = y + dy >= 0 && y + dy < bh;
      for (let x = 0; x < bw; x++, i++) {
        const sideOf = f(smoothstep(uUp[i], -t.seam, t.seam)), share = k === 0 ? f(1 - sideOf) : sideOf;
        borrowed[i] = f(borrowed[i] + f(share * g[i]));
        if (inY && x + dx >= 0 && x + dx < bw) have[i] = f(have[i] + f(share * s[i]));
      }
    }
  });
  let changed = false;
  const half = f(0.5);
  od = out.data;              // looked up again: a view taken before the Mats above were made may no longer reach the photo
  for (let y = 0, i = 0; y < bh; y++) {
    for (let x = 0, p = 3 * ((y + y0) * W + x0); x < bw; x++, i++, p += 3) {
      let g = f(borrowed[i] * have[i]);                       // no skin to borrow from: left smooth
      g = f(g * Math.min(1, f(limit / f(Math.abs(g) + f(1e-6)))));
      for (let ch = 0; ch < 3; ch++) {
        const j = 3 * i + ch;
        const value = f(f(f(box[j] + f(paint[i] * f(f(changeUp[j] + g) - grain[j]))) * k255) + half);
        const byte = value < 0 ? 0 : value > 255 ? 255 : Math.trunc(value);
        if (byte !== od[p + ch]) { od[p + ch] = byte; changed = true; }
      }
    }
  }
  return changed;
}

// smoothstep as numpy works it out on float32 values: here the result is cut into yes and
// no further on, so it has to come out the same on both sides.
function step32(x, lo, hi) {
  let s = f(f(x - f(lo)) / f(hi - lo));
  s = s < 0 ? 0 : s > 1 ? 1 : s;
  return f(f(s * s) * f(3 - f(2 * s)));
}

// A smooth open curve through the points (Catmull-Rom), both ends included (lip._spline).
function spline(points, steps = 8) {
  const n = points.length, q = [points[0], ...points, points[n - 1]], out = [];
  for (let i = 1; i < q.length - 2; i++) {
    const p0 = q[i - 1], p1 = q[i], p2 = q[i + 1], p3 = q[i + 2];
    for (let j = 0; j < steps; j++) {
      const s = j / steps;
      out.push([0, 1].map((k) => 0.5 * ((2.0 * p1[k]) + (p2[k] - p0[k]) * s + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * s * s
        + (3.0 * p1[k] - p0[k] - 3.0 * p2[k] + p3[k]) * s * s * s)));
    }
  }
  out.push(points[n - 1]);
  return out;
}

// 255 inside the polygon through these points, rounded as numpy rounds, on a grid w by h (bytes).
function polygon(cv, points, w, h) {
  const mat = cv.Mat.zeros(h, w, cv.CV_8U);
  const m = cv.matFromArray(points.length, 1, cv.CV_32SC2, points.flatMap((p) => [pyRound(p[0]), pyRound(p[1])])), polys = new cv.MatVector();
  polys.push_back(m);
  cv.fillPoly(mat, polys, new cv.Scalar(255));
  const out = new Uint8Array(mat.data);
  mat.delete(); m.delete(); polys.delete();
  return out;
}

const roundKernel = (cv, radius) => {
  const r = Math.max(1, pyRound(radius));
  return cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(2 * r + 1, 2 * r + 1));
};

// Bytes grown by the kernel, or with gaps smaller than it closed.
function morphed(cv, data, w, h, kernel, close = false) {
  const src = cv.matFromArray(h, w, cv.CV_8U, data), dst = new cv.Mat();
  if (close) cv.morphologyEx(src, dst, cv.MORPH_CLOSE, kernel);
  else cv.dilate(src, dst, kernel);
  const out = new Uint8Array(dst.data);
  src.delete(); dst.delete();
  return out;
}

// seed spread through within (both 0 or 1), neighbour to neighbour, all eight, until it
// stops, or after as many rounds as the desktop's takes at most (lip._grown).
function grown(seed, within, w, h) {
  const out = new Uint8Array(w * h);
  let edge = [];
  for (let i = 0; i < w * h; i++) if (seed[i] && within[i]) { out[i] = 1; edge.push(i); }
  for (let round = 0; round < w + h && edge.length; round++) {
    const next = [];
    for (const i of edge) {
      const x = i % w, y = (i - x) / w;
      for (let ny = Math.max(0, y - 1); ny <= Math.min(h - 1, y + 1); ny++) {
        for (let nx = Math.max(0, x - 1); nx <= Math.min(w - 1, x + 1); nx++) {
          const j = ny * w + nx;
          if (!out[j] && within[j]) { out[j] = 1; next.push(j); }
        }
      }
    }
    edge = next;
  }
  return out;
}

// data (three values a pixel), known where weight is 1, carried everywhere: the widest
// radius first, each finer one taking over where enough of what it sees is known (lip._fill_in).
function fillIn(cv, data, weight, w, h, radii, unit, lo, hi) {
  const n = w * h, tiny = f(1e-6), weighted = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let ch = 0; ch < 3; ch++) weighted[3 * i + ch] = f(data[3 * i + ch] * weight[i]);
  let out = null;
  for (const r of [...radii].sort((a, b) => b - a)) {
    const seen = blurred(cv, weight, w, h, 1, r * unit), here = blurred(cv, weighted, w, h, 3, r * unit);
    for (let i = 0; i < n; i++) {
      const den = f(seen[i] + tiny);
      for (let ch = 0; ch < 3; ch++) here[3 * i + ch] = f(here[3 * i + ch] / den);
    }
    if (!out) { out = here; continue; }
    for (let i = 0; i < n; i++) {
      const c = step32(seen[i], lo, hi), rest = f(1 - c);
      for (let ch = 0; ch < 3; ch++) out[3 * i + ch] = f(f(out[3 * i + ch] * rest) + f(here[3 * i + ch] * c));
    }
  }
  return out;
}

// For one layer of the grain's pieces, along one side of the box: which piece each
// column (or row) lies in, where inside it (-0.5..0.5), and that place's part of the weight.
function pieces(n, first, origin, size, shift) {
  const piece = new Float64Array(n), inside = new Float64Array(n), wave = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const a = (k + first - origin) / size + shift;
    piece[k] = Math.floor(a + 0.5);
    inside[k] = a - piece[k];
    wave[k] = Math.cos(Math.PI * inside[k]);
  }
  return { piece, inside, wave };
}
const LAYERS = [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]];     // four layers of pieces, each half a piece along

// A full moustache on one face (a frameOf), changed in place in the Mat out: the desktop's
// lip._clear_full, step by step. Returns whether anything changed. seen, if given,
// receives what was found on the way (the box, the hair, the replaced area...), for
// looking at it.
export function clearFull(cv, out, fr, mask, amount, keep, seen = null) {
  const t = TUNE, hv = FULL, t32 = TUNE32, h32 = FULL32;
  const W = out.cols, H = out.rows, d = fr.eyes;
  // The box: as the thin path's, and further down beside the mouth for the moustache's ends.
  const xs = [], ys = [];
  for (const su of [-1, 1]) for (const v of [hv.clear_lo - 0.1, 1.2]) { const p = fr.place(su * t.fit_wide, v); xs.push(p[0]); ys.push(p[1]); }
  for (const c of fr.cheeks) for (const s of [-1, 1]) { xs.push(c[0] + s * t.cheek * d); ys.push(c[1] + s * t.cheek * d); }
  const pad = 3 * t.meet * d + 2;
  const x0 = Math.max(0, Math.floor(Math.min(...xs) - pad)), x1 = Math.min(W, Math.ceil(Math.max(...xs) + pad) + 1);
  const y0 = Math.max(0, Math.floor(Math.min(...ys) - pad)), y1 = Math.min(H, Math.ceil(Math.max(...ys) + pad) + 1);
  if (x1 - x0 < 8 || y1 - y0 < 8) return false;
  const bw = x1 - x0, bh = y1 - y0, bn = bw * bh, scale = Math.min(1, GRID_EYES / d);
  const gw = scale < 1 ? Math.max(8, pyRound(bw * scale)) : bw, gh = scale < 1 ? Math.max(8, pyRound(bh * scale)) : bh, gn = gw * gh;
  const fx = bw / gw, fy = bh / gh, eyes = d / ((fx + fy) / 2);

  const k255 = f(255), tiny = f(1e-4), least = f(1e-3);
  const luma = (a, j) => f(f(f(a[j] * f(LUMA_B)) + f(a[j + 1] * f(LUMA_G))) + f(a[j + 2] * f(LUMA_R)));
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
  const small = scale < 1 ? resized(cv, box, bw, bh, 3, gw, gh, cv.INTER_AREA) : box;
  const skin = scale < 1 ? resized(cv, boxMask, bw, bh, 1, gw, gh, cv.INTER_AREA) : boxMask;

  const u = new Float32Array(gn), v = new Float32Array(gn), px = new Float64Array(gn), py = new Float64Array(gn);
  for (let y = 0, i = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++, i++) {
      px[i] = x0 + (x + 0.5) * fx - 0.5; py[i] = y0 + (y + 0.5) * fy - 0.5;
      const uv = fr.uv(px[i], py[i]);
      u[i] = uv[0]; v[i] = uv[1];
    }
  }

  // The lips, outlined smoothly: here on the copy for the tone, further down at the photo's size for the edge.
  const upper = spline(fr.lips.slice(0, UPPER.length)), lower = spline(fr.lips.slice(UPPER.length));
  const outline = [...upper, ...lower.slice(1, -1)];
  const lips = polygon(cv, outline.map((p) => [(p[0] - x0 + 0.5) / fx - 0.5, (p[1] - y0 + 0.5) / fy - 0.5]), gw, gh);
  const valid = blurred(cv, Float32Array.from(lips, (b) => f(b / k255)), gw, gh, 1, Math.max(0.6, t.lip_edge * eyes));
  for (let i = 0; i < gn; i++) valid[i] = f(1 - valid[i]);

  const sigma = t.tone * eyes, lit = new Float32Array(3 * gn);
  for (let i = 0; i < gn; i++) { lit[3 * i] = f(small[3 * i] * valid[i]); lit[3 * i + 1] = f(small[3 * i + 1] * valid[i]); lit[3 * i + 2] = f(small[3 * i + 2] * valid[i]); }
  const toneSum = blurred(cv, lit, gw, gh, 3, sigma), toneDen = blurred(cv, valid, gw, gh, 1, sigma);
  const tone = new Float32Array(3 * gn), toneY = new Float32Array(gn);
  for (let i = 0; i < gn; i++) {
    const den = f(toneDen[i] + tiny);
    tone[3 * i] = f(toneSum[3 * i] / den); tone[3 * i + 1] = f(toneSum[3 * i + 1] / den); tone[3 * i + 2] = f(toneSum[3 * i + 2] / den);
    toneY[i] = luma(tone, 3 * i);
  }

  const cheek = [0, 0, 0], r2 = (t.cheek * d) ** 2, onCheek = new Uint8Array(gn), wellSkin = f(0.9);
  for (const c of fr.cheeks) {
    const found = [[], [], []];
    for (let i = 0; i < gn; i++) {
      if ((px[i] - c[0]) ** 2 + (py[i] - c[1]) ** 2 < r2 && skin[i] > wellSkin) {
        found[0].push(tone[3 * i]); found[1].push(tone[3 * i + 1]); found[2].push(tone[3 * i + 2]);
        onCheek[i] = 1;
      }
    }
    if (found[0].length < 30) return false;
    for (let ch = 0; ch < 3; ch++) cheek[ch] += medianOf(found[ch]);
  }
  for (let ch = 0; ch < 3; ch++) cheek[ch] = f(f(cheek[ch]) / f(2));

  // A first reading, as the thin path's (the table fitted from above), to find the hair by.
  const shape = new Float32Array(gn), at = [], clearEdge = f(0.98);
  for (let i = 0; i < gn; i++) {
    shape[i] = tableAt(u[i], v[i]);
    if (Math.abs(u[i]) < t32.fit_wide && v[i] > t32.fit_lo && v[i] < t32.fit_hi && valid[i] > clearEdge && skin[i] > 0.5) at.push(i);
  }
  if (at.length < 200) return false;
  const step = Math.max(1, Math.floor(at.length / t.fit_most)), fitted = [], fittedU = [];
  for (let k = 0; k < at.length; k += step) {
    const i = at[k];
    fitted.push(f(Math.log(f(Math.max(toneY[i], least) / Math.max(shape[i], least)))));
    fittedU.push(u[i]);
  }
  const [a, b] = fitFromAbove(fitted, fittedU);
  const e = t.red, powers = [1 + e, 1 + e, 1 - e * (0.701 / 0.299)].map(f);
  const cheekY = f(f(f(cheek[0] * f(LUMA_B)) + f(cheek[1] * f(LUMA_G))) + f(cheek[2] * f(LUMA_R)));
  const shadeBy = f((1 - t.drop) / Math.max(cheekY, 1e-3));
  const dark = new Float32Array(gn), shaded = new Float32Array(3);
  for (let i = 0; i < gn; i++) {
    const level = f(shape[i] * f(Math.exp(f(f(a) + f(f(b) * u[i])))));
    const shade = Math.max(f(level * shadeBy), least);
    for (let ch = 0; ch < 3; ch++) shaded[ch] = f(cheek[ch] * f(Math.pow(shade, powers[ch])));
    const firstY = luma(shaded, 0);
    dark[i] = f(f(firstY - toneY[i]) / f(firstY + tiny));
  }

  // How rough the skin is, against the cheeks': strands are rough, shade is not.
  const rough = new Float32Array(gn);
  {
    const squared = new Float32Array(gn);
    for (let i = 0; i < gn; i++) { const fine = f(f(luma(small, 3 * i) - toneY[i]) * valid[i]); squared[i] = f(fine * fine); }
    const size = hv.rough_at * eyes, sum = blurred(cv, squared, gw, gh, 1, size), share = blurred(cv, valid, gw, gh, 1, size), cheeks = [];
    for (let i = 0; i < gn; i++) {
      rough[i] = f(Math.sqrt(f(Math.max(sum[i], 0) / f(share[i] + tiny))));
      if (onCheek[i]) cheeks.push(rough[i]);
    }
    const by = f(Math.max(f(medianOf(cheeks)), 1e-5));
    for (let i = 0; i < gn; i++) rough[i] = f(rough[i] / by);
  }

  // Where: from the dense core, through hair, to its thin ends, within bounds.
  const bounds = new Uint8Array(gn), onSkin = new Uint8Array(gn), core = new Uint8Array(gn), between = new Uint8Array(gn);
  let hairy = new Uint8Array(gn);
  const someSkin = f(0.02);
  for (let i = 0; i < gn; i++) {
    const au = Math.abs(u[i]), low = f(f(-hv.drop) * step32(au, hv.drop_from, hv.drop_to));
    bounds[i] = au < h32.wide && v[i] < h32.top && v[i] > low ? 1 : 0;
    onSkin[i] = valid[i] > 0.5 && skin[i] > someSkin && bounds[i] ? 1 : 0;
    if (!onSkin[i]) continue;
    core[i] = dark[i] > h32.core && au < h32.core_wide && v[i] > h32.core_lo && v[i] < h32.core_hi ? 1 : 0;
    hairy[i] = (dark[i] > h32.hair_lo && rough[i] > h32.rough) || dark[i] > h32.hair_hi ? 1 : 0;
    between[i] = rough[i] > h32.mid_rough && au < h32.mid_wide && v[i] > 0 ? 1 : 0;
  }
  const bridge = roundKernel(cv, hv.bridge * eyes);
  hairy = morphed(cv, hairy, gw, gh, bridge, true);
  for (let i = 0; i < gn; i++) { hairy[i] = hairy[i] && onSkin[i] ? 1 : 0; between[i] = hairy[i] || between[i] ? 1 : 0; }
  let region = grown(core, between, gw, gh);
  if (!region.some((r) => r)) { bridge.delete(); return false; }
  {
    const reach = roundKernel(cv, hv.tail_reach * eyes), near = morphed(cv, region, gw, gh, reach);
    reach.delete();
    let tails = new Uint8Array(gn);
    for (let i = 0; i < gn; i++) tails[i] = dark[i] > h32.tail_dark && rough[i] > h32.tail_rough && onSkin[i] && near[i] ? 1 : 0;
    tails = morphed(cv, tails, gw, gh, bridge, true);
    bridge.delete();
    for (let i = 0; i < gn; i++) tails[i] = (tails[i] && onSkin[i]) || region[i] ? 1 : 0;
    region = grown(region, tails, gw, gh);
  }
  // The replaced area: the hair and a little past it, soft-edged. The skin mask has the lips cut out
  // with a hard edge, so the area runs on over them: the lip's own soft edge is applied further down.
  const paint = new Float32Array(gn), by = f(Math.min(1, amount));
  {
    const grow = roundKernel(cv, hv.grow * eyes), wide = morphed(cv, region, gw, gh, grow), lipsWide = morphed(cv, lips, gw, gh, grow);
    grow.delete();
    const soft = blurred(cv, Float32Array.from(wide), gw, gh, 1, hv.soft * eyes);
    const inBounds = blurred(cv, Float32Array.from(bounds), gw, gh, 1, hv.bound_edge * eyes);
    let most = 0;
    for (let i = 0; i < gn; i++) {
      const here = f(Math.min(f(soft[i] * f(1.5)), 1) * Math.min(f(inBounds[i] * 2), 1));
      paint[i] = f(f(here * (skin[i] > someSkin || lipsWide[i] ? 1 : 0)) * by);
      if (paint[i] > most) most = paint[i];
    }
    if (!(most > 1e-3)) return false;
  }

  // The tone: the table's shape in the cheeks' colour, matched channel by channel to the clear skin around.
  // (the table's shape is rounded first: over a whole lip its creases would show, the one down the middle most)
  const prior = new Float32Array(3 * gn), final = new Float32Array(3 * gn), change = new Float32Array(3 * gn);
  const clearSkin = new Float32Array(gn);
  {
    const rounded = blurred(cv, shape, gw, gh, 1, hv.shape_soft * eyes);
    const off = roundKernel(cv, hv.clear_off * eyes), hair = new Uint8Array(gn);
    for (let i = 0; i < gn; i++) hair[i] = region[i] || hairy[i] || dark[i] > h32.hair_hi ? 1 : 0;
    const nearHair = morphed(cv, hair, gw, gh, off);
    off.delete();
    const offset = new Float32Array(3 * gn), spanLo = f(Math.log(hv.span[0])), spanHi = f(Math.log(hv.span[1]));
    for (let i = 0; i < gn; i++) {
      clearSkin[i] = valid[i] > clearEdge && skin[i] > 0.5 && !nearHair[i] && Math.abs(u[i]) < t32.fit_wide
        && v[i] > h32.clear_lo && v[i] < h32.clear_hi ? 1 : 0;
      const base = Math.max(rounded[i], least);
      for (let ch = 0; ch < 3; ch++) {
        const j = 3 * i + ch;
        prior[j] = f(cheek[ch] * f(Math.pow(base, powers[ch])));
        offset[j] = Math.min(Math.max(f(Math.log(f(Math.max(tone[j], least) / Math.max(prior[j], least)))), spanLo), spanHi);
      }
    }
    const carried = fillIn(cv, offset, clearSkin, gw, gh, hv.fill, eyes, hv.fill_lo, hv.fill_hi);
    const hairSoft = blurred(cv, Float32Array.from(region), gw, gh, 1, hv.soft * eyes);
    for (let i = 0; i < gn; i++) {
      for (let ch = 0; ch < 3; ch++) final[3 * i + ch] = f(prior[3 * i + ch] * f(Math.exp(carried[3 * i + ch])));
      const finalY = luma(final, 3 * i);
      // Hair goes whatever its tone; the clear skin beside it only ever gets lighter.
      const lift = Math.max(step32(f(f(finalY - toneY[i]) / f(finalY + tiny)), -t.dark_lo, t.dark_lo), Math.min(f(hairSoft[i] * 2), 1));
      for (let ch = 0; ch < 3; ch++) change[3 * i + ch] = f(lift * f(final[3 * i + ch] - tone[3 * i + ch]));
    }
  }

  const up = (a_, channels) => (scale < 1 ? resized(cv, a_, gw, gh, channels, bw, bh, cv.INTER_LINEAR) : a_);
  // 1 between two curves that run the same way, at the photo's size.
  const filled = (curveA, curveB) => polygon(cv, [...curveA, ...[...curveB].reverse()].map((p) => [p[0] - x0, p[1] - y0]), bw, bh);

  // At the photo's own size: the lip's edge.
  const lipsFull = polygon(cv, outline.map((p) => [p[0] - x0, p[1] - y0]), bw, bh);
  const lipsSoft = blurred(cv, Float32Array.from(lipsFull, (b) => f(b / k255)), bw, bh, 1, Math.max(0.6, hv.lip_soft * d));
  const ok = new Float32Array(bn);
  for (let i = 0; i < bn; i++) ok[i] = f(1 - lipsSoft[i]);
  let paintFull = up(paint, 1), hold = null;
  if (keep) {
    const ax0 = Math.max(x0, keep.x0), ay0 = Math.max(y0, keep.y0), ax1 = Math.min(x1, keep.x0 + keep.w), ay1 = Math.min(y1, keep.y0 + keep.h);
    if (ax1 > ax0 && ay1 > ay0) {
      hold = new Float32Array(bn).fill(1);
      for (let y = ay0; y < ay1; y++) {
        for (let x = ax0; x < ax1; x++) hold[(y - y0) * bw + x - x0] = f(1 + Math.min(keep.data[(y - keep.y0) * keep.w + x - keep.x0], 0));
      }
      paintFull = paintFull.map((p, i) => f(p * hold[i]));
    }
  }

  // The grain: pieces of each cheek's own patch, laid over its half of the lip side by side and
  // blended into each other, each piece from a different part of the patch. So it never comes
  // from a fold, the nose or the backdrop, and no piece of cheek shows twice in a row.
  const toneUp = up(tone, 3), changeUp = up(change, 3), uUp = up(u, 1);
  const grain = new Float32Array(3 * bn), grainY = new Float32Array(bn), usable = new Float32Array(bn), quiet = [];
  for (let y = 0, i = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++, i++) {
      for (let ch = 0; ch < 3; ch++) grain[3 * i + ch] = f(box[3 * i + ch] - toneUp[3 * i + ch]);
      grainY[i] = f(luma(grain, 3 * i) * ok[i]);
      usable[i] = boxMask[i] > wellSkin ? ok[i] : 0;
      if (boxMask[i] > wellSkin && fr.cheeks.some((c) => (x + x0 - c[0]) ** 2 + (y + y0 - c[1]) ** 2 < r2)) quiet.push(Math.abs(grainY[i]));
    }
  }
  const limit = f(t.grain_most * percentile(quiet, 25));
  const size = hv.tile * d, wander = hv.tile_wander * d, borrowed = new Float32Array(bn), have = new Float32Array(bn);
  fr.cheeks.forEach((c, k) => {
    const side = k === 0 ? -1 : 1;
    const hx = fr.centre[0] + side * t.side * fr.half[0] + t.side_up * fr.up[0], hy = fr.centre[1] + side * t.side * fr.half[1] + t.side_up * fr.up[1];
    const total = new Float64Array(bn), got = new Float64Array(bn), squares = new Float64Array(bn);
    LAYERS.forEach(([ox, oy], layer) => {
      const across = pieces(bw, x0, hx, size, ox), down = pieces(bh, y0, hy, size, oy);
      for (let y = 0, i = 0; y < bh; y++) {
        for (let x = 0; x < bw; x++, i++) {
          const weight = (across.wave[x] * down.wave[y]) ** 2;
          const turn = (across.piece[x] * 3.0 + down.piece[y] * 5.0 + layer * 2.0) % WANDER.length, pick = turn < 0 ? turn + WANDER.length : turn;
          const sx = pyRound(c[0] - x0 + across.inside[x] * size + WANDER[pick][0] * wander);
          const sy = pyRound(c[1] - y0 + down.inside[y] * size + WANDER[pick][1] * wander);
          const j = Math.min(bh - 1, Math.max(0, sy)) * bw + Math.min(bw - 1, Math.max(0, sx));
          total[i] += weight * grainY[j];
          if (sx >= 0 && sx < bw && sy >= 0 && sy < bh) got[i] += weight * usable[j];
          squares[i] += weight * weight;
        }
      }
    });
    for (let i = 0; i < bn; i++) {
      const sideOf = step32(uUp[i], -t.seam, t.seam), share = k === 0 ? f(1 - sideOf) : sideOf;       // 0: the right cheek's half, 1: the left's
      borrowed[i] = f(borrowed[i] + f(share * f(total[i] / Math.sqrt(Math.max(squares[i], 1e-6)))));   // blended, and as strong as one piece
      have[i] = f(have[i] + f(share * f(got[i])));
    }
  });
  for (let i = 0; i < bn; i++) {
    const g = f(borrowed[i] * have[i]);                         // no skin to borrow from: left smooth
    borrowed[i] = f(g * Math.min(1, f(limit / f(Math.abs(g) + f(1e-6)))));
  }

  // The lip's side of the edge: hair lying over the upper lip takes the lip's own colour, carried
  // along the lip from where it is clear (as bright as the lip's brighter half: hair only darkens).
  const inner = spline(fr.inner);
  const away = upper.map((p, k) => {
    const ax = p[0] - inner[k][0], ay = p[1] - inner[k][1], length = Math.hypot(ax, ay) + 1e-6;
    return [ax / length, ay / length];
  });
  const middle = away[upper.length >> 1];                       // at its corners the lip has no thickness to point along
  for (let k = 0; k < 3; k++) { away[k] = middle; away[upper.length - 1 - k] = middle; }
  const reach = 4.0 * hv.lip_soft * d, far = upper.map((p, k) => [p[0] + away[k][0] * reach, p[1] + away[k][1] * reach]);
  const boxY = new Float32Array(bn), known = new Float32Array(bn);
  {
    const whole = filled(far, inner), within = filled(upper, inner), inLip = [];
    for (let i = 0; i < bn; i++) {
      boxY[i] = luma(box, 3 * i);
      whole[i] = whole[i] && lipsSoft[i] > 0.5 ? 1 : 0;
      if (within[i]) inLip.push(boxY[i]);
    }
    let bright = 0;
    if (inLip.length) { const half = f(medianOf(inLip)); bright = medianOf(inLip.filter((y) => y > half)); }
    const brightBy = f(Math.max(bright, 1e-3));
    let sum = 0;
    for (let i = 0; i < bn; i++) { known[i] = f(whole[i] * step32(f(boxY[i] / brightBy), hv.lip_keep[0], hv.lip_keep[1])); sum += known[i]; }
    if (!(sum > 10)) for (let i = 0; i < bn; i++) known[i] = whole[i];
  }
  const lipTone = fillIn(cv, box, known, bw, bh, hv.lip_fill, d, hv.fill_lo, hv.fill_hi);
  const strand = Math.max(0.6, hv.lip_strand * d);
  let onLip = blurred(cv, boxY, bw, bh, 1, strand);
  const band = blurred(cv, Float32Array.from(filled(far, upper.map((p, k) => [p[0] + hv.lip_band * (inner[k][0] - p[0]), p[1] + hv.lip_band * (inner[k][1] - p[1])])),
    (b) => (b ? 1 : 0)), bw, bh, 1, Math.max(0.6, 0.5 * hv.lip_soft * d));
  {
    const beside = blurred(cv, paintFull, bw, bh, 1, hv.lip_reach * d);
    for (let i = 0; i < bn; i++) {
      const darker = step32(f(1 - f(onLip[i] / f(luma(lipTone, 3 * i) + tiny))), hv.lip_dark[0], hv.lip_dark[1]);
      onLip[i] = f(f(darker * band[i]) * Math.min(f(beside[i] * 3), 1));
    }
    onLip = blurred(cv, onLip, bw, bh, 1, strand);
    for (let i = 0; i < bn; i++) {
      onLip[i] = f(Math.min(f(onLip[i] * f(1.5)), 1) * by);
      if (hold) onLip[i] = f(onLip[i] * hold[i]);
    }
  }

  if (seen) {
    Object.assign(seen, {
      box: [x0, y0, x1, y1], grid: [gw, gh], u, v, dark, rough, hairy, region, paint, paintFull, clearSkin, final, prior, tone,
      have, lipsSoft, lipsFull, onLip, lipTone, band,
    });
  }
  let changed = false;
  // The photo's bytes, looked up again: the Mats made since the top may have moved the heap, and a
  // view taken before that no longer reaches it (what is written through it is lost).
  const od = out.data, half = f(0.5), lipGrain = f(hv.lip_grain);
  for (let y = 0, i = 0; y < bh; y++) {
    for (let x = 0, p = 3 * ((y + y0) * W + x0); x < bw; x++, i++, p += 3) {
      const g = borrowed[i];
      for (let ch = 0; ch < 3; ch++) {
        const j = 3 * i + ch;
        const newSkin = f(box[j] + f(paintFull[i] * f(f(changeUp[j] + g) - grain[j])));
        const newLip = f(box[j] + f(onLip[i] * f(f(lipTone[j] + f(lipGrain * g)) - box[j])));
        const value = f(f(f(f(newSkin * ok[i]) + f(newLip * lipsSoft[i])) * k255) + half);
        const byte = value < 0 ? 0 : value > 255 ? 255 : Math.trunc(value);
        if (byte !== od[p + ch]) { od[p + ch] = byte; changed = true; }
      }
    }
  }
  return changed;
}

// bgr (CV_8UC3) with a thin moustache taken off each face's upper lip: a new Mat, or bgr
// itself when nothing changes. points: each face's landmarks or null. mask: the smoothing
// mask ({ x0, y0, w, h, data }). amount: 0..1. keep: the brush's Restore edit ({ x0, y0,
// w, h, data } in -1..1), where the lip stays as it is. force: skip the questions whether
// the face is posed for it and whether there is a moustache (the parity check). full: a
// full, dense moustache goes too (its own way, clearFull); without it one is left alone.
// A thin moustache comes out the same with or without full.
export function clear(cv, bgr, points, mask, amount, keep = null, force = false, full = false) {
  if (!(amount > 0) || !points || !points.length) return bgr;
  const out = copyMat(cv, bgr);
  let changed = false;
  for (const pts of points) {
    if (pts && clearFace(cv, out, pts, mask, amount, keep, force, full)) changed = true;
  }
  if (changed) return out;
  out.delete();
  return bgr;
}
