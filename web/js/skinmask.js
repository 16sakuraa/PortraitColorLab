// Skin mask for smoothing - a line-for-line port of smoother.build_skin_mask
// and its helpers (skin_tone_mask, lip_mask, eye_mask, brow_mask).
//
// 1 where skin should be smoothed: the face ovals, minus non-skin colours,
// minus the eyes, brows and lips themselves (found from how they differ from
// the surrounding skin). Computed only around the faces; zero elsewhere.

import { copyMat } from './cv.js';

// Python's round(): halves go to the even neighbour.
export function pyRound(v) {
  const f = Math.floor(v), d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

const kernel = (cv, size) => {
  const k = Math.max(3, Math.trunc(size)) | 1;
  return cv.Mat.ones(k, k, cv.CV_8U);
};

function runs(flags, maxGap) {
  const out = [];
  let start = -1, prev = -1;
  for (let i = 0; i < flags.length; i++) {
    if (!flags[i]) continue;
    if (start < 0) { start = prev = i; continue; }
    if (i - prev > maxGap + 1) { out.push([start, prev]); start = i; }
    prev = i;
  }
  if (start >= 0) out.push([start, prev]);
  return out;
}

function percentile(values, p) {                 // numpy 'linear' method
  const a = Float32Array.from(values).sort();
  const pos = (a.length - 1) * p / 100, lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}
const median = (values) => percentile(values, 50);

// Copy of a sub-rectangle (so neighbourhood ops see it as a standalone image,
// like a numpy slice passed to cv2).
function crop(cv, mat, x0, y0, x1, y1) {
  const v = mat.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
  const c = copyMat(cv, v);
  v.delete();
  return c;
}

function lab8(cv, bgr) {
  const lab = new cv.Mat();
  cv.cvtColor(bgr, lab, cv.COLOR_BGR2Lab);
  return lab;
}

export function skinToneMask(cv, bgr) {
  const ycc = new cv.Mat();
  cv.cvtColor(bgr, ycc, cv.COLOR_BGR2YCrCb);
  const m = new cv.Mat(bgr.rows, bgr.cols, cv.CV_8U);
  const s = ycc.data, d = m.data;
  for (let i = 0, j = 0; j < d.length; i += 3, j++) {
    d[j] = s[i + 1] >= 133 && s[i + 1] <= 173 && s[i + 2] >= 77 && s[i + 2] <= 127 ? 255 : 0;
  }
  ycc.delete();
  const k3 = cv.Mat.ones(3, 3, cv.CV_8U), k7 = cv.Mat.ones(7, 7, cv.CV_8U);
  cv.morphologyEx(m, m, cv.MORPH_OPEN, k3);
  cv.morphologyEx(m, m, cv.MORPH_CLOSE, k7);
  k3.delete(); k7.delete();
  return m;
}

const clampWin = (bgr, x0, y0, x1, y1) => [
  Math.trunc(Math.max(0, x0)), Math.trunc(Math.max(0, y0)),
  Math.trunc(Math.min(bgr.cols, x1)), Math.trunc(Math.min(bgr.rows, y1))];

// ---------- lips ----------

export function lipMask(cv, bgr, f) {
  const h = bgr.rows, w = bgr.cols;
  const eyeD = Math.hypot(f[6] - f[4], f[7] - f[5]) || f[2] * 0.4;
  const mw = Math.max(Math.hypot(f[12] - f[10], f[13] - f[11]), 0.6 * eyeD);
  const c = [(f[10] + f[12]) / 2, (f[11] + f[13]) / 2];
  const angle = Math.atan2(f[13] - f[11], f[12] - f[10]) * 180 / Math.PI;
  const x0 = Math.trunc(Math.max(0, c[0] - 0.85 * mw)), x1 = Math.trunc(Math.min(w, c[0] + 0.85 * mw));
  const y0 = Math.trunc(Math.max(0, c[1] - 0.3 * mw)), y1 = Math.trunc(Math.min(h, c[1] + 0.8 * mw));
  const ww = Math.max(1, x1 - x0), wh = Math.max(1, y1 - y0);
  const out = cv.Mat.zeros(wh, ww, cv.CV_8U);
  if (x1 - x0 < 8 || y1 - y0 < 8) return { mask: out, x0, y0, measured: false };

  let ecx = c[0] - x0, ecy = c[1] - y0 + 0.12 * mw, halfW = 0.6 * mw, halfH = 0.28 * mw, measured = false;

  // Skin reference: skin-tone pixels in the face box, outside the window.
  const fx0 = Math.trunc(Math.max(0, f[0])), fy0 = Math.trunc(Math.max(0, f[1]));
  const fx1 = Math.trunc(Math.min(w, f[0] + f[2])), fy1 = Math.trunc(Math.min(h, f[1] + f[3]));
  const box = crop(cv, bgr, fx0, fy0, fx1, fy1);
  const tone = skinToneMask(cv, box), boxLab = lab8(cv, box);
  const bw = box.cols, bh = box.rows;
  const ey0 = Math.max(0, y0 - fy0), ey1 = Math.max(0, y1 - fy0), ex0 = Math.max(0, x0 - fx0), ex1 = Math.max(0, x1 - fx0);
  let Ls = [], As = [];
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      const i = y * bw + x;
      if (!tone.data[i] || (y >= ey0 && y < ey1 && x >= ex0 && x < ex1)) continue;
      Ls.push(boxLab.data[3 * i]); As.push(boxLab.data[3 * i + 1]);
    }
  }
  if (Ls.length < 50) {
    Ls = []; As = [];
    for (let i = 0; i < bw * bh; i++) { Ls.push(boxLab.data[3 * i]); As.push(boxLab.data[3 * i + 1]); }
  }
  box.delete(); tone.delete(); boxLab.delete();
  const skinL = median(Ls), skinA = median(As);
  const madA = median(As.map((a) => Math.abs(a - skinA))) || 1.0;

  const win = crop(cv, bgr, x0, y0, x1, y1), lab = lab8(cv, win);
  win.delete();
  const ind0 = new cv.Mat(wh, ww, cv.CV_8U);
  const lipThr = skinA + Math.max(6.0, 3.0 * madA);
  for (let i = 0; i < ww * wh; i++) {
    const L = lab.data[3 * i], A = lab.data[3 * i + 1], B = lab.data[3 * i + 2];
    const lips = A > lipThr, teeth = L > skinL + 20 && Math.abs(A - 128) + Math.abs(B - 128) < 28;
    ind0.data[i] = lips || teeth ? 1 : 0;
  }
  lab.delete();
  const k = kernel(cv, mw * 0.03);
  cv.morphologyEx(ind0, ind0, cv.MORPH_OPEN, k);
  k.delete();
  const ind = ind0.data;

  const cx = pyRound(c[0] - x0);
  const cl = Math.max(0, cx - Math.trunc(0.3 * mw)), cr = Math.min(ww, cx + Math.trunc(0.3 * mw));
  let rows = new Float64Array(wh);
  if (cr > cl) for (let y = 0; y < wh; y++) { let s = 0; for (let x = cl; x < cr; x++) s += ind[y * ww + x]; rows[y] = s / (cr - cl); }
  const kk = Math.max(1, Math.trunc(mw * 0.03)), off = Math.floor((kk - 1) / 2);
  const conv = new Float64Array(wh);          // np.convolve(rows, ones(k)/k, 'same')
  for (let i = 0; i < wh; i++) {
    let s = 0;
    for (let m = i + off - kk + 1; m <= i + off; m++) if (m >= 0 && m < wh) s += rows[m];
    conv[i] = s / kk;
  }
  rows = conv;
  const sumOf = (r) => { let s = 0; for (let i = r[0]; i <= r[1]; i++) s += rows[i]; return s; };
  const maxOf = (r) => { let s = -Infinity; for (let i = r[0]; i <= r[1]; i++) s = Math.max(s, rows[i]); return s; };
  const bands = runs(Array.from(rows, (v) => v > 0.3), Math.trunc(0.06 * mw))
    .filter((r) => { const d = (r[0] + r[1]) / 2 - ecy; return d >= -0.4 * mw && d <= 0.45 * mw; });
  if (bands.length) {
    let [top, bot] = bands.reduce((a, b) => (sumOf(b) > sumOf(a) ? b : a));
    let grew = true;
    while (grew) {
      grew = false;
      for (const r of bands) {
        const gap = Math.max(r[0] - bot, top - r[1]);
        if (gap > 0 && gap <= 0.15 * mw && maxOf(r) >= 0.45 && Math.max(bot, r[1]) - Math.min(top, r[0]) <= 0.65 * mw) {
          top = Math.min(top, r[0]); bot = Math.max(bot, r[1]); grew = true;
        }
      }
    }
    if (bot - top >= 0.1 * mw && bot - top <= 0.9 * mw) {
      ecy = (top + bot) / 2;
      halfH = Math.max((bot - top) / 2 + 0.06 * mw, 0.2 * mw);
      const cols = new Array(ww);
      for (let x = 0; x < ww; x++) { let s = 0; for (let y = top; y <= bot; y++) s += ind[y * ww + x]; cols[x] = s / (bot - top + 1) > 0.3; }
      const colRuns = runs(cols, Math.trunc(0.1 * mw));
      if (colRuns.length) {
        const dist = (r) => (r[0] <= cx && cx <= r[1] ? 0 : Math.min(Math.abs(r[0] - cx), Math.abs(r[1] - cx)));
        const [lft, rgt] = colRuns.reduce((a, b) => (dist(b) < dist(a) ? b : a));
        if (rgt - lft >= 0.4 * mw) {
          ecx = Math.min(Math.max((lft + rgt) / 2, cx - 0.15 * mw), cx + 0.15 * mw);
          halfW = Math.min(Math.max((rgt - lft) / 2 + 0.06 * mw, 0.45 * mw), 0.8 * mw);
        }
      }
      measured = true;
    }
  }
  ind0.delete();
  cv.ellipse(out, new cv.Point(Math.trunc(ecx), Math.trunc(ecy)), new cv.Size(Math.trunc(halfW), Math.trunc(halfH)),
    angle, 0, 360, new cv.Scalar(255), -1);
  return { mask: out, x0, y0, measured };
}

// ---------- eyes ----------

function fillHoles(cv, blob) {
  const contours = new cv.MatVector(), hier = new cv.Mat();
  cv.findContours(blob, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  const out = copyMat(cv, blob);
  cv.drawContours(out, contours, -1, new cv.Scalar(1), -1);
  contours.delete(); hier.delete();
  return out;
}

export function eyeMask(cv, bgr, ex, ey, eyeD, angle) {
  const [x0, y0, x1, y1] = clampWin(bgr, ex - 0.55 * eyeD, ey - 0.38 * eyeD, ex + 0.55 * eyeD, ey + 0.3 * eyeD);
  const ww = Math.max(1, x1 - x0), wh = Math.max(1, y1 - y0);
  const sx = Math.trunc(ex - x0), sy = Math.trunc(ey - y0);
  if (x1 - x0 < 8 || y1 - y0 < 8) return { mask: cv.Mat.zeros(wh, ww, cv.CV_8U), x0, y0, measured: false };
  const win = crop(cv, bgr, x0, y0, x1, y1), lab = lab8(cv, win);
  win.delete();
  const n = ww * wh, Ls = new Float32Array(n);
  for (let i = 0; i < n; i++) Ls[i] = lab.data[3 * i];
  const skinL = median(Ls);
  const darkThr = Math.min(skinL - 35, percentile(Ls, 12));
  const cand = new cv.Mat(wh, ww, cv.CV_8U);
  for (let i = 0; i < n; i++) {
    const L = lab.data[3 * i], A = lab.data[3 * i + 1], B = lab.data[3 * i + 2];
    cand.data[i] = L < darkThr || (L > skinL + 8 && Math.abs(A - 128) + Math.abs(B - 128) < 20) ? 1 : 0;
  }
  lab.delete();
  let k = kernel(cv, eyeD * 0.06); cv.morphologyEx(cand, cand, cv.MORPH_CLOSE, k); k.delete();
  k = kernel(cv, eyeD * 0.02); cv.morphologyEx(cand, cand, cv.MORPH_OPEN, k); k.delete();
  const bound = cv.Mat.zeros(wh, ww, cv.CV_8U);
  cv.ellipse(bound, new cv.Point(sx, Math.trunc(sy - 0.06 * eyeD)),
    new cv.Size(Math.trunc(0.5 * eyeD), Math.trunc(0.19 * eyeD)), angle, 0, 360, new cv.Scalar(1), -1);
  cv.bitwise_and(cand, bound, cand);
  bound.delete();

  const labels = new cv.Mat();
  const nLab = cv.connectedComponents(cand, labels);
  cand.delete();
  const L32 = labels.data32S, r = Math.max(2, Math.trunc(0.1 * eyeD));
  const counts = new Map();
  for (let y = Math.max(0, sy - r); y < Math.min(wh, sy + r); y++) {
    for (let x = Math.max(0, sx - r); x < Math.min(ww, sx + r); x++) {
      const l = L32[y * ww + x];
      if (l > 0) counts.set(l, (counts.get(l) || 0) + 1);
    }
  }
  if (nLab > 1 && counts.size) {
    let best = -1, bestN = -1;
    for (const [l, c] of [...counts].sort((a, b) => a[0] - b[0])) if (c > bestN) { best = l; bestN = c; }
    const blob = new cv.Mat(wh, ww, cv.CV_8U);
    let minX = ww, maxX = -1, minY = wh, maxY = -1;
    for (let i = 0; i < n; i++) {
      const on = L32[i] === best;
      blob.data[i] = on ? 1 : 0;
      if (on) { const x = i % ww, y = (i - x) / ww; minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
    }
    labels.delete();
    if (maxX - minX >= 0.3 * eyeD && maxX - minX <= 1.0 * eyeD && maxY - minY >= 0.07 * eyeD) {
      const filled = fillHoles(cv, blob);
      blob.delete();
      const out = new cv.Mat();
      k = kernel(cv, eyeD * 0.04); cv.dilate(filled, out, k); k.delete(); filled.delete();
      for (let i = 0; i < n; i++) out.data[i] *= 255;
      return { mask: out, x0, y0, measured: true };
    }
    blob.delete();
  } else {
    labels.delete();
  }
  const out = cv.Mat.zeros(wh, ww, cv.CV_8U);
  cv.ellipse(out, new cv.Point(sx, Math.trunc(sy - 0.03 * eyeD)),
    new cv.Size(Math.trunc(0.32 * eyeD), Math.trunc(0.15 * eyeD)), angle, 0, 360, new cv.Scalar(255), -1);
  return { mask: out, x0, y0, measured: false };
}

// ---------- brows ----------

export function browMask(cv, bgr, ex, ey, eyeD, angle) {
  const [x0, y0, x1, y1] = clampWin(bgr, ex - 0.6 * eyeD, ey - 0.75 * eyeD, ex + 0.6 * eyeD, ey - 0.12 * eyeD);
  const ww = Math.max(1, x1 - x0), wh = Math.max(1, y1 - y0);
  const out = cv.Mat.zeros(wh, ww, cv.CV_8U);
  if (x1 - x0 < 8 || y1 - y0 < 8) return { mask: out, x0, y0, measured: false };
  const win = crop(cv, bgr, x0, y0, x1, y1), lab = lab8(cv, win);
  win.delete();
  const n = ww * wh, Ls = new Float32Array(n);
  for (let i = 0; i < n; i++) Ls[i] = lab.data[3 * i];
  lab.delete();
  const thr = median(Ls) - 18;
  const dark = new cv.Mat(wh, ww, cv.CV_8U);
  for (let i = 0; i < n; i++) dark.data[i] = Ls[i] < thr ? 1 : 0;
  let k = kernel(cv, eyeD * 0.02); cv.morphologyEx(dark, dark, cv.MORPH_OPEN, k); k.delete();
  const D = dark.data;

  const cx = Math.trunc(ex - x0);
  const cl = Math.max(0, cx - Math.trunc(0.3 * eyeD)), cr = Math.min(ww, cx + Math.trunc(0.3 * eyeD));
  const rows = new Float64Array(wh);
  if (cr > cl) for (let y = 0; y < wh; y++) { let s = 0; for (let x = cl; x < cr; x++) s += D[y * ww + x]; rows[y] = s / (cr - cl); }
  const sumOf = (r) => { let s = 0; for (let i = r[0]; i <= r[1]; i++) s += rows[i]; return s; };
  const bands = runs(Array.from(rows, (v) => v > 0.25), Math.trunc(0.03 * eyeD))
    .filter((r) => r[1] - r[0] >= 0.04 * eyeD && r[1] - r[0] <= 0.3 * eyeD);
  if (bands.length) {
    let [top, bot] = bands.reduce((a, b) => (sumOf(b) > sumOf(a) ? b : a));
    const pad = Math.trunc(0.04 * eyeD);
    top = Math.max(0, top - pad); bot = Math.min(wh - 1, bot + pad);
    const band = crop(cv, dark, 0, top, ww, bot + 1);
    k = kernel(cv, eyeD * 0.05); cv.morphologyEx(band, band, cv.MORPH_CLOSE, k); k.delete();
    const bh = bot - top + 1, cols = new Array(ww);
    for (let x = 0; x < ww; x++) { let s = 0; for (let y = 0; y < bh; y++) s += band.data[y * ww + x]; cols[x] = s / bh > 0.15; }
    const over = runs(cols, Math.trunc(0.05 * eyeD)).filter((r) => r[0] <= cx + 0.2 * eyeD && r[1] >= cx - 0.2 * eyeD);
    if (over.length) {
      const lft = Math.min(...over.map((r) => r[0])), rgt = Math.max(...over.map((r) => r[1]));
      if (rgt - lft >= 0.3 * eyeD) {
        for (let y = 0; y < bh; y++) for (let x = lft; x <= rgt; x++) out.data[(y + top) * ww + x] = band.data[y * ww + x];
        band.delete(); dark.delete();
        const dil = new cv.Mat();
        k = kernel(cv, eyeD * 0.03); cv.dilate(out, dil, k); k.delete(); out.delete();
        for (let i = 0; i < n; i++) dil.data[i] *= 255;
        return { mask: dil, x0, y0, measured: true };
      }
    }
    band.delete();
  }
  dark.delete();
  cv.ellipse(out, new cv.Point(Math.trunc(ex - x0), Math.trunc(ey - y0 - 0.3 * eyeD)),
    new cv.Size(Math.trunc(0.30 * eyeD), Math.trunc(0.13 * eyeD)), angle, 0, 360, new cv.Scalar(255), -1);
  return { mask: out, x0, y0, measured: false };
}

// ---------- the mask ----------

// Returns { x0, y0, w, h, data: Float32Array } (values 0..1; zero outside).
export function buildSkinMask(cv, bgr, faces, feather = 1.0) {
  const H = bgr.rows, W = bgr.cols;
  const diag = Math.hypot(H, W);
  const k = Math.max(3, Math.trunc(diag * 0.01 * Math.max(feather, 0.01))) | 1;
  const found = faces.length > 0;

  // Work only around the faces (everything else is 0 anyway).
  let X0 = 0, Y0 = 0, X1 = W, Y1 = H;
  if (found) {
    X0 = W; Y0 = H; X1 = 0; Y1 = 0;
    for (const f of faces) {
      const cx = f[0] + f[2] / 2, cy = f[1] + f[3] * 0.55, ax = f[2] * 0.52, ay = f[3] * 0.72;
      X0 = Math.min(X0, cx - ax); X1 = Math.max(X1, cx + ax); Y0 = Math.min(Y0, cy - ay); Y1 = Math.max(Y1, cy + ay);
    }
    const m = k + 32;
    X0 = Math.max(0, Math.floor(X0) - m); Y0 = Math.max(0, Math.floor(Y0) - m);
    X1 = Math.min(W, Math.ceil(X1) + m); Y1 = Math.min(H, Math.ceil(Y1) + m);
  }
  const RW = X1 - X0, RH = Y1 - Y0;
  const region = crop(cv, bgr, X0, Y0, X1, Y1);
  let area = cv.Mat.zeros(RH, RW, cv.CV_8U);
  const protect = cv.Mat.zeros(RH, RW, cv.CV_8U);
  const paint = ({ mask, x0, y0 }) => {
    for (let y = 0; y < mask.rows; y++) {
      const ry = y + y0 - Y0;
      if (ry < 0 || ry >= RH) continue;
      for (let x = 0; x < mask.cols; x++) {
        const rx = x + x0 - X0;
        if (rx >= 0 && rx < RW && mask.data[y * mask.cols + x]) protect.data[ry * RW + rx] = 255;
      }
    }
    mask.delete();
  };

  for (const f of faces) {
    const eyeD = Math.hypot(f[6] - f[4], f[7] - f[5]) || f[2] * 0.4;
    cv.ellipse(area, new cv.Point(Math.trunc(f[0] + f[2] / 2 - X0), Math.trunc(f[1] + f[3] * 0.55 - Y0)),
      new cv.Size(Math.trunc(f[2] * 0.52), Math.trunc(f[3] * 0.72)), 0, 0, 360, new cv.Scalar(255), -1);
    const angle = Math.atan2(f[7] - f[5], f[6] - f[4]) * 180 / Math.PI;
    for (const [ex, ey] of [[f[4], f[5]], [f[6], f[7]]]) {
      paint(eyeMask(cv, bgr, ex, ey, eyeD, angle));
      paint(browMask(cv, bgr, ex, ey, eyeD, angle));
    }
    paint(lipMask(cv, bgr, f));
  }

  const tone = skinToneMask(cv, region);
  region.delete();
  if (!found) {
    area.delete();
    area = tone;
  } else {
    const k9 = cv.Mat.ones(9, 9, cv.CV_8U);
    cv.dilate(tone, tone, k9);
    k9.delete();
    cv.bitwise_and(area, tone, area);
    tone.delete();
  }

  const soft = new cv.Mat();
  area.convertTo(soft, cv.CV_32F, 1 / 255);
  area.delete();
  cv.GaussianBlur(soft, soft, new cv.Size(k, k), 0);
  if (found) {
    const ks = Math.max(3, Math.floor(k / 4)) | 1;
    const kg = kernel(cv, ks), guard = new cv.Mat();
    cv.dilate(protect, guard, kg);
    kg.delete();
    const g = new cv.Mat();
    guard.convertTo(g, cv.CV_32F, 1 / 255);
    guard.delete();
    cv.GaussianBlur(g, g, new cv.Size(ks, ks), 0);
    const s = soft.data32F, gd = g.data32F;
    for (let i = 0; i < s.length; i++) s[i] = Math.fround(s[i] * Math.fround(1.0 - gd[i]));
    g.delete();
  }
  protect.delete();
  const data = new Float32Array(soft.data32F);
  soft.delete();
  for (let i = 0; i < data.length; i++) data[i] = data[i] < 0 ? 0 : data[i] > 1 ? 1 : data[i];
  return { x0: X0, y0: Y0, w: RW, h: RH, data };
}
