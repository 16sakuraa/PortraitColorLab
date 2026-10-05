// Learned blemish removal - a line-for-line port of skin_smoother/blemish.py.
//
// Per face: cut out an FFHQ-framed 1024x1024 crop from the YuNet points, run
// the network, and warp its *correction* back onto the full-resolution photo.
// Overlapping crops (people close together) are blended, never added.

import { copyMat } from './cv.js';

export const SIZE = 1024;
const MIN_QUAD = 96;

export function ffhqQuad(f) {
  const e1 = [f[4], f[5]], e2 = [f[6], f[7]], m1 = [f[10], f[11]], m2 = [f[12], f[13]];
  const eyeAvg = [(e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2];
  const mouthAvg = [(m1[0] + m2[0]) / 2, (m1[1] + m2[1]) / 2];
  const em = [mouthAvg[0] - eyeAvg[0], mouthAvg[1] - eyeAvg[1]];
  let ee = [e2[0] - e1[0], e2[1] - e1[1]];
  if (ee[0] * em[1] - ee[1] * em[0] < 0) ee = [-ee[0], -ee[1]];
  let x = [ee[0] + em[1], ee[1] - em[0]];                 // ee - flipud(em) * [-1, 1]
  const xn = Math.hypot(x[0], x[1]);
  const scale = Math.max(Math.hypot(ee[0], ee[1]) * 2.0, Math.hypot(em[0], em[1]) * 1.8) * 1.046;
  x = [x[0] / xn * scale, x[1] / xn * scale];
  const y = [-x[1], x[0]];                                 // flipud(x) * [-1, 1]
  const c = [eyeAvg[0] + em[0] * 0.1, eyeAvg[1] + em[1] * 0.1];
  return [[c[0] - x[0] - y[0], c[1] - x[1] - y[1]], [c[0] - x[0] + y[0], c[1] - x[1] + y[1]],
    [c[0] + x[0] + y[0], c[1] + x[1] + y[1]], [c[0] + x[0] - y[0], c[1] + x[1] - y[1]]];
}

// Does the landmark layout look like a face? (see blemish._plausible)
export function plausible(f) {
  const eyeAvg = [(f[4] + f[6]) / 2, (f[5] + f[7]) / 2];
  const mouthAvg = [(f[10] + f[12]) / 2, (f[11] + f[13]) / 2];
  const ee = [f[6] - f[4], f[7] - f[5]], em = [mouthAvg[0] - eyeAvg[0], mouthAvg[1] - eyeAvg[1]];
  const eyeD = Math.hypot(ee[0], ee[1]), emD = Math.hypot(em[0], em[1]);
  if (eyeD < 0.28 * f[2] || emD < 1e-6) return false;
  const cos = Math.abs(ee[0] * em[0] + ee[1] * em[1]) / (eyeD * emD);
  const tNose = ((f[8] - eyeAvg[0]) * em[0] + (f[9] - eyeAvg[1]) * em[1]) / (emD * emD);
  return emD / eyeD >= 0.6 && emD / eyeD <= 2.2 && cos <= 0.12 && tNose >= 0.35 && tNose <= 1.0;
}

// Crop squares of the faces worth processing; largest first, a detection whose
// landmarks sit inside an already chosen face's box is the same face again.
export function selectFaces(faces) {
  const chosen = [];
  const sorted = [...faces].sort((a, b) => b[2] * b[3] - a[2] * a[3]);
  for (const f of sorted) {
    if (!plausible(f)) continue;
    const quad = ffhqQuad(f);
    if (Math.hypot(quad[3][0] - quad[0][0], quad[3][1] - quad[0][1]) < MIN_QUAD) continue;
    let cx = 0, cy = 0;
    for (let k = 0; k < 5; k++) { cx += f[4 + 2 * k]; cy += f[5 + 2 * k]; }
    cx /= 5; cy /= 5;
    if (chosen.some(([g]) => g[0] <= cx && cx <= g[0] + g[2] && g[1] <= cy && cy <= g[1] + g[3])) continue;
    chosen.push([f, quad]);
  }
  return chosen.map(([, q]) => q);
}

let WEIGHTS = null;            // cv.Mat CV_32FC2: [blend, fade] per crop pixel

function cropWeights(cv) {
  const fadePx = Math.floor(SIZE / 24);
  const ramp = new Float32Array(SIZE);
  for (let i = 0; i < SIZE; i++) {
    let r = Math.min(Math.max(Math.min(i, SIZE - 1 - i) / fadePx, 0), 1);
    ramp[i] = Math.fround(r * r * (3 - 2 * r));
  }
  const data = new Float32Array(SIZE * SIZE * 2);
  const half = (SIZE - 1) / 2;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const fade = Math.min(ramp[y], ramp[x]);
      const rr = Math.hypot(Math.fround(y - half), Math.fround(x - half)) / (SIZE / 2);
      const i = 2 * (y * SIZE + x);
      data[i] = fade * Math.exp(-((rr / 0.6) ** 2));
      data[i + 1] = fade;
    }
  }
  return cv.matFromArray(SIZE, SIZE, cv.CV_32FC2, data);
}

// One face: returns { x0, y0, corr: Mat CV_32FC3, wts: Mat CV_32FC2 } or null.
async function correctionForFace(cv, ort, session, bgr, quad) {
  const h = bgr.rows, w = bgr.cols;
  const qsize = Math.hypot(quad[3][0] - quad[0][0], quad[3][1] - quad[0][1]);
  if (qsize < MIN_QUAD) return null;
  const xs = quad.map((p) => p[0]), ys = quad.map((p) => p[1]);
  const x0 = Math.trunc(Math.max(0, Math.floor(Math.min(...xs))));
  const y0 = Math.trunc(Math.max(0, Math.floor(Math.min(...ys))));
  const x1 = Math.trunc(Math.min(w, Math.ceil(Math.max(...xs)) + 1));
  const y1 = Math.trunc(Math.min(h, Math.ceil(Math.max(...ys)) + 1));
  if (x1 - x0 < 16 || y1 - y0 < 16) return null;

  const roiView = bgr.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
  let src = roiView, f = 1.0;
  if (qsize > SIZE * 1.2) {
    f = SIZE / qsize;
    src = new cv.Mat();
    cv.resize(roiView, src, new cv.Size(0, 0), f, f, cv.INTER_AREA);
  }
  const q = quad.map(([px, py]) => [Math.fround((px - x0) * f), Math.fround((py - y0) * f)]);
  const srcPts = cv.matFromArray(3, 1, cv.CV_32FC2, [...q[0], ...q[1], ...q[3]]);
  const dstPts = cv.matFromArray(3, 1, cv.CV_32FC2, [0, 0, 0, SIZE, SIZE, 0]);
  const M = cv.getAffineTransform(srcPts, dstPts);           // CV_64F 2x3
  srcPts.delete(); dstPts.delete();
  const crop = new cv.Mat();
  cv.warpAffine(src, crop, M, new cv.Size(SIZE, SIZE), cv.INTER_LINEAR, cv.BORDER_REFLECT);
  if (src !== roiView) src.delete();
  roiView.delete();

  // Network: BGR float NCHW in [0,1] -> correction = clip(out)*255 - crop.
  const plane = SIZE * SIZE, input = new Float32Array(3 * plane), cd = crop.data;
  for (let i = 0; i < plane; i++) {
    input[i] = cd[3 * i] / 255; input[plane + i] = cd[3 * i + 1] / 255; input[2 * plane + i] = cd[3 * i + 2] / 255;
  }
  const res = await session.run({ input: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) });
  const o = res.output.data;
  const corrData = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let k = 0; k < 3; k++) {
      const v = Math.min(Math.max(o[k * plane + i], 0), 1);
      corrData[3 * i + k] = Math.fround(Math.fround(v * 255) - cd[3 * i + k]);
    }
  }
  res.output.dispose?.();
  crop.delete();
  const corrSmall = cv.matFromArray(SIZE, SIZE, cv.CV_32FC3, corrData);

  if (!WEIGHTS) WEIGHTS = cropWeights(cv);
  // Map back onto the full-resolution region (inverse of the crop warp).
  const Mf = copyMat(cv, M);
  for (const i of [0, 1, 3, 4]) Mf.data64F[i] *= f;
  M.delete();
  const back = (mat) => {
    const dst = new cv.Mat();
    cv.warpAffine(mat, dst, Mf, new cv.Size(x1 - x0, y1 - y0), cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
      cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
    return dst;
  };
  const corr = back(corrSmall), wts = back(WEIGHTS);
  corrSmall.delete(); Mf.delete();
  return { x0, y0, corr, wts };
}

// All faces -> { x0, y0, w, h, data: Float32Array (h*w*3, 0..255 units) } or null.
export async function correction(cv, ort, session, bgr, faces) {
  const parts = [];
  for (const quad of selectFaces(faces)) {
    const p = await correctionForFace(cv, ort, session, bgr, quad);
    if (p) parts.push(p);
  }
  if (!parts.length) return { corr: null, count: 0 };
  const X0 = Math.min(...parts.map((p) => p.x0)), Y0 = Math.min(...parts.map((p) => p.y0));
  const X1 = Math.max(...parts.map((p) => p.x0 + p.corr.cols)), Y1 = Math.max(...parts.map((p) => p.y0 + p.corr.rows));
  const W = X1 - X0, H = Y1 - Y0;
  const num = new Float32Array(W * H * 3), den = new Float32Array(W * H), fade = new Float32Array(W * H);
  for (const p of parts) {
    const c = p.corr.data32F, wt = p.wts.data32F, pw = p.corr.cols, ph = p.corr.rows;
    for (let y = 0; y < ph; y++) {
      for (let x = 0; x < pw; x++) {
        const i = y * pw + x, j = (y + p.y0 - Y0) * W + (x + p.x0 - X0);
        const b = wt[2 * i];
        num[3 * j] += c[3 * i] * b; num[3 * j + 1] += c[3 * i + 1] * b; num[3 * j + 2] += c[3 * i + 2] * b;
        den[j] += b;
        if (wt[2 * i + 1] > fade[j]) fade[j] = wt[2 * i + 1];
      }
    }
    p.corr.delete(); p.wts.delete();
  }
  for (let j = 0; j < W * H; j++) {
    const s = fade[j] / Math.max(den[j], 1e-6);
    num[3 * j] *= s; num[3 * j + 1] *= s; num[3 * j + 2] *= s;
  }
  return { corr: { x0: X0, y0: Y0, w: W, h: H, data: num }, count: parts.length };
}

// bgr + amount * corr -> new BGR Mat (blemish.apply).
export function apply(cv, bgr, corr, amount = 1.0) {
  amount = Math.min(Math.max(amount, 0), 1);
  const out = copyMat(cv, bgr);
  if (!corr || amount <= 0) return out;
  const d = out.data, W = bgr.cols, c = corr.data;
  amount = Math.fround(amount);                          // NumPy casts the scalar to float32
  for (let y = 0; y < corr.h; y++) {
    for (let x = 0; x < corr.w; x++) {
      const i = 3 * ((y + corr.y0) * W + (x + corr.x0)), j = 3 * (y * corr.w + x);
      for (let k = 0; k < 3; k++) {
        const v = Math.fround(Math.fround(d[i + k] + Math.fround(amount * c[j + k])) + 0.5);
        d[i + k] = v <= 0 ? 0 : v >= 255 ? 255 : Math.trunc(v);
      }
    }
  }
  return out;
}

// Same correction at a smaller size (for the on-screen preview).
export function scaleCorrection(cv, corr, s) {
  if (!corr) return null;
  const m = cv.matFromArray(corr.h, corr.w, cv.CV_32FC3, corr.data);
  const w = Math.max(1, Math.round(corr.w * s)), h = Math.max(1, Math.round(corr.h * s));
  const dst = new cv.Mat();
  cv.resize(m, dst, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
  const out = { x0: Math.round(corr.x0 * s), y0: Math.round(corr.y0 * s), w, h, data: new Float32Array(dst.data32F) };
  m.delete(); dst.delete();
  return out;
}
