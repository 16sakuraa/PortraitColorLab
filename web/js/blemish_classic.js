// Blemish removal with classic image processing - a port of
// skin_smoother/blemish_classic.py. No network and no model file.
//
// Finds small dark or red spots with a morphological black-hat on the green
// channel, keeps what looks like a spot, then fills each from the skin around it
// and reduces the fine detail inside it. Gives the same kind of correction as
// blemish.js, so blemish.apply and blemish.scaleCorrection work on it unchanged.
//
// Allocating a Mat can move the WebAssembly heap, so values that must outlive
// the next allocation are copied into plain arrays first.

import { median, pyRound } from './skinmask.js';
import {
  TUNE as SMOOTH, WORK_PIXELS, STRIP_ROWS, MIN_SPLIT_SIGMA,
  detailStrip, faceWidths, medianOf, scaleRegions, softBlur,
} from './smooth.js';

const WORK_FACE_PX = 450, MIN_FACE_PX = 80, MIN_SPOT_PX = 4;

// Same settings as blemish_classic.TUNE; see the notes there.
export const TUNE = {
  presmooth: 0.003,
  window: 0.045,
  edge_margin: 0.015,
  threshold: 7.0,
  min_contrast: 0.035,
  max_contrast: 0.25,
  max_red: 0.1,
  min_size: 0.007,
  max_size: 0.04,
  max_aspect: 2.5,
  min_fill: 0.4,
  isolation: 0.6,
  island_size: 0.05,
  grow: 0.008,
  fill_sigma: 0.015,
  feather: 0.003,
  detail_cut: 0.8,
  near_face: 0.5,
};

const odd = (v, minimum = 3) => Math.max(minimum, pyRound(v)) | 1;
const ellipse = (cv, size) => cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size, size));

function byteMat(cv, rows, cols, test) {
  const m = new cv.Mat(rows, cols, cv.CV_8U), v = m.data;
  for (let i = 0; i < v.length; i++) v[i] = test(i) ? 1 : 0;
  return m;
}

// Connected shapes of a CV_8U Mat, as plain arrays. Deletes the Mat.
function shapes(cv, mat) {
  const labels = new cv.Mat(), stats = new cv.Mat(), centroids = new cv.Mat();
  const n = cv.connectedComponentsWithStats(mat, labels, stats, centroids, 8, cv.CV_32S);
  const out = { n, labels: new Int32Array(labels.data32S), stats: new Int32Array(stats.data32S) };
  mat.delete(); labels.delete(); stats.delete(); centroids.delete();
  return out;
}

// Spots on one group of similarly sized faces: { spots: CV_8U Mat of 0/1 or null, count }.
// green, red: CV_32F 0..1. mask: plain array, zero outside this group. faceW: face width in these pixels.
function findSpots(cv, green, red, mask, faceW) {
  const rows = green.rows, cols = green.cols, n = rows * cols, t = TUNE;
  const none = { spots: null, count: 0 };

  // A light blur first: spots are larger than pixel grain, and it makes the
  // statistics below continuous. On raw 8-bit values the median moves in whole
  // steps, so one rounding difference could shift the threshold a lot.
  const blur = Math.max(0.7, faceW * t.presmooth);
  const softGreen = softBlur(cv, green, blur), softRed = softBlur(cv, red, blur);
  const se = ellipse(cv, odd(faceW * t.window)), bhMat = new cv.Mat(), bhRedMat = new cv.Mat();
  cv.morphologyEx(softGreen, bhMat, cv.MORPH_BLACKHAT, se);
  cv.morphologyEx(softRed, bhRedMat, cv.MORPH_BLACKHAT, se);
  se.delete(); softGreen.delete();
  const bh = new Float32Array(bhMat.data32F);
  bhMat.delete();
  // How much darker than its surroundings in red: a share, so it holds on any skin tone.
  const redDip = new Float32Array(n);
  {
    const b = bhRedMat.data32F, s = softRed.data32F;
    for (let i = 0; i < n; i++) redDip[i] = b[i] / (s[i] + b[i] + 1e-6);
  }
  bhRedMat.delete(); softRed.delete();

  const margin = Math.max(1, pyRound(faceW * t.edge_margin));
  const inside = byteMat(cv, rows, cols, (i) => mask[i] > 0.9), coreMat = new cv.Mat();
  const box = cv.Mat.ones(2 * margin + 1, 2 * margin + 1, cv.CV_8U);
  cv.erode(inside, coreMat, box);
  inside.delete(); box.delete();
  const core = new Uint8Array(coreMat.data);
  coreMat.delete();

  let size = 0;
  for (let i = 0; i < n; i++) if (core[i]) size++;
  if (size < 100) return none;
  const values = new Float32Array(size);
  for (let i = 0, j = 0; i < n; i++) if (core[i]) values[j++] = bh[i];
  const med = median(values);
  const mad = median(values.map((v) => Math.abs(v - med)));
  const threshold = Math.max(t.min_contrast, med + t.threshold * mad);

  // Shapes at a lower threshold show what a candidate is part of. A real spot
  // stays a small island; on hair, stubble or a crease it joins up into
  // something much larger.
  const lower = t.isolation * threshold;
  const loose = shapes(cv, byteMat(cv, rows, cols, (i) => bh[i] > lower && mask[i] > 0.3));
  const longest = faceW * t.island_size;
  const tooBig = new Uint8Array(loose.n);
  for (let i = 0; i < loose.n; i++) {
    tooBig[i] = Math.max(loose.stats[5 * i + 2], loose.stats[5 * i + 3]) > longest ? 1 : 0;
  }

  const found = shapes(cv, byteMat(cv, rows, cols, (i) => bh[i] > threshold && core[i]));
  const minArea = Math.max(MIN_SPOT_PX, Math.PI * (faceW * t.min_size / 2) ** 2);
  const maxArea = Math.PI * (faceW * t.max_size / 2) ** 2;
  const keep = new Uint8Array(found.n);
  let count = 0;
  for (let i = 1; i < found.n; i++) {
    const s = 5 * i, x = found.stats[s], y = found.stats[s + 1], w = found.stats[s + 2], h = found.stats[s + 3], area = found.stats[s + 4];
    if (area < minArea || area > maxArea) continue;
    if (Math.max(w, h) > t.max_aspect * Math.min(w, h) || area < t.min_fill * w * h) continue;
    // Which island it's in (the one holding most of it), and its darkest point.
    const tally = new Map();
    let peak = -Infinity, peakRed = -Infinity;
    for (let yy = y; yy < y + h; yy++) {
      for (let xx = x, p = yy * cols + x; xx < x + w; xx++, p++) {
        if (found.labels[p] !== i) continue;
        tally.set(loose.labels[p], (tally.get(loose.labels[p]) || 0) + 1);
        if (bh[p] > peak) peak = bh[p];
        if (redDip[p] > peakRed) peakRed = redDip[p];
      }
    }
    let island = 0, most = -1;
    for (const [label, c] of [...tally].sort((a, b) => a[0] - b[0])) if (c > most) { island = label; most = c; }
    if (tooBig[island] || peak > t.max_contrast || peakRed > t.max_red) continue;
    keep[i] = 1;
    count++;
  }
  if (!count) return none;

  const marked = byteMat(cv, rows, cols, (i) => keep[found.labels[i]]), spots = new cv.Mat();
  const grow = ellipse(cv, odd(faceW * t.grow));
  cv.dilate(marked, spots, grow);
  marked.delete(); grow.delete();
  return { spots, count };
}

// { corr: { x0, y0, w, h, data: Float32Array (h*w*3, 0..255 units) } or null, count }
// count is the number of spots. Compute once per photo; blemish.apply is then cheap.
export function correction(cv, bgr, faces, mask) {
  const none = { corr: null, count: 0 };
  if (!faces || !faces.length) return none;
  const t = TUNE, H = bgr.rows, W = bgr.cols, diag = Math.hypot(H, W);

  // Stay near the faces: the mask may cover arms and legs too, which would use up
  // the working pixels and leave the faces too small to find spots on. Boxes here
  // are in the mask's own pixels.
  const near = faces.map((f) => {
    const reach = t.near_face * Math.max(f[2], f[3]);
    return [
      Math.max(0, Math.trunc(Math.max(0, f[0] - reach)) - mask.x0), Math.max(0, Math.trunc(Math.max(0, f[1] - reach)) - mask.y0),
      Math.min(mask.w, Math.trunc(Math.min(W, f[0] + f[2] + reach)) - mask.x0), Math.min(mask.h, Math.trunc(Math.min(H, f[1] + f[3] + reach)) - mask.y0),
    ];
  });
  let minX = Infinity, maxX = -1, minY = Infinity, maxY = -1;
  for (const [bx0, by0, bx1, by1] of near) {
    for (let y = by0; y < by1; y++) {
      for (let x = bx0; x < bx1; x++) {
        if (mask.data[y * mask.w + x] > 1e-3) {
          if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y;
        }
      }
    }
  }
  if (maxX < 0) return none;
  const widths = faceWidths(faces, diag), widest = Math.max(...widths);
  const pad = Math.ceil(widest * t.fill_sigma * 4) + 8;
  const y0 = Math.max(0, minY + mask.y0 - pad), y1 = Math.min(H, maxY + mask.y0 + 1 + pad);
  const x0 = Math.max(0, minX + mask.x0 - pad), x1 = Math.min(W, maxX + mask.x0 + 1 + pad);
  const roi = bgr.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
  const rh = roi.rows, rw = roi.cols;

  const maskR = new Float32Array(rw * rh);
  for (const [bx0, by0, bx1, by1] of near) {
    for (let my = by0; my < by1; my++) {
      const y = my + mask.y0 - y0;
      if (y < 0 || y >= rh) continue;
      for (let mx = bx0; mx < bx1; mx++) {
        const x = mx + mask.x0 - x0;
        if (x >= 0 && x < rw) maskR[y * rw + x] = mask.data[my * mask.w + mx];
      }
    }
  }

  const scale = Math.min(1.0, WORK_FACE_PX / widest, Math.sqrt(WORK_PIXELS / (rh * rw)));
  let reduced = roi, maskW = maskR;
  if (scale < 1) {
    const size = new cv.Size(Math.max(1, pyRound(rw * scale)), Math.max(1, pyRound(rh * scale)));
    reduced = new cv.Mat();
    cv.resize(roi, reduced, size, 0, 0, cv.INTER_AREA);
    const full = cv.matFromArray(rh, rw, cv.CV_32F, maskR), small = new cv.Mat();
    cv.resize(full, small, size, 0, 0, cv.INTER_AREA);
    maskW = new Float32Array(small.data32F);
    full.delete(); small.delete();
  }
  const work = new cv.Mat();
  reduced.convertTo(work, cv.CV_32F, 1 / 255);
  if (reduced !== roi) reduced.delete();
  const rows = work.rows, cols = work.cols, n = rows * cols;
  const planes = new cv.MatVector();
  cv.split(work, planes);
  const green = planes.get(1), red = planes.get(2);
  planes.get(0).delete();
  planes.delete();

  const sigma = Math.max(MIN_SPLIT_SIGMA, medianOf(widths) * SMOOTH.split);
  let low;
  if (sigma * scale >= 0.5) {
    const blurred = softBlur(cv, work, sigma * scale);
    low = new Float32Array(blurred.data32F);
    blurred.delete();
  } else {
    low = new Float32Array(work.data32F);
  }
  work.delete();

  const delta = new Float32Array(3 * n), softAll = new Float32Array(n);
  let count = 0;
  for (const { width, weight } of scaleRegions(cv, faces, widths, scale, x0, y0, rows, cols)) {
    const faceW = width * scale;
    if (faceW < MIN_FACE_PX) continue;
    const groupMask = weight ? maskW.map((m, i) => (weight[i] > 0.5 ? m : 0)) : maskW;
    const { spots, count: found } = findSpots(cv, green, red, groupMask, faceW);
    if (!found) continue;
    count += found;

    const spotsF = new cv.Mat(), softMat = new cv.Mat();
    spots.convertTo(spotsF, cv.CV_32F);
    spots.delete();
    cv.GaussianBlur(spotsF, softMat, new cv.Size(0, 0), Math.max(0.7, faceW * t.feather));
    const soft = new Float32Array(softMat.data32F);
    const isSpot = new Float32Array(spotsF.data32F);
    spotsF.delete(); softMat.delete();
    for (let i = 0; i < n; i++) soft[i] = Math.min(soft[i] * 1.5, 1);

    // Read only surrounding skin: not the spots, and not hair or eyes.
    const valid = maskW.map((m, i) => (1 - isSpot[i]) * m);
    const validMat = cv.matFromArray(rows, cols, cv.CV_32F, valid);
    const weighted = new Float32Array(3 * n);
    for (let i = 0, j = 0; i < n; i++, j += 3) {
      weighted[j] = low[j] * valid[i]; weighted[j + 1] = low[j + 1] * valid[i]; weighted[j + 2] = low[j + 2] * valid[i];
    }
    const weightedMat = cv.matFromArray(rows, cols, cv.CV_32FC3, weighted);
    const sigmaFill = Math.max(1.5, faceW * t.fill_sigma);
    const numMat = softBlur(cv, weightedMat, sigmaFill), denMat = softBlur(cv, validMat, sigmaFill);
    const num = numMat.data32F, den = denMat.data32F;
    for (let i = 0, j = 0; i < n; i++, j += 3) {
      const d = den[i] + 1e-6, s = soft[i];
      delta[j] += s * (num[j] / d - low[j]);
      delta[j + 1] += s * (num[j + 1] / d - low[j + 1]);
      delta[j + 2] += s * (num[j + 2] / d - low[j + 2]);
      if (s > softAll[i]) softAll[i] = s;
    }
    validMat.delete(); weightedMat.delete(); numMat.delete(); denMat.delete();
  }
  green.delete(); red.delete();
  if (!count) { roi.delete(); return none; }

  // Back to full resolution, cut to the area the spots cover.
  let deltaFull = delta, softFull = softAll;
  if (scale < 1) {
    const size = new cv.Size(rw, rh);
    const up = (rowsData, type) => {
      const small = cv.matFromArray(rows, cols, type, rowsData), big = new cv.Mat();
      cv.resize(small, big, size, 0, 0, cv.INTER_LINEAR);
      const copy = new Float32Array(big.data32F);
      small.delete(); big.delete();
      return copy;
    };
    deltaFull = up(delta, cv.CV_32FC3);
    softFull = up(softAll, cv.CV_32F);
  }
  let bx0 = rw, bx1 = 0, by0 = rh, by1 = 0;
  for (let y = 0, i = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++, i++) {
      if (softFull[i] > 0) {
        if (x < bx0) bx0 = x; if (x >= bx1) bx1 = x + 1; if (y < by0) by0 = y; if (y >= by1) by1 = y + 1;
      }
    }
  }
  const bw = bx1 - bx0, bh = by1 - by0;
  const data = new Float32Array(3 * bw * bh);
  const roiBox = roi.roi(new cv.Rect(bx0, by0, bw, bh));
  const cut = t.detail_cut * 255;
  for (let top = 0; top < bh; top += STRIP_ROWS) {
    const bottom = Math.min(bh, top + STRIP_ROWS);
    const { high } = detailStrip(cv, roiBox, top, bottom, sigma);
    for (let y = top, p = 0; y < bottom; y++) {
      const from = (y + by0) * rw + bx0;
      for (let x = 0; x < bw; x++, p += 3) {
        const i = from + x, j = 3 * i, s = cut * softFull[i], o = 3 * (y * bw + x);
        data[o] = deltaFull[j] * 255 - s * high[p];
        data[o + 1] = deltaFull[j + 1] * 255 - s * high[p + 1];
        data[o + 2] = deltaFull[j + 2] * 255 - s * high[p + 2];
      }
    }
  }
  roiBox.delete(); roi.delete();
  return { corr: { x0: x0 + bx0, y0: y0 + by0, w: bw, h: bh, data }, count };
}
