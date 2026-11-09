// Skin smoothing - a port of smoother.smooth_face and its helpers.
//
// The region is split into a tone layer and a detail layer, using sizes relative
// to the face. The tone layer gets guided filters (brightness over a small radius,
// colour over a wider one) with the large-scale part of the change removed again,
// so shading stays. The detail layer is treated by how strong it is at each pixel:
// pores follow Keep texture, stronger detail such as fine lines is reduced more,
// and the strongest (real edges, lashes, stray hairs) is left alone.
//
// Full-resolution work runs a strip of rows at a time, so a large photo never
// needs a whole-region float copy. Allocating a Mat can move the WebAssembly
// heap, so views like mat.data32F are read after the allocations they follow.

import { median, pyRound } from './skinmask.js';
import { copyMat } from './cv.js';

export const WORK_PIXELS = 1.5e6;
export const STRIP_ROWS = 256;

// Same settings as smoother.TUNE; see the notes there.
export const TUNE = {
  split: 0.006,
  split_edge: 0.08,
  luma_radius: 0.015,
  luma_edge: 0.045,
  chroma_radius: 0.05,
  chroma_edge: 0.05,
  contour: 1.0,
  contour_ratio: 1.5,
  line_lo: 2.5,
  line_hi: 5.0,
  line_cut: 0.8,
  keep_lo: 9.0,
  keep_hi: 16.0,
  keep_colour_lo: 3.5,
  keep_colour_hi: 7.0,
  pore_size: 0.0012,
  pore_band: 2.0,
  pore_lean: 0.5,
  pore_depth: 1.0,
  pore_min_level: 0.006,
  shine_lift: 0.02,
  shine_span: 0.08,
  shine_lo: 2.0,
  shine_hi: 4.0,
  shine_colour_lo: 1.0,
  shine_colour_hi: 2.0,
};
export const MIN_SPLIT_SIGMA = 0.6;
const MIN_PORE_SIGMA = 0.6, MIN_PORE_GAIN = 0.15, PORE_SEED = 0x9E3779B9 | 0, PORE_PATCH = 256, SOFT_BLUR_MIN = 1.4;
const NO_FACE_WIDTH = 0.25, GROUP_RATIO = 1.5, MIN_DETAIL_LEVEL = 0.002;
const SPLIT_RADIUS_PER_SIGMA = 1.2;
const LEVEL_BANDS = 8, LEVEL_BAND_ROWS = 32;
const LUMA_B = 0.114, LUMA_G = 0.587, LUMA_R = 0.299;
// Print size (smoother.PRINT_GRAIN): a small print keeps coarser grain, because
// detail finer than about one printed pixel disappears in it.
export const PRINT_GRAIN = { large: 1.0, small: 2.0 };
export const MAX_GRAIN = 2.5;
export const MIN_GRAIN = 0.25;   // finer grain, for smoothing a painted area again (heal.js's Smooth again)
// Lighting (smoother.LIGHTING): hard or patchy light leaves mottled light and
// shadow, evened out by a wider brightness radius; shading is still kept.
export const LIGHTING = { soft: 1.0, hard: 2.0 };
export const MAX_LIGHT = 3.0;

const clamp01 = (v) => Math.min(Math.max(v, 0), 1);

export function smoothstep(x, lo, hi) {
  let t = (x - lo) / (hi - lo);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return t * t * (3 - 2 * t);
}

// Guided filter (He, Sun, Tang, "Guided Image Filtering", ECCV 2010): an
// edge-preserving blur of src that follows the edges of guide. Both are float
// Mats with the same channels (each channel is filtered on its own).
export function guidedFilter(cv, guide, src, radius, eps) {
  const size = new cv.Size(2 * radius + 1, 2 * radius + 1), anchor = new cv.Point(-1, -1);
  const box = (m) => { const o = new cv.Mat(); cv.boxFilter(m, o, cv.CV_32F, size, anchor, true, cv.BORDER_DEFAULT); return o; };
  const boxOfProduct = (x, y) => {
    const p = new cv.Mat();
    cv.multiply(x, y, p);
    const o = box(p);
    p.delete();
    return o;
  };
  const rows = guide.rows, cols = guide.cols, type = guide.type(), n = rows * cols * guide.channels();
  const meanG = box(guide), meanS = box(src);
  const boxGG = boxOfProduct(guide, guide), boxGS = boxOfProduct(guide, src);
  const a = new cv.Mat(rows, cols, type), b = new cv.Mat(rows, cols, type);
  {
    const mg = meanG.data32F, ms = meanS.data32F, gg = boxGG.data32F, gs = boxGS.data32F, av = a.data32F, bv = b.data32F;
    for (let i = 0; i < n; i++) {
      const varG = gg[i] - mg[i] * mg[i], cov = gs[i] - mg[i] * ms[i];
      const ai = cov / (varG + eps);
      av[i] = ai;
      bv[i] = ms[i] - ai * mg[i];
    }
  }
  meanG.delete(); meanS.delete(); boxGG.delete(); boxGS.delete();
  const boxA = box(a), boxB = box(b);
  a.delete(); b.delete();
  const out = new cv.Mat(rows, cols, type);
  {
    const av = boxA.data32F, bv = boxB.data32F, g = guide.data32F, o = out.data32F;
    for (let i = 0; i < n; i++) o[i] = av[i] * g[i] + bv[i];
  }
  boxA.delete(); boxB.delete();
  return out;
}

// Three box blurs: practically a Gaussian of this sigma, at a cost that doesn't
// grow with it (the wide blurs here would otherwise dominate the run time).
// Returns a new Mat.
export function softBlur(cv, a, sigma) {
  const radius = Math.max(1, pyRound((Math.sqrt(4 * sigma * sigma + 1) - 1) / 2));
  const size = new cv.Size(2 * radius + 1, 2 * radius + 1), anchor = new cv.Point(-1, -1);
  const out = new cv.Mat();
  cv.boxFilter(a, out, cv.CV_32F, size, anchor, true, cv.BORDER_DEFAULT);
  cv.boxFilter(out, out, cv.CV_32F, size, anchor, true, cv.BORDER_DEFAULT);
  cv.boxFilter(out, out, cv.CV_32F, size, anchor, true, cv.BORDER_DEFAULT);
  return out;
}

export const faceWidths = (faces, diag) => (faces && faces.length ? faces.map((f) => f[2]) : [NO_FACE_WIDTH * diag]);

// The size (px) smoothFace splits tone from detail at, for a photo this size: the grain
// it keeps. The light brush (heal.js) paints tone only, at the same size.
export function detailSigma(rows, cols, faces, grain = 1) {
  grain = Math.min(Math.max(grain, MIN_GRAIN), MAX_GRAIN);
  return Math.max(MIN_SPLIT_SIGMA, medianOf(faceWidths(faces, Math.hypot(rows, cols))) * TUNE.split * grain);
}

export function medianOf(values) {
  const a = [...values].sort((x, y) => x - y), mid = a.length >> 1;
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

// [{ width, weight }]: faces of similar size share one smoothing pass. Very
// different sizes get separate passes, blended by a soft map of which face each
// pixel is nearest to. Most photos have one group, which needs no map (null).
export function scaleRegions(cv, faces, widths, scale, x0, y0, rows, cols) {
  const order = widths.map((_, i) => i).sort((a, b) => widths[a] - widths[b]);
  const groups = [];
  for (const i of order) {
    const last = groups[groups.length - 1];
    if (last && widths[i] <= GROUP_RATIO * widths[last[0]]) last.push(i);
    else groups.push([i]);
  }
  const meanWidth = groups.map((g) => Math.exp(g.reduce((s, i) => s + Math.log(widths[i]), 0) / g.length));
  if (groups.length === 1) return [{ width: meanWidth[0], weight: null }];

  const n = rows * cols, best = new Float32Array(n).fill(Infinity), nearest = new Int32Array(n);
  groups.forEach((members, g) => {
    for (const i of members) {
      const f = faces[i];
      const cx = (f[0] + f[2] / 2 - x0) * scale, cy = (f[1] + f[3] / 2 - y0) * scale, size = f[2] * scale;
      for (let y = 0, p = 0; y < rows; y++) {
        const dy = y - cy;
        for (let x = 0; x < cols; x++, p++) {
          const dx = x - cx, d = Math.sqrt(dx * dx + dy * dy) / size;
          if (d < best[p]) { best[p] = d; nearest[p] = g; }
        }
      }
    }
  });
  // Soft edges, but narrower than the smallest face so it fully gets its own pass.
  const sigma = Math.max(2.0, 0.25 * Math.min(...widths) * scale);
  const maps = groups.map((_, g) => {
    const m = new cv.Mat(rows, cols, cv.CV_32F), blurred = new cv.Mat();
    const v = m.data32F;
    for (let i = 0; i < n; i++) v[i] = nearest[i] === g ? 1 : 0;
    cv.GaussianBlur(m, blurred, new cv.Size(0, 0), sigma);
    const copy = new Float32Array(blurred.data32F);
    m.delete(); blurred.delete();
    return copy;
  });
  const total = new Float32Array(n).fill(1e-6);
  for (const m of maps) for (let i = 0; i < n; i++) total[i] += m[i];
  for (const m of maps) for (let i = 0; i < n; i++) m[i] /= total[i];
  return groups.map((_, g) => ({ width: meanWidth[g], weight: maps[g] }));
}

// What to add to each pixel's tone layer, on a reduced copy of the region:
// a CV_32FC3 Mat (B, G, R), working rows by full width. Caller deletes it.
// light widens the brightness radius for hard light (LIGHTING).
function toneChange(cv, roi, maskMat, faces, widths, x0, y0, sigma, scale, strength, light = 1) {
  const rh = roi.rows, rw = roi.cols;
  let reduced = roi, maskW = maskMat;
  if (scale < 1) {
    const size = new cv.Size(Math.max(1, pyRound(rw * scale)), Math.max(1, pyRound(rh * scale)));
    reduced = new cv.Mat();
    cv.resize(roi, reduced, size, 0, 0, cv.INTER_AREA);
    maskW = new cv.Mat();
    cv.resize(maskMat, maskW, size, 0, 0, cv.INTER_AREA);
  }
  const work = new cv.Mat();
  reduced.convertTo(work, cv.CV_32F, 1 / 255);
  if (reduced !== roi) reduced.delete();
  let toneInput = work;
  if (sigma * scale >= 0.5) {
    // Only from skin, or the backdrop's colour would mix into the skin at its edge.
    const s = sigma * scale, count = work.rows * work.cols;
    const weighted = new cv.Mat(work.rows, work.cols, cv.CV_32FC3);
    {
      const wv = weighted.data32F, v = work.data32F, m = maskW.data32F;
      for (let i = 0, j = 0; i < count; i++, j += 3) { wv[j] = v[j] * m[i]; wv[j + 1] = v[j + 1] * m[i]; wv[j + 2] = v[j + 2] * m[i]; }
    }
    work.delete();
    const sum = softBlur(cv, weighted, s), share = softBlur(cv, maskW, s);
    weighted.delete();
    toneInput = new cv.Mat(sum.rows, sum.cols, cv.CV_32FC3);
    {
      const tv = toneInput.data32F, sv = sum.data32F, d = share.data32F;
      for (let i = 0, j = 0; i < count; i++, j += 3) {
        const den = d[i] + 1e-4;
        tv[j] = sv[j] / den; tv[j + 1] = sv[j + 1] / den; tv[j + 2] = sv[j + 2] / den;
      }
    }
    sum.delete(); share.delete();
  }
  const ycc = new cv.Mat(), planes = new cv.MatVector();
  cv.cvtColor(toneInput, ycc, cv.COLOR_BGR2YCrCb);
  cv.split(ycc, planes);
  toneInput.delete(); ycc.delete();
  const Y = planes.get(0), Cr = planes.get(1), Cb = planes.get(2);
  planes.delete();
  const rows = Y.rows, cols = Y.cols, n = rows * cols;

  const dY = new Float32Array(n), dCr = new Float32Array(n), dCb = new Float32Array(n);
  const epsY = (TUNE.luma_edge * (0.5 + strength)) ** 2, epsC = (TUNE.chroma_edge * (0.5 + strength)) ** 2;

  // Blurred mask per sigma, as a plain array: the same one serves both colour planes.
  const maskBlurs = new Map();
  const maskBlur = (sigma) => {
    if (!maskBlurs.has(sigma)) {
      const blurred = softBlur(cv, maskW, sigma);
      maskBlurs.set(sigma, new Float32Array(blurred.data32F));
      blurred.delete();
    }
    return maskBlurs.get(sigma);
  };

  const addChange = (total, plane, radius, eps, weight) => {
    // Normalized convolution (Knutsson & Westin, 1993) reads only from skin, so
    // hair and background can't leak in. First for the filter's input: skin and a
    // coloured backdrop can be equally bright, and then the brightness guide can't
    // keep the backdrop's colour off the skin beside it. Everything outside the
    // skin takes the colour of the skin nearby.
    const nearby = maskBlur(radius), weighted = new cv.Mat();
    cv.multiply(plane, maskW, weighted);
    const nearbySum = softBlur(cv, weighted, radius);
    weighted.delete();
    const here = new cv.Mat(rows, cols, cv.CV_32F);
    {
      const hv = here.data32F, pv = plane.data32F, mv = maskW.data32F, sv = nearbySum.data32F;
      for (let i = 0; i < n; i++) hv[i] = mv[i] * pv[i] + (1 - mv[i]) * (sv[i] / (nearby[i] + 1e-4));
    }
    nearbySum.delete();
    // Then for the large-scale part of the change.
    const sigmaC = TUNE.contour_ratio * radius, den = maskBlur(sigmaC);
    const filtered = guidedFilter(cv, Y, here, radius, eps), d = new cv.Mat(), product = new cv.Mat();
    cv.subtract(filtered, here, d);
    filtered.delete(); here.delete();
    cv.multiply(d, maskW, product);
    const num = softBlur(cv, product, sigmaC);
    product.delete();
    const dv = d.data32F, nv = num.data32F, c = TUNE.contour;
    if (weight) for (let i = 0; i < n; i++) total[i] += weight[i] * (dv[i] - c * nv[i] / (den[i] + 1e-4));
    else for (let i = 0; i < n; i++) total[i] += dv[i] - c * nv[i] / (den[i] + 1e-4);
    d.delete(); num.delete();
  };

  for (const { width, weight } of scaleRegions(cv, faces, widths, scale, x0, y0, rows, cols)) {
    const rY = Math.max(1, pyRound(width * scale * TUNE.luma_radius * light));
    const rC = Math.max(2, pyRound(width * scale * TUNE.chroma_radius));
    addChange(dY, Y, rY, epsY, weight);
    addChange(dCr, Cr, rC, epsC, weight);
    addChange(dCb, Cb, rC, epsC, weight);
  }
  Y.delete(); Cr.delete(); Cb.delete();
  if (maskW !== maskMat) maskW.delete();

  // The YCrCb change expressed in B, G, R (OpenCV's float conversion).
  let delta = new cv.Mat(rows, cols, cv.CV_32FC3);
  {
    const v = delta.data32F;
    for (let i = 0, j = 0; i < n; i++, j += 3) {
      v[j] = dY[i] + 1.773 * dCb[i];
      v[j + 1] = dY[i] - 0.714 * dCr[i] - 0.344 * dCb[i];
      v[j + 2] = dY[i] + 1.403 * dCr[i];
    }
  }
  if (cols !== rw) {
    const wide = new cv.Mat();
    cv.resize(delta, wide, new cv.Size(rw, rows), 0, 0, cv.INTER_LINEAR);
    delta.delete();
    delta = wide;
  }
  return delta;
}

// Rows y0..y1 of the region as floats, and their detail layer (both plain
// arrays, 3 values per pixel). The split is edge-preserving: with a plain blur,
// part of every strong edge would land in the detail layer, and reducing detail
// would then soften it. Filters with enough extra rows that strip edges can't show.
export function detailStrip(cv, roi, y0, y1, sigma) {
  const radius = Math.max(1, pyRound(SPLIT_RADIUS_PER_SIGMA * sigma)), margin = 2 * radius + 1;
  const a = Math.max(0, y0 - margin), b = Math.min(roi.rows, y1 + margin), cols = roi.cols;
  const view = roi.roi(new cv.Rect(0, a, cols, b - a));
  const part = new cv.Mat();
  view.convertTo(part, cv.CV_32F, 1 / 255);
  view.delete();
  const low = guidedFilter(cv, part, part, radius, TUNE.split_edge ** 2);
  const skip = 3 * cols * (y0 - a), n = 3 * cols * (y1 - y0);
  const pv = part.data32F, lv = low.data32F;
  const values = pv.slice(skip, skip + n), high = new Float32Array(n);
  for (let i = 0; i < n; i++) high[i] = values[i] - lv[skip + i];
  part.delete(); low.delete();
  return { values, high };
}

// Evenly spaced bands of rows [y0, y1] through skin well inside the mask: enough
// for a stable median (smoother._sample_bands). Empty without such skin.
function sampleBands(mask, rw, rh) {
  const rows = [];
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      if (mask[y * rw + x] > 0.9) { rows.push(y); break; }
    }
  }
  if (!rows.length) return [];
  const bands = [];
  for (let i = 0; i < LEVEL_BANDS; i++) {
    const y0 = rows[Math.floor((2 * i + 1) * rows.length / (2 * LEVEL_BANDS))];
    bands.push([y0, Math.min(rh, y0 + LEVEL_BAND_ROWS)]);
  }
  return bands;
}

// Median brightness detail on skin well inside the mask, in 0..1 units. Skin
// detail strength varies with exposure, sharpness and resolution, so the detail
// bands are multiples of this rather than fixed values.
function detailLevel(cv, roi, mask, sigma) {
  const rw = roi.cols, samples = [];
  for (const [y0, y1] of sampleBands(mask, rw, roi.rows)) {
    const { high } = detailStrip(cv, roi, y0, y1, sigma);
    for (let y = y0, p = 0; y < y1; y++) {
      for (let x = 0; x < rw; x++, p += 3) {
        if (mask[y * rw + x] > 0.9) samples.push(Math.abs(LUMA_B * high[p] + LUMA_G * high[p + 1] + LUMA_R * high[p + 2]));
      }
    }
  }
  return samples.length ? Math.max(MIN_DETAIL_LEVEL, median(samples)) : MIN_DETAIL_LEVEL;
}

// Median brightness (0..1) of the tone layer on skin well inside the mask, on the
// same rows as detailLevel; null without such skin (smoother._skin_tone). Reduce
// shine treats skin brighter than this as where shine can be.
function skinTone(cv, roi, mask, sigma) {
  const rw = roi.cols, samples = [];
  for (const [y0, y1] of sampleBands(mask, rw, roi.rows)) {
    const { values, high } = detailStrip(cv, roi, y0, y1, sigma);
    for (let y = y0, p = 0; y < y1; y++) {
      for (let x = 0; x < rw; x++, p += 3) {
        if (mask[y * rw + x] > 0.9) {
          samples.push(LUMA_B * (values[p] - high[p]) + LUMA_G * (values[p + 1] - high[p + 1]) + LUMA_R * (values[p + 2] - high[p + 2]));
        }
      }
    }
  }
  return samples.length ? median(samples) : null;
}

// Noise with structure at one size only (a plain array): noise in -1..1 from each
// pixel's position, so every run and both apps get the same values (the hash is
// the finalizer of MurmurHash3, see smoother._position_noise). Finer grain is then
// blurred away and anything larger subtracted, so it can't read as blotches.
function poreBand(cv, x0, y0, w, h, sigma) {
  const noise = new cv.Mat(h, w, cv.CV_32F);
  {
    const v = noise.data32F;
    for (let y = 0, p = 0; y < h; y++) {
      const row = ((y + y0) & 0xFFFF) << 16;
      for (let x = 0; x < w; x++, p++) {
        let k = (row | ((x + x0) & 0xFFFF)) ^ PORE_SEED;
        k ^= k >>> 16;
        k = Math.imul(k, 0x85EBCA6B);
        k ^= k >>> 13;
        k = Math.imul(k, 0xC2B2AE35);
        k ^= k >>> 16;
        v[p] = (k >>> 8) / 8388608 - 1;
      }
    }
  }
  let fine;
  if (sigma >= SOFT_BLUR_MIN) fine = softBlur(cv, noise, sigma);
  else { fine = new cv.Mat(); cv.GaussianBlur(noise, fine, new cv.Size(0, 0), sigma); }
  noise.delete();
  const coarse = softBlur(cv, fine, TUNE.pore_band * sigma);
  const band = new Float32Array(w * h);
  {
    const f = fine.data32F, c = coarse.data32F;
    for (let i = 0; i < band.length; i++) band[i] = f[i] - c[i];
  }
  fine.delete(); coarse.delete();
  return band;
}

// Pores are pits: deepen the dark side and flatten the light side.
const leanDark = (v) => v * (v < 0 ? 1 + TUNE.pore_lean : 1 - TUNE.pore_lean);

// { sigma, gain, spread, mean, weight } per group of similar face sizes. Faces too
// small to show pores get fainter ones, then none, so they don't pick up a grain
// that's coarse for their size. The fade is about how much real pores weaken when a
// photo is shrunk, so a preview shows roughly what the export will look like at that
// size. spread and mean are measured on a fixed patch, which keeps every strip on
// the same scale. weight is workRows x rw, or null. grain enlarges the pores for a
// small print.
function poreGroups(cv, regions, rw, workRows, workCols, grain = 1) {
  const groups = [];
  for (const { width, weight } of regions) {
    const size = width * TUNE.pore_size * grain, gain = Math.min(1, size / MIN_PORE_SIGMA);
    if (gain < MIN_PORE_GAIN) continue;
    const sigma = Math.max(MIN_PORE_SIGMA, size);
    const band = poreBand(cv, 0, 0, PORE_PATCH, PORE_PATCH, sigma);
    const spread = median(band.map(Math.abs));
    let total = 0;
    for (let i = 0; i < band.length; i++) total += leanDark(band[i] / spread);
    let resized = null;
    if (weight) {
      const m = cv.matFromArray(workRows, workCols, cv.CV_32F, weight), r = new cv.Mat();
      cv.resize(m, r, new cv.Size(rw, workRows), 0, 0, cv.INTER_LINEAR);
      resized = new Float32Array(r.data32F);
      m.delete(); r.delete();
    }
    groups.push({ sigma, gain, spread, mean: total / band.length, weight: resized });
  }
  return groups;
}

// One group's pattern for a strip: median size 1, averaging zero, leaning dark.
// Made with enough extra pixels that strip edges can't show.
function poreStrip(cv, group, x0, y0, w, h) {
  const { sigma, gain, spread, mean } = group;
  const reach = Math.ceil((4 + 3 * TUNE.pore_band) * sigma) + 4, stride = w + 2 * reach;
  const band = poreBand(cv, x0 - reach, y0 - reach, stride, h + 2 * reach, sigma);
  const out = new Float32Array(w * h);
  for (let y = 0, p = 0; y < h; y++) {
    let q = (y + reach) * stride + reach;
    for (let x = 0; x < w; x++, p++, q++) out[p] = gain * (leanDark(band[q] / spread) - mean);
  }
  return out;
}

// 0..1 for rows y0..y1: how strongly each pixel sits by a hard edge or a hair. The
// detail layer can't show this, because its edge-preserving split keeps strong
// edges out of it. A plain box blur doesn't, so what it changes does.
function nearEdge(cv, roi, y0, y1, radius, lo, hi) {
  const a = Math.max(0, y0 - radius), b = Math.min(roi.rows, y1 + radius), cols = roi.cols, rows = b - a;
  const view = roi.roi(new cv.Rect(0, a, cols, rows));
  const part = new cv.Mat(), luma = new cv.Mat(rows, cols, cv.CV_32F), blurred = new cv.Mat();
  view.convertTo(part, cv.CV_32F, 1 / 255);
  view.delete();
  {
    const src = part.data32F, l = luma.data32F, n = rows * cols;
    for (let i = 0, q = 0; i < n; i++, q += 3) l[i] = LUMA_B * src[q] + LUMA_G * src[q + 1] + LUMA_R * src[q + 2];
  }
  part.delete();
  cv.boxFilter(luma, blurred, cv.CV_32F, new cv.Size(2 * radius + 1, 2 * radius + 1), new cv.Point(-1, -1), true, cv.BORDER_DEFAULT);
  const out = new Float32Array(cols * (y1 - y0));
  {
    const l = luma.data32F, m = blurred.data32F, skip = cols * (y0 - a);
    for (let i = 0; i < out.length; i++) out[i] = smoothstep(Math.abs(l[skip + i] - m[skip + i]), lo, hi);
  }
  luma.delete(); blurred.delete();
  return out;
}

// bgr: cv.Mat (CV_8UC3); mask: { x0, y0, w, h, data } from buildSkinMask;
// faces: YuNet rows in this image's pixels; pores: 0..1, synthetic pore texture
// (scaled to the detail measured on this photo, sized to the face, brightness
// only, and following the smoothing: through the mask, in proportion to strength,
// not on anything left alone); grain: MIN_GRAIN..MAX_GRAIN, size of the grain kept and
// added, for the print size (PRINT_GRAIN; 1 = large print or screen); shine: 0..1,
// how much more to remove of bright, colourless grain on the brighter skin (the
// glints on oily skin; dark pores, freckles and coloured marks are untouched);
// light: 1..MAX_LIGHT, how far brightness is evened out, for the lighting
// (LIGHTING; 1 = soft studio light).
export function smoothFace(cv, bgr, strength, texture, mask, faces, pores = 0, grain = 1, shine = 0, light = 1) {
  strength = clamp01(strength);
  if (strength <= 0) return copyMat(cv, bgr);
  texture = clamp01(texture);
  pores = clamp01(pores);
  grain = Math.min(Math.max(grain, MIN_GRAIN), MAX_GRAIN);
  shine = clamp01(shine);
  light = Math.min(Math.max(light, 1), MAX_LIGHT);
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
  const widths = faceWidths(faces, diag);
  const pad = Math.ceil(Math.max(...widths) * TUNE.chroma_radius) + 8;
  const y0 = Math.max(0, minY + mask.y0 - pad), y1 = Math.min(H, maxY + mask.y0 + 1 + pad);
  const x0 = Math.max(0, minX + mask.x0 - pad), x1 = Math.min(W, maxX + mask.x0 + 1 + pad);
  const roi = bgr.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0));
  const rh = roi.rows, rw = roi.cols;

  // The mask, cut to this region.
  const maskR = new Float32Array(rw * rh);
  for (let y = 0; y < rh; y++) {
    const my = y + y0 - mask.y0;
    if (my < 0 || my >= mask.h) continue;
    for (let x = 0; x < rw; x++) {
      const mx = x + x0 - mask.x0;
      if (mx >= 0 && mx < mask.w) maskR[y * rw + x] = mask.data[my * mask.w + mx];
    }
  }

  const scale = Math.min(1.0, Math.sqrt(WORK_PIXELS / (rh * rw)));
  const sigma = Math.max(MIN_SPLIT_SIGMA, medianOf(widths) * TUNE.split * grain);

  const maskMat = cv.matFromArray(rh, rw, cv.CV_32F, maskR);
  const tone = toneChange(cv, roi, maskMat, faces, widths, x0, y0, sigma, scale, strength, light);
  maskMat.delete();
  const workRows = tone.rows;
  const level = detailLevel(cv, roi, maskR, sigma);
  let groups = [];
  if (pores > 0) {
    const workCols = scale < 1 ? Math.max(1, pyRound(rw * scale)) : rw;
    groups = poreGroups(cv, scaleRegions(cv, faces, widths, scale, x0, y0, workRows, workCols), rw, workRows, workCols, grain);
  }
  const poreAmount = pores * TUNE.pore_depth * Math.max(level, TUNE.pore_min_level);
  const edgeRadius = Math.max(1, pyRound(SPLIT_RADIUS_PER_SIGMA * sigma));
  const toneMid = shine > 0 ? skinTone(cv, roi, maskR, sigma) : null;
  const keepLo = TUNE.keep_lo * level, keepHi = TUNE.keep_hi * level;

  // Tone-change rows either side of a full-resolution row: linear between them.
  const between = (y) => {
    const pos = (y + 0.5) * workRows / rh - 0.5, below = Math.floor(pos);
    const i0 = Math.min(Math.max(below, 0), workRows - 1);
    return { i0, i1: Math.min(i0 + 1, workRows - 1), frac: pos - below };
  };
  // What pores add to each pixel of a strip, or null without pores.
  const poreRows = (top, bottom) => {
    if (!groups.length) return null;
    const rows = bottom - top, all = new Float32Array(rw * rows);
    const edge = nearEdge(cv, roi, top, bottom, edgeRadius, keepLo, keepHi);
    for (const group of groups) {
      const pattern = poreStrip(cv, group, x0, y0 + top, rw, rows), wt = group.weight;
      for (let y = top, p = 0; y < bottom; y++) {
        const { i0, i1, frac } = between(y), r0 = i0 * rw, r1 = i1 * rw;
        for (let x = 0; x < rw; x++, p++) {
          const w = wt ? wt[r0 + x] * (1 - frac) + wt[r1 + x] * frac : 1;
          all[p] += poreAmount * pattern[p] * w * (1 - edge[p]);
        }
      }
    }
    return all;
  };

  const colourLo = TUNE.keep_colour_lo * level, colourHi = TUNE.keep_colour_hi * level;
  const lineLo = TUNE.line_lo * level, lineHi = TUNE.line_hi * level, lineCut = TUNE.line_cut;
  const glintLo = TUNE.shine_lo * level, glintHi = TUNE.shine_hi * level;
  const glintColourLo = TUNE.shine_colour_lo * level, glintColourHi = TUNE.shine_colour_hi * level;
  const shinyLo = TUNE.shine_lift, shinyHi = TUNE.shine_lift + TUNE.shine_span;

  const out = copyMat(cv, bgr);
  for (let top = 0; top < rh; top += STRIP_ROWS) {
    const bottom = Math.min(rh, top + STRIP_ROWS);
    const { values, high } = detailStrip(cv, roi, top, bottom, sigma);
    const added = poreRows(top, bottom);
    const o = out.data, t = tone.data32F;
    for (let y = top, p = 0; y < bottom; y++) {
      const { i0, i1, frac } = between(y);
      const r0 = 3 * i0 * rw, r1 = 3 * i1 * rw;
      let j = 3 * ((y + y0) * W + x0);
      for (let x = 0, m = y * rw; x < rw; x++, m++, p += 3, j += 3) {
        const a = maskR[m];
        if (a === 0) continue;                       // outside the mask: unchanged
        const hb = high[p], hg = high[p + 1], hr = high[p + 2];
        const luma = LUMA_B * hb + LUMA_G * hg + LUMA_R * hr;
        const amount = Math.abs(luma);
        const colour = Math.max(Math.abs(hb - luma), Math.abs(hg - luma), Math.abs(hr - luma));
        const keep = Math.max(smoothstep(amount, keepLo, keepHi), smoothstep(colour, colourLo, colourHi));
        let removed = 1 - texture * (1 - lineCut * smoothstep(amount, lineLo, lineHi));
        if (toneMid !== null) {
          // Glints: brighter than their surroundings, nearly colourless, on the brighter skin.
          const toneLuma = LUMA_B * (values[p] - hb) + LUMA_G * (values[p + 1] - hg) + LUMA_R * (values[p + 2] - hr);
          const glint = smoothstep(luma, glintLo, glintHi)
            * (1 - smoothstep(colour, glintColourLo, glintColourHi))
            * smoothstep(toneLuma - toneMid, shinyLo, shinyHi);
          removed = removed + (1 - removed) * (shine * glint);
        }
        // A real edge is left alone completely: the tone change was worked out
        // on a blurred copy, so applying it there would soften the edge.
        const here = a * strength * (1 - keep), pore = added ? added[p / 3] : 0;
        for (let c = 0; c < 3; c++) {
          const change = t[r0 + 3 * x + c] * (1 - frac) + t[r1 + 3 * x + c] * frac;
          const v = (values[p + c] + here * (change - removed * high[p + c] + pore)) * 255 + 0.5;
          o[j + c] = v <= 0 ? 0 : v >= 255 ? 255 : Math.trunc(v);
        }
      }
    }
  }
  roi.delete(); tone.delete();
  return out;
}

export const RESMOOTH_GRAIN = 0.5;      // Smooth again keeps half the grain the rest of the skin keeps,
export const RESMOOTH_STRENGTH = 1.0;   // at full strength: grain alone changes little, strength does

// smoothed (smoothFace of bgr), with a Smooth again edit (withEdits' again: { x0, y0, w,
// h, data } in 0..1) smoothed once more, at full strength and with finer grain, as far
// as the edit goes: coarse grain, blotches or stubble shadow left by the first pass
// become tone and are smoothed, while the finest grain stays as Keep texture says. It
// also smooths where the mask left skin out. A new Mat (smoother.smooth_again).
export function smoothAgain(cv, bgr, smoothed, region, mask, faces, texture, pores = 0, grain = 1, shine = 0, light = 1) {
  const { x0, y0, w, h } = region, edit = region.data, data = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const my = y + y0 - mask.y0;
    for (let x = 0; x < w; x++) {
      const mx = x + x0 - mask.x0;
      const m = my >= 0 && my < mask.h && mx >= 0 && mx < mask.w ? mask.data[my * mask.w + mx] : 0;
      data[y * w + x] = Math.max(m, edit[y * w + x]);
    }
  }
  const fine = smoothFace(cv, bgr, RESMOOTH_STRENGTH, texture, { x0, y0, w, h, data }, faces, pores,
    grain * RESMOOTH_GRAIN, shine, light);
  const out = copyMat(cv, smoothed), d = out.data, fd = fine.data, W = bgr.cols, f = Math.fround, half = f(0.5);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const e = edit[y * w + x];
      for (let k = 0, i = 3 * ((y + y0) * W + x + x0); k < 3; k++, i++) {
        const now = d[i], v = f(f(now + f(e * f(fd[i] - now))) + half);
        d[i] = v <= 0 ? 0 : v >= 255 ? 255 : Math.trunc(v);
      }
    }
  }
  fine.delete();
  return out;
}
