// Finishing steps of a studio retouch that aren't smoothing: taking large shiny patches
// down (the other half of Reduce shine), evening the skin under the eyes, balancing the
// skin's brightness, and tidying stray hairs against a plain backdrop. All are off unless asked for, and are applied to the smoothed
// photo, before the light brush's strokes. The same steps as the desktop app's look.py;
// see the notes there.

import { copyMat } from './cv.js';
import { pyRound, median } from './skinmask.js';
import { softBlur, smoothstep, faceWidths, medianOf, scaleRegions } from './smooth.js';
import { underEyeZones } from './face_landmarks.js';

// Same settings as look.BALANCE, look.EYES and look.MATTE.
export const MATTE = { tone: 0.012, from: 0.7, to: 0.8, cut: 0.6, span: 0.04 };
const MATTE_GRID = 250000, MATTE_BORDER = 0.05;
// Same settings as look.TIDY.
export const TIDY = {
  gap: 0.012, reach: 0.22, below: 0.15, fill: 0.03, thin: 0.02, lo: 5.0, hi: 12.0, busy_lo: 3.0, busy_hi: 6.0, feather: 0.004,
  crown: 0.25, rim: 0.03, mix_lo: 0.03, mix_hi: 1.25, stray: 0.4, stray_floor: 6.0, other: 25.0,
};
const TIDY_FACE_PX = 600, TIDY_MIN_HAIR = 50;
// Same settings as look.LEVEL.
export const LEVEL = { pivot: 0.25, above: 0.15, neck: 0.75, side: 1.6, side_end: 2.6, most: 8.0 };
const LUMA_B = 0.114, LUMA_G = 0.587, LUMA_R = 0.299;
export const BALANCE = { centre: 0.575, slack: 0.05, pull: 0.7, limit: 0.12 };
export const EYES = { feather: 0.05, tone: 0.03, even: 0.1, below: 0.3, wide: 0.07, lift: 0.12, limit: 0.25 };
const MIN_LUMA = 0.05, MAX_LUMA = 0.95, SURE_SKIN = 0.9, SOFT_BLUR_MIN = 1.4;
const f = Math.fround;

// Calls each(i) with the offset into bgr.data of every pixel of the faces' skin: skin
// well inside the mask ({ x0, y0, w, h, data }), inside the face boxes (anywhere in the
// mask without faces).
function eachFacesSkin(bgr, mask, faces, each) {
  const W = bgr.cols, H = bgr.rows;
  const boxes = (faces || []).map((b) => [Math.max(0, Math.trunc(b[0])), Math.max(0, Math.trunc(b[1])),
    Math.min(W, Math.trunc(b[0] + b[2])), Math.min(H, Math.trunc(b[1] + b[3]))]);
  for (let y = 0; y < mask.h; y++) {
    const py = y + mask.y0;
    for (let x = 0; x < mask.w; x++) {
      if (!(mask.data[y * mask.w + x] > SURE_SKIN)) continue;
      const px = x + mask.x0;
      if (boxes.length && !boxes.some((b) => px >= b[0] && px < b[2] && py >= b[1] && py < b[3])) continue;
      each(3 * (py * W + px));
    }
  }
}

// How many of the faces' skin pixels have each brightness (114 B + 587 G + 299 R), and
// the brightness at a rank among them (0 = the darkest).
function skinLevels(bgr, mask, faces) {
  const d = bgr.data, counts = new Uint32Array(255001);
  let n = 0;
  eachFacesSkin(bgr, mask, faces, (i) => { counts[114 * d[i] + 587 * d[i + 1] + 299 * d[i + 2]]++; n++; });
  const at = (rank) => { let seen = 0; for (let v = 0; ; v++) { seen += counts[v]; if (seen > rank) return v; } };
  return { n, at };
}

// Median brightness (0..1, the file's own values) of the faces' skin. null without such skin.
export function skinLuma(bgr, mask, faces) {
  const { n, at } = skinLevels(bgr, mask, faces);
  if (!n) return null;
  // The median as numpy gives it: the middle value, or the average of the middle two.
  const mid = n % 2 ? at((n - 1) / 2) : (at(n / 2 - 1) + at(n / 2)) / 2;
  return mid / 255000;
}

// The colour ([B, G, R], 0..1) of the faces' brighter skin: the average of the skin
// between MATTE's two shares of the way up it, ordered by brightness. matte pulls
// brighter skin tone back towards it. null without such skin.
export function brightColour(bgr, mask, faces) {
  const { n, at } = skinLevels(bgr, mask, faces);
  if (!n) return null;
  const lo = at(Math.trunc(MATTE.from * (n - 1))), hi = at(Math.trunc(MATTE.to * (n - 1)));
  const d = bgr.data, sum = [0, 0, 0];
  let count = 0;
  eachFacesSkin(bgr, mask, faces, (i) => {
    const level = 114 * d[i] + 587 * d[i + 1] + 299 * d[i + 2];
    if (level < lo || level > hi) return;
    sum[0] += d[i]; sum[1] += d[i + 1]; sum[2] += d[i + 2];
    count++;
  });
  return sum.map((s) => s / (count * 255));
}

// brightColour for each face on its own (null for a face without such skin), or for all
// the skin in the mask, as a list of one, without faces.
export function brightColours(bgr, mask, faces) {
  if (!faces || !faces.length) return [brightColour(bgr, mask, null)];
  return faces.map((face) => brightColour(bgr, mask, [face]));
}

// bgr with large shiny patches on skin taken down: a new Mat, or bgr itself when nothing
// changes. mask: the smoothing mask ({ x0, y0, w, h, data }). colours: brightColours of
// the photo as it was opened, one per face (each face's skin, and the body skin nearest
// it, is judged against its own), or one colour for everything.
export function matte(cv, bgr, mask, faces, colours, amount) {
  if (!colours || !(amount > 0)) return bgr;
  if (colours.length === 3 && typeof colours[0] === 'number') colours = [colours];   // one colour for everything
  if (colours.every((c) => !c)) return bgr;
  amount = Math.min(1, amount);
  const t = MATTE, W = bgr.cols, H = bgr.rows;
  let minX = Infinity, maxX = -1, minY = Infinity, maxY = -1;
  for (let y = 0, k = 0; y < mask.h; y++) {
    for (let x = 0; x < mask.w; x++, k++) {
      if (mask.data[k] > 1e-3) { if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y; }
    }
  }
  if (maxX < 0) return bgr;
  const widths = faceWidths(faces, Math.hypot(H, W));
  const sigma = Math.max(SOFT_BLUR_MIN, medianOf(widths) * t.tone);
  const pad = Math.ceil(3 * sigma) + 2;
  const y0 = Math.max(0, minY + mask.y0 - pad), y1 = Math.min(H, maxY + mask.y0 + 1 + pad);
  const x0 = Math.max(0, minX + mask.x0 - pad), x1 = Math.min(W, maxX + mask.x0 + 1 + pad);
  const w = x1 - x0, h = y1 - y0, n = w * h;
  const out = copyMat(cv, bgr), d = out.data, k255 = f(255);
  const box = new Float32Array(3 * n), lit = new Float32Array(3 * n), skin = new Float32Array(n);
  for (let y = Math.max(y0, mask.y0); y < Math.min(y1, mask.y0 + mask.h); y++) {
    for (let x = Math.max(x0, mask.x0); x < Math.min(x1, mask.x0 + mask.w); x++) {
      skin[(y - y0) * w + x - x0] = mask.data[(y - mask.y0) * mask.w + x - mask.x0];
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0, i = y * w, p = 3 * ((y + y0) * W + x0); x < w; x++, i++, p += 3) {
      for (let c = 0; c < 3; c++) { box[3 * i + c] = f(d[p + c] / k255); lit[3 * i + c] = f(box[3 * i + c] * skin[i]); }
    }
  }
  // The tone of skin only: hair and backdrop beside it have no say.
  const tone = blurred(cv, lit, w, h, 3, sigma), den = blurred(cv, skin, w, h, 1, sigma);
  const tiny = f(1e-4), half = f(0.5), share = f(amount * t.cut);
  for (let i = 0; i < n; i++) {
    const under = f(den[i] + tiny);
    for (let c = 0; c < 3; c++) tone[3 * i + c] = f(tone[3 * i + c] / under);
  }
  // How far the tone is above a colour, per channel, where the skin as a whole is
  // brighter than it and not where one channel is (a blush is redder than it, and no
  // brighter), added into excess, times a weight per pixel if there is one.
  const excess = new Float32Array(3 * n);
  const above = (colour, weight) => {
    const ref = colour.map(f), refLuma = f(f(f(ref[0] * f(LUMA_B)) + f(ref[1] * f(LUMA_G))) + f(ref[2] * f(LUMA_R)));
    for (let i = 0, j = 0; i < n; i++, j += 3) {
      const luma = f(f(f(tone[j] * f(LUMA_B)) + f(tone[j + 1] * f(LUMA_G))) + f(tone[j + 2] * f(LUMA_R)));
      const gate = f(smoothstep(f(luma - refLuma), 0, t.span));
      if (!(gate > 0)) continue;
      const by = weight ? f(gate * weight[i]) : gate;
      for (let c = 0; c < 3; c++) excess[j + c] = f(excess[j + c] + f(Math.max(f(tone[j + c] - ref[c]), 0) * by));
    }
  };
  let regions = [{ weight: null }];
  if (faces && colours.length === faces.length && faces.length > 1) {
    const grid = Math.min(1, Math.sqrt(MATTE_GRID / (h * w)));
    const rows = Math.max(1, pyRound(h * grid)), cols = Math.max(1, pyRound(w * grid));
    regions = scaleRegions(cv, faces, widths, grid, x0, y0, rows, cols, true, MATTE_BORDER).map(({ weight }) => ({ weight, rows, cols }));
  }
  if (!regions[0].weight) {
    above(colours.find((c) => c), null);
  } else {
    // Each face's skin, and the body skin nearest it, against that face's own colour.
    regions.forEach(({ weight, rows, cols }, k) => {
      if (!colours[k]) return;
      const small = cv.matFromArray(rows, cols, cv.CV_32F, weight), big = new cv.Mat();
      cv.resize(small, big, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR);
      above(colours[k], big.data32F);
      small.delete(); big.delete();
    });
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0, i = y * w, p = 3 * ((y + y0) * W + x0); x < w; x++, i++, p += 3) {
      const by = f(skin[i] * share);
      for (let c = 0; c < 3; c++) {
        const v = f(f(f(box[3 * i + c] - f(excess[3 * i + c] * by)) * k255) + half);
        d[p + c] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
  }
  return out;
}

// Where balance moves a face of this brightness to, at this amount (0..1).
export function wantedLuma(luma, amount) {
  const t = BALANCE, off = luma - t.centre;
  const move = -t.pull * Math.sign(off) * Math.max(Math.abs(off) - t.slack, 0);
  return luma + Math.min(Math.max(amount, 0), 1) * Math.max(-t.limit, Math.min(t.limit, move));
}

// bgr with the skin's brightness balanced: a new Mat, or bgr itself when nothing
// changes. area: all the skin ({ x0, y0, w, h, data }). luma: skinLuma of the photo as
// it was opened.
export function balance(cv, bgr, area, luma, amount) {
  if (luma == null || !(amount > 0) || !(luma > MIN_LUMA && luma < MAX_LUMA) || !area) return bgr;
  const want = wantedLuma(luma, amount);
  if (Math.abs(want - luma) < 1e-6) return bgr;
  const gamma = Math.log(want) / Math.log(luma);
  const curve = new Float32Array(256);
  for (let v = 0; v < 256; v++) curve[v] = Math.pow(v / 255, gamma) * 255;
  const out = copyMat(cv, bgr), d = out.data, W = bgr.cols, half = f(0.5);
  for (let y = 0; y < area.h; y++) {
    for (let x = 0, k = y * area.w, i = 3 * ((y + area.y0) * W + area.x0); x < area.w; x++, k++, i += 3) {
      const a = area.data[k];
      if (!(a > 0)) continue;
      for (let c = 0; c < 3; c++) {
        const now = d[i + c], mixed = f(f(now + f(a * f(curve[now] - now))) + half);
        d[i + c] = mixed < 0 ? 0 : mixed > 255 ? 255 : mixed;      // a Uint8Array drops the fraction
      }
    }
  }
  return out;
}

// A float array (w x h, 1 or 3 channels) blurred by softBlur.
function blurred(cv, data, w, h, channels, sigma) {
  const src = cv.matFromArray(h, w, channels === 3 ? cv.CV_32FC3 : cv.CV_32F, data);
  const dst = softBlur(cv, src, sigma);
  const out = new Float32Array(dst.data32F);
  src.delete(); dst.delete();
  return out;
}

// a read from dx, dy away: out[y][x] = a[y + dy][x + dx], zero where that's outside.
function moved(a, w, h, channels, dx, dy) {
  const out = new Float32Array(a.length);
  const ya = Math.max(0, -dy), yb = Math.min(h, h - dy), xa = Math.max(0, -dx), xb = Math.min(w, w - dx);
  if (xb <= xa) return out;
  for (let y = ya; y < yb; y++) {
    out.set(a.subarray(channels * ((y + dy) * w + xa + dx), channels * ((y + dy) * w + xb + dx)), channels * (y * w + xa));
  }
  return out;
}

// bgr with the skin under the eyes evened towards the cheek below: a new Mat, or bgr
// itself when nothing changes. points: each face's landmarks or null. mask: the
// smoothing mask ({ x0, y0, w, h, data }), whose skin (no eyes, lashes or brows) is
// where tone is read from, and where the change shows.
export function underEyes(cv, bgr, points, mask, amount) {
  if (!(amount > 0) || !points || !points.length) return bgr;
  amount = Math.min(1, amount);
  const t = EYES, W = bgr.cols, H = bgr.rows;
  let out = bgr;
  for (const pts of points) {
    if (!pts) continue;
    const { zones, dx, dy, eyeD } = underEyeZones(cv, pts, W, H);
    const sx = pyRound(t.below * dx), sy = pyRound(t.below * dy);
    const feather = Math.max(SOFT_BLUR_MIN, t.feather * eyeD), fine = Math.max(SOFT_BLUR_MIN, t.tone * eyeD);
    const wide = Math.max(SOFT_BLUR_MIN, t.wide * eyeD), even = Math.max(SOFT_BLUR_MIN, t.even * eyeD);
    const pad = Math.ceil(3 * Math.max(feather, wide, even)) + Math.max(Math.abs(sx), Math.abs(sy)) + 2;
    for (const zone of zones) {
      const zw = zone.mask.cols, zh = zone.mask.rows;
      const x0 = Math.max(0, zone.x0 - pad), y0 = Math.max(0, zone.y0 - pad);
      const x1 = Math.min(W, zone.x0 + zw + pad), y1 = Math.min(H, zone.y0 + zh + pad);
      const w = x1 - x0, h = y1 - y0, n = w * h;
      const inside = new Float32Array(n), skin = new Float32Array(n);
      for (let y = 0; y < zh; y++) {
        for (let x = 0; x < zw; x++) inside[(y + zone.y0 - y0) * w + x + zone.x0 - x0] = f(zone.mask.data[y * zw + x] / f(255));
      }
      zone.mask.delete();
      for (let y = Math.max(y0, mask.y0); y < Math.min(y1, mask.y0 + mask.h); y++) {
        for (let x = Math.max(x0, mask.x0); x < Math.min(x1, mask.x0 + mask.w); x++) {
          skin[(y - y0) * w + x - x0] = mask.data[(y - mask.y0) * mask.w + x - mask.x0];
        }
      }
      const weight = blurred(cv, inside, w, h, 1, feather);
      let most = 0;
      for (let i = 0; i < n; i++) { weight[i] = f(weight[i] * skin[i]); if (weight[i] > most) most = weight[i]; }
      if (!(most > 1e-3)) continue;
      if (out === bgr) out = copyMat(cv, bgr);
      const d = out.data, box = new Float32Array(3 * n), lit = new Float32Array(3 * n), k255 = f(255);
      for (let y = 0; y < h; y++) {
        for (let x = 0, i = y * w, p = 3 * ((y + y0) * W + x0); x < w; x++, i++, p += 3) {
          for (let c = 0; c < 3; c++) { box[3 * i + c] = f(d[p + c] / k255); lit[3 * i + c] = f(box[3 * i + c] * skin[i]); }
        }
      }
      const tone = blurred(cv, lit, w, h, 3, fine), toneSkin = blurred(cv, skin, w, h, 1, fine);
      const flat = blurred(cv, lit, w, h, 3, even), flatSkin = blurred(cv, skin, w, h, 1, even);
      // Where there's little skin below (a photo's edge, glasses, hair), nothing to go by.
      const seen = moved(blurred(cv, skin, w, h, 1, wide), w, h, 1, sx, sy);
      const cheek = moved(blurred(cv, lit, w, h, 3, wide), w, h, 3, sx, sy);
      const tiny = f(1e-4), lift = f(t.lift), limit = f(t.limit), half = f(0.5), amt = f(amount);
      for (let y = 0; y < h; y++) {
        for (let x = 0, i = y * w, p = 3 * ((y + y0) * W + x0); x < w; x++, i++, p += 3) {
          const sure = Math.min(Math.max(f(f(seen[i] * 2) - half), 0), 1), by = f(f(amt * weight[i]));
          for (let c = 0; c < 3; c++) {
            const j = 3 * i + c;
            const tn = f(tone[j] / f(toneSkin[i] + tiny)), fl = f(flat[j] / f(flatSkin[i] + tiny)), ch = f(cheek[j] / f(seen[i] + tiny));
            const target = f(fl + f(f(lift * sure) * f(ch - fl)));
            const change = f(Math.min(Math.max(f(target - tn), -limit), limit) * by);
            const v = f(f(f(box[j] + change) * k255) + half);
            d[p + c] = v < 0 ? 0 : v > 255 ? 255 : v;
          }
        }
      }
    }
  }
  return out;
}

// What tidy needs of the segmentation model's result (segment() in skin_segment.js):
// { x0, y0, x1, y1, w, h, scores } (person scores, w x h, over that region of the photo),
// or null without it.
export const personOf = (seg) => (seg ? { x0: seg.x0, y0: seg.y0, x1: seg.x1, y1: seg.y1, w: seg.ww, h: seg.wh, scores: seg.person } : null);

const boxKernel = (cv, r) => { const side = 2 * Math.max(1, pyRound(r)) + 1; return cv.Mat.ones(side, side, cv.CV_8U); };

// bgr with stray hairs around the heads replaced by the backdrop: a new Mat, or bgr
// itself when nothing changes. person: personOf the photo's segmentation. keep: a brush
// edit ({ x0, y0, w, h, data }, -1..1) whose Restore strokes leave the photo as it was.
export function tidy(cv, bgr, person, faces, amount, keep = null) {
  if (!person || !(amount > 0) || !faces || !faces.length) return bgr;
  amount = Math.min(1, amount);
  const t = TIDY, { x0, y0, x1, y1 } = person, rw = x1 - x0, rh = y1 - y0;
  if (rw < 8 || rh < 8) return bgr;
  const faceW = medianOf(faces.map((face) => face[2])), scale = Math.min(1, TIDY_FACE_PX / faceW);
  const ww = scale < 1 ? Math.max(1, pyRound(rw * scale)) : rw, wh = scale < 1 ? Math.max(1, pyRound(rh * scale)) : rh, n = ww * wh;
  const fw = faceW * scale;
  const roi = bgr.roi(new cv.Rect(x0, y0, rw, rh));
  const small = new cv.Mat(), workMat = new cv.Mat();
  if (scale < 1) cv.resize(roi, small, new cv.Size(ww, wh), 0, 0, cv.INTER_AREA); else roi.copyTo(small);
  small.convertTo(workMat, cv.CV_32F);
  const work = new Float32Array(workMat.data32F);
  small.delete(); workMat.delete();

  // The person's outline at this size, the ring around it, and the heads' part of it.
  const scoreMat = cv.matFromArray(person.h, person.w, cv.CV_32F, person.scores), sized = new cv.Mat();
  cv.resize(scoreMat, sized, new cv.Size(ww, wh), 0, 0, cv.INTER_LINEAR);
  const body = new cv.Mat(wh, ww, cv.CV_8U);
  { const v = sized.data32F, b = body.data; for (let i = 0; i < n; i++) b[i] = v[i] > 0.5 ? 1 : 0; }
  scoreMat.delete(); sized.delete();
  const grown = (r) => { const k = boxKernel(cv, r), out = new cv.Mat(); cv.dilate(body, out, k); const copy = new Uint8Array(out.data); k.delete(); out.delete(); return copy; };
  const near = grown(t.gap * fw), far = grown(t.reach * fw);
  const inner = (() => { const k = boxKernel(cv, t.rim * fw), out = new cv.Mat(); cv.erode(body, out, k); const copy = new Uint8Array(out.data); k.delete(); out.delete(); return copy; })();
  const bodyData = new Uint8Array(body.data);
  body.delete();
  const heads = new Uint8Array(n), crown = new Uint8Array(n);
  const within = (v, most) => Math.trunc(Math.min(most, Math.max(0, v)));
  for (const face of faces) {
    const hy1 = within((face[1] + face[3] * (1 + t.below) - y0) * scale, wh);
    const cy1 = within((face[1] + face[3] * t.crown - y0) * scale, wh);
    const hx0 = within((face[0] - face[2] - x0) * scale, ww), hx1 = within((face[0] + 2 * face[2] - x0) * scale, ww);
    for (let y = 0; y < hy1; y++) heads.fill(1, y * ww + hx0, y * ww + hx1);
    for (let y = 0; y < cy1; y++) crown.fill(1, y * ww + hx0, y * ww + hx1);
  }
  // The hair's colour: the person's rim at the top of each head.
  const hairPx = [[], [], []];
  for (let i = 0; i < n; i++) {
    if (bodyData[i] && !inner[i] && crown[i]) { hairPx[0].push(work[3 * i]); hairPx[1].push(work[3 * i + 1]); hairPx[2].push(work[3 * i + 2]); }
  }
  if (hairPx[0].length < TIDY_MIN_HAIR) { roi.delete(); return bgr; }
  const hair = hairPx.map((v) => f(median(Float32Array.from(v))));
  const ring = new Uint8Array(n);
  let ringSize = 0;
  for (let i = 0; i < n; i++) { ring[i] = far[i] && !near[i] && heads[i] ? 1 : 0; ringSize += ring[i]; }
  if (ringSize < 100) { roi.delete(); return bgr; }

  const sigma = Math.max(SOFT_BLUR_MIN, t.fill * fw), tiny = f(1e-4);
  // The backdrop's colour everywhere, from the pixels marked valid nearby.
  const backdrop = (valid) => {
    const lit = new Float32Array(3 * n);
    for (let i = 0, j = 0; i < n; i++, j += 3) { lit[j] = f(work[j] * valid[i]); lit[j + 1] = f(work[j + 1] * valid[i]); lit[j + 2] = f(work[j + 2] * valid[i]); }
    const num = blurred(cv, lit, ww, wh, 3, sigma), den = blurred(cv, valid, ww, wh, 1, sigma);
    for (let i = 0, j = 0; i < n; i++, j += 3) { const under = f(den[i] + tiny); num[j] = f(num[j] / under); num[j + 1] = f(num[j + 1] / under); num[j + 2] = f(num[j + 2] / under); }
    return num;
  };
  // How far each pixel is from the backdrop, and how much of a strand it is (0..1):
  // different from the backdrop, and too thin to survive an opening.
  const thin = boxKernel(cv, t.thin * fw);
  const strands = (fill) => {
    const dist = new Float32Array(n);
    for (let i = 0, j = 0; i < n; i++, j += 3) {
      dist[i] = Math.max(Math.abs(f(work[j] - fill[j])), Math.abs(f(work[j + 1] - fill[j + 1])), Math.abs(f(work[j + 2] - fill[j + 2])));
    }
    const m = cv.matFromArray(wh, ww, cv.CV_32F, dist), opened = new cv.Mat();
    cv.morphologyEx(m, opened, cv.MORPH_OPEN, thin);
    const o = opened.data32F, found = new Float32Array(n);
    for (let i = 0; i < n; i++) found[i] = f(smoothstep(Math.min(dist[i], f(f(dist[i] - o[i]) * f(1.5))), t.lo, t.hi));
    m.delete(); opened.delete();
    return { dist, found };
  };
  // 1 for a pixel that is the backdrop with some of the hair's colour mixed in, -1 for
  // one that is clearly some other colour (a logo, a seam, a lamp) and differs strongly
  // from the backdrop, 0 for the rest: too close to the backdrop's colour to say, or
  // another colour but weak, like the halo sharpening leaves beside a hair.
  const hairlike = (fill, i) => {
    const j = 3 * i;
    let dot = 0, len = 0, size = 0;
    const diff = [0, 0, 0], towards = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      diff[c] = f(work[j + c] - fill[j + c]); towards[c] = f(hair[c] - fill[j + c]);
      dot += diff[c] * towards[c]; len += towards[c] * towards[c]; size += diff[c] * diff[c];
    }
    const share = dot / (len + 1e-3);
    let rest = 0;
    for (let c = 0; c < 3; c++) { const r = diff[c] - share * towards[c]; rest += r * r; }
    if (Math.sqrt(rest) >= t.stray * Math.sqrt(size) + t.stray_floor || share >= t.mix_hi || share <= -t.mix_lo) {
      return Math.sqrt(size) >= t.other ? -1 : 0;
    }
    return share > t.mix_lo ? 1 : 0;
  };
  const outside = new Float32Array(n);
  for (let i = 0; i < n; i++) outside[i] = near[i] ? 0 : 1;
  const first = strands(backdrop(outside)).found;
  let fill = backdrop(outside.map((v, i) => (first[i] < 0.2 ? v : 0)));      // the backdrop without the thin things in it
  const { dist, found } = strands(fill);
  thin.delete();
  // The faint edge of a thin thing of another colour could pass for hair: nothing right
  // beside one counts.
  const like = new Int8Array(n);
  for (let i = 0; i < n; i++) like[i] = found[i] > 0 ? hairlike(fill, i) : 0;
  const other = (() => {
    const m = new cv.Mat(wh, ww, cv.CV_8U), wide = new cv.Mat(), k = boxKernel(cv, t.thin * fw / 2);
    { const v = m.data; for (let i = 0; i < n; i++) v[i] = found[i] > 0.5 && like[i] < 0 ? 1 : 0; }
    cv.dilate(m, wide, k);
    const copy = new Uint8Array(wide.data);
    m.delete(); wide.delete(); k.delete();
    return copy;
  })();
  let most = 0;
  const quiet = [];
  for (let i = 0; i < n; i++) {
    found[i] = ring[i] && like[i] > 0 && !other[i] ? found[i] : 0;
    if (found[i] > most) most = found[i];
    if (ring[i] && found[i] < 0.2) quiet.push(dist[i]);
  }
  // A busy background has no backdrop to tidy against.
  const rough = quiet.length ? median(Float32Array.from(quiet)) : t.busy_hi;
  const plain = 1 - smoothstep(f(rough), t.busy_lo, t.busy_hi);
  if (plain <= 0 || !(most > 0)) { roi.delete(); return bgr; }

  let soft;
  {
    const m = cv.matFromArray(wh, ww, cv.CV_32F, found), wide = new cv.Mat(), blur = new cv.Mat(), k3 = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.dilate(m, wide, k3);
    cv.GaussianBlur(wide, blur, new cv.Size(0, 0), Math.max(0.7, t.feather * fw));
    soft = new Float32Array(blur.data32F);
    for (let i = 0; i < n; i++) soft[i] = f(Math.min(Math.max(f(soft[i] * f(1.3)), 0), 1) * ring[i]);
    m.delete(); wide.delete(); blur.delete(); k3.delete();
  }
  if (scale < 1) {
    const up = (data, type) => {
      const m = cv.matFromArray(wh, ww, type, data), big = new cv.Mat();
      cv.resize(m, big, new cv.Size(rw, rh), 0, 0, cv.INTER_LINEAR);
      const copy = new Float32Array(big.data32F);
      m.delete(); big.delete();
      return copy;
    };
    soft = up(soft, cv.CV_32F);
    fill = up(fill, cv.CV_32FC3);
  }
  if (keep) {
    const ax0 = Math.max(x0, keep.x0), ay0 = Math.max(y0, keep.y0), ax1 = Math.min(x1, keep.x0 + keep.w), ay1 = Math.min(y1, keep.y0 + keep.h);
    for (let y = ay0; y < ay1; y++) {
      for (let x = ax0; x < ax1; x++) {
        const i = (y - y0) * rw + x - x0;
        soft[i] = f(soft[i] * f(1 + Math.min(keep.data[(y - keep.y0) * keep.w + x - keep.x0], 0)));
      }
    }
  }
  roi.delete();
  const out = copyMat(cv, bgr), d = out.data, W = bgr.cols, share = f(amount * plain), half = f(0.5);
  for (let y = 0; y < rh; y++) {
    for (let x = 0, i = y * rw, p = 3 * ((y + y0) * W + x0); x < rw; x++, i++, p += 3) {
      const by = f(share * soft[i]);
      if (!(by > 0)) continue;
      for (let c = 0; c < 3; c++) {
        const now = d[p + c], v = f(f(now + f(by * f(fill[3 * i + c] - now))) + half);
        d[p + c] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
  }
  return out;
}

// One face's tilt in degrees: the eye line's angle, positive when the eye on the right
// of the photo is lower. pts: x0, y0, x1, y1, ...
export function headTilt(pts) {
  const rx = (pts[66] + pts[266]) / 2, ry = (pts[67] + pts[267]) / 2;        // points 33 and 133
  const lx = (pts[526] + pts[724]) / 2, ly = (pts[527] + pts[725]) / 2;      // points 263 and 362
  return Math.atan2(ly - ry, lx - rx) * 180 / Math.PI;
}

// bgr with each tilted head turned upright by this share of its tilt: a new Mat, or bgr
// itself when nothing changes. points: each face's landmarks or null. Everything above
// the jaw turns as one piece about a point just under the chin, and the turn fades out
// down the neck.
export function straighten(cv, bgr, points, amount) {
  if (!(amount > 0) || !points || !points.length) return bgr;
  amount = Math.min(1, amount);
  const t = LEVEL, W = bgr.cols, H = bgr.rows, CHIN = 152;
  let out = bgr;
  for (const pts of points) {
    if (!pts) continue;
    const rx = (pts[66] + pts[266]) / 2, ry = (pts[67] + pts[267]) / 2, lx = (pts[526] + pts[724]) / 2, ly = (pts[527] + pts[725]) / 2;
    const d = Math.hypot(lx - rx, ly - ry);
    if (!(d > 4)) continue;
    const tilt = Math.max(-t.most, Math.min(t.most, headTilt(pts)));
    const turn = tilt * Math.PI / 180 * amount;          // the output is the photo turned back by this
    if (Math.abs(turn) < 1e-4) continue;
    const ax = (lx - rx) / d, ay = (ly - ry) / d;        // along the eyes
    let dx = -ay, dy = ax;                               // down the face ...
    const mx = (rx + lx) / 2, my = (ry + ly) / 2;
    if ((pts[2 * CHIN] - mx) * dx + (pts[2 * CHIN + 1] - my) * dy < 0) { dx = -dx; dy = -dy; }   // ... on the chin's side
    const px = pts[2 * CHIN] + t.pivot * d * dx, py = pts[2 * CHIN + 1] + t.pivot * d * dy;
    const reach = t.side_end * d, xs = [], ys = [];
    for (const sx of [-1, 1]) {
      for (const u of [t.neck * d, -(H + W)]) { xs.push(px + sx * reach * ax + u * dx); ys.push(py + sx * reach * ay + u * dy); }
    }
    const x0 = Math.max(0, Math.floor(Math.min(...xs))), x1 = Math.min(W, Math.ceil(Math.max(...xs)) + 1);
    const y0 = Math.max(0, Math.floor(Math.min(...ys))), y1 = Math.min(H, Math.ceil(Math.max(...ys)) + 1);
    const w = x1 - x0, h = y1 - y0;
    if (w < 2 || h < 2) continue;
    const mapX = new cv.Mat(h, w, cv.CV_32F), mapY = new cv.Mat(h, w, cv.CV_32F), moved = new Uint8Array(w * h);
    const vx = mapX.data32F, vy = mapY.data32F;
    const fpx = f(px), fpy = f(py), fdx = f(dx), fdy = f(dy), fax = f(ax), fay = f(ay), fturn = f(turn);
    for (let y = 0, i = 0; y < h; y++) {
      const gy = f(f(y + y0) - fpy);
      for (let x = 0; x < w; x++, i++) {
        const gx = f(f(x + x0) - fpx);
        const down = f(f(gx * fdx) + f(gy * fdy)), across = Math.abs(f(f(gx * fax) + f(gy * fay)));
        const weight = f(f(1 - f(smoothstep(down, -t.above * d, t.neck * d))) * f(1 - f(smoothstep(across, t.side * d, t.side_end * d))));
        const angle = f(fturn * weight), cos = f(Math.cos(angle)), sin = f(Math.sin(angle));
        // Where each pixel of the result comes from: its place turned forward again about the pivot.
        vx[i] = f(f(fpx + f(cos * gx)) - f(sin * gy));
        vy[i] = f(f(fpy + f(sin * gx)) + f(cos * gy));
        moved[i] = weight > 0 ? 1 : 0;
      }
    }
    const turned = new cv.Mat();
    cv.remap(out, turned, mapX, mapY, cv.INTER_CUBIC, cv.BORDER_REPLICATE);
    if (out === bgr) out = copyMat(cv, bgr);
    const from = turned.data, to = out.data;
    for (let y = 0, i = 0; y < h; y++) {
      for (let x = 0, p = 3 * ((y + y0) * W + x0); x < w; x++, i++, p += 3) {
        if (moved[i]) { to[p] = from[3 * i]; to[p + 1] = from[3 * i + 1]; to[p + 2] = from[3 * i + 2]; }
      }
    }
    mapX.delete(); mapY.delete(); turned.delete();
  }
  return out;
}

// The steps in order: shiny patches (shine, with the faces and brightColours), then
// under-eyes, then skin tone, then stray hairs (strays, with the faces, personOf and the
// brush's Restore edit), then straightening the heads (level). bgr itself with all of
// them off (the default).
export function finish(cv, bgr, mask, area, points, luma, eyes = 0, tone = 0, shine = 0, faces = null, bright = null,
  strays = 0, person = null, keep = null, level = 0) {
  const steps = [
    (m) => matte(cv, m, mask, faces, bright, shine),
    (m) => underEyes(cv, m, points, mask, eyes),
    (m) => balance(cv, m, area || mask, luma, tone),
    (m) => tidy(cv, m, person, faces, strays, keep),
    (m) => straighten(cv, m, points, level),
  ];
  let out = bgr;
  for (const step of steps) {
    const next = step(out);
    if (next !== out && out !== bgr) out.delete();
    out = next;
  }
  return out;
}
