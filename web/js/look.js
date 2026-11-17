// Two finishing steps of a studio retouch that aren't smoothing: evening the skin
// under the eyes, and balancing the skin's brightness. Both are off unless asked for,
// and are applied to the smoothed photo, before the light brush's strokes. The same
// steps as the desktop app's look.py; see the notes there.

import { copyMat } from './cv.js';
import { pyRound } from './skinmask.js';
import { softBlur } from './smooth.js';
import { underEyeZones } from './face_landmarks.js';

// Same settings as look.BALANCE and look.EYES.
export const BALANCE = { centre: 0.575, slack: 0.05, pull: 0.7, limit: 0.12 };
export const EYES = { feather: 0.05, tone: 0.03, even: 0.1, below: 0.3, wide: 0.07, lift: 0.12, limit: 0.25 };
const MIN_LUMA = 0.05, MAX_LUMA = 0.95, SURE_SKIN = 0.9, SOFT_BLUR_MIN = 1.4;
const f = Math.fround;

// Median brightness (0..1, the file's own values) of the faces' skin: skin well inside
// the mask ({ x0, y0, w, h, data }), inside the face boxes (anywhere in the mask without
// faces). null without such skin.
export function skinLuma(bgr, mask, faces) {
  const W = bgr.cols, H = bgr.rows, d = bgr.data;
  const boxes = (faces || []).map((b) => [Math.max(0, Math.trunc(b[0])), Math.max(0, Math.trunc(b[1])),
    Math.min(W, Math.trunc(b[0] + b[2])), Math.min(H, Math.trunc(b[1] + b[3]))]);
  const counts = new Uint32Array(255001);
  let n = 0;
  for (let y = 0; y < mask.h; y++) {
    const py = y + mask.y0;
    for (let x = 0; x < mask.w; x++) {
      if (!(mask.data[y * mask.w + x] > SURE_SKIN)) continue;
      const px = x + mask.x0;
      if (boxes.length && !boxes.some((b) => px >= b[0] && px < b[2] && py >= b[1] && py < b[3])) continue;
      const i = 3 * (py * W + px);
      counts[114 * d[i] + 587 * d[i + 1] + 299 * d[i + 2]]++;
      n++;
    }
  }
  if (!n) return null;
  // The median as numpy gives it: the middle value, or the average of the middle two.
  const at = (rank) => { let seen = 0; for (let v = 0; ; v++) { seen += counts[v]; if (seen > rank) return v; } };
  const mid = n % 2 ? at((n - 1) / 2) : (at(n / 2 - 1) + at(n / 2)) / 2;
  return mid / 255000;
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

// Both steps, under-eyes first: bgr itself with both off (the default).
export function finish(cv, bgr, mask, area, points, luma, eyes = 0, tone = 0) {
  const lifted = underEyes(cv, bgr, points, mask, eyes);
  const out = balance(cv, lifted, area || mask, luma, tone);
  if (out !== lifted && lifted !== bgr) lifted.delete();
  return out;
}
