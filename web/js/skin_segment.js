// Skin segmentation - a port of skin_smoother/skin_segment.py.
//
// Google's MediaPipe multiclass selfie model labels every pixel as background, hair,
// body skin, face skin, clothes or other. It sees only 256x256 pixels, so faces run as
// crops, with a pass over the whole photo for body skin. The results are blended at
// working resolution and snapped to the photo's edges with a guided filter. Body skin
// is kept only on a person with a detected face and near that face's skin colour.
//
// segment() runs the model once per photo. skinArea() is cheap, so Face only can
// switch without running the model again.
//
// Allocating a Mat can move the WebAssembly heap, so views like mat.data32F are read
// after the allocations they follow.

import { pyRound, median } from './skinmask.js';
import { WORK_PIXELS, guidedFilter, medianOf, smoothstep } from './smooth.js';

const SIZE = 256, BACKGROUND = 0, BODY = 2, FACE = 3, MAX_CROPS = 8, MIN_FACE_SCORE = 0.6;

// Same settings as skin_segment.TUNE; see the notes there.
export const TUNE = {
  crop_reach: 1.0,
  crop_faces: 3.5,
  whole_share: 0.6,
  whole_face: 32.0,
  crop_feather: 0.1,
  prob_lo: 0.2,
  prob_hi: 0.8,
  snap_radius: 0.02,
  snap_edge: 0.03,
  snap_lo: 0.25,
  snap_hi: 0.75,
  colour_face: 0.8,
  colour_lo: 10.0,
  colour_hi: 20.0,
};

// Class scores for part of a photo: the planes one after another, adding up to 1.
async function scores(cv, ort, session, bgr, x0, y0, x1, y1) {
  const view = bgr.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0)), small = new cv.Mat();
  cv.resize(view, small, new cv.Size(SIZE, SIZE), 0, 0, cv.INTER_AREA);
  view.delete();
  const plane = SIZE * SIZE, input = new Float32Array(3 * plane), d = small.data;
  for (let i = 0, j = 0; i < plane; i++, j += 3) {     // RGB, -1..1
    input[i] = (d[j + 2] - 127.5) / 127.5;
    input[plane + i] = (d[j + 1] - 127.5) / 127.5;
    input[2 * plane + i] = (d[j] - 127.5) / 127.5;
  }
  small.delete();
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) });
  const tensor = out[session.outputNames[0]], logits = tensor.data, result = new Float32Array(6 * plane);
  for (let i = 0; i < plane; i++) {
    let top = -Infinity, total = 0;
    for (let c = 0; c < 6; c++) if (logits[c * plane + i] > top) top = logits[c * plane + i];
    for (let c = 0; c < 6; c++) { const e = Math.exp(logits[c * plane + i] - top); result[c * plane + i] = e; total += e; }
    for (let c = 0; c < 6; c++) result[c * plane + i] /= total;
  }
  tensor.dispose?.();
  return result;
}

// Where to run the model: [{ x0, y0, x1, y1, members }], largest faces first. A face
// joins a crop it overlaps while the crop stays small enough for its smallest face to
// fill a useful part of the model's input.
export function crops(faces, w, h) {
  const t = TUNE, clusters = [];
  const order = faces.map((_, i) => i).sort((a, b) => faces[b][2] - faces[a][2]);
  for (const i of order) {
    const f = faces[i], reach = t.crop_reach * Math.max(f[2], f[3]);
    const cx = f[0] + f[2] / 2, cy = f[1] + f[3] / 2;
    const x0 = cx - reach, y0 = cy - reach, x1 = cx + reach, y1 = cy + reach;
    let joined = false;
    for (const c of clusters) {
      const nx0 = Math.min(c.x0, x0), ny0 = Math.min(c.y0, y0), nx1 = Math.max(c.x1, x1), ny1 = Math.max(c.y1, y1);
      const smallest = Math.min(c.smallest, f[2]);
      if (x0 < c.x1 && c.x0 < x1 && y0 < c.y1 && c.y0 < y1 && Math.max(nx1 - nx0, ny1 - ny0) <= t.crop_faces * smallest) {
        Object.assign(c, { x0: nx0, y0: ny0, x1: nx1, y1: ny1, smallest });
        c.members.push(i);
        joined = true;
        break;
      }
    }
    if (!joined) clusters.push({ x0, y0, x1, y1, smallest: f[2], members: [i] });
  }
  return clusters.map((c) => {
    const half = Math.max(c.x1 - c.x0, c.y1 - c.y0) / 2;     // square, so a face keeps its shape
    const cx = (c.x0 + c.x1) / 2, cy = (c.y0 + c.y1) / 2;
    return {
      x0: Math.trunc(Math.max(0, cx - half)), y0: Math.trunc(Math.max(0, cy - half)),
      x1: Math.trunc(Math.min(w, cx + half)), y1: Math.trunc(Math.min(h, cy + half)), members: c.members,
    };
  });
}

// 1 in the middle, fading to 0 over span at each end that's marked open.
function ramp(n, span, openStart, openEnd) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const up = openStart ? (i + 0.5) / span : 1, down = openEnd ? (n - 0.5 - i) / span : 1;
    out[i] = Math.min(Math.max(Math.min(up, down), 0), 1);
  }
  return out;
}

// The central half of a face box, in working pixels.
function middle(f, rx0, ry0, sx, sy, ww, wh) {
  return [
    Math.max(0, Math.trunc((f[0] + 0.25 * f[2] - rx0) * sx)), Math.max(0, Math.trunc((f[1] + 0.25 * f[3] - ry0) * sy)),
    Math.min(ww, Math.trunc((f[0] + 0.75 * f[2] - rx0) * sx)), Math.min(wh, Math.trunc((f[1] + 0.75 * f[3] - ry0) * sy)),
  ];
}

// Run the model on a photo's faces: class scores for a region, at working resolution.
export async function segment(cv, ort, session, bgr, faces) {
  if (!session || !faces.length) return null;
  const t = TUNE, h = bgr.rows, w = bgr.cols;
  const rects = crops(faces, w, h);
  const whole = Math.max(...faces.map((f) => f[2])) * SIZE / Math.max(h, w) >= t.whole_face;

  // Crops past the limit aren't run; a crop nearly as big as the photo adds nothing.
  const covered = [], runs = [];
  for (const r of rects.slice(0, MAX_CROPS)) {
    covered.push(...r.members);
    if (!(whole && (r.x1 - r.x0) * (r.y1 - r.y0) >= t.whole_share * h * w)) runs.push(r);
  }

  // Work on the whole photo if it has a pass of its own, else only where crops are.
  let rx0 = 0, ry0 = 0, rx1 = w, ry1 = h;
  if (!whole && runs.length) {
    rx0 = Math.min(...runs.map((r) => r.x0)); ry0 = Math.min(...runs.map((r) => r.y0));
    rx1 = Math.max(...runs.map((r) => r.x1)); ry1 = Math.max(...runs.map((r) => r.y1));
  }
  const rw = rx1 - rx0, rh = ry1 - ry0, scale = Math.min(1, Math.sqrt(WORK_PIXELS / (rh * rw)));
  const ww = Math.max(1, pyRound(rw * scale)), wh = Math.max(1, pyRound(rh * scale)), n = ww * wh;
  const view = bgr.roi(new cv.Rect(rx0, ry0, rw, rh));
  let work;
  if (scale < 1) {
    const small = new cv.Mat();
    cv.resize(view, small, new cv.Size(ww, wh), 0, 0, cv.INTER_AREA);
    work = new Uint8Array(small.data);
    small.delete();
  } else {
    const copy = new cv.Mat();
    view.copyTo(copy);
    work = new Uint8Array(copy.data);
    copy.delete();
  }
  view.delete();
  const sx = ww / rw, sy = wh / rh;

  const planes = (all, cols, rows) => [FACE, BODY, BACKGROUND].map((c) => {
    const src = cv.matFromArray(SIZE, SIZE, cv.CV_32F, all.subarray(c * SIZE * SIZE, (c + 1) * SIZE * SIZE)), big = new cv.Mat();
    cv.resize(src, big, new cv.Size(cols, rows), 0, 0, cv.INTER_LINEAR);
    const copy = new Float32Array(big.data32F);
    src.delete(); big.delete();
    return copy;
  });

  let s;
  if (whole) s = planes(await scores(cv, ort, session, bgr, 0, 0, w, h), ww, wh);
  else s = [new Float32Array(n), new Float32Array(n), new Float32Array(n).fill(1)];

  // Crops take over from the whole-photo pass where they reach, fading at their
  // edges. Where crops overlap they're averaged by weight.
  if (runs.length) {
    const num = s.map(() => new Float32Array(n)), den = new Float32Array(n), cover = new Float32Array(n);
    for (const r of runs) {
      const cx0 = Math.trunc((r.x0 - rx0) * sx), cx1 = Math.min(ww, Math.ceil((r.x1 - rx0) * sx));
      const cy0 = Math.trunc((r.y0 - ry0) * sy), cy1 = Math.min(wh, Math.ceil((r.y1 - ry0) * sy));
      const cols = cx1 - cx0, rows = cy1 - cy0, span = Math.max(1, t.crop_feather * Math.max(cols, rows));
      // No fade at the photo's own edges: a face there would lose its side.
      const across = ramp(cols, span, r.x0 > 0, r.x1 < w), down = ramp(rows, span, r.y0 > 0, r.y1 < h);
      const ps = planes(await scores(cv, ort, session, bgr, r.x0, r.y0, r.x1, r.y1), cols, rows);
      for (let y = 0, p = 0; y < rows; y++) {
        const at = (y + cy0) * ww + cx0;
        for (let x = 0; x < cols; x++, p++) {
          const weight = Math.min(down[y], across[x]), i = at + x;
          num[0][i] += weight * ps[0][p]; num[1][i] += weight * ps[1][p]; num[2][i] += weight * ps[2][p];
          den[i] += weight;
          if (weight > cover[i]) cover[i] = weight;
        }
      }
    }
    for (let c = 0; c < s.length; c++) {
      for (let i = 0; i < n; i++) s[c][i] = cover[i] * (num[c][i] / Math.max(den[i], 1e-6)) + (1 - cover[i]) * s[c][i];
    }
  }
  const [face, body, background] = s;

  // Faces the model doesn't clearly see use the older method rather than a hole in the mask.
  const trusted = [];
  for (const i of covered) {
    const [x0, y0, x1, y1] = middle(faces[i], rx0, ry0, sx, sy, ww, wh);
    if (x1 <= x0 || y1 <= y0) continue;
    let total = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) total += face[y * ww + x];
    if (total / ((x1 - x0) * (y1 - y0)) >= MIN_FACE_SCORE) trusted.push(i);
  }
  const person = background.map((b) => 1 - b);
  return { x0: rx0, y0: ry0, x1: rx1, y1: ry1, ww, wh, work, face, body, person, covered: trusted };
}

// 0..1 for each working pixel: could body skin here belong to a detected person? It
// has to be on a connected person shape that contains a detected face, and near that
// face's skin colour (a and b only, since shade varies with the light).
function bodyKeep(cv, seg, faces) {
  const t = TUNE, { ww, wh } = seg, n = ww * wh;
  const sx = ww / (seg.x1 - seg.x0), sy = wh / (seg.y1 - seg.y0);
  const isPerson = new cv.Mat(wh, ww, cv.CV_8U), labelsMat = new cv.Mat();
  { const v = isPerson.data; for (let i = 0; i < n; i++) v[i] = seg.person[i] > 0.5 ? 1 : 0; }
  const shapes = cv.connectedComponents(isPerson, labelsMat, 8, cv.CV_32S);
  isPerson.delete();
  const labels = new Int32Array(labelsMat.data32S);
  labelsMat.delete();
  const workMat = cv.matFromArray(wh, ww, cv.CV_8UC3, seg.work), labMat = new cv.Mat();
  cv.cvtColor(workMat, labMat, cv.COLOR_BGR2Lab);
  workMat.delete();
  const lab = new Uint8Array(labMat.data);
  labMat.delete();

  const onPerson = new Float32Array(shapes), nearest = new Float32Array(n).fill(1e9);
  for (const f of faces) {
    const [x0, y0, x1, y1] = middle(f, seg.x0, seg.y0, sx, sy, ww, wh);
    if (x1 <= x0 || y1 <= y0) continue;
    const As = [], Bs = [];
    for (let y = y0; y < y1; y++) {
      for (let x = x0, i = y * ww + x0; x < x1; x++, i++) {
        onPerson[labels[i]] = 1;
        if (seg.face[i] > t.colour_face) { As.push(lab[3 * i + 1]); Bs.push(lab[3 * i + 2]); }
      }
    }
    if (As.length < 16) continue;
    const a0 = median(As), b0 = median(Bs);
    for (let i = 0; i < n; i++) {
      const d = Math.max(Math.abs(lab[3 * i + 1] - a0), Math.abs(lab[3 * i + 2] - b0));
      if (d < nearest[i]) nearest[i] = d;
    }
  }
  onPerson[0] = 0;
  const keep = new Float32Array(n);
  for (let i = 0; i < n; i++) keep[i] = onPerson[labels[i]] * (1 - smoothstep(nearest[i], t.colour_lo, t.colour_hi));
  return keep;
}

// The skin the model found: { x0, y0, w, h, data: Float32Array in 0..1 } in photo
// pixels, sharp where the photo has an edge, cut to the skin's box. Or null.
export function skinArea(cv, seg, faces, faceOnly = false) {
  const t = TUNE, { ww, wh } = seg, n = ww * wh, rw = seg.x1 - seg.x0, rh = seg.y1 - seg.y0;
  const keep = faceOnly ? null : bodyKeep(cv, seg, faces);
  const areaMat = new cv.Mat(wh, ww, cv.CV_32F);
  {
    const v = areaMat.data32F;
    for (let i = 0; i < n; i++) v[i] = smoothstep(keep ? seg.face[i] + seg.body[i] * keep[i] : seg.face[i], t.prob_lo, t.prob_hi);
  }
  const width = medianOf(faces.map((f) => f[2])) * ww / rw;
  const workMat = cv.matFromArray(wh, ww, cv.CV_8UC3, seg.work), workF = new cv.Mat(), gray = new cv.Mat();
  workMat.convertTo(workF, cv.CV_32F, 1 / 255);
  workMat.delete();
  cv.cvtColor(workF, gray, cv.COLOR_BGR2GRAY);
  workF.delete();
  const snapped = guidedFilter(cv, gray, areaMat, Math.max(2, pyRound(width * t.snap_radius)), t.snap_edge ** 2);
  gray.delete(); areaMat.delete();
  let a0 = ww, a1 = 0, b0 = wh, b1 = 0;
  {
    const v = snapped.data32F;
    for (let y = 0, i = 0; y < wh; y++) {
      for (let x = 0; x < ww; x++, i++) {
        v[i] = smoothstep(v[i], t.snap_lo, t.snap_hi);
        if (v[i] > 0) { if (x < a0) a0 = x; if (x >= a1) a1 = x + 1; if (y < b0) b0 = y; if (y >= b1) b1 = y + 1; }
      }
    }
  }
  if (a1 === 0) { snapped.delete(); return null; }
  const sx = rw / ww, sy = rh / wh;
  a0 = Math.max(0, a0 - 1); a1 = Math.min(ww, a1 + 1);
  b0 = Math.max(0, b0 - 1); b1 = Math.min(wh, b1 + 1);
  const x0 = Math.trunc(a0 * sx), x1 = Math.min(rw, Math.ceil(a1 * sx));
  const y0 = Math.trunc(b0 * sy), y1 = Math.min(rh, Math.ceil(b1 * sy));
  const view = snapped.roi(new cv.Rect(a0, b0, a1 - a0, b1 - b0)), full = new cv.Mat();
  cv.resize(view, full, new cv.Size(x1 - x0, y1 - y0), 0, 0, cv.INTER_LINEAR);
  view.delete(); snapped.delete();
  const data = new Float32Array(full.data32F);
  full.delete();
  return { x0: seg.x0 + x0, y0: seg.y0 + y0, w: x1 - x0, h: y1 - y0, data };
}
