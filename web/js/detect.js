// Face detection with YuNet, matching the desktop app (smoother._detect_faces,
// which uses OpenCV's FaceDetectorYN):
//   * photos are searched on a copy shrunk to at most 1280 px (INTER_AREA);
//   * the image is padded to a multiple of 32 and fed as raw BGR 0..255;
//   * outputs are decoded at strides 8/16/32, score = sqrt(cls * obj),
//     score threshold 0.6, NMS IoU 0.3 over the 50 best.
// Each face is a row like OpenCV's: [x, y, w, h, right eye x, y, left eye x, y,
// nose x, y, right mouth corner x, y, left mouth corner x, y, score].

export const DETECT_MAX = 1280;
const SCORE_THR = 0.6, NMS_THR = 0.3, TOP_K = 50, STRIDES = [8, 16, 32], SAME_FACE_OVERLAP = 0.3;

export async function createSession(ort, url, providers) {
  return ort.InferenceSession.create(url, { executionProviders: providers });
}

export async function detectFaces(cv, ort, session, bgr) {
  const h = bgr.rows, w = bgr.cols;
  const s = Math.min(1, DETECT_MAX / Math.max(h, w));
  let img = bgr;
  if (s < 1) {
    img = new cv.Mat();
    cv.resize(bgr, img, new cv.Size(Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))), 0, 0, cv.INTER_AREA);
  }
  const dw = img.cols, dh = img.rows;
  const padW = (Math.floor((dw - 1) / 32) + 1) * 32, padH = (Math.floor((dh - 1) / 32) + 1) * 32;

  // NCHW float32, BGR order, zero padding at the bottom/right.
  const plane = padW * padH;
  const input = new Float32Array(3 * plane);
  const d = img.data;
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const i = (y * dw + x) * 3, o = y * padW + x;
      input[o] = d[i]; input[plane + o] = d[i + 1]; input[2 * plane + o] = d[i + 2];
    }
  }
  if (img !== bgr) img.delete();

  const out = await session.run({ input: new ort.Tensor('float32', input, [1, 3, padH, padW]) });
  const faces = [];
  for (const stride of STRIDES) {
    const cols = padW / stride, rows = padH / stride;
    const cls = out[`cls_${stride}`].data, obj = out[`obj_${stride}`].data;
    const bbox = out[`bbox_${stride}`].data, kps = out[`kps_${stride}`].data;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const score = Math.sqrt(Math.min(Math.max(cls[idx], 0), 1) * Math.min(Math.max(obj[idx], 0), 1));
        if (score < SCORE_THR) continue;
        const cx = (c + bbox[idx * 4]) * stride, cy = (r + bbox[idx * 4 + 1]) * stride;
        const bw = Math.exp(bbox[idx * 4 + 2]) * stride, bh = Math.exp(bbox[idx * 4 + 3]) * stride;
        const row = [cx - bw / 2, cy - bh / 2, bw, bh];
        for (let n = 0; n < 5; n++) {
          row.push((kps[idx * 10 + 2 * n] + c) * stride, (kps[idx * 10 + 2 * n + 1] + r) * stride);
        }
        row.push(score);
        faces.push(row);
      }
    }
  }
  for (const t of Object.values(out)) t.dispose?.();

  const kept = nms(faces);
  if (s < 1) for (const f of kept) for (let k = 0; k < 14; k++) f[k] /= s;
  return separateFaces(kept);
}

// Does the landmark layout look like a face? (see smoother._plausible)
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

// Only one row per real face, in the same order (see smoother._separate_faces). A row
// whose landmarks sit inside a larger face's box, or whose box is largely inside it,
// is that face again; a row that doesn't look like a face goes too, unless it's all
// there is.
export function separateFaces(faces) {
  if (faces.length < 2) return faces;
  const sameFace = (f, g) => {
    let cx = 0, cy = 0;
    for (let k = 0; k < 5; k++) { cx += f[4 + 2 * k]; cy += f[5 + 2 * k]; }
    cx /= 5; cy /= 5;
    if (g[0] <= cx && cx <= g[0] + g[2] && g[1] <= cy && cy <= g[1] + g[3]) return true;
    const across = Math.min(f[0] + f[2], g[0] + g[2]) - Math.max(f[0], g[0]);
    const down = Math.min(f[1] + f[3], g[1] + g[3]) - Math.max(f[1], g[1]);
    return across > 0 && down > 0 && across * down >= SAME_FACE_OVERLAP * f[2] * f[3];
  };
  const order = faces.map((_, i) => i).sort((a, b) => faces[b][2] * faces[b][3] - faces[a][2] * faces[a][3]);
  const kept = [];
  for (const i of order) {
    if (plausible(faces[i]) && !kept.some((k) => sameFace(faces[i], faces[k]))) kept.push(i);
  }
  if (!kept.length) kept.push(order[0]);
  return kept.sort((a, b) => a - b).map((i) => faces[i]);
}

// Greedy NMS on integer boxes, like cv::dnn::NMSBoxes.
function nms(faces) {
  const order = faces.map((f, i) => i).sort((a, b) => faces[b][14] - faces[a][14]).slice(0, TOP_K);
  const box = (f) => [Math.trunc(f[0]), Math.trunc(f[1]), Math.trunc(f[2]), Math.trunc(f[3])];
  const iou = (a, b) => {
    const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    return inter / (a[2] * a[3] + b[2] * b[3] - inter);
  };
  const keep = [];
  for (const i of order) {
    const bi = box(faces[i]);
    if (keep.every((k) => iou(bi, box(faces[k])) <= NMS_THR)) keep.push(i);
  }
  return keep.map((i) => faces[i]);
}
