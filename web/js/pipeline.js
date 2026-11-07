// The whole retouch, in the same order as the desktop app (serve.py):
//   mask = skin mask of the original; heal blemishes; smooth through the mask.
// analyze() does the expensive, slider-independent part once per photo;
// render() applies the slider values.

import { loadCv } from './cv.js';
import { createSession, detectFaces } from './detect.js';
import * as blemish from './blemish.js';
import * as blemishClassic from './blemish_classic.js';
import { buildSkinMask } from './skinmask.js';
import { segment, skinArea } from './skin_segment.js';
import { landmarks, zones } from './face_landmarks.js';
import { smoothFace } from './smooth.js';
import * as maskEdit from './mask_edit.js';

const LIB = new URL('../lib/', import.meta.url).href;
const MODELS = new URL('../models/', import.meta.url).href;

let state = null;

// Loads OpenCV.js, onnxruntime-web and the models. Uses the graphics chip
// (WebGPU) when the browser offers it, otherwise the CPU (WebAssembly).
// blemishEngine: 'classic' (blemish_classic.js, no model) or 'network' (blemish.js,
// whose model is non-commercial only).
export async function init(onProgress = () => {}, { blemishEngine = 'classic' } = {}) {
  if (state) return state;
  onProgress('Loading image tools…');
  const cv = await loadCv(LIB + 'opencv/opencv.js');
  onProgress('Loading face models…');
  const ort = await import(LIB + 'ort/ort.webgpu.min.mjs');
  ort.env.wasm.wasmPaths = LIB + 'ort/';
  let engine = 'wasm';
  if (globalThis.navigator?.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) engine = 'webgpu';
    } catch { /* no usable GPU */ }
  }
  const make = async (name) => {
    try {
      return await createSession(ort, MODELS + name, [engine]);
    } catch (e) {
      if (engine === 'wasm') throw e;
      engine = 'wasm';                         // GPU present but unusable: fall back
      return createSession(ort, MODELS + name, ['wasm']);
    }
  };
  const detector = await make('face_detection_yunet.onnx');
  const healer = blemishEngine === 'network' ? await make('blemish_ffhqr.onnx') : null;
  // Without this model the skin mask falls back to its older method.
  const segmenter = await make('selfie_multiclass_256x256.onnx').catch(() => null);
  // Without this one, eyes, brows and lips are found the older way too.
  const landmarker = await make('face_landmarks.onnx').catch(() => null);
  state = { cv, ort, detector, healer, segmenter, landmarker, engine, blemishEngine };
  return state;
}

// healed counts faces for the network and spots for the classic engine (healedUnit).
// faceOnly leaves body skin out of the mask. previous is this photo's last analysis,
// so switching faceOnly doesn't have to run the models again.
export async function analyze(bgr, { faceOnly = false, previous = null } = {}) {
  const { cv, ort, detector, healer, segmenter, landmarker, blemishEngine } = state;
  const faces = previous ? previous.faces : await detectFaces(cv, ort, detector, bgr);
  const seg = previous ? previous.seg : await segment(cv, ort, segmenter, bgr, faces);
  const points = previous ? previous.points : await landmarks(cv, ort, landmarker, bgr, faces);
  const area = seg ? skinArea(cv, seg, faces, faceOnly) : null;
  const outlines = points.map((p) => p && zones(cv, p, bgr.cols, bgr.rows));
  const mask = buildSkinMask(cv, bgr, faces, 1.0, area, seg ? seg.covered : [], outlines);
  const classic = blemishEngine === 'classic';
  let corr, count;
  if (classic) ({ corr, count } = blemishClassic.correction(cv, bgr, faces, mask));
  else if (previous) ({ corr, healed: count } = previous);     // the network doesn't use the mask
  else ({ corr, count } = await blemish.correction(cv, ort, healer, bgr, faces));
  return { faces, seg, points, mask, corr, healed: count, healedUnit: classic ? 'spots' : 'faces' };
}

// params: { blemish: 0..1, strength: 0..1, texture: 0..1, pores: 0..1 }
export function render(bgr, analysis, params) {
  const { cv } = state;
  const healed = blemish.apply(cv, bgr, analysis.corr, params.blemish);
  const out = smoothFace(cv, healed, params.strength, params.texture, analysis.mask, analysis.faces, params.pores);
  healed.delete();
  return out;
}

// A brush layer for a photo of this size holding these strokes (mask_edit.js).
// previous is the last one, which saves repainting strokes it already holds.
export function editLayer(w, h, strokes, previous = null) {
  return maskEdit.layerFor(state.cv, w, h, strokes, previous);
}

// An analysis with the brush's edits applied, for an image of this size.
export function withEdits(analysis, layer, w, h) {
  if (!layer || !layer.strokes.length) return analysis;
  const region = maskEdit.regionOf(state.cv, layer, w, h);
  if (!region) return analysis;
  return { ...analysis, mask: maskEdit.applyToMask(analysis.mask, region), corr: maskEdit.applyToCorrection(analysis.corr, region) };
}

// A mask as one byte per pixel of an image this size, for drawing it over the photo.
export function maskBytes(mask, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < mask.h; y++) {
    const to = (y + mask.y0) * w + mask.x0, from = y * mask.w;
    for (let x = 0; x < mask.w; x++) out[to + x] = mask.data[from + x] * 255 + 0.5;
  }
  return out;
}

// The analysis of a photo, scaled down to a preview of it (same look,
// without running the networks again).
export function scaleAnalysis(analysis, s, previewW, previewH) {
  const { cv } = state;
  const m = analysis.mask;
  const mat = cv.matFromArray(m.h, m.w, cv.CV_32F, m.data);
  const w = Math.max(1, Math.round(m.w * s)), h = Math.max(1, Math.round(m.h * s));
  const small = new cv.Mat();
  cv.resize(mat, small, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
  const x0 = Math.min(previewW - 1, Math.round(m.x0 * s)), y0 = Math.min(previewH - 1, Math.round(m.y0 * s));
  const mask = { x0, y0, w: Math.min(w, previewW - x0), h: Math.min(h, previewH - y0), data: null };
  mask.data = new Float32Array(mask.w * mask.h);
  for (let y = 0; y < mask.h; y++) for (let x = 0; x < mask.w; x++) mask.data[y * mask.w + x] = small.data32F[y * w + x];
  mat.delete(); small.delete();
  let corr = blemish.scaleCorrection(cv, analysis.corr, s);
  if (corr) {   // clip to the preview
    const cw = Math.min(corr.w, previewW - corr.x0), ch = Math.min(corr.h, previewH - corr.y0);
    if (cw <= 0 || ch <= 0) corr = null;
    else if (cw !== corr.w || ch !== corr.h) {
      const d = new Float32Array(cw * ch * 3);
      for (let y = 0; y < ch; y++) d.set(corr.data.subarray(3 * y * corr.w, 3 * (y * corr.w + cw)), 3 * y * cw);
      corr = { ...corr, w: cw, h: ch, data: d };
    }
  }
  const faces = analysis.faces.map((f) => f.map((v, k) => (k < 14 ? v * s : v)));  // the last value is the score
  return { ...analysis, mask, corr, faces };
}

export const engine = () => state?.engine;
