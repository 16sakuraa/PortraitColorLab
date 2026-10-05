// The whole retouch, in the same order as the desktop app (serve.py):
//   mask = skin mask of the original; heal blemishes; smooth through the mask.
// analyze() does the expensive, slider-independent part once per photo;
// render() applies the slider values.

import { loadCv } from './cv.js';
import { createSession, detectFaces } from './detect.js';
import * as blemish from './blemish.js';
import { buildSkinMask } from './skinmask.js';
import { smoothFace } from './smooth.js';

const LIB = new URL('../lib/', import.meta.url).href;
const MODELS = new URL('../models/', import.meta.url).href;

let state = null;

// Loads OpenCV.js, onnxruntime-web and both models. Uses the graphics chip
// (WebGPU) when the browser offers it, otherwise the CPU (WebAssembly).
export async function init(onProgress = () => {}) {
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
  const healer = await make('blemish_ffhqr.onnx');
  state = { cv, ort, detector, healer, engine };
  return state;
}

export async function analyze(bgr) {
  const { cv, ort, detector, healer } = state;
  const faces = await detectFaces(cv, ort, detector, bgr);
  const mask = buildSkinMask(cv, bgr, faces);
  const { corr, count } = await blemish.correction(cv, ort, healer, bgr, faces);
  return { faces, mask, corr, healed: count };
}

// params: { blemish: 0..1, strength: 0..1, texture: 0..1 }
export function render(bgr, analysis, params) {
  const { cv } = state;
  const healed = blemish.apply(cv, bgr, analysis.corr, params.blemish);
  const out = smoothFace(cv, healed, params.strength, params.texture, analysis.mask);
  healed.delete();
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
  return { ...analysis, mask, corr };
}

export const engine = () => state?.engine;
