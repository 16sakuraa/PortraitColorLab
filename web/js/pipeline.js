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
import { landmarks, zones, earZones } from './face_landmarks.js';
import { smoothFace, smoothAgain, detailSigma, PRINT_GRAIN, LIGHTING } from './smooth.js';
import * as maskEdit from './mask_edit.js';
import * as heal from './heal.js';
import * as look from './look.js';
import * as lip from './lip.js';

const LIB = new URL('../lib/', import.meta.url).href;
const MODELS = new URL('../models/', import.meta.url).href;

let state = null;

// Loads OpenCV.js, onnxruntime-web and the models. Uses the graphics chip
// (WebGPU) when the browser offers it, otherwise the CPU (WebAssembly).
// blemishEngine: 'classic' (blemish_classic.js, no model) or 'network' (blemish.js,
// whose model is non-commercial only).
export async function init(onProgress = () => {}, { blemishEngine = 'classic' } = {}) {
  if (state) return state;
  onProgress('Loading image tools…', 'progress.tools');
  const cv = await loadCv(LIB + 'opencv/opencv.js');
  onProgress('Loading face models…', 'progress.models');
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
  const ears = points.map((p) => p && earZones(cv, p, bgr.cols, bgr.rows));
  // The mask, and all the skin with the eyes, brows, lips and ears in (look.js's steps go by it).
  const whole = area && faceOnly ? skinArea(cv, seg, faces, false) : area;
  const { area: allSkin, ...mask } = buildSkinMask(cv, bgr, faces, 1.0, area, seg ? seg.covered : [], outlines, ears, whole);
  const skin = { x0: mask.x0, y0: mask.y0, w: mask.w, h: mask.h, data: allSkin };
  const classic = blemishEngine === 'classic';
  let corr, count;
  if (classic) ({ corr, count } = blemishClassic.correction(cv, bgr, faces, mask));
  else if (previous) ({ corr, healed: count } = previous);     // the network doesn't use the mask
  else ({ corr, count } = await blemish.correction(cv, ort, healer, bgr, faces));
  return {
    faces, seg, points, mask, corr, healed: count, healedUnit: classic ? 'spots' : 'faces',
    area: skin, luma: look.skinLuma(bgr, mask, faces), bright: look.brightColours(bgr, mask, faces),
    person: look.personOf(seg),
  };
}

// params: { blemish: 0..1, strength: 0..1, texture: 0..1, pores: 0..1, shine: 0..1 (optional),
// print: 'large' | 'small', lighting: 'soft' | 'hard', eyes: 0..1, tone: 0..1 and strays: 0..1
// (optional, look.js's under-eyes, skin tone and stray hairs), lip: 0..1 (optional, lip.js's thin
// moustache; lipFull takes a full one too; lipForce skips its questions), level: 0..1 (optional, straighten
// the head; left out while wantMask is set), strokes (optional, the brush's) }.
// The parity tool passes grain and light (numbers) instead of print and lighting.
// The analysis already holds the strokes' mask edits (withEdits); the heal and spot
// strokes are painted here, after the spot fills and before smoothing, Smooth again
// after the smoothing, then under-eyes and skin tone, and the light strokes last.
export function render(bgr, analysis, params) {
  const { cv } = state;
  const spotless = blemish.apply(cv, bgr, analysis.corr, params.blemish);
  // A thin moustache, if asked for (and a full one, if that is asked for too): before the healing
  // brush, so its strokes go on top, and not where the brush's Restore says to leave the photo alone.
  const filled = lip.clear(cv, spotless, analysis.points, analysis.mask, params.lip ?? 0, analysis.restored ?? null,
    params.lipForce ?? false, params.lipFull ?? false);
  if (filled !== spotless) spotless.delete();
  const healed = heal.apply(cv, filled, params.strokes);
  if (healed !== filled) filled.delete();
  const grain = params.grain ?? PRINT_GRAIN[params.print] ?? 1;
  const light = params.light ?? LIGHTING[params.lighting] ?? 1;
  let smoothed = smoothFace(cv, healed, params.strength, params.texture, analysis.mask, analysis.faces,
    params.pores, grain, params.shine ?? 0, light);
  if (analysis.again) {
    const again = smoothAgain(cv, healed, smoothed, analysis.again, analysis.mask, analysis.faces, params.texture,
      params.pores, grain, params.shine ?? 0, light);
    smoothed.delete();
    smoothed = again;
  }
  healed.delete();
  // Shiny patches (the other half of Reduce shine), under-eyes, skin tone and stray
  // hairs, all off unless asked for, on the smoothed result.
  const finished = look.finish(cv, smoothed, analysis.mask, analysis.area, analysis.points, analysis.luma,
    params.eyes ?? 0, params.tone ?? 0, params.shine ?? 0, analysis.faces, analysis.bright,
    params.strays ?? 0, analysis.person, analysis.restored ?? null,
    params.wantMask ? 0 : params.level ?? 0);       // off while the brush is in use: it would move the photo under the strokes
  if (finished !== smoothed) { smoothed.delete(); smoothed = finished; }
  // The light brush paints tone on the smoothed result, keeping the grain it kept.
  const out = heal.applyLight(cv, smoothed, params.strokes, detailSigma(bgr.rows, bgr.cols, analysis.faces, grain));
  if (out !== smoothed) smoothed.delete();
  return out;
}

// A brush layer for a photo of this size holding these strokes (mask_edit.js).
// previous is the last one, which saves repainting strokes it already holds.
export function editLayer(w, h, strokes, previous = null) {
  return maskEdit.layerFor(state.cv, w, h, strokes, previous);
}

// An analysis with the brush's edits applied, for an image of this size.
// The spot fills follow their own layer once there's an Undo smoothing stroke, and
// Smooth again its own (again: null without one). The skin area follows the fills'.
export function withEdits(analysis, layer, w, h) {
  if (!layer || !layer.strokes.length) return analysis;
  const region = maskEdit.regionOf(state.cv, layer, w, h);
  const fills = layer.fills ? maskEdit.regionOf(state.cv, { ...layer, data: layer.fills }, w, h) : region;
  const again = layer.again ? maskEdit.regionOf(state.cv, { ...layer, data: layer.again }, w, h) : null;
  if (!region && !fills && !again) return analysis;
  return {
    ...analysis, mask: maskEdit.applyToMask(analysis.mask, region),
    corr: maskEdit.applyToCorrection(analysis.corr, fills), again,
    area: analysis.area && maskEdit.applyToMask(analysis.area, fills),
    restored: fills,               // the brush's Restore strokes, for the step outside the skin
  };
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
  const shrunk = (m) => {
    const mat = cv.matFromArray(m.h, m.w, cv.CV_32F, m.data);
    const w = Math.max(1, Math.round(m.w * s)), h = Math.max(1, Math.round(m.h * s));
    const small = new cv.Mat();
    cv.resize(mat, small, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
    const x0 = Math.min(previewW - 1, Math.round(m.x0 * s)), y0 = Math.min(previewH - 1, Math.round(m.y0 * s));
    const out = { x0, y0, w: Math.min(w, previewW - x0), h: Math.min(h, previewH - y0), data: null };
    out.data = new Float32Array(out.w * out.h);
    for (let y = 0; y < out.h; y++) for (let x = 0; x < out.w; x++) out.data[y * out.w + x] = small.data32F[y * w + x];
    mat.delete(); small.delete();
    return out;
  };
  const mask = shrunk(analysis.mask);
  const area = analysis.area && shrunk(analysis.area);
  const points = analysis.points && analysis.points.map((p) => p && p.map((v) => v * s));
  // The person's outline keeps its scores, over the region's place in the preview.
  let person = analysis.person;
  if (person) {
    const px0 = Math.min(previewW - 1, Math.round(person.x0 * s)), py0 = Math.min(previewH - 1, Math.round(person.y0 * s));
    person = { ...person, x0: px0, y0: py0, x1: Math.max(px0 + 1, Math.min(previewW, Math.round(person.x1 * s))),
      y1: Math.max(py0 + 1, Math.min(previewH, Math.round(person.y1 * s))) };
  }
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
  return { ...analysis, mask, corr, faces, area, points, person };
}

export const engine = () => state?.engine;
