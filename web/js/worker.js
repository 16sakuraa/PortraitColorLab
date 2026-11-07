// Background worker: all image work happens here so the page stays responsive.
//
// Messages in:  init {blemishEngine} | open {file} | render {id, params} | export {params}
// Messages out: progress {text} | ready {engine} | opened {...} | analysed {healed, healedUnit}
//               | rendered {id, blob, maskBytes?, maskW?, maskH?} | exported {blob, ext} | error {message}
// params may hold strokes (see mask_edit.js) and wantMask (send the mask back too).

import * as pipeline from './pipeline.js';
import { matFromImage, imageFromMat } from './cv.js';
import { decode, encode, encodePreview } from './photo_io.js';

const PREVIEW_MAX = 1600;
let st = null;
let photo = null;   // { full, meta, analysis, preview, previewAnalysis, scale, faceOnly, layer }

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const progress = (text) => post({ type: 'progress', text });

function release() {
  if (!photo) return;
  photo.full.delete();
  photo.preview.delete();
  photo = null;
}

async function open(file) {
  release();
  progress('Reading photo…');
  const { image, meta } = await decode(file);
  const full = matFromImage(st.cv, image);
  progress('Finding faces, skin and blemishes…');
  const t0 = performance.now();
  const analysis = await pipeline.analyze(full);
  const s = Math.min(1, PREVIEW_MAX / Math.max(full.cols, full.rows));
  let preview = full;
  if (s < 1) {
    preview = new st.cv.Mat();
    st.cv.resize(full, preview, new st.cv.Size(Math.round(full.cols * s), Math.round(full.rows * s)), 0, 0, st.cv.INTER_AREA);
  } else {
    preview = new st.cv.Mat();
    full.copyTo(preview);
  }
  const previewAnalysis = s < 1 ? pipeline.scaleAnalysis(analysis, s, preview.cols, preview.rows) : analysis;
  photo = { full, meta, analysis, preview, previewAnalysis, scale: s, faceOnly: false, layer: null };
  const before = await encodePreview(imageFromMat(st.cv, preview), meta);
  post({
    type: 'opened', width: full.cols, height: full.rows, faces: analysis.faces.length,
    healed: analysis.healed, healedUnit: analysis.healedUnit, seconds: (performance.now() - t0) / 1000, before,
    colourProfile: meta.format === 'jpeg' ? meta.icc.length > 0 : !!meta.chunks?.length,
  });
}

// Face only changes the mask, and blemishes are looked for inside the mask.
async function useFaceOnly(faceOnly) {
  if (photo.faceOnly === faceOnly) return;
  photo.faceOnly = faceOnly;
  progress('Updating the skin mask…');
  photo.analysis = await pipeline.analyze(photo.full, { faceOnly, previous: photo.analysis });
  photo.previewAnalysis = photo.scale < 1
    ? pipeline.scaleAnalysis(photo.analysis, photo.scale, photo.preview.cols, photo.preview.rows) : photo.analysis;
  post({ type: 'analysed', healed: photo.analysis.healed, healedUnit: photo.analysis.healedUnit });
}

// The analysis with the brush's strokes applied, for an image this size.
function edited(analysis, params, w, h) {
  photo.layer = pipeline.editLayer(photo.full.cols, photo.full.rows, params.strokes || [], photo.layer);
  return pipeline.withEdits(analysis, photo.layer, w, h);
}

async function render(id, params) {
  if (!photo) return;
  await useFaceOnly(params.faceOnly);
  const { preview } = photo;
  const analysis = edited(photo.previewAnalysis, params, preview.cols, preview.rows);
  const out = pipeline.render(preview, analysis, params);
  const blob = await encodePreview(imageFromMat(st.cv, out), photo.meta);
  out.delete();
  if (params.wantMask) {
    const maskBytes = pipeline.maskBytes(analysis.mask, preview.cols, preview.rows);
    post({ type: 'rendered', id, blob, maskBytes, maskW: preview.cols, maskH: preview.rows }, [maskBytes.buffer]);
  } else {
    post({ type: 'rendered', id, blob });
  }
}

async function exportFull(params) {
  if (!photo) return;
  await useFaceOnly(params.faceOnly);
  progress('Processing full resolution…');
  const analysis = edited(photo.analysis, params, photo.full.cols, photo.full.rows);
  const out = pipeline.render(photo.full, analysis, params);
  const image = imageFromMat(st.cv, out);
  out.delete();
  progress('Saving…');
  const { blob, ext } = await encode(image, photo.meta);
  post({ type: 'exported', blob, ext });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      st = await pipeline.init(progress, { blemishEngine: data.blemishEngine });
      post({ type: 'ready', engine: st.engine });
    } else if (data.type === 'open') {
      await open(data.file);
    } else if (data.type === 'render') {
      await render(data.id, data.params);
    } else if (data.type === 'export') {
      await exportFull(data.params);
    }
  } catch (e) {
    post({ type: 'error', message: e?.message || String(e) });
  }
};
