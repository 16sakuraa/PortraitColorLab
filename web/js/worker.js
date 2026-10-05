// Background worker: all image work happens here so the page stays responsive.
//
// Messages in:  init | open {file} | render {id, params} | export {params}
// Messages out: progress {text} | ready {engine} | opened {...} | rendered {id, blob}
//               | exported {blob, ext} | error {message}

import * as pipeline from './pipeline.js';
import { matFromImage, imageFromMat } from './cv.js';
import { decode, encode, encodePreview } from './photo_io.js';

const PREVIEW_MAX = 1600;
let st = null;
let photo = null;   // { full, meta, analysis, preview, previewAnalysis }

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
  progress('Finding faces and blemishes…');
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
  photo = { full, meta, analysis, preview, previewAnalysis };
  const before = await encodePreview(imageFromMat(st.cv, preview), meta);
  post({
    type: 'opened', width: full.cols, height: full.rows, faces: analysis.faces.length,
    healed: analysis.healed, seconds: (performance.now() - t0) / 1000, before,
    colourProfile: meta.format === 'jpeg' ? meta.icc.length > 0 : !!meta.chunks?.length,
  });
}

async function render(id, params) {
  if (!photo) return;
  const out = pipeline.render(photo.preview, photo.previewAnalysis, params);
  const blob = await encodePreview(imageFromMat(st.cv, out), photo.meta);
  out.delete();
  post({ type: 'rendered', id, blob });
}

async function exportFull(params) {
  if (!photo) return;
  progress('Processing full resolution…');
  const out = pipeline.render(photo.full, photo.analysis, params);
  const image = imageFromMat(st.cv, out);
  out.delete();
  progress('Saving…');
  const { blob, ext } = await encode(image, photo.meta);
  post({ type: 'exported', blob, ext });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      st = await pipeline.init(progress);
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
