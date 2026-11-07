// Page logic: hands photos to the worker and shows its results.

import { attachLoupe } from './loupe.js';
import { attachMaskBrush } from './mask_brush.js';

const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
// The brush first: it wraps the picture, and the magnifier belongs inside that.
const brush = attachMaskBrush({
  image: $('after'),
  button: $('editmask'),
  onChange: () => requestRender(),
  onModeChange: (editing) => { if (editing) { loupe.hide(); requestRender(); } },
});
const loupe = attachLoupe([$('before'), $('after')]);

let ready = false, hasPhoto = false, engine = '';
let photoName = 'photo';
let renderId = 0, shownId = 0, busy = false, pending = false;
const urls = { before: null, after: null };

function status(text, spinning = false) {
  $('status').textContent = text;
  $('status').classList.toggle('busy', spinning);
}

function show(which, blob) {
  const old = urls[which];
  urls[which] = URL.createObjectURL(blob);
  $(which).src = urls[which];
  // Swap the magnifier over once the new picture is ready, then free the old one.
  $(which).decode().catch(() => {}).finally(() => {
    loupe.refresh();
    if (old) URL.revokeObjectURL(old);
  });
}

const params = () => ({
  blemish: +$('blemish').value / 100,
  strength: +$('strength').value / 100,
  texture: +$('texture').value / 100,
  pores: +$('pores').value / 100,
  faceOnly: $('faceonly').checked,
  strokes: brush.strokes,
  wantMask: brush.editing,
});

function setControls(enabled) {
  for (const id of ['blemish', 'strength', 'texture', 'pores', 'faceonly', 'editmask', 'download']) $(id).disabled = !enabled;
}

let summary = '', opened = null;
function describe({ healed, healedUnit }) {
  const faces = opened.faces === 1 ? '1 face' : `${opened.faces} faces`;
  const treated = !healed ? ''
    : healedUnit === 'spots' ? ` · ${healed === 1 ? '1 spot' : `${healed} spots`} removed`
      : ` · blemishes treated on ${healed}`;
  summary = `${opened.width}×${opened.height} · ${faces} found` + treated +
    (opened.colourProfile ? ' · colour profile kept' : '') +
    ` · analysed in ${opened.seconds.toFixed(1)} s`;
  status(summary);
}

let debounce = null;
function scheduleRender() {
  if (!hasPhoto) return;
  clearTimeout(debounce);
  debounce = setTimeout(requestRender, 120);
}
function requestRender() {
  if (busy) { pending = true; return; }
  busy = true;
  worker.postMessage({ type: 'render', id: ++renderId, params: params() });
}

worker.onmessage = ({ data }) => {
  switch (data.type) {
    case 'progress':
      status(data.text, true);
      break;
    case 'ready':
      ready = true;
      engine = data.engine === 'webgpu' ? 'graphics chip (WebGPU)' : 'CPU — slower; a browser with WebGPU (Chrome or Edge) is faster';
      status(`Ready · running on the ${engine}.`);
      break;
    case 'opened': {
      hasPhoto = true;
      show('before', data.before);
      $('drop').classList.add('hidden');
      $('workspace').classList.remove('hidden');
      $('reset').classList.remove('hidden');
      setControls(true);
      opened = data;
      describe(data);
      requestRender();
      break;
    }
    case 'analysed':
      describe(data);
      break;
    case 'rendered':
      busy = false;
      if (data.id >= shownId) {
        shownId = data.id;
        show('after', data.blob);
        if (data.maskBytes) brush.setMaskBytes(data.maskW, data.maskH, data.maskBytes);
      }
      if (pending) { pending = false; requestRender(); }
      break;
    case 'exported': {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(data.blob);
      a.download = `${photoName}_retouched${data.ext}`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
      setControls(true);
      status(`${summary} · saved ${a.download} (${(data.blob.size / 1e6).toFixed(1)} MB)`);
      break;
    }
    case 'error':
      busy = false;
      setControls(hasPhoto);
      status('Something went wrong: ' + data.message);
      break;
  }
};

function openFile(file) {
  if (!file) return;
  if (!ready) { status('Still loading, one moment…', true); setTimeout(() => openFile(file), 500); return; }
  photoName = file.name.replace(/\.[^.]+$/, '') || 'photo';
  brush.reset();
  setControls(false);
  status('Reading photo…', true);
  worker.postMessage({ type: 'open', file });
}

$('drop').addEventListener('click', () => $('file').click());
$('drop').addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file').click(); } });
$('drop').addEventListener('dragover', (e) => { e.preventDefault(); $('drop').classList.add('over'); });
$('drop').addEventListener('dragleave', () => $('drop').classList.remove('over'));
$('drop').addEventListener('drop', (e) => { e.preventDefault(); $('drop').classList.remove('over'); openFile(e.dataTransfer.files[0]); });
$('file').addEventListener('change', (e) => openFile(e.target.files[0]));

for (const [id, label] of [['blemish', 'bval'], ['strength', 'sval'], ['texture', 'tval'], ['pores', 'pval']]) {
  $(id).addEventListener('input', () => { $(label).textContent = $(id).value; scheduleRender(); });
}

$('faceonly').addEventListener('change', scheduleRender);

$('download').addEventListener('click', () => {
  setControls(false);
  status('Processing full resolution…', true);
  worker.postMessage({ type: 'export', params: params() });
});

$('reset').addEventListener('click', () => {
  hasPhoto = false;
  brush.reset();
  loupe.hide();
  $('drop').classList.remove('hidden');
  $('workspace').classList.add('hidden');
  $('reset').classList.add('hidden');
  $('file').value = '';
  setControls(false);
  status(`Ready · running on the ${engine}.`);
});

status('Loading (about 60 MB the first time, then cached)…', true);
// ?blemish=network uses the blemish network instead (non-commercial licence).
const blemishEngine = new URLSearchParams(location.search).get('blemish') === 'network' ? 'network' : 'classic';
worker.postMessage({ type: 'init', blemishEngine });
