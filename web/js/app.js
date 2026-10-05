// Page logic: hands photos to the worker and shows its results.

const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

let ready = false, hasPhoto = false, engine = '';
let photoName = 'photo';
let renderId = 0, shownId = 0, busy = false, pending = false;
const urls = { before: null, after: null };

function status(text, spinning = false) {
  $('status').textContent = text;
  $('status').classList.toggle('busy', spinning);
}

function show(which, blob) {
  if (urls[which]) URL.revokeObjectURL(urls[which]);
  urls[which] = URL.createObjectURL(blob);
  $(which).src = urls[which];
}

const params = () => ({
  blemish: +$('blemish').value / 100,
  strength: +$('strength').value / 100,
  texture: +$('texture').value / 100,
});

function setControls(enabled) {
  for (const id of ['blemish', 'strength', 'texture', 'download']) $(id).disabled = !enabled;
}

let summary = '';
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
      const faces = data.faces === 1 ? '1 face' : `${data.faces} faces`;
      summary = `${data.width}×${data.height} · ${faces} found` +
        (data.healed ? ` · blemishes treated on ${data.healed}` : '') +
        (data.colourProfile ? ' · colour profile kept' : '') +
        ` · analysed in ${data.seconds.toFixed(1)} s`;
      status(summary);
      requestRender();
      break;
    }
    case 'rendered':
      busy = false;
      if (data.id >= shownId) { shownId = data.id; show('after', data.blob); }
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

for (const [id, label] of [['blemish', 'bval'], ['strength', 'sval'], ['texture', 'tval']]) {
  $(id).addEventListener('input', () => { $(label).textContent = $(id).value; scheduleRender(); });
}

$('download').addEventListener('click', () => {
  setControls(false);
  status('Processing full resolution…', true);
  worker.postMessage({ type: 'export', params: params() });
});

$('reset').addEventListener('click', () => {
  hasPhoto = false;
  $('drop').classList.remove('hidden');
  $('workspace').classList.add('hidden');
  $('reset').classList.add('hidden');
  $('file').value = '';
  setControls(false);
  status(`Ready · running on the ${engine}.`);
});

status('Loading (about 40 MB the first time, then cached)…', true);
worker.postMessage({ type: 'init' });
