// Page logic: hands photos to the worker and shows its results.

import { attachCompareZoom } from './compare_zoom.js';
import { attachHistory } from './history.js';
import { attachMaskBrush, attachSteps } from './mask_brush.js';
import { t, translatePage, attachLangSwitch, onLangChange } from './i18n.js';

const $ = (id) => document.getElementById(id);
// The page in the language last chosen (or the browser's), with a TH | EN switch.
translatePage();
attachLangSwitch($('actions'));
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
// The brush first: it wraps the picture, and the comparison's zoom works inside that.
const brush = attachMaskBrush({
  image: $('after'),
  button: $('editmask'),
  onChange: () => requestRender(),
  onModeChange: (editing) => { if (editing) requestRender(); },
  toolsHost: $('brushbox'),
  labels: [t('brush.use'), t('brush.done')],
});
// Scroll on either photo to zoom both into the same spot.
const zoom = attachCompareZoom([$('before'), $('after')]);
// The settings and brush tools in three steps, like DM Portrait's: Skin, Smooth, Finish.
const steps = attachSteps({ panel: document.querySelector('.tools'), brush, navBefore: $('historybox') });
// Every change listed, to go back to any of them, with Undo and Redo.
const history = attachHistory({ host: $('historybox'), brush, render: () => requestRender() });

let ready = false, hasPhoto = false, engine = '';
let photoName = 'photo';
let renderId = 0, shownId = 0, busy = false, pending = false;
const urls = { before: null, after: null };

// text: a function giving the words, so a language switch can say it again.
let lastStatus = null;
function status(text, spinning = false) {
  lastStatus = { text, spinning };
  $('status').textContent = text();
  $('status').classList.toggle('busy', spinning);
}
onLangChange(() => { if (lastStatus) status(lastStatus.text, lastStatus.spinning); });

function show(which, blob) {
  const old = urls[which];
  urls[which] = URL.createObjectURL(blob);
  $(which).src = urls[which];
  // Free the old picture once the new one is ready.
  $(which).decode().catch(() => {}).finally(() => {
    if (old) URL.revokeObjectURL(old);
  });
}

const params = () => ({
  blemish: +$('blemish').value / 100,
  strength: +$('strength').value / 100,
  texture: +$('texture').value / 100,
  shine: +$('shine').value / 100,
  pores: +$('pores').value / 100,
  print: $('print').value,
  lighting: $('lighting').value,
  faceOnly: $('faceonly').checked,
  strokes: brush.strokes,
  wantMask: brush.editing,
});

function setControls(enabled) {
  for (const id of ['blemish', 'strength', 'texture', 'shine', 'pores', 'lighting', 'print', 'faceonly', 'editmask', 'download']) $(id).disabled = !enabled;
}

let summary = () => '', opened = null;
const readyText = () => t('status.ready', { engine: t(engine) });
function describe({ healed, healedUnit }) {
  const o = opened;
  summary = () => {
    const parts = [`${o.width}×${o.height}`, o.faces === 1 ? t('summary.faces1') : t('summary.faces', { n: o.faces })];
    if (healed) {
      const unit = healedUnit === 'spots' ? 'summary.spots' : 'summary.healed';
      parts.push(healed === 1 ? t(`${unit}1`) : t(unit, { n: healed }));
    }
    if (o.colourProfile) parts.push(t('summary.profile'));
    parts.push(t('summary.time', { s: o.seconds.toFixed(1) }));
    return parts.join(' · ');
  };
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
      status(() => (data.key ? t(data.key) : data.text), true);
      break;
    case 'ready':
      ready = true;
      engine = data.engine === 'webgpu' ? 'engine.gpu' : 'engine.cpu';
      status(readyText);
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
      history.start();
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
        if (data.maskBytes) brush.setMaskBytes(data.maskW, data.maskH, data.maskBytes, data.upTo);
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
      history.markSaved(exporting);     // closing the tab now loses nothing
      const saved = { name: a.download, mb: (data.blob.size / 1e6).toFixed(1) };
      status(() => `${summary()} · ${t('summary.saved', saved)}`);
      break;
    }
    case 'error':
      busy = false;
      setControls(hasPhoto);
      status(() => t('status.wrong', { message: data.message }));
      break;
  }
};

// Leaving a photo whose current state hasn't been downloaded asks first.
const mayLeave = () => !hasPhoto || !history.unsaved || confirm(t('confirm.newPhoto'));

function openFile(file) {
  if (!file || !mayLeave()) return;
  if (!ready) { status(() => t('status.stillLoading'), true); setTimeout(() => openFile(file), 500); return; }
  photoName = file.name.replace(/\.[^.]+$/, '') || 'photo';
  brush.reset();
  steps.setStep(1);
  zoom.reset();
  setControls(false);
  status(() => t('progress.reading'), true);
  worker.postMessage({ type: 'open', file });
}

$('drop').addEventListener('click', () => $('file').click());
$('drop').addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file').click(); } });
$('drop').addEventListener('dragover', (e) => { e.preventDefault(); $('drop').classList.add('over'); });
$('drop').addEventListener('dragleave', () => $('drop').classList.remove('over'));
$('drop').addEventListener('drop', (e) => { e.preventDefault(); $('drop').classList.remove('over'); openFile(e.dataTransfer.files[0]); });
$('file').addEventListener('change', (e) => openFile(e.target.files[0]));

for (const [id, label] of [['blemish', 'bval'], ['strength', 'sval'], ['texture', 'tval'], ['shine', 'rval'], ['pores', 'pval']]) {
  $(id).addEventListener('input', () => { $(label).textContent = $(id).value; scheduleRender(); });
}

$('faceonly').addEventListener('change', scheduleRender);
$('print').addEventListener('change', scheduleRender);
$('lighting').addEventListener('change', scheduleRender);

let exporting = null;      // the History entry being downloaded
$('download').addEventListener('click', () => {
  exporting = history.point;
  setControls(false);
  status(() => t('progress.full'), true);
  worker.postMessage({ type: 'export', params: params() });
});

$('reset').addEventListener('click', () => {
  if (!mayLeave()) return;
  hasPhoto = false;
  brush.reset();
  history.clear();
  steps.setStep(1);
  zoom.reset();
  $('drop').classList.remove('hidden');
  $('workspace').classList.add('hidden');
  $('reset').classList.add('hidden');
  $('file').value = '';
  setControls(false);
  status(readyText);
});

status(() => t('status.loading'), true);
// ?blemish=network uses the blemish network instead (non-commercial licence).
const blemishEngine = new URLSearchParams(location.search).get('blemish') === 'network' ? 'network' : 'classic';
worker.postMessage({ type: 'init', blemishEngine });
