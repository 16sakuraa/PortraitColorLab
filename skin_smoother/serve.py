"""
Local web UI for the skin smoother.

Starts a small web server (Python standard library only) and opens a browser
page where you can upload a photo, adjust Strength / Keep-texture, compare
before and after, and download the result. While you edit, the page shows a
preview-size copy (PREVIEW_MAX), so every slider move and brush stroke comes back
quickly; Download processes the full-resolution photo. All of it happens on this
computer; nothing leaves it.

Run:  python serve.py   (or double-click run.bat)
"""

from __future__ import annotations
import base64
import json
import os
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import smoother
import blemish
import heal
import mask_edit
import photo_io
import updater
from version import VERSION

MASK_IMAGE_MAX = 1600     # longest side of the mask picture sent to the page for the brush
# Longest side of the picture worked on while editing, as in the web app (worker.js):
# about 8 times fewer pixels than a DSLR photo, so a stroke takes a fraction of the time.
PREVIEW_MAX = 1600

# Reading a photo, finding its skin and its spots happen once per photo and Face only
# setting: moving a slider or painting a stroke reuses them.
_analysis_cache: dict = {}
_analysis_lock = threading.Lock()


def _analysed(image_url: str, face_only: bool, preview: bool = False):
    """(img, info, faces, mask, corr, spot count) for an uploaded photo. With preview,
    the photo scaled down to PREVIEW_MAX with its analysis to match (scaled_analysis):
    the faces, skin and spots are still found on the full-size photo, as the web app
    does, so the preview shows what the download will."""
    key = (hash(image_url), face_only)
    with _analysis_lock:
        if key not in _analysis_cache:
            _analysis_cache.clear()
            img, info = photo_io.decode(_decode_data_url(image_url))
            if img is None:
                raise ValueError("could not decode image")
            faces = smoother._detect_faces(img)
            mask = smoother.build_skin_mask(img, faces=faces, face_only=face_only)
            corr, count = blemish.correction(img, faces, mask) if blemish.have_model() else (None, 0)
            _analysis_cache[key] = (img, info, faces, mask, corr, count)
        if not preview:
            return _analysis_cache[key]
        if (key, "preview") not in _analysis_cache:
            _analysis_cache[(key, "preview")] = scaled_analysis(_analysis_cache[key], PREVIEW_MAX)
        return _analysis_cache[(key, "preview")]


def scaled_analysis(analysed, longest: int):
    """_analysed's (img, info, faces, mask, corr, spot count) for the photo scaled down to
    at most `longest` pixels on its long side, without finding anything again: the
    photo, mask and spot fills shrunk (INTER_AREA), the face rows' positions scaled and
    their scores kept. The fills are cut to the smaller photo. Like scaleAnalysis in the
    web app's pipeline.js. The same tuple when the photo is already that small."""
    img, info, faces, mask, corr, count = analysed
    h, w = img.shape[:2]
    s = min(1.0, longest / max(h, w))
    if s >= 1.0:
        return analysed
    pw, ph = max(1, round(w * s)), max(1, round(h * s))
    small = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_AREA)
    if faces is not None:
        faces = np.array(faces, np.float32)
        faces[:, :14] *= s                  # box and landmarks; the last value is the score
    if corr is not None:
        x0, y0, values = corr
        fills = cv2.resize(values, (max(1, round(values.shape[1] * s)), max(1, round(values.shape[0] * s))),
                           interpolation=cv2.INTER_AREA)
        x0, y0 = min(pw - 1, round(x0 * s)), min(ph - 1, round(y0 * s))
        corr = x0, y0, fills[:ph - y0, :pw - x0]
    return small, info, faces, mask, corr, count


def _picture(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _mask_image(mask: np.ndarray) -> str:
    """A mask as a small greyscale PNG data URL."""
    h, w = mask.shape[:2]
    s = min(1.0, MASK_IMAGE_MAX / max(h, w))
    if s < 1.0:
        mask = cv2.resize(mask, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


WEB_SCRIPTS = ("mask_brush.js", "i18n.js", "compare_zoom.js", "history.js")   # the web app's scripts this page uses


def _web_script(name: str) -> bytes:
    """One of WEB_SCRIPTS: bundled inside the .exe, or from the web app beside this folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, "..", "web", "js", name)]
    if getattr(sys, "frozen", False):
        candidates.insert(0, os.path.join(sys._MEIPASS, "web", name))
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return fh.read()
    raise FileNotFoundError(name)


PAGE ="""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Skin Smoother</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><defs><linearGradient id='g' x2='0' y2='1'><stop offset='0' stop-color='%236a95f7'/><stop offset='1' stop-color='%233b63d6'/></linearGradient></defs><rect x='3' y='3' width='94' height='94' rx='22' fill='url(%23g)'/><g fill='%23fff'><circle cx='45' cy='38.5' r='15.5'/><path d='M15 83a30 24 0 0 1 60 0z'/></g><path fill='%23ffe79e' d='M77.5 11Q79.1 22.9 91 24.5Q79.1 26.1 77.5 38Q75.9 26.1 64 24.5Q75.9 22.9 77.5 11z'/></svg>">
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 system-ui, Segoe UI, Arial, sans-serif;
         background: #f4f5f7; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) {
    body { background: #16181c; color: #e8e8e8; }
    .panel, header, .tools { background: #22252b !important; }
    .drop { border-color: #3a3f47 !important; }
    .tools section + section { border-top-color: #ffffff1f !important; }
  }
  header { background: #fff; padding: 14px 20px; border-bottom: 1px solid #0001;
           display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  header h1 { font-size: 17px; margin: 0; font-weight: 650; }
  .actions { display: flex; gap: 10px; margin-left: auto; }
  .ctl { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
  .ctl label { font-size: 12px; opacity: .75; display: flex; justify-content: space-between; }
  .check { display: flex; align-items: center; gap: 6px; font-size: 12px; opacity: .75; cursor: pointer; }
  input[type=range] { width: 100%; margin: 4px 0; accent-color: #4b7bec; }
  select { width: 100%; font: inherit; font-size: 13px; padding: 3px 4px; border-radius: 6px; }
  button { font: inherit; padding: 8px 16px; border: 0; border-radius: 8px;
           background: #4b7bec; color: #fff; cursor: pointer; }
  button.secondary { background: #6b7280; }
  button:disabled { opacity: .45; cursor: default; }
  /* Photos on the left, the settings in a panel on the right that stays in view. */
  .layout { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 18px; align-items: start;
            padding: 18px; }
  main { min-width: 0; }
  .tools { background: #fff; border-radius: 12px; padding: 16px; position: sticky; top: 12px;
           max-height: calc(100vh - 24px); overflow-y: auto; display: grid; gap: 16px; }
  .tools section { display: grid; gap: 12px; align-content: start; }
  .tools section + section { border-top: 1px solid #0000001a; padding-top: 16px; }
  .tools h2 { font-size: 13px; font-weight: 650; margin: 0; }
  .tools .note { margin: 0; font-size: 12px; opacity: .7; }
  .tools button { justify-self: start; }
  /* While painting, the panel holds only the brush. */
  body.mask-editing .tools > section:not(#brushbox):not(#historybox) { display: none; }
  body.mask-editing .tools > #brushbox { border-top: 0 !important; padding-top: 0; }
  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; }
    .tools { order: -1; position: static; max-height: none; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }
    .tools section + section { border-top: 0; padding-top: 0; }
  }
  .drop { border: 2px dashed #c7ccd4; border-radius: 12px; padding: 40px;
          text-align: center; background: #fff8; cursor: pointer; }
  .drop.small { padding: 14px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .panel { background: #fff; border-radius: 12px; padding: 10px; }
  .panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
              opacity: .6; margin: 4px 6px 10px; }
  .panel img { width: 100%; border-radius: 8px; display: block; }
  .hint { font-size: 12px; opacity: .6; margin: 8px 2px 0; }
  @media (hover: none) { .hint { display: none; } }
  .status { font-size: 13px; opacity: .8; margin-top: 10px; min-height: 18px; }
  .busy::after { content: ""; display: inline-block; width: 10px; height: 10px; margin-left: 8px; border-radius: 50%;
                 border: 2px solid currentColor; border-top-color: transparent; animation: spin .8s linear infinite;
                 vertical-align: -1px; }
  /* A new photo on its way: shown dimmed under a spinner, never the last photo's retouch. */
  .loading { position: absolute; inset: 0; z-index: 3; display: flex; flex-direction: column; align-items: center;
             justify-content: center; gap: 12px; padding: 16px; text-align: center; font-size: 14px;
             background: #000000a0; color: #fff; }
  .loading::before { content: ""; width: 30px; height: 30px; border-radius: 50%; border: 3px solid #fff;
                     border-top-color: transparent; animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .version { font-size: 12px; opacity: .55; }
  .update { padding: 9px 20px; background: #fff4cc; color: #4d3b00; font-size: 13px; }
  @media (prefers-color-scheme: dark) { .update { background: #3d3415; color: #f3e3a6; } }
  .hidden { display: none; }
  @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>Skin Smoother</h1>
  <span class="version" id="version">version __VERSION__</span>
  <div class="actions" id="actions">
    <button id="reset" class="secondary hidden" data-i18n="page.newPhoto">New photo</button>
    <button id="download" disabled data-i18n="page.download">Download</button>
  </div>
</header>
<div id="update" class="update hidden"></div>

<div class="layout">
<main>
  <div id="drop" class="drop">
    <p data-i18n-html="page.dropHtml"><strong>Click to choose a photo</strong> or drag it here</p>
    <p style="opacity:.6;font-size:13px" data-i18n="page.dropNoteExe">JPG, PNG, BMP, WEBP — stays on your computer</p>
    <input id="file" type="file" accept="image/*" class="hidden">
  </div>

  <div id="workspace" class="hidden">
    <div class="grid">
      <div class="panel"><h2 data-i18n="page.original">Original</h2><img id="before"></div>
      <div class="panel"><h2 data-i18n="page.retouched">Retouched</h2><img id="after"></div>
    </div>
    <p class="hint" data-i18n="page.zoomHint">Scroll on either photo to zoom both into the same spot · drag to move around · double-click to see the whole photo</p>
    <div class="status" id="status"></div>
  </div>
</main>

<aside class="tools" aria-label="Retouch settings" data-i18n-aria="page.settingsAria">
  <section data-step="1">
    <h2 data-i18n="panel.skinPhoto">Skin and photo</h2>
    <div class="ctl">
      <label for="lighting" data-i18n="ctl.lighting">Lighting</label>
      <select id="lighting" data-i18n-title="ctl.lightingTitle" title="Hard or patchy light (on-camera flash, a bare window) leaves mottled light and shadow; this evens it out more and keeps the face's shading. It won't blend a sharp shadow edge.">
        <option value="soft" selected data-i18n="ctl.lightingSoft">Soft (studio)</option>
        <option value="hard" data-i18n="ctl.lightingHard">Hard or patchy (flash)</option>
      </select>
    </div>
    <div class="ctl">
      <label for="print" data-i18n="ctl.print">Print size</label>
      <select id="print" data-i18n-title="ctl.printTitle" title="A small print keeps coarser grain, so the skin still has texture at that size. It looks coarser here.">
        <option value="large" selected data-i18n="ctl.printLarge">Large print or screen</option>
        <option value="small" data-i18n="ctl.printSmall">Small print (1-inch ID)</option>
      </select>
    </div>
    <label class="check"><input id="faceonly" type="checkbox"> <span data-i18n="ctl.faceOnly">Face only</span></label>
    <p class="note" data-i18n="note.step1">Tinted skin is what gets smoothed. Paint the skin area to add skin the mask missed, or take out what shouldn't be smoothed.</p>
  </section>
  <section data-step="2">
    <h2 data-i18n="panel.smoothing">Smoothing</h2>
    <div class="ctl">
      <label><span data-i18n="ctl.blemish">Remove blemishes</span> <span id="bval">100</span></label>
      <input id="blemish" type="range" min="0" max="100" value="100">
    </div>
    <div class="ctl">
      <label><span data-i18n="ctl.strength">Strength</span> <span id="sval">60</span></label>
      <input id="strength" type="range" min="0" max="100" value="60">
    </div>
    <div class="ctl">
      <label><span data-i18n="ctl.texture">Keep texture</span> <span id="tval">60</span></label>
      <input id="texture" type="range" min="0" max="100" value="60">
    </div>
    <div class="ctl">
      <label data-i18n-title="ctl.shineTitle" title="Tames the bright glints of oily skin. Dark pores, freckles and coloured marks are left alone."><span data-i18n="ctl.shine">Reduce shine</span> <span id="rval">0</span></label>
      <input id="shine" type="range" min="0" max="100" value="0">
    </div>
    <div class="ctl">
      <label><span data-i18n="ctl.pores">Add pores</span> <span id="pval">0</span></label>
      <input id="pores" type="range" min="0" max="100" value="0">
    </div>
    <p class="note" data-i18n="note.step2">The light brush evens out blotches, dark circles and seams between light and shadow with a colour you pick from the skin, Nose line paints a soft highlight down the nose, and Smooth again smooths leftover coarse grain or stubble shadow once more.</p>
  </section>
  <section data-step="3">
    <h2 data-i18n="panel.finishing">Finishing</h2>
    <p class="note" data-i18n="note.step3">Bring back sharpness around the eyes, nostrils and jawline with Undo smoothing, cover scars with Heal, and click leftover spots away with Remove spot. Then press Download.</p>
  </section>
  <section id="brushbox">
    <h2 data-i18n="panel.brush">Brush</h2>
    <button id="editmask" class="secondary" disabled>Use brush</button>
  </section>
  <section id="historybox">
    <h2 data-i18n="panel.history">History</h2>
  </section>
</aside>
</div>

<script type="module">
import { attachMaskBrush, attachSteps } from '/mask_brush.js';
import { t, translatePage, attachLangSwitch, onLangChange } from '/i18n.js';
import { attachCompareZoom } from '/compare_zoom.js';
import { attachHistory } from '/history.js';

const $ = id => document.getElementById(id);
// The page in the language last chosen (or the browser's), with a TH | EN switch.
translatePage();
attachLangSwitch($('actions'));
let original = null;      // data URL of the source image
let originalName = 'photo';
let busy = false, pending = false, pendingFull = false, modelMissing = false;
let needBefore = false;   // the Original at preview size, asked for once per photo
const brush = attachMaskBrush({
  image: $('after'),
  button: $('editmask'),
  onChange: () => runProcess(),
  onModeChange: (editing) => { if (editing) runProcess(); },
  toolsHost: $('brushbox'),
  labels: [t('brush.use'), t('brush.done')],
});
// The settings and brush tools in three steps, like DM Portrait's: Skin, Smooth, Finish.
const steps = attachSteps({ panel: document.querySelector('.tools'), brush, navBefore: $('historybox') });
// Every change listed, to go back to any of them, with Undo and Redo.
const history = attachHistory({ host: $('historybox'), brush, render: () => runProcess() });
// Scroll on either photo to zoom both into the same spot.
const zoom = attachCompareZoom([$('before'), $('after')]);

// While a new photo's first result is on its way, the Retouched side shows that photo
// dimmed under a spinner and these words (an i18n key), not the last photo's retouch.
const loading = Object.assign(document.createElement('div'), { className: 'loading hidden' });
$('after').closest('.mask-clip').append(loading);
function setLoading(key){
  loading.classList.toggle('hidden', !key);
  if(key){ loading.dataset.i18n = key; loading.textContent = t(key); }
}

// text: a function giving the words, so a language switch can say it again.
// spinning: a spinner after them, for the longer waits.
let lastStatus = null;
function setStatus(text, spinning = false){
  lastStatus = { text, spinning };
  $('status').textContent = text();
  $('status').classList.toggle('busy', spinning);
}

// The program checks for a newer version when it starts (updater.py); show what it finds.
let update = { version: '__VERSION__', state: 'off' };
function showUpdate(){
  $('version').textContent = t('page.version', { v: update.version });
  const key = { downloading: 'update.downloading', ready: 'update.ready', failed: 'update.failed' }[update.state];
  const text = key && update.latest ? t(key, { v: update.latest, reason: update.reason || '' }) : (update.message || '');
  $('update').textContent = text;
  $('update').classList.toggle('hidden', !text);
}
showUpdate();
async function watchUpdate(){
  try {
    update = await (await fetch('/update')).json();
    showUpdate();
    if (update.state === 'checking' || update.state === 'downloading') setTimeout(watchUpdate, 3000);
  } catch(err) { /* the program has been closed */ }
}
watchUpdate();

onLangChange(() => {
  showUpdate();
  if (lastStatus) setStatus(lastStatus.text, lastStatus.spinning);
  if (modelMissing) $('bval').textContent = t('status.modelMissing');
});

$('drop').addEventListener('click', () => $('file').click());
$('drop').addEventListener('dragover', e => { e.preventDefault(); $('drop').style.opacity=.7; });
$('drop').addEventListener('dragleave', () => $('drop').style.opacity=1);
$('drop').addEventListener('drop', e => {
  e.preventDefault(); $('drop').style.opacity=1;
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});
$('file').addEventListener('change', e => { if(e.target.files[0]) loadFile(e.target.files[0]); });

// Leaving a photo whose current state hasn't been downloaded asks first.
const mayLeave = () => !original || !history.unsaved || confirm(t('confirm.newPhoto'));

function loadFile(f){
  if (!mayLeave()) return;
  const r = new FileReader();
  r.onload = () => {
    original = r.result;
    needBefore = true;
    originalName = f.name.replace(/\\.[^.]+$/, '') || 'photo';
    brush.reset();
    steps.setStep(1);
    zoom.reset();
    $('editmask').disabled = true;
    $('before').src = original;
    $('after').src = original;          // under the loading cover until its retouch arrives
    setLoading('progress.finding');
    setStatus(() => t('progress.finding'), true);
    history.start();
    $('drop').classList.add('hidden');
    $('workspace').classList.remove('hidden');
    $('reset').classList.remove('hidden');
    process();
  };
  r.readAsDataURL(f);
}

$('reset').addEventListener('click', () => {
  if (!mayLeave()) return;
  original = null;
  setLoading(null);
  brush.reset();
  history.clear();
  steps.setStep(1);
  zoom.reset();
  $('drop').classList.remove('hidden');
  $('workspace').classList.add('hidden');
  $('reset').classList.add('hidden');
  $('download').disabled = true;
  $('editmask').disabled = true;
  $('file').value = '';
});

const valueLabel = {strength: 'sval', texture: 'tval', shine: 'rval', pores: 'pval', blemish: 'bval'};
['strength','texture','shine','pores','blemish'].forEach(id => {
  $(id).addEventListener('input', () => {
    $(valueLabel[id]).textContent = $(id).value;
    process();
  });
});
$('faceonly').addEventListener('change', process);
$('print').addEventListener('change', process);
$('lighting').addEventListener('change', process);

let debounce = null;
function process(){
  if(!original) return;
  clearTimeout(debounce);
  debounce = setTimeout(runProcess, 180);
}

// The status line after a result: faces, spots removed, and the photo's size.
function summary(data){
  const parts = [(data.faces === 1 ? t('summary.faces1') : t('summary.faces', { n: data.faces }))
                 + (data.model ? '' : t('status.noFaceModel'))];
  if (data.blemish_count) {
    const unit = data.blemish_unit === 'faces' ? 'summary.healed' : 'summary.spots';
    parts.push(data.blemish_count === 1 ? t(unit + '1') : t(unit, { n: data.blemish_count }));
  }
  parts.push(data.w + '×' + data.h);
  return parts.join('  ·  ');
}

// While editing, the photo is worked on at preview size (the server's PREVIEW_MAX), so
// each slider move and stroke comes back quickly. Download (full) processes the whole
// photo and saves it.
async function runProcess(full = false){
  if(!original) return;
  if(busy){ if(full) pendingFull = true; else pending = true; return; }
  busy = true;
  const wantBefore = needBefore && !full;     // the first result for this photo: it's found first
  setStatus(() => full ? t('progress.full') : wantBefore ? t('progress.finding') : t('status.processing'),
            full || wantBefore);
  $('download').disabled = true;
  const source = original, sent = brush.strokes, upTo = sent.length ? sent[sent.length - 1].id : 0;
  const point = history.point;      // the History entry this result shows
  // When it doesn't work: Download can be tried again (the preview still stands), and a
  // new photo that couldn't be retouched shows nothing under Retouched.
  const failed = () => {
    if(full) $('download').disabled = false;
    else if(!loading.classList.contains('hidden')){ setLoading(null); $('after').removeAttribute('src'); }
  };
  try {
    const res = await fetch('/process', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        image: source,
        strength: +$('strength').value / 100,
        texture: +$('texture').value / 100,
        shine: +$('shine').value / 100,
        pores: +$('pores').value / 100,
        print: $('print').value,
        lighting: $('lighting').value,
        face_only: $('faceonly').checked,
        blemish: +$('blemish').value / 100,
        mask_strokes: sent,
        preview: !full,
        want_mask: !full && brush.editing,
        want_before: wantBefore
      })
    });
    const data = await res.json();
    if(source !== original){ /* another photo was opened meanwhile: nothing to show */ }
    else if(data.error){
      failed();
      setStatus(() => t('status.error', { message: data.error }));
    }
    else if(full){
      const a = document.createElement('a');
      a.href = data.image; a.download = originalName + '_retouched' + (data.ext || '.jpg'); a.click();
      history.markSaved(point);       // closing the tab now loses nothing
      $('download').disabled = false;
      setStatus(() => summary(data));
    }
    else {
      $('after').src = data.image;
      setLoading(null);
      if(data.before_image){ $('before').src = data.before_image; needBefore = false; }
      if(data.mask_image){
        const picture = new Image();
        picture.onload = () => brush.setMaskImage(picture, upTo);
        picture.src = data.mask_image;
      }
      $('download').disabled = false;
      $('editmask').disabled = false;
      if(!data.blemish_model){
        modelMissing = true;
        $('blemish').disabled = true; $('bval').textContent = t('status.modelMissing');
      }
      setStatus(() => summary(data));
    }
  } catch(err){
    if(source === original){
      failed();
      setStatus(() => t('status.error', { message: String(err) }));
    }
  }
  busy = false;
  if(pendingFull){ pendingFull = false; runProcess(true); }    // a waiting edit runs after it
  else if(pending){ pending = false; runProcess(); }
}
$('download').addEventListener('click', () => runProcess(true));
</script>
</body>
</html>
"""


def _decode_data_url(data_url: str) -> bytes:
    """The uploaded file's original bytes (colour profile, EXIF etc. intact)."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.replace("__VERSION__", VERSION).encode("utf-8"), "text/html; charset=utf-8")
        elif self.path.lstrip("/") in WEB_SCRIPTS:
            self._send(200, _web_script(self.path.lstrip("/")), "text/javascript; charset=utf-8")
        elif self.path == "/update":
            self._send(200, json.dumps(updater.status()).encode("utf-8"), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/process":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            strength = float(payload.get("strength", 0.6))
            texture = float(payload.get("texture", 0.6))
            pores = float(payload.get("pores", 0.0))
            grain = smoother.PRINT_GRAIN.get(payload.get("print"), 1.0)
            shine = float(payload.get("shine", 0.0))
            light = smoother.LIGHTING.get(payload.get("lighting"), 1.0)
            face_only = bool(payload.get("face_only", False))
            blemish_amt = float(payload.get("blemish", 0.0))
            # The page asks for a preview while editing and the full size to download.
            preview = bool(payload.get("preview", False))

            full_h, full_w = _analysed(payload["image"], face_only)[0].shape[:2]
            img, info, faces, mask, corr, spots = _analysed(payload["image"], face_only, preview)
            n_faces = 0 if faces is None else len(faces)

            # The brush's strokes change the mask, and take restored areas out of the fills.
            strokes = payload.get("mask_strokes") or []
            again = None
            if strokes:
                region, fills, again = mask_edit.edits(strokes, img.shape[1], img.shape[0])
                mask = mask_edit.apply_to_mask(mask, region)
                corr = mask_edit.apply_to_correction(corr, fills)

            # Heal blemishes first, then the healing brush's strokes, then smooth (the
            # usual retouching order).
            healed, blemish_count = img, 0
            if blemish_amt > 0 and blemish.have_model():
                blemish_count = spots
                healed = blemish.apply(img, corr, blemish_amt)
            healed = heal.apply(healed, strokes)
            out = smoother.smooth_face(healed, strength=strength, texture=texture,
                                       mask=mask, faces=faces, pores=pores, grain=grain, shine=shine,
                                       light=light)
            # Smooth again: once more, with finer grain, where it was painted.
            if again is not None:
                out = smoother.smooth_again(healed, out, again, mask, faces, texture, pores, grain, shine, light)
            # The light brush paints tone on the smoothed result, keeping the grain it kept.
            out = heal.apply_light(out, strokes, smoother.detail_sigma(img.shape, faces, grain))

            # Save like the original: same colour profile, EXIF, DPI and JPEG
            # quality, so colours and file size match what was uploaded. A preview
            # only has to show right on screen.
            fmt = photo_io.output_format(info)
            mime, ext = ("image/png", ".png") if fmt == "PNG" else ("image/jpeg", ".jpg")
            if preview:
                image = _picture(photo_io.encode_preview(out, info), "image/jpeg")
            else:
                image = _picture(photo_io.encode(out, info, fmt), mime)
            resp = {
                "image": image,
                "ext": ext,
                "faces": n_faces,
                "model": smoother.have_face_model(),
                "blemish_model": blemish.have_model(),
                "blemish_text": blemish.count_text(blemish_count) if blemish_count else "",
                # for the page to say in its own language: spots removed, or faces treated
                "blemish_count": blemish_count,
                "blemish_unit": "spots" if blemish.ENGINE == "classic" else "faces",
                "w": full_w, "h": full_h,       # the photo's own size, for the status line
            }
            if payload.get("want_mask"):
                resp["mask_image"] = _mask_image(mask)
            if preview and payload.get("want_before"):
                # The Original beside it at the same size, so zooming in compares like with like.
                resp["before_image"] = _picture(photo_io.encode_preview(img, info), "image/jpeg")
            self._send(200, json.dumps(resp).encode("utf-8"), "application/json")
        except Exception as e:
            self._send(200, json.dumps({"error": str(e)}).encode("utf-8"),
                       "application/json")


def _free_port(start: int = 8000, tries: int = 20) -> int:
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def main():
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("=" * 48)
    print(f"  Skin Smoother {VERSION} is running.")
    print(f"  Open:  {url}")
    print("  Close this window to stop.")
    if not smoother.have_face_model():
        print("  (Tip: run get_model.py for face-aware masking.)")
    print("=" * 48, flush=True)
    updater.start()
    if not os.environ.get("SKINSMOOTHER_NO_BROWSER"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
