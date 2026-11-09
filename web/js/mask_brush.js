// The brush for editing the retouch by hand, and the retouch's steps: the same interface
// on the web page and on the desktop app's page, so it imports only its words
// (i18n.js, in English and Thai) and brings its own styles.
//
// Pressing the page's brush button tints the treated skin over the retouched photo,
// in a larger view that zooms and pans. Restore (a stroke's 'protect' mode) brings
// back the original, Undo smoothing ('unsmooth') brings back its sharpness but keeps
// the spot fills, heals and light brush, and Add skin smooths more, each at the
// Strength set. Strokes (see mask_edit.js for their form) are shown straight away on
// the tint, then the page asks for a new result and hands back the real mask with
// setMaskBytes or setMaskImage.
// Heal, Remove spot and Light change the photo instead (heal.js): Alt-click picks where
// Heal copies from, or the colour Light paints with, and the strokes carry it. These
// show as a patch until a result that includes them arrives (the page passes the last
// stroke's id).
// attachSteps splits the panel into steps, like DM Portrait's: each step shows its own
// settings and brush tools (setModes). Steps are only views, so going back loses nothing.

import { t, has, translatePage, onLangChange } from './i18n.js';

const TINT = [25, 211, 197], TINT_ALPHA = 0.42, TINT_CSS = `rgba(${TINT.join(',')},${TINT_ALPHA})`;
const MAX_ZOOM = 10, MIN_VIEW = 320, VIEW_MARGIN = 28;   // px; the view fills the window under the tools

const STYLE = `
  .mask-clip { position: relative; }
  .mask-stage { position: relative; transform-origin: 0 0; }
  .mask-canvas, .mask-cursor, .mask-tools { display: none; }
  body.mask-editing .grid { grid-template-columns: 1fr; }
  body.mask-editing .grid > .panel:first-child, body.mask-editing .hint { display: none; }
  body.mask-editing .mask-clip { overflow: hidden; border-radius: 8px; background: #0003; touch-action: none; user-select: none; }
  body.mask-editing .mask-stage { position: absolute; left: 0; top: 0; }
  body.mask-editing .mask-stage img { width: 100%; height: 100%; border-radius: 0; pointer-events: none; }
  body.mask-editing .mask-canvas { display: block; position: absolute; inset: 0; width: 100%; height: 100%; cursor: none; touch-action: none; }
  body.mask-editing .mask-canvas.hide-tint { opacity: 0; }
  body.mask-editing.mask-panning .mask-canvas { cursor: grab; }
  .mask-cursor { position: absolute; left: 0; top: 0; border-radius: 50%; pointer-events: none; z-index: 2;
                 border: 1.5px solid #fff; box-shadow: 0 0 0 1px #000a, inset 0 0 0 1px #000a; }
  body.mask-editing .mask-cursor.on { display: block; }
  body.mask-editing .mask-tools { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; margin: 0 2px 10px; font-size: 12px; }
  .mask-tools label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
  .mask-tools input[type=range] { width: 120px; }
  .mask-tools button { padding: 4px 10px; font-size: 12px; }
  .mask-row { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  body.mask-editing .mask-tools.hosted { display: grid; gap: 10px; margin: 0; }
  .mask-tools.hosted label { justify-content: space-between; }
  .mask-tools.hosted label.mask-check { justify-content: flex-start; }
  .mask-tools.hosted input[type=range] { width: auto; flex: 1; min-width: 0; }
  .mask-mode { display: inline-flex; border-radius: 8px; overflow: hidden; justify-self: start; }
  .mask-mode button { border-radius: 0; background: #6b7280; }
  .mask-mode button.on { background: #4b7bec; }
  .mask-hint { flex-basis: 100%; opacity: .7; }
  .mask-hint.warn { opacity: 1; color: #d97706; }
  .mask-tools:not([data-mode="heal"]):not([data-mode="light"]):not([data-mode="line"]) .mask-point-only { display: none; }
  .mask-tools:not([data-mode="heal"]) .mask-heal-only { display: none; }
  .mask-tools:not([data-mode="light"]):not([data-mode="line"]) .mask-light-only { display: none; }
  .mask-tools[data-mode="heal"] .mask-tint-only, .mask-tools[data-mode="spot"] .mask-tint-only,
  .mask-tools[data-mode="light"] .mask-tint-only, .mask-tools[data-mode="line"] .mask-tint-only,
  .mask-tools[data-mode="resmooth"] .mask-tint-only { display: none; }
  .mask-tools button.secondary.on { background: #4b7bec; }
  .mask-swatch { width: 18px; height: 18px; border-radius: 50%; border: 1.5px solid #fff; box-shadow: 0 0 0 1px #000a;
                 background: repeating-conic-gradient(#999 0 25%, #ddd 0 50%) 50% / 8px 8px; }
  body.mask-editing.mask-picking .mask-canvas { cursor: crosshair; }
  .mask-source { position: absolute; left: 0; top: 0; width: 16px; height: 16px; margin: -8px 0 0 -8px; display: none;
                 border-radius: 50%; pointer-events: none; z-index: 2; border: 1.5px solid #fff; box-shadow: 0 0 0 1px #000a; }
  .mask-source::before, .mask-source::after { content: ""; position: absolute; background: #fff; box-shadow: 0 0 0 1px #000a; }
  .mask-source::before { left: 50%; top: -5px; bottom: -5px; width: 1.5px; margin-left: -.75px; }
  .mask-source::after { top: 50%; left: -5px; right: -5px; height: 1.5px; margin-top: -.75px; }
  body.mask-editing .mask-source.on { display: block; }
  .step-tabs { display: flex; border-radius: 8px; overflow: hidden; grid-column: 1 / -1; }
  .step-tabs button { flex: 1; border-radius: 0; background: #6b7280; padding: 7px 4px; font-size: 13px; }
  .step-tabs button.on { background: #4b7bec; font-weight: 600; }
  .step-nav { display: flex; gap: 8px; grid-column: 1 / -1; }
  .step-nav button:last-child { margin-left: auto; }
  [data-step] { border-top: 0 !important; padding-top: 0 !important; }
  .step-off { display: none !important; }
`;
const PATCH_CSS = 'rgba(255,255,255,0.4)';   // a heal or spot stroke waiting for its result
const IMAGE_MODES = ['heal', 'spot', 'light', 'line', 'resmooth'];   // tools that hide the tint
const PATCH_STROKES = ['heal', 'spot', 'light', 'resmooth'];         // strokes shown as a patch until their result
const MASK_MODES = ['protect', 'unsmooth', 'skin'];
// The modes: which take a point (Set source, Pick colour) and which paint the picked
// colour. Their words are in i18n.js, under mode.<name>.label, .title (the button's
// tooltip), .strength (the Strength slider's), .hint, and .point, .pointTitle, .noPoint
// and .next where they have those.
const MODES = {
  protect: {}, unsmooth: {}, skin: {}, heal: { point: true }, spot: {},
  light: { point: true, colour: true }, line: { point: true, colour: true }, resmooth: {},
};
const DEFAULT_AMOUNT = { light: 20, line: 25 };   // Strength (%) each mode starts at; the rest 100
const DEFAULT_SOFT = { light: 100, line: 100 };   // Softness (%) each mode starts at; the rest 30

// The retouch's steps, in DM Portrait's order. Their words are in i18n.js under
// <key>.title, .button and .about; words: a step's own words for a mode (an i18n.js
// key prefix used instead of mode.<name>).
export const STEPS = [
  { key: 'step1', modes: ['skin', 'protect'], words: { protect: 'mode.removeSkin' } },
  { key: 'step2', modes: ['light', 'line', 'resmooth'] },
  { key: 'step3', modes: ['unsmooth', 'protect', 'heal', 'spot'] },
];

const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};
// Marks a node's text and tooltip with i18n.js keys, for translatePage.
const worded = (node, text, title = null) => {
  if (text) node.dataset.i18n = text;
  if (title) node.dataset.i18nTitle = title;
  return node;
};

// Brush size slider 1..100 to a share of the long side, finer at the small end.
const sizeOf = (v) => 0.002 + 0.1 * (v / 100) ** 2;
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

function addStyle() {
  if (!document.getElementById('mask-brush-style')) {
    document.head.append(el('style', { id: 'mask-brush-style', textContent: STYLE }));
  }
}

// toolsHost: where the tools go (a side panel); without it they sit above the photo.
// labels: the button's text when off and when painting.
export function attachMaskBrush({ image, button, onChange, onModeChange = () => {}, toolsHost = null,
                                  labels = ['Edit mask', 'Done editing'] }) {
  addStyle();
  const clip = el('div', { className: 'mask-clip' }), stage = el('div', { className: 'mask-stage' });
  const canvas = el('canvas', { className: 'mask-canvas' }), cursor = el('div', { className: 'mask-cursor' });
  const sourceMark = el('div', { className: 'mask-source' });
  image.parentElement.insertBefore(clip, image);
  stage.append(image, canvas);
  clip.append(stage, cursor, sourceMark);
  const ctx = canvas.getContext('2d');
  const base = document.createElement('canvas');      // the mask as last received

  // ---------- tools ----------
  const modeButtons = {};
  for (const m of Object.keys(MODES)) modeButtons[m] = el('button', { type: 'button' });
  const maskGroup = el('span', { className: 'mask-mode' }), imageGroup = el('span', { className: 'mask-mode' });
  const pointBtn = el('button', { type: 'button', className: 'secondary' });
  const alignedInput = worded(el('input', { type: 'checkbox', checked: true }), null, 'brush.alignedTitle');
  const swatchEl = worded(el('span', { className: 'mask-swatch mask-light-only' }), null, 'brush.swatch');
  const amountInput = el('input', { type: 'range', min: 5, max: 100, step: 5, value: 100 });
  const amountValue = el('span', { textContent: '100%' });
  const sizeInput = worded(el('input', { type: 'range', min: 1, max: 100, value: 24 }), null, 'brush.sizeTitle');
  const softInput = worded(el('input', { type: 'range', min: 0, max: 100, value: 30 }), null, 'brush.softTitle');
  const undoBtn = worded(el('button', { type: 'button', className: 'secondary', disabled: true }), 'brush.undo', 'brush.undoTitle');
  const resetBtn = worded(el('button', { type: 'button', className: 'secondary', disabled: true }), 'brush.reset', 'brush.resetTitle');
  const fitBtn = worded(el('button', { type: 'button', className: 'secondary' }), 'brush.fit', 'brush.fitTitle');
  const tintInput = worded(el('input', { type: 'checkbox', checked: true }), null, 'brush.tintTitle');
  const hint = el('span', { className: 'mask-hint' });
  const word = (key) => worded(el('span'), key);
  const tools = el('div', { className: 'mask-tools' },
    maskGroup, imageGroup,
    el('span', { className: 'mask-row mask-point-only' }, pointBtn, swatchEl,
      el('label', { className: 'mask-check mask-heal-only' }, alignedInput, word('brush.aligned'))),
    el('label', {}, word('brush.strength'), amountInput, amountValue),
    el('label', {}, word('brush.size'), sizeInput),
    el('label', {}, word('brush.softness'), softInput),
    el('span', { className: 'mask-row' }, undoBtn, resetBtn, fitBtn),
    el('label', { className: 'mask-check mask-tint-only' }, tintInput, word('brush.tint')),
    hint);
  translatePage(tools);
  if (toolsHost) {
    tools.classList.add('hosted');
    toolsHost.append(tools);
  } else {
    clip.parentElement.insertBefore(tools, clip);
  }
  button.textContent = labels[0];

  // ---------- state ----------
  let editing = false, mode = 'protect', strokes = [], nextId = 1, version = 0;
  let current = null, panning = null, spaceDown = false;
  let fitW = 0, fitH = 0, clipW = 0, zoom = 1, tx = 0, ty = 0;
  // Heal: the point picked to copy from and the offset kept while Aligned. Light and Nose
  // line: the point their colour comes from, and that colour as seen on screen (for show
  // only). Nose line: the line after its first click. Whether the next click picks the
  // point, and where the pointer is (all 0..1 of the photo).
  let source = null, alignedOffset = null, pick = null, swatch = null, lineDraft = null;
  let picking = false, altDown = false, hover = null;
  let awaiting = [];                  // strokes not yet in a result, shown as a patch
  // Who hears of each stroke and Reset (the History panel), its undo if it has one,
  // and the words for the stroke being painted (an i18n.js key).
  const editListeners = new Set();
  let externalUndo = null, canUndo = false, currentLabel = null;
  const swatches = new Map();         // light stroke id -> its colour on screen
  let visible = Object.keys(MODES), names = {};
  const amounts = {}, softs = {};     // the Strength and Softness last used in each mode

  const imageMode = () => IMAGE_MODES.includes(mode);
  // A mode's words in the current language: the step's own (names), else the mode's.
  const info = (m) => {
    const word = (field) => {
      const own = names[m] && `${names[m]}.${field}`;
      return t(own && has(own) ? own : `mode.${m}.${field}`);
    };
    return {
      ...MODES[m], label: word('label'), title: word('title'), strength: word('strength'), hint: word('hint'),
      point: MODES[m].point ? word('point') : null, pointTitle: word('pointTitle'), noPoint: word('noPoint'), next: word('next'),
    };
  };
  const setHint = (text = null, warn = true) => {
    hint.textContent = (text || info(mode).hint) + t('brush.move');
    hint.classList.toggle('warn', !!text && warn);
  };
  const dropLine = () => {
    if (!lineDraft) return;
    if (current === lineDraft) current = null;
    lineDraft = null;
    drawTint();
  };
  const setPicking = (on) => {
    picking = on;
    pointBtn.classList.toggle('on', on);
    document.body.classList.toggle('mask-picking', (on || altDown) && !!MODES[mode].point);
  };
  const setMode = (m) => {
    dropLine();
    amounts[mode] = +amountInput.value;
    softs[mode] = +softInput.value;
    mode = m;
    softInput.value = softs[m] ?? DEFAULT_SOFT[m] ?? 30;
    for (const [name, b] of Object.entries(modeButtons)) b.classList.toggle('on', m === name);
    tools.dataset.mode = m;
    const about = info(m);
    amountInput.title = about.strength;
    amountInput.value = amounts[m] ?? DEFAULT_AMOUNT[m] ?? 100;
    amountValue.textContent = `${amountInput.value}%`;
    if (about.point) { pointBtn.textContent = about.point; pointBtn.title = about.pointTitle; }
    setPicking(false);
    setHint();
    canvas.classList.toggle('hide-tint', !imageMode() && !tintInput.checked);
    drawTint();
    showSource();
  };
  // Which modes the tools offer, in this order, with per-step labels (see STEPS). The
  // current mode stays if it's offered and keep is set; otherwise the first one is chosen.
  const setModes = (list, labelsFor = {}, keep = true) => {
    visible = list;
    names = labelsFor;
    for (const [m, b] of Object.entries(modeButtons)) b.hidden = !list.includes(m);
    for (const m of list) {
      const b = modeButtons[m], about = info(m);
      b.textContent = about.label;
      b.title = about.title;
      (MASK_MODES.includes(m) ? maskGroup : imageGroup).append(b);
    }
    maskGroup.style.display = list.some((m) => MASK_MODES.includes(m)) ? '' : 'none';
    imageGroup.style.display = list.some((m) => IMAGE_MODES.includes(m)) ? '' : 'none';
    setMode(keep && list.includes(mode) ? mode : list[0]);
  };

  const syncButtons = () => {
    undoBtn.disabled = externalUndo ? !canUndo : !strokes.length;
    resetBtn.disabled = !strokes.length;
  };
  const tell = (edit) => { for (const fn of editListeners) fn(edit); };
  const changed = () => { version++; syncButtons(); onChange(); };

  // ---------- view ----------
  function applyView() {
    if (!editing) { stage.style.cssText = ''; clip.style.cssText = ''; return; }
    const w = fitW * zoom, h = fitH * zoom;
    tx = w <= clipW ? (clipW - w) / 2 : clamp(tx, clipW - w, 0);
    ty = h <= fitH ? (fitH - h) / 2 : clamp(ty, fitH - h, 0);
    stage.style.width = `${fitW}px`;
    stage.style.height = `${fitH}px`;
    stage.style.transform = `translate(${tx}px, ${ty}px) scale(${zoom})`;
    clip.style.height = `${fitH}px`;
    drawCursorSize();
    showSource();
  }

  function fit() {
    if (!editing) return;
    clipW = clip.parentElement.clientWidth - 2 * parseFloat(getComputedStyle(clip.parentElement).paddingLeft || 0);
    const aspect = (image.naturalWidth || 3) / (image.naturalHeight || 4);
    fitH = Math.max(MIN_VIEW, window.innerHeight - (toolsHost ? 0 : tools.offsetHeight) - VIEW_MARGIN);
    fitW = fitH * aspect;
    if (fitW > clipW) { fitW = clipW; fitH = fitW / aspect; }
    applyView();
  }

  const longPx = () => Math.max(fitW, fitH) * zoom;
  function drawCursorSize() {
    const d = sizeOf(+sizeInput.value) * longPx();
    cursor.style.width = cursor.style.height = `${d}px`;
    cursor.style.marginLeft = cursor.style.marginTop = `${-d / 2}px`;
  }

  // ---------- drawing ----------
  // The tint, while a mode that edits it is chosen, and heal, spot and light strokes
  // still waiting for their result.
  function drawTint() {
    if (!base.width) return;
    if (canvas.width !== base.width || canvas.height !== base.height) { canvas.width = base.width; canvas.height = base.height; }
    ctx.globalCompositeOperation = 'source-over';
    ctx.filter = 'none';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!imageMode()) ctx.drawImage(base, 0, 0);
    for (const stroke of current ? [...awaiting, current] : awaiting) drawStroke(stroke);
  }

  function drawStroke(stroke) {
    const w = canvas.width, h = canvas.height, c = stroke.points, d = stroke.size * Math.max(w, h);
    const patch = PATCH_STROKES.includes(stroke.mode);
    ctx.save();
    ctx.globalAlpha = stroke.amount ?? 1;   // part strength: part of the tint added or taken away
    ctx.globalCompositeOperation = stroke.mode === 'skin' || patch ? 'source-over' : 'destination-out';
    ctx.strokeStyle = patch ? swatches.get(stroke.id) || PATCH_CSS : stroke.mode === 'skin' ? TINT_CSS : '#000';
    ctx.lineCap = ctx.lineJoin = 'round';
    ctx.lineWidth = d;
    if (stroke.soft > 0) ctx.filter = `blur(${stroke.soft * d * 0.25}px)`;
    ctx.beginPath();
    ctx.moveTo(c[0] * w, c[1] * h);
    for (let i = 2; i + 1 < c.length; i += 2) ctx.lineTo(c[i] * w, c[i + 1] * h);
    if (c.length === 2) ctx.lineTo(c[0] * w + 0.01, c[1] * h);
    ctx.stroke();
    ctx.restore();
  }

  // The picked colour as the photo on screen shows it, for the swatch and the patch
  // shown while waiting. The retouch itself reads the colour from the photo it works on.
  function sampleSwatch() {
    swatch = null;
    try {
      const w = image.naturalWidth, h = image.naturalHeight;
      if (w && pick) {
        const r = Math.max(1, Math.round(Math.max(w, h) * 0.002)), n = 2 * r + 1;
        const work = el('canvas', { width: n, height: n });
        const g = work.getContext('2d', { willReadFrequently: true });
        g.drawImage(image, Math.round(pick[0] * w) - r, Math.round(pick[1] * h) - r, n, n, 0, 0, n, n);
        const d = g.getImageData(0, 0, n, n).data, sum = [0, 0, 0];
        let count = 0;
        for (let i = 0; i < d.length; i += 4) if (d[i + 3]) { sum[0] += d[i]; sum[1] += d[i + 1]; sum[2] += d[i + 2]; count++; }
        if (count) swatch = `rgb(${sum.map((v) => Math.round(v / count)).join(',')})`;
      }
    } catch { /* the swatch is only for show */ }
    swatchEl.style.background = swatch || '';
  }

  const position = (e) => {
    const r = canvas.getBoundingClientRect();
    return [clamp((e.clientX - r.left) / r.width, 0, 1), clamp((e.clientY - r.top) / r.height, 0, 1)];
  };

  function moveCursor(e) {
    const r = clip.getBoundingClientRect();
    cursor.style.transform = `translate(${e.clientX - r.left}px, ${e.clientY - r.top}px)`;
    cursor.classList.add('on');
  }

  // Where Heal copies from: the point picked, or, once Aligned has an offset or while
  // painting, the brush plus that offset. For Light and Nose line, the point their
  // colour comes from.
  function showSource() {
    let at = null;
    if (editing && mode === 'heal' && source) {
      const offset = current?.offset ?? (alignedInput.checked ? alignedOffset : null);
      const p = current ? current.points.slice(-2) : hover;
      at = offset && p ? [p[0] + offset[0], p[1] + offset[1]] : source;
    } else if (editing && MODES[mode].colour) {
      at = pick;
    }
    if (!at) { sourceMark.classList.remove('on'); return; }
    const r = canvas.getBoundingClientRect(), box = clip.getBoundingClientRect();
    sourceMark.style.transform = `translate(${r.left - box.left + at[0] * r.width}px, ${r.top - box.top + at[1] * r.height}px)`;
    sourceMark.classList.add('on');
  }

  // ---------- pointer ----------
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    try { canvas.setPointerCapture(e.pointerId); } catch { /* strokes still work inside the view */ }
    if (e.button !== 0 || spaceDown) {
      panning = { x: e.clientX, y: e.clientY, tx, ty };
      document.body.classList.add('mask-panning');
      return;
    }
    const [x, y] = position(e);
    if (MODES[mode].point && (e.altKey || picking)) {
      if (mode === 'heal') { source = [x, y]; alignedOffset = null; }
      else { pick = [x, y]; sampleSwatch(); }
      setPicking(false);
      setHint();
      showSource();
      return;
    }
    if ((mode === 'heal' && !source) || (MODES[mode].colour && !pick)) { setHint(info(mode).noPoint); return; }
    if (lineDraft) {
      // Nose line: the second click ends the line, and letting go finishes it.
      lineDraft.points = [lineDraft.points[0], lineDraft.points[1], x, y];
      current = lineDraft;
      lineDraft = null;
      setHint();
      drawTint();
      return;
    }
    // A nose line is a light stroke between two clicks.
    current = { id: nextId++, mode: mode === 'line' ? 'light' : mode, size: sizeOf(+sizeInput.value),
                soft: +softInput.value / 100, points: mode === 'line' ? [x, y, x, y] : [x, y] };
    currentLabel = names[mode] ? `${names[mode]}.label` : `mode.${mode}.label`;
    if (mode === 'heal') {
      // Aligned keeps the first stroke's distance to the source for the strokes after it.
      const offset = alignedInput.checked && alignedOffset ? alignedOffset : [source[0] - x, source[1] - y];
      if (alignedInput.checked) alignedOffset = offset;
      current.offset = offset;
    } else if (MODES[mode].colour) {
      current.pick = pick;
      if (swatch) swatches.set(current.id, swatch);
    }
    const amount = +amountInput.value / 100;
    if (amount < 1) current.amount = amount;   // full strokes stay as they always were
    if (mode === 'line') {
      lineDraft = current;
      setHint(info('line').next, false);
    }
    drawTint();
    showSource();
  });
  canvas.addEventListener('pointermove', (e) => {
    moveCursor(e);
    hover = position(e);
    if (mode === 'heal') showSource();
    if (panning) {
      tx = panning.tx + e.clientX - panning.x;
      ty = panning.ty + e.clientY - panning.y;
      applyView();
    } else if (lineDraft) {
      // The line follows the pointer until the second click.
      lineDraft.points[2] = hover[0];
      lineDraft.points[3] = hover[1];
      drawTint();
    } else if (current) {
      const [x, y] = position(e), c = current.points;
      // Not every tiny movement: this keeps strokes small, and the layer paints fast.
      if (Math.hypot((x - c[c.length - 2]) * canvas.width, (y - c[c.length - 1]) * canvas.height) >= 1) {
        c.push(x, y);
        drawTint();
      }
    }
  });
  const finish = () => {
    if (panning) { panning = null; document.body.classList.remove('mask-panning'); }
    if (current && current !== lineDraft) {    // a nose line waits for its second click
      if (PATCH_STROKES.includes(current.mode)) awaiting = [...awaiting, current];
      strokes = [...strokes, current];
      current = null;
      tell({ kind: 'stroke', label: currentLabel });   // before the page asks for a result
      changed();
      showSource();
    }
  };
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointercancel', () => { lineDraft = null; current = null; drawTint(); finish(); });
  canvas.addEventListener('pointerleave', () => { cursor.classList.remove('on'); hover = null; showSource(); });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = clip.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const next = clamp(zoom * (e.deltaY < 0 ? 1.25 : 0.8), 1, MAX_ZOOM);
    tx = mx - (mx - tx) * (next / zoom);
    ty = my - (my - ty) * (next / zoom);
    zoom = next;
    applyView();
  }, { passive: false });

  // ---------- controls ----------
  for (const [m, b] of Object.entries(modeButtons)) b.addEventListener('click', () => setMode(m));
  pointBtn.addEventListener('click', () => setPicking(!picking));
  alignedInput.addEventListener('change', () => { alignedOffset = null; showSource(); });
  sizeInput.addEventListener('input', drawCursorSize);
  amountInput.addEventListener('input', () => { amountValue.textContent = `${amountInput.value}%`; });
  const forget = () => { awaiting = awaiting.filter((s) => strokes.includes(s)); drawTint(); };
  const undo = () => {
    if (lineDraft) { dropLine(); setHint(); return; }    // a half-drawn nose line goes first
    if (externalUndo) { externalUndo(); return; }        // the History panel's undo
    if (strokes.length) { strokes = strokes.slice(0, -1); forget(); changed(); }
  };
  undoBtn.addEventListener('click', undo);
  const clearStrokes = () => { strokes = []; forget(); };
  resetBtn.addEventListener('click', () => { clearStrokes(); tell({ kind: 'reset' }); changed(); });
  const resetView = () => { zoom = 1; applyView(); };
  fitBtn.addEventListener('click', resetView);
  tintInput.addEventListener('change', () => canvas.classList.toggle('hide-tint', !imageMode() && !tintInput.checked));
  window.addEventListener('resize', fit);
  // Alt shows the picking cursor, and mustn't move the focus to the browser's menu.
  const holdAlt = (down) => { altDown = down; setPicking(picking); };
  window.addEventListener('blur', () => holdAlt(false));
  window.addEventListener('keydown', (e) => {
    if (!editing || /INPUT|TEXTAREA/.test(document.activeElement?.tagName || '') && document.activeElement.type !== 'range') return;
    if (e.key === 'Alt') { holdAlt(true); e.preventDefault(); }
    else if (e.code === 'Space') {
      // A focused button would take Space as a click, like Done editing just pressed.
      if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
      spaceDown = true;
      document.body.classList.add('mask-panning');
      e.preventDefault();
    }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !e.shiftKey && (lineDraft || !externalUndo)) {
      undo();                         // with a History panel, it takes Ctrl+Z, except for a half-drawn line
      e.preventDefault();
      e.stopImmediatePropagation();
    }
    else if (e.key === '[') { sizeInput.value = Math.max(1, +sizeInput.value - 4); drawCursorSize(); }
    else if (e.key === ']') { sizeInput.value = Math.min(100, +sizeInput.value + 4); drawCursorSize(); }
    else if (e.key === '0') resetView();
    else if (e.key.toLowerCase() === 'm') { tintInput.checked = !tintInput.checked; tintInput.dispatchEvent(new Event('change')); }
    else if (e.key.toLowerCase() === 'x') setMode(visible[(visible.indexOf(mode) + 1) % visible.length]);
    else if (e.key === 'Escape' && lineDraft) { dropLine(); setHint(); }
  });
  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space') { spaceDown = false; if (!panning) document.body.classList.remove('mask-panning'); }
    if (e.key === 'Alt') { holdAlt(false); if (editing) e.preventDefault(); }
  });
  setModes(Object.keys(MODES));
  // Another language: the same tools and mode, in the new words.
  onLangChange(() => {
    translatePage(tools);
    setModes(visible, names, true);
    button.textContent = editing ? labels[1] : labels[0];
  });

  function setEditing(on) {
    if (on === editing) return;
    dropLine();
    editing = on;
    document.body.classList.toggle('mask-editing', on);
    button.textContent = on ? labels[1] : labels[0];
    zoom = 1;
    if (on) {
      fit();
      // The tools at the top with the whole view below them, or, with the tools in a
      // side panel that stays in view, the view at the top.
      (toolsHost ? clip : tools).scrollIntoView({ block: 'start' });
    } else {
      applyView();
    }
    setPicking(false);
    showSource();
    onModeChange(on);
  }
  button.addEventListener('click', () => setEditing(!editing));
  image.addEventListener('load', () => { if (editing) fit(); });

  return {
    get strokes() { return strokes; },
    get version() { return version; },
    get editing() { return editing; },
    setEditing,
    setModes,
    // Strokes from elsewhere (the History panel): shown at once; the page renders them.
    setStrokes(list) {
      dropLine();
      current = null;
      strokes = list;                 // lists are never changed in place, so they can be shared
      awaiting = awaiting.filter((s) => strokes.includes(s));
      version++;
      syncButtons();
      drawTint();
    },
    // fn({ kind: 'stroke', label: an i18n.js key } or { kind: 'reset' }) after each.
    onEdit(fn) { editListeners.add(fn); },
    // Undo (its button and Ctrl+Z) goes to fn instead; setUndoState says if it can.
    setUndo(fn) {
      externalUndo = fn;
      undoBtn.dataset.i18nTitle = 'history.undoTitle';
      undoBtn.title = t('history.undoTitle');
      syncButtons();
    },
    setUndoState(on) { canUndo = on; syncButtons(); },
    // The button's text when off and when painting.
    setLabels(next) {
      labels = next;
      button.textContent = editing ? labels[1] : labels[0];
    },
    // A new photo: no strokes, no points picked, and out of edit mode.
    reset() {
      strokes = []; current = null; awaiting = []; base.width = 0;
      source = alignedOffset = pick = swatch = lineDraft = null;
      swatches.clear();
      swatchEl.style.background = '';
      syncButtons();
      setHint();
      setEditing(false);
    },
    // The real mask, one byte per pixel. upTo: the id of the last stroke the result
    // holds, so strokes up to it stop showing as waiting (all, if missing).
    setMaskBytes(w, h, bytes, upTo = null) {
      awaiting = upTo == null ? [] : awaiting.filter((s) => s.id > upTo);
      base.width = w; base.height = h;
      const picture = new ImageData(w, h), d = picture.data, a = TINT_ALPHA;
      for (let i = 0, j = 0; i < bytes.length; i++, j += 4) {
        d[j] = TINT[0]; d[j + 1] = TINT[1]; d[j + 2] = TINT[2]; d[j + 3] = bytes[i] * a;
      }
      base.getContext('2d').putImageData(picture, 0, 0);
      drawTint();
    },
    // The real mask, as a loaded picture whose brightness is the mask.
    setMaskImage(picture, upTo = null) {
      const w = picture.naturalWidth, h = picture.naturalHeight;
      const work = document.createElement('canvas');
      work.width = w; work.height = h;
      const c = work.getContext('2d', { willReadFrequently: true });
      c.drawImage(picture, 0, 0);
      const pixels = c.getImageData(0, 0, w, h).data, bytes = new Uint8Array(w * h);
      for (let i = 0, j = 0; i < bytes.length; i++, j += 4) bytes[i] = pixels[j];
      this.setMaskBytes(w, h, bytes, upTo);
    },
  };
}

// The retouch in steps, like DM Portrait's: tabs at the top of the panel and Back / Next
// at its bottom. A step shows the panel's sections marked data-step="n" (the others
// always show) and its own brush tools, starting on the first, and brush button text. Steps are only views:
// everything is redone from the original, so going back to change a setting loses
// nothing. onStep(n) runs after each change. Back / Next go at the panel's end, or
// before navBefore.
export function attachSteps({ panel, brush, steps = STEPS, onStep = () => {}, navBefore = null }) {
  addStyle();
  const title = (i) => t(`${steps[i].key}.title`);
  const tabButtons = steps.map(() => el('button', { type: 'button' }));
  const tabs = el('div', { className: 'step-tabs' }, ...tabButtons);
  const back = el('button', { type: 'button', className: 'secondary' });
  const next = el('button', { type: 'button' });
  const nav = el('div', { className: 'step-nav' }, back, next);
  panel.prepend(tabs);
  if (navBefore) panel.insertBefore(nav, navBefore);
  else panel.append(nav);
  let step = 0;
  // The tabs, Back / Next and the brush button, in the current language.
  function words() {
    tabButtons.forEach((b, k) => { b.textContent = `${k + 1} ${title(k)}`; b.title = t(`${steps[k].key}.about`); });
    back.textContent = step > 0 ? t('steps.back', { step: title(step - 1) }) : '';
    next.textContent = step < steps.length - 1 ? t('steps.next', { step: title(step + 1) }) : '';
    brush.setLabels([t(`${steps[step].key}.button`), t('brush.done')]);
  }
  function show(i) {
    step = clamp(i, 0, steps.length - 1);
    tabButtons.forEach((b, k) => b.classList.toggle('on', k === step));
    for (const section of panel.querySelectorAll('[data-step]')) section.classList.toggle('step-off', +section.dataset.step !== step + 1);
    back.classList.toggle('step-off', step === 0);
    next.classList.toggle('step-off', step === steps.length - 1);
    brush.setModes(steps[step].modes, steps[step].words || {}, false);   // each step opens on its first tool
    words();
    onStep(step + 1);
  }
  tabButtons.forEach((b, k) => b.addEventListener('click', () => show(k)));
  back.addEventListener('click', () => show(step - 1));
  next.addEventListener('click', () => show(step + 1));
  onLangChange(words);
  show(0);
  return { get step() { return step + 1; }, setStep: (n) => show(n - 1) };
}
