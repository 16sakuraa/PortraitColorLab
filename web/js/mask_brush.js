// The brush for editing the skin mask by hand: the same interface on the web page and
// on the desktop app's page, so it imports nothing and brings its own styles.
//
// Pressing the page's Edit mask button tints the treated skin over the retouched
// photo, in a larger view that zooms and pans. Strokes (see mask_edit.js for their
// form) are shown straight away on the tint, then the page asks for a new result and
// hands back the real mask with setMaskBytes or setMaskImage.

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
  .mask-mode { display: inline-flex; border-radius: 8px; overflow: hidden; }
  .mask-mode button { border-radius: 0; background: #6b7280; }
  .mask-mode button.on { background: #4b7bec; }
  .mask-hint { flex-basis: 100%; opacity: .7; }
`;

const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};

// Brush size slider 1..100 to a share of the long side, finer at the small end.
const sizeOf = (v) => 0.002 + 0.1 * (v / 100) ** 2;
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

export function attachMaskBrush({ image, button, onChange, onModeChange = () => {} }) {
  if (!document.getElementById('mask-brush-style')) {
    document.head.append(el('style', { id: 'mask-brush-style', textContent: STYLE }));
  }
  const clip = el('div', { className: 'mask-clip' }), stage = el('div', { className: 'mask-stage' });
  const canvas = el('canvas', { className: 'mask-canvas' }), cursor = el('div', { className: 'mask-cursor' });
  image.parentElement.insertBefore(clip, image);
  stage.append(image, canvas);
  clip.append(stage, cursor);
  const ctx = canvas.getContext('2d');
  const base = document.createElement('canvas');      // the mask as last received

  // ---------- tools ----------
  const protectBtn = el('button', { type: 'button', textContent: 'Protect', title: 'Never smooth here, and don\'t fill spots here' });
  const skinBtn = el('button', { type: 'button', textContent: 'Add skin', title: 'Smooth here, even if the automatic mask left it out' });
  const sizeInput = el('input', { type: 'range', min: 1, max: 100, value: 24, title: 'Brush size ( [ and ] )' });
  const softInput = el('input', { type: 'range', min: 0, max: 100, value: 30, title: 'How soft the brush\'s edge is' });
  const undoBtn = el('button', { type: 'button', className: 'secondary', textContent: 'Undo', disabled: true, title: 'Undo the last stroke (Ctrl+Z)' });
  const resetBtn = el('button', { type: 'button', className: 'secondary', textContent: 'Reset', disabled: true, title: 'Remove all strokes' });
  const fitBtn = el('button', { type: 'button', className: 'secondary', textContent: 'Fit', title: 'Show the whole photo (0)' });
  const tintInput = el('input', { type: 'checkbox', checked: true, title: 'Show the tint (M)' });
  const tools = el('div', { className: 'mask-tools' },
    el('span', { className: 'mask-mode' }, protectBtn, skinBtn),
    el('label', {}, 'Size', sizeInput),
    el('label', {}, 'Softness', softInput),
    undoBtn, resetBtn, fitBtn,
    el('label', {}, tintInput, 'Show tint'),
    el('span', { className: 'mask-hint', textContent: 'Tinted skin gets smoothed. Scroll to zoom, hold Space or the right mouse button to move around. Add skin smooths an area but doesn\'t look for spots in it.' }));
  clip.parentElement.insertBefore(tools, clip);

  // ---------- state ----------
  let editing = false, mode = 'protect', strokes = [], nextId = 1, version = 0;
  let current = null, panning = null, spaceDown = false;
  let fitW = 0, fitH = 0, clipW = 0, zoom = 1, tx = 0, ty = 0;

  const setMode = (m) => {
    mode = m;
    protectBtn.classList.toggle('on', m === 'protect');
    skinBtn.classList.toggle('on', m === 'skin');
  };
  setMode('protect');

  const syncButtons = () => { undoBtn.disabled = resetBtn.disabled = !strokes.length; };
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
  }

  function fit() {
    if (!editing) return;
    clipW = clip.parentElement.clientWidth - 2 * parseFloat(getComputedStyle(clip.parentElement).paddingLeft || 0);
    const aspect = (image.naturalWidth || 3) / (image.naturalHeight || 4);
    fitH = Math.max(MIN_VIEW, window.innerHeight - tools.offsetHeight - VIEW_MARGIN);
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
  function drawTint() {
    if (!base.width) return;
    if (canvas.width !== base.width || canvas.height !== base.height) { canvas.width = base.width; canvas.height = base.height; }
    ctx.globalCompositeOperation = 'source-over';
    ctx.filter = 'none';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(base, 0, 0);
    for (const stroke of current ? [current] : []) drawStroke(stroke);
  }

  function drawStroke(stroke) {
    const w = canvas.width, h = canvas.height, c = stroke.points, d = stroke.size * Math.max(w, h);
    ctx.save();
    ctx.globalCompositeOperation = stroke.mode === 'skin' ? 'source-over' : 'destination-out';
    ctx.strokeStyle = stroke.mode === 'skin' ? TINT_CSS : '#000';
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

  const position = (e) => {
    const r = canvas.getBoundingClientRect();
    return [clamp((e.clientX - r.left) / r.width, 0, 1), clamp((e.clientY - r.top) / r.height, 0, 1)];
  };

  function moveCursor(e) {
    const r = clip.getBoundingClientRect();
    cursor.style.transform = `translate(${e.clientX - r.left}px, ${e.clientY - r.top}px)`;
    cursor.classList.add('on');
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
    if (e.button === 0) {
      const [x, y] = position(e);
      current = { id: nextId++, mode, size: sizeOf(+sizeInput.value), soft: +softInput.value / 100, points: [x, y] };
      drawTint();
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    moveCursor(e);
    if (panning) {
      tx = panning.tx + e.clientX - panning.x;
      ty = panning.ty + e.clientY - panning.y;
      applyView();
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
    if (current) {
      strokes = [...strokes, current];
      current = null;
      changed();
    }
  };
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointercancel', () => { current = null; drawTint(); finish(); });
  canvas.addEventListener('pointerleave', () => cursor.classList.remove('on'));
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
  protectBtn.addEventListener('click', () => setMode('protect'));
  skinBtn.addEventListener('click', () => setMode('skin'));
  sizeInput.addEventListener('input', drawCursorSize);
  const undo = () => { if (strokes.length) { strokes = strokes.slice(0, -1); changed(); } };
  undoBtn.addEventListener('click', undo);
  const clearStrokes = () => { strokes = []; };
  resetBtn.addEventListener('click', () => { clearStrokes(); changed(); });
  const resetView = () => { zoom = 1; applyView(); };
  fitBtn.addEventListener('click', resetView);
  tintInput.addEventListener('change', () => canvas.classList.toggle('hide-tint', !tintInput.checked));
  window.addEventListener('resize', fit);
  window.addEventListener('keydown', (e) => {
    if (!editing || /INPUT|TEXTAREA/.test(document.activeElement?.tagName || '') && document.activeElement.type !== 'range') return;
    if (e.code === 'Space') {
      // A focused button would take Space as a click, like Done editing just pressed.
      if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
      spaceDown = true;
      document.body.classList.add('mask-panning');
      e.preventDefault();
    }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { undo(); e.preventDefault(); }
    else if (e.key === '[') { sizeInput.value = Math.max(1, +sizeInput.value - 4); drawCursorSize(); }
    else if (e.key === ']') { sizeInput.value = Math.min(100, +sizeInput.value + 4); drawCursorSize(); }
    else if (e.key === '0') resetView();
    else if (e.key.toLowerCase() === 'm') { tintInput.checked = !tintInput.checked; tintInput.dispatchEvent(new Event('change')); }
    else if (e.key.toLowerCase() === 'x') setMode(mode === 'protect' ? 'skin' : 'protect');
  });
  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space') { spaceDown = false; if (!panning) document.body.classList.remove('mask-panning'); }
  });

  function setEditing(on) {
    if (on === editing) return;
    editing = on;
    document.body.classList.toggle('mask-editing', on);
    button.textContent = on ? 'Done editing' : 'Edit mask';
    zoom = 1;
    if (on) {
      fit();
      tools.scrollIntoView({ block: 'start' });   // the tools at the top, the whole view below them
    } else {
      applyView();
    }
    onModeChange(on);
  }
  button.addEventListener('click', () => setEditing(!editing));
  image.addEventListener('load', () => { if (editing) fit(); });

  return {
    get strokes() { return strokes; },
    get version() { return version; },
    get editing() { return editing; },
    setEditing,
    // A new photo: no strokes, and out of edit mode.
    reset() {
      strokes = []; current = null; base.width = 0;
      syncButtons();
      setEditing(false);
    },
    // The real mask, one byte per pixel.
    setMaskBytes(w, h, bytes) {
      base.width = w; base.height = h;
      const picture = new ImageData(w, h), d = picture.data, a = TINT_ALPHA;
      for (let i = 0, j = 0; i < bytes.length; i++, j += 4) {
        d[j] = TINT[0]; d[j + 1] = TINT[1]; d[j + 2] = TINT[2]; d[j + 3] = bytes[i] * a;
      }
      base.getContext('2d').putImageData(picture, 0, 0);
      drawTint();
    },
    // The real mask, as a loaded picture whose brightness is the mask.
    setMaskImage(picture) {
      const w = picture.naturalWidth, h = picture.naturalHeight;
      const work = document.createElement('canvas');
      work.width = w; work.height = h;
      const c = work.getContext('2d', { willReadFrequently: true });
      c.drawImage(picture, 0, 0);
      const pixels = c.getImageData(0, 0, w, h).data, bytes = new Uint8Array(w * h);
      for (let i = 0, j = 0; i < bytes.length; i++, j += 4) bytes[i] = pixels[j];
      this.setMaskBytes(w, h, bytes);
    },
  };
}
