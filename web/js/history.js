// The History panel, like DM Portrait's: every change to the retouch - a slider let go,
// a setting, a brush stroke, clearing the strokes - is listed, and clicking one goes
// back to the photo as it was then. Undo and Redo (Ctrl+Z, and Ctrl+Y or Ctrl+Shift+Z)
// step through the list. A new change after going back replaces the ones below it.
//
// Each entry keeps the settings and the brush's stroke list. The brush never changes a
// stroke list in place, so entries share strokes instead of copying them. Shared by the
// web app and the desktop app's page, which use the same ids for their controls; it
// imports only its words.
//
// While a photo is open and the photo as it now is hasn't been downloaded (the page
// calls markSaved on Download), closing the tab or reloading asks first, with the
// browser's own "Leave site?" box.

import { t, onLangChange } from './i18n.js';

const MAX_ENTRIES = 300;   // the oldest changes (after Photo opened) drop off beyond this
// The settings recorded, by element id, with the word for each; sliders show their
// value in a label beside them.
const FIELDS = [
  { id: 'blemish', key: 'ctl.blemish', shows: 'bval' },
  { id: 'strength', key: 'ctl.strength', shows: 'sval' },
  { id: 'texture', key: 'ctl.texture', shows: 'tval' },
  { id: 'shine', key: 'ctl.shine', shows: 'rval' },
  { id: 'pores', key: 'ctl.pores', shows: 'pval' },
  { id: 'lighting', key: 'ctl.lighting' },
  { id: 'print', key: 'ctl.print' },
  { id: 'faceonly', key: 'ctl.faceOnly' },
];

const STYLE = `
  .history { display: grid; gap: 8px; }
  .history-row { display: flex; gap: 8px; }
  .history-row button { padding: 4px 10px; font-size: 12px; }
  .history-list { position: relative; list-style: none; margin: 0; padding: 0; max-height: 210px; overflow-y: auto; font-size: 12px; }
  .history-list button { all: unset; display: block; box-sizing: border-box; width: 100%; padding: 3px 8px;
                         border-radius: 6px; cursor: pointer; overflow-wrap: anywhere; }
  .history-list button:hover { background: #8884; }
  .history-list li.now button { background: #4b7bec; color: #fff; }
  .history-list li.undone button { opacity: .45; }
  .history-list li.empty { opacity: .6; padding: 3px 8px; }
  .history-hint { margin: 0; font-size: 11px; opacity: .65; }
`;

const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};
const $ = (id) => document.getElementById(id);

// host: where the panel goes. brush: attachMaskBrush's (for its strokes). render: shows
// the photo for the controls and strokes as they now are.
export function attachHistory({ host, brush, render }) {
  if (!document.getElementById('history-style')) {
    document.head.append(el('style', { id: 'history-style', textContent: STYLE }));
  }
  const undoBtn = el('button', { type: 'button', className: 'secondary', disabled: true });
  const redoBtn = el('button', { type: 'button', className: 'secondary', disabled: true });
  const list = el('ol', { className: 'history-list' });
  const hint = el('p', { className: 'history-hint' });
  host.append(el('div', { className: 'history' }, el('div', { className: 'history-row' }, undoBtn, redoBtn), list, hint));

  let entries = [], now = -1, applying = false;

  const read = () => Object.fromEntries(FIELDS.map(({ id }) => {
    const input = $(id);
    return [id, input.type === 'checkbox' ? input.checked : input.value];
  }));
  function write(settings) {
    for (const { id, shows } of FIELDS) {
      const input = $(id);
      if (input.type === 'checkbox') input.checked = settings[id];
      else input.value = settings[id];
      if (shows && !input.disabled) $(shows).textContent = input.value;
    }
  }
  // What a settings change says, e.g. "Strength 75" or "Lighting: Soft (studio)".
  function said(id, value) {
    const field = FIELDS.find((f) => f.id === id), input = $(id), name = t(field.key);
    if (input.type === 'checkbox') return `${name}: ${t(value ? 'history.on' : 'history.off')}`;
    if (input.tagName === 'SELECT') return `${name}: ${[...input.options].find((o) => o.value === value)?.textContent ?? value}`;
    return `${name} ${value}`;
  }

  function draw() {
    undoBtn.textContent = t('history.undo');
    undoBtn.title = t('history.undoTitle');
    redoBtn.textContent = t('history.redo');
    redoBtn.title = t('history.redoTitle');
    hint.textContent = t('history.hint');
    undoBtn.disabled = now <= 0;
    redoBtn.disabled = now < 0 || now >= entries.length - 1;
    brush.setUndoState(now > 0);
    list.replaceChildren(...(entries.length ? entries.map((entry, i) => {
      const item = el('li', { className: i === now ? 'now' : i > now ? 'undone' : '' },
        el('button', { type: 'button', textContent: `${i + 1}. ${entry.label()}` }));
      item.firstChild.addEventListener('click', () => go(i));
      return item;
    }) : [el('li', { className: 'empty', textContent: t('history.empty') })]));
    // Keep the current entry in view, without scrolling the page.
    const item = list.children[now];
    if (item) {
      if (item.offsetTop < list.scrollTop) list.scrollTop = item.offsetTop;
      else if (item.offsetTop + item.offsetHeight > list.scrollTop + list.clientHeight) {
        list.scrollTop = item.offsetTop + item.offsetHeight - list.clientHeight;
      }
    }
  }

  function push(label) {
    if (applying || now < 0) return;
    entries = entries.slice(0, now + 1);
    entries.push({ label, settings: read(), strokes: brush.strokes });
    if (entries.length > MAX_ENTRIES) entries.splice(1, 1);
    now = entries.length - 1;
    draw();
  }

  function go(i) {
    if (i < 0 || i >= entries.length || i === now) return;
    now = i;
    applying = true;
    write(entries[i].settings);
    brush.setStrokes(entries[i].strokes);
    applying = false;
    draw();
    render();
  }

  // A slider records when it's let go; a select or the checkbox when it changes.
  for (const { id } of FIELDS) {
    $(id).addEventListener('change', () => {
      if (applying || now < 0) return;
      const value = read()[id];
      if (entries[now].settings[id] === value) return;   // let go where it started
      push(() => said(id, value));
    });
  }
  brush.onEdit((edit) => push(edit.kind === 'reset' ? () => t('history.cleared') : () => t(edit.label)));
  brush.setUndo(() => go(now - 1));
  undoBtn.addEventListener('click', () => go(now - 1));
  redoBtn.addEventListener('click', () => go(now + 1));
  window.addEventListener('keydown', (e) => {
    if (now < 0 || !(e.ctrlKey || e.metaKey) || e.altKey) return;
    const focus = document.activeElement;
    if (focus?.tagName === 'TEXTAREA' || (focus?.tagName === 'INPUT' && !['range', 'checkbox'].includes(focus.type))) return;
    const key = e.key.toLowerCase();
    if (key === 'z' && !e.shiftKey) { go(now - 1); e.preventDefault(); }
    else if (key === 'y' || (key === 'z' && e.shiftKey)) { go(now + 1); e.preventDefault(); }
  });
  onLangChange(draw);
  draw();

  // The entry that was current at the last download: closing loses nothing while it
  // still is. Any change, or going back, makes another entry current.
  let saved = null;
  const unsaved = () => now >= 0 && entries[now] !== saved;
  window.addEventListener('beforeunload', (e) => {
    if (!unsaved()) return;
    e.preventDefault();
    e.returnValue = '';        // older browsers need this to ask
  });

  return {
    // A photo was opened: the list starts again from it, with the settings as they are.
    start() {
      entries = [{ label: () => t('history.opened'), settings: read(), strokes: brush.strokes }];
      now = 0;
      saved = null;
      draw();
    },
    // No photo: an empty list.
    clear() {
      entries = [];
      now = -1;
      saved = null;
      draw();
    },
    // The photo as it is now: hand it to markSaved once it's been downloaded.
    get point() { return entries[now] ?? null; },
    markSaved(point = entries[now] ?? null) { saved = point; },
    get unsaved() { return unsaved(); },
    undo: () => go(now - 1),
    redo: () => go(now + 1),
    get length() { return entries.length; },
    get now() { return now; },
  };
}
