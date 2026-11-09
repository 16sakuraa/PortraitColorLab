// Zoom for the side-by-side comparison: scroll on either photo to zoom both into the
// same spot, drag to move around, double-click to see the whole photo again. Shared by
// the web app and the desktop app's page, so it imports nothing and brings its own
// styles. While the brush is painting (body.mask-editing) it stands aside: the brush
// has its own view, which zooms and pans by itself.
//
// The zoom is kept as shares of the photo's frame (CSS translate percentages refer to
// the photo itself), so both photos stay in step at any panel width.

const MAX_ZOOM = 10, STEP = 1.25;

const STYLE = `
  .zoom-clip { position: relative; overflow: hidden; border-radius: 8px; }
  body:not(.mask-editing) img.zoomable {
    transform-origin: 0 0; cursor: zoom-in; user-select: none;
    transform: translate(var(--zoom-x, 0%), var(--zoom-y, 0%)) scale(var(--zoom, 1));
  }
  body:not(.mask-editing) .zoomed img.zoomable { cursor: grab; }
  body.zoom-dragging img.zoomable { cursor: grabbing !important; }
  .zoom-badge { position: absolute; right: 8px; top: 8px; z-index: 2; display: none; pointer-events: none;
                font-size: 11px; line-height: 1.5; padding: 0 7px; border-radius: 8px; background: #000000a6; color: #fff; }
  body:not(.mask-editing) .zoomed .zoom-badge { display: block; }
`;

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const painting = () => document.body.classList.contains('mask-editing');

// images: the photos to keep in step. Each gets a frame that clips it: the brush's, if
// it has one, else a new one around it.
export function attachCompareZoom(images) {
  if (!document.getElementById('compare-zoom-style')) {
    document.head.append(Object.assign(document.createElement('style'), { id: 'compare-zoom-style', textContent: STYLE }));
  }
  // The scale, and the photo's top-left corner as a share of its frame (0 or below).
  let zoom = 1, x = 0, y = 0, drag = null;
  const badges = [];
  const frames = images.map((img) => {
    img.classList.add('zoomable');
    img.draggable = false;
    let frame = img.closest('.mask-clip');
    if (!frame) {
      frame = document.createElement('div');
      img.parentElement.insertBefore(frame, img);
      frame.append(img);
    }
    frame.classList.add('zoom-clip');
    const badge = Object.assign(document.createElement('span'), { className: 'zoom-badge' });
    frame.append(badge);
    badges.push(badge);
    return frame;
  });

  function apply() {
    x = clamp(x, 1 - zoom, 0);
    y = clamp(y, 1 - zoom, 0);
    for (const img of images) {
      img.style.setProperty('--zoom', String(zoom));
      img.style.setProperty('--zoom-x', `${x * 100}%`);
      img.style.setProperty('--zoom-y', `${y * 100}%`);
    }
    for (const frame of frames) frame.classList.toggle('zoomed', zoom > 1);
    for (const badge of badges) badge.textContent = `${Math.round(zoom * 10) / 10}×`;
  }

  for (const frame of frames) {
    frame.addEventListener('wheel', (e) => {
      if (painting()) return;
      e.preventDefault();
      // Zoom about the point under the pointer: it stays put, on both photos.
      const r = frame.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
      const next = clamp(zoom * (e.deltaY < 0 ? STEP : 1 / STEP), 1, MAX_ZOOM);
      x = px - (px - x) * (next / zoom);
      y = py - (py - y) * (next / zoom);
      zoom = next;
      apply();
    }, { passive: false });
    frame.addEventListener('pointerdown', (e) => {
      if (painting() || e.button !== 0 || zoom <= 1) return;
      const r = frame.getBoundingClientRect();
      drag = { id: e.pointerId, sx: e.clientX, sy: e.clientY, x, y, w: r.width, h: r.height };
      try { frame.setPointerCapture(e.pointerId); } catch { /* still moves inside the frame */ }
      document.body.classList.add('zoom-dragging');
      e.preventDefault();
    });
    frame.addEventListener('pointermove', (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      x = drag.x + (e.clientX - drag.sx) / drag.w;
      y = drag.y + (e.clientY - drag.sy) / drag.h;
      apply();
    });
    const stop = () => { drag = null; document.body.classList.remove('zoom-dragging'); };
    frame.addEventListener('pointerup', stop);
    frame.addEventListener('pointercancel', stop);
    frame.addEventListener('dblclick', () => { if (!painting()) reset(); });
  }

  function reset() {
    zoom = 1; x = 0; y = 0;
    apply();
  }
  apply();
  return {
    reset,
    get zoom() { return zoom; },
  };
}
