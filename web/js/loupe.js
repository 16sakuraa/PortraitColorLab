// Synchronised magnifier: hovering over either picture shows a round zoom
// window at the cursor, and the same spot magnified on the other picture,
// so original and retouched detail can be compared directly. Scroll to zoom.
//
// The lens is a CSS background of the picture's own image, so the browser
// colour-manages it exactly like the picture (Adobe RGB photos stay right).

export function attachLoupe(images, { zoom = 3, minZoom = 1.5 } = {}) {
  const state = { x: 0, y: 0, on: false, zoom };
  const parts = images.map((img) => {
    const lens = document.createElement('div');
    lens.className = 'loupe';
    lens.setAttribute('aria-hidden', 'true');
    const label = document.createElement('span');
    label.className = 'loupe-zoom';
    lens.append(label);
    img.parentElement.append(lens);
    return { img, lens, label };
  });

  // Up to twice the preview's own pixel size; beyond that it only gets blurry.
  const maxZoom = () => {
    const img = images[0];
    return Math.max(4, (2 * (img.naturalWidth || 1)) / (img.clientWidth || 1));
  };

  function draw() {
    for (const { img, lens, label } of parts) {
      if (!state.on || !img.currentSrc) { lens.classList.remove('on'); continue; }
      const w = img.clientWidth, h = img.clientHeight, s = lens.offsetWidth;
      const px = state.x * w, py = state.y * h, z = state.zoom;
      lens.style.left = `${img.offsetLeft + px}px`;
      lens.style.top = `${img.offsetTop + py}px`;
      lens.style.backgroundImage = `url("${img.currentSrc}")`;
      lens.style.backgroundSize = `${w * z}px ${h * z}px`;
      lens.style.backgroundPosition = `${s / 2 - px * z}px ${s / 2 - py * z}px`;
      label.textContent = `${Math.round(z * 10) / 10}×`;
      lens.classList.add('on');
    }
  }

  let frame = 0;
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(() => { frame = 0; draw(); });
  };

  for (const { img } of parts) {
    img.addEventListener('pointermove', (e) => {
      if (e.pointerType === 'touch') return;          // don't block scrolling on phones
      const r = img.getBoundingClientRect();
      state.x = (e.clientX - r.left) / r.width;
      state.y = (e.clientY - r.top) / r.height;
      state.on = state.x >= 0 && state.x <= 1 && state.y >= 0 && state.y <= 1;
      schedule();
    });
    img.addEventListener('pointerleave', () => { state.on = false; schedule(); });
    img.addEventListener('wheel', (e) => {
      if (!state.on) return;
      e.preventDefault();
      state.zoom = Math.min(maxZoom(), Math.max(minZoom, state.zoom * (e.deltaY < 0 ? 1.25 : 0.8)));
      schedule();
    }, { passive: false });
  }

  return {
    refresh: schedule,                                  // call after a picture changes
    hide() { state.on = false; schedule(); },
  };
}
