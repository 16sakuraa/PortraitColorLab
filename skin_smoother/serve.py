"""
Local web UI for the skin smoother.

Starts a small web server (Python standard library only) and opens a browser
page where you can upload a photo, adjust Strength / Keep-texture, compare
before and after, and download the result. Full-resolution processing happens
on the server; nothing leaves your computer.

Run:  python serve.py   (or double-click run.bat)
"""

from __future__ import annotations
import base64
import json
import os
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import smoother
import blemish
import photo_io

# The blemish network runs once per photo; slider moves reuse its correction.
_blemish_cache: dict = {}
_blemish_lock = threading.Lock()


def _blemish_correction(key: int, img: np.ndarray, faces):
    with _blemish_lock:
        if key not in _blemish_cache:
            _blemish_cache.clear()
            _blemish_cache[key] = blemish.correction(img, faces)
        return _blemish_cache[key]


PAGE ="""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Skin Smoother</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 system-ui, Segoe UI, Arial, sans-serif;
         background: #f4f5f7; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) {
    body { background: #16181c; color: #e8e8e8; }
    .panel, header { background: #22252b !important; }
    .drop { border-color: #3a3f47 !important; }
  }
  header { background: #fff; padding: 14px 20px; border-bottom: 1px solid #0001;
           display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  header h1 { font-size: 17px; margin: 0; font-weight: 650; }
  .controls { display: flex; gap: 22px; align-items: center; flex-wrap: wrap; margin-left: auto; }
  .ctl { display: flex; flex-direction: column; gap: 2px; min-width: 170px; }
  .ctl label { font-size: 12px; opacity: .75; display: flex; justify-content: space-between; }
  input[type=range] { width: 180px; accent-color: #4b7bec; }
  button { font: inherit; padding: 8px 16px; border: 0; border-radius: 8px;
           background: #4b7bec; color: #fff; cursor: pointer; }
  button.secondary { background: #6b7280; }
  button:disabled { opacity: .45; cursor: default; }
  main { padding: 18px; }
  .drop { border: 2px dashed #c7ccd4; border-radius: 12px; padding: 40px;
          text-align: center; background: #fff8; cursor: pointer; }
  .drop.small { padding: 14px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 16px; }
  .panel { background: #fff; border-radius: 12px; padding: 10px; }
  .panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
              opacity: .6; margin: 4px 6px 10px; }
  .panel img { width: 100%; border-radius: 8px; display: block; }
  .status { font-size: 13px; opacity: .8; margin-top: 10px; min-height: 18px; }
  .hidden { display: none; }
  @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>Skin Smoother</h1>
  <div class="controls">
    <div class="ctl">
      <label>Remove blemishes <span id="bval">100</span></label>
      <input id="blemish" type="range" min="0" max="100" value="100">
    </div>
    <div class="ctl">
      <label>Strength <span id="sval">60</span></label>
      <input id="strength" type="range" min="0" max="100" value="60">
    </div>
    <div class="ctl">
      <label>Keep texture <span id="tval">35</span></label>
      <input id="texture" type="range" min="0" max="100" value="35">
    </div>
    <label style="display:flex;align-items:center;gap:6px;font-size:13px;white-space:nowrap;">
      <input id="gpu" type="checkbox" checked> Use GPU
    </label>
    <button id="download" disabled>Download</button>
    <button id="reset" class="secondary hidden">New image</button>
  </div>
</header>

<main>
  <div id="drop" class="drop">
    <p><strong>Click to choose a photo</strong> or drag it here</p>
    <p style="opacity:.6;font-size:13px">JPG, PNG, BMP, WEBP — stays on your computer</p>
    <input id="file" type="file" accept="image/*" class="hidden">
  </div>

  <div id="workspace" class="hidden">
    <div class="grid">
      <div class="panel"><h2>Original</h2><img id="before"></div>
      <div class="panel"><h2>Smoothed</h2><img id="after"></div>
    </div>
    <div class="status" id="status"></div>
  </div>
</main>

<script>
const $ = id => document.getElementById(id);
let original = null;      // data URL of the source image
let originalName = 'photo';
let busy = false, pending = false;

const modelNote = () => {};

function setStatus(t){ $('status').textContent = t; }

$('drop').addEventListener('click', () => $('file').click());
$('drop').addEventListener('dragover', e => { e.preventDefault(); $('drop').style.opacity=.7; });
$('drop').addEventListener('dragleave', () => $('drop').style.opacity=1);
$('drop').addEventListener('drop', e => {
  e.preventDefault(); $('drop').style.opacity=1;
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});
$('file').addEventListener('change', e => { if(e.target.files[0]) loadFile(e.target.files[0]); });

function loadFile(f){
  const r = new FileReader();
  r.onload = () => {
    original = r.result;
    originalName = f.name.replace(/\\.[^.]+$/, '') || 'photo';
    $('before').src = original;
    $('drop').classList.add('hidden');
    $('workspace').classList.remove('hidden');
    $('reset').classList.remove('hidden');
    process();
  };
  r.readAsDataURL(f);
}

$('reset').addEventListener('click', () => {
  original = null;
  $('drop').classList.remove('hidden');
  $('workspace').classList.add('hidden');
  $('reset').classList.add('hidden');
  $('download').disabled = true;
  $('file').value = '';
});

const valueLabel = {strength: 'sval', texture: 'tval', blemish: 'bval'};
['strength','texture','blemish'].forEach(id => {
  $(id).addEventListener('input', () => {
    $(valueLabel[id]).textContent = $(id).value;
    process();
  });
});
$('gpu').addEventListener('change', process);

let debounce = null;
function process(){
  if(!original) return;
  clearTimeout(debounce);
  debounce = setTimeout(runProcess, 180);
}

async function runProcess(){
  if(!original) return;
  if(busy){ pending = true; return; }
  busy = true; setStatus('Processing…'); $('download').disabled = true;
  try {
    const res = await fetch('/process', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        image: original,
        strength: +$('strength').value / 100,
        texture: +$('texture').value / 100,
        blemish: +$('blemish').value / 100,
        gpu: $('gpu').checked
      })
    });
    const data = await res.json();
    if(data.error){ setStatus('Error: ' + data.error); }
    else {
      $('after').src = data.image;
      $('download').disabled = false;
      $('download').onclick = () => {
        const a = document.createElement('a');
        a.href = data.image; a.download = originalName + '_retouched' + (data.ext || '.jpg'); a.click();
      };
      if(!data.gpu_available){ $('gpu').checked = false; $('gpu').disabled = true; }
      if(!data.blemish_model){
        $('blemish').disabled = true; $('bval').textContent = 'model missing';
      }
      setStatus(data.faces + ' face(s) detected'
        + (data.model ? '' : ' — skin-tone mode (no face model installed)')
        + (data.blemish_faces ? '  ·  blemishes healed on ' + data.blemish_faces + ' face(s)' : '')
        + '  ·  ' + (data.gpu ? 'GPU (OpenCL)' : 'CPU')
        + '  ·  ' + data.w + '×' + data.h);
    }
  } catch(err){ setStatus('Error: ' + err); }
  busy = false;
  if(pending){ pending = false; runProcess(); }
}
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
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/process":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            img, info = photo_io.decode(_decode_data_url(payload["image"]))
            if img is None:
                raise ValueError("could not decode image")
            strength = float(payload.get("strength", 0.6))
            texture = float(payload.get("texture", 0.35))
            blemish_amt = float(payload.get("blemish", 0.0))
            gpu_eff = smoother.set_opencl(bool(payload.get("gpu", False)))

            mask = smoother.build_skin_mask(img)
            faces = smoother._detect_faces(img)
            n_faces = 0 if faces is None else len(faces)

            # Heal blemishes first, then smooth (the usual retouching order).
            healed, blemish_faces = img, 0
            if blemish_amt > 0 and blemish.have_model():
                corr, blemish_faces = _blemish_correction(hash(payload["image"]), img, faces)
                healed = blemish.apply(img, corr, blemish_amt)
            out = smoother.smooth_face(healed, strength=strength, texture=texture, mask=mask)

            # Save like the original: same colour profile, EXIF, DPI and JPEG
            # quality, so colours and file size match what was uploaded.
            fmt = photo_io.output_format(info)
            b64 = base64.b64encode(photo_io.encode(out, info, fmt)).decode("ascii")
            mime, ext = ("image/png", ".png") if fmt == "PNG" else ("image/jpeg", ".jpg")
            resp = {
                "image": f"data:{mime};base64," + b64,
                "ext": ext,
                "faces": n_faces,
                "model": smoother.have_face_model(),
                "blemish_model": blemish.have_model(),
                "blemish_faces": blemish_faces,
                "gpu": gpu_eff,
                "gpu_available": smoother.opencl_available(),
                "w": img.shape[1], "h": img.shape[0],
            }
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
    print("  Skin Smoother is running.")
    print(f"  Open:  {url}")
    print("  Close this window to stop.")
    if not smoother.have_face_model():
        print("  (Tip: run get_model.py for face-aware masking.)")
    print(f"  GPU (OpenCL): {'available' if smoother.opencl_available() else 'not available (CPU only)'}")
    print("=" * 48)
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
