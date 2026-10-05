# PortraitColorLab

Portrait retouching for studio headshots: learned **blemish removal** and
face-aware **skin smoothing**, as a desktop app and as a web app that runs
entirely in the browser.

**Web app:** https://16sakuraa.github.io/PortraitColorLab/

Photos are never uploaded: all processing happens on the visitor's own
computer. The results keep the original's colour profile (e.g. Adobe RGB from
a DSLR), camera data and print resolution.

## What's here

| Folder | What |
|---|---|
| `web/` | Browser version (static site, deployed by `.github/workflows/pages.yml`). Runs the same steps as the desktop app with OpenCV.js and ONNX Runtime Web; uses the graphics chip (WebGPU) when available, otherwise the CPU. |
| `skin_smoother/` | Desktop app (Python): web UI (`run.bat` / `serve.py`), Tk app (`app.py`), command line (`smoother.py`), standalone `.exe` build (`build_exe.bat`). |
| `retouch_model/` | Training code for the blemish model (PyTorch, RTX GPU). Training data is not included. |

## Web app vs desktop app

Both run the same algorithm. `web/tools/parity.py` compares them stage by
stage on a test set; on the last run every final image matched the desktop
within 69-90 dB PSNR (differences of one brightness level on 0.02-0.23 % of
pixels), on both WebGPU and the CPU fallback. Saved JPEGs use the original's
estimated quality and colour sampling, so file size and quality stay close to
the original.

Typical speed on a 24-megapixel photo: about 3 s to analyse and 3 s to save
with WebGPU. Without WebGPU the blemish model takes roughly 3-4 s per face.

## Running the web app locally

```
python web/tools/devserver.py
```

then open http://127.0.0.1:8765/. (Opening `index.html` as a file does not
work: browsers block the background worker and model files there.)

## Licence

The source code in this repository is released under the **MIT licence**
(`LICENSE`). That does not cover:

- the blemish model (`*/models/blemish_ffhqr.onnx`), trained on FFHQR/FFHQ and
  licensed **CC BY-NC-SA 4.0 (non-commercial)**. Do not use it for paid work;
  retrain on your own before/after photos for that (`retouch_model/README.md`);
- the YuNet face model and the libraries bundled in `web/lib/`, which keep
  their own licences.

See `THIRD_PARTY_NOTICES.md` for details.
