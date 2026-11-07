# PortraitColorLab

Portrait retouching for studio headshots: **blemish removal** and face-aware
**skin smoothing**, as a desktop app and as a web app that runs entirely in the
browser.

**Web app:** https://16sakuraa.github.io/PortraitColorLab/

Photos are never uploaded: all processing happens on the visitor's own
computer. The results keep the original's colour profile (e.g. Adobe RGB from
a DSLR), camera data and print resolution.

## What it does

- **Skin mask:** a segmentation model finds face and body skin, and leaves out
  hair, glasses, clothes and backdrops. Eyes, brows and lips are protected. Tick
  **Face only** to leave out the neck and body.
- **Remove blemishes:** finds small red marks with image processing and fills
  them from the skin around them. Moles and stubble are left alone.
- **Strength and Keep texture:** smooth the skin's tone, and keep its shape and
  pores. Sizes follow the face, so results don't depend on the crop.
- **Add pores:** adds fine synthetic pore texture, for soft skin or strong
  smoothing. Off by default.
- **Edit mask:** a brush for when the automatic mask is wrong: paint **Protect**
  to keep an area as it is, or **Add skin** to smooth it. In the web app and the
  `.exe`.

## What's here

| Folder | What |
|---|---|
| `web/` | Browser version (static site, deployed by `.github/workflows/pages.yml`). Runs the same steps as the desktop app with OpenCV.js and ONNX Runtime Web; uses the graphics chip (WebGPU) when available, otherwise the CPU. |
| `skin_smoother/` | Desktop app (Python): web UI (`run.bat` / `serve.py`), Tk app (`app.py`), command line (`smoother.py`), standalone `.exe` build (`build_exe.bat`). The `.exe` is published on the [Releases page](https://github.com/16sakuraa/PortraitColorLab/releases/latest) by `.github/workflows/exe-release.yml` and updates itself. |
| `retouch_model/` | Training code for the optional blemish network (PyTorch, RTX GPU). Training data is not included. |

## Web app vs desktop app

Both run the same algorithm. `web/tools/parity.py` compares them stage by stage
on a test set; on the last run the skin masks differed by 0.00003 or less on
average and every final image matched the desktop at 80 dB PSNR or better, on both
WebGPU and the CPU fallback. Saved JPEGs use the original's quality settings and colour sampling, so
file size and quality stay close to the original.

Typical speed on a 20-megapixel photo: about 3 s to analyse and 4 s to save. The
first visit downloads about 60 MB, which is then cached.

## Running the web app locally

```
python web/tools/devserver.py
```

then open http://127.0.0.1:8765/. (Opening `index.html` as a file does not
work: browsers block the background worker and model files there.)

## Licence

The source code in this repository is released under the **MIT licence**
(`LICENSE`). That does not cover:

- the optional blemish network (`*/models/blemish_ffhqr.onnx`), trained on
  FFHQR/FFHQ and licensed **CC BY-NC-SA 4.0 (non-commercial)**. The default blemish
  removal doesn't use it. It only runs if you ask for it (`?blemish=network` on the
  web, `BLEMISH_ENGINE=network` on the desktop). Do not use it for paid work; retrain
  on your own before/after photos for that (`retouch_model/README.md`);
- the YuNet face model, Google's selfie segmentation and face landmark models (Apache
  2.0, converted to ONNX) and the libraries bundled in `web/lib/`, which keep their own
  licences.

See `THIRD_PARTY_NOTICES.md` for details.
