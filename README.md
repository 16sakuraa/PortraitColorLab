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
  hair, glasses, clothes and backdrops. Eyes, brows, lips and ears are protected. Tick
  **Face only** to leave out the neck and body.
- **Remove blemishes:** finds small red marks with image processing and fills
  them from the skin around them. Moles and stubble are left alone.
- **Strength and Keep texture:** smooth the skin's tone, and keep its shape and
  pores. Sizes follow the face, so results don't depend on the crop.
- **Reduce shine:** tames the bright glints of oily skin: grain that is brighter
  than its surroundings, nearly colourless, and on the brighter parts of the face.
  Dark pores, freckles and coloured marks are left alone. Off by default.
- **Add pores:** adds fine synthetic pore texture, for soft skin or strong
  smoothing. Off by default.
- **Lighting:** for hard or patchy light (on-camera flash, a bare window), evens
  out mottled light and shadow a few millimetres across, while keeping the face's
  shading so it doesn't look puffy. It doesn't blend a sharp shadow edge. Soft
  studio light, the default, needs nothing extra.
- **Print size:** for a small print such as a 1-inch ID photo, keeps coarser skin
  grain, because grain finer than a printed dot disappears at that size. The
  print then keeps as much texture as Keep texture asks for. It looks coarser on
  screen.
- **Brush:** paint **Restore** to bring back the original, all of it or part
  (Strength), for example around the eyes or on a mole, or **Add skin** to smooth
  an area the automatic mask left out. In the web app and the `.exe`.

The settings sit in a panel on the right of the window, grouped as Retouch, Photo
and Brush; on a narrow screen they move above the photos.

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
