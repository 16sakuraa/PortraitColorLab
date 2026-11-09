# PortraitColorLab

Portrait retouching for studio headshots: **blemish removal** and face-aware
**skin smoothing**, as a desktop app and as a web app that runs entirely in the
browser.

**Web app:** https://16sakuraa.github.io/PortraitColorLab/

Photos are never uploaded: all processing happens on the visitor's own
computer. The results keep the original's colour profile (e.g. Adobe RGB from
a DSLR), camera data and print resolution.

## How to use

1. **Open it.** Use the [web app](https://16sakuraa.github.io/PortraitColorLab/) in a
   current desktop browser, or on Windows download **SkinSmoother.exe** from the
   [latest release](https://github.com/16sakuraa/PortraitColorLab/releases/latest) and
   double-click it. The `.exe` opens the same page in your browser; keep its console
   window open while you work. The first time, Windows may say "Windows protected your
   PC" because the file isn't code-signed: click **More info**, then **Run anyway**.
2. **Drop in a photo**, a JPEG or PNG. The app finds the faces and the skin by itself.
3. **Go through the three steps** in the panel on the right: **1 Skin**, **2 Smooth**,
   **3 Finish**. The sliders do most of the work; the brushes fix what the automatic
   pass gets wrong or misses.
4. **Check before you save.** Scroll on either photo to zoom the before / after pair,
   and look at the eyes, hairline, glasses, ears and jaw, where the automatic skin area
   is most likely to be off.
5. **Download.** You get a new file, `<name>_retouched.jpg`; the original is never
   changed.

For official ID or passport photos, check the issuing office's rules: many don't accept
photos edited to change how someone looks. Removing passing marks and evening out the
skin is usually the most they allow.

## AI in this app

- **AI finds, it doesn't paint.** Three small neural networks run on your own computer,
  only to find things: the YuNet face detector finds the faces, Google's MediaPipe selfie
  segmentation tells skin from hair, clothes and background, and Google's Face Mesh V2
  outlines the eyes, brows and lips so smoothing leaves them alone.
- **The retouch itself is ordinary image processing.** Blemish removal, smoothing,
  Reduce shine, Lighting and every brush only blend and move pixels of the same photo;
  Add pores adds a fixed, computed grain pattern. There is no generative AI: nothing is
  made up, the shape of the face and its features stay as they are, and the same photo
  with the same settings gives the same result each time. The app can't remove a beard,
  change hair or reshape a face.
- **Your photos stay with you.** The models are inside the `.exe`, or downloaded once by
  the web page and cached. Photos are never uploaded and never used to train anything.
  The `.exe`'s only request over the internet is its update check, which asks GitHub for
  the list of releases.
- **The models can be wrong.** The skin area can take in a strand of hair or miss skin
  behind glasses or in deep shadow, and Google reports lower accuracy on the darkest
  skin tones. Check the result and correct it with the brush (**Add skin**,
  **Remove skin**, **Restore**).
- **One exception, off by default:** an optional network, trained on the FFHQR dataset,
  can do the blemish removal instead, and that one does paint its own fix. It only runs
  if you switch it on, and its licence doesn't allow paid work (see [Licence](#licence)).
- **Built with AI help.** The code was written with an AI coding assistant, Anthropic's
  Claude in Claude Code, under the maintainer's direction; every commit made that way
  says so in a `Co-Authored-By` line. The unit checks (`skin_smoother/test_retouch.py`)
  and the web / desktop comparison (`web/tools/parity.py`) test what it wrote.

Model sources and licences are in `THIRD_PARTY_NOTICES.md`.

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
  (Strength), for example on a mole; **Undo smoothing** to bring back the original's
  sharpness, around the eyes, the nostrils or the jawline, while keeping the removed
  spots and heals; or **Add skin** to smooth an area the automatic mask left out.
  **Heal** covers a scar, cut or large spot
  with skin copied from a clean area you Alt-click, matched to the light and
  shading where you paint (seamless cloning). **Remove spot** fills a spot the
  automatic removal missed from the skin around it, in one click. The **light
  brush** paints a colour picked from the skin (Alt-click) over blotches, dark
  circles and seams between light and shadow, changing only the tone so the skin's
  grain stays (frequency-separation painting); **Nose line** paints the same way along
  a line between two clicks, such as a highlight down the nose. **Smooth again**
  smooths a painted area once more, at full strength with half the grain, for coarse
  grain, blotches or stubble shadow that are left. In the web app and the `.exe`.

The settings sit in a panel on the right of the window, in three steps: **1 Skin**
(Lighting, Print size, Face only, and the brush for the skin area), **2 Smooth** (the
sliders, the light brush, Nose line and Smooth again) and **3 Finish** (Undo
smoothing, Restore, Heal, Remove spot). Tabs and Back / Next move between them; everything is redone from the
original, so going back to change a setting keeps all brush work. On a narrow screen
the panel moves above the photos. Scroll on either photo of the before / after pair to
zoom both into the same spot; drag to move around, double-click to see the whole photo.
**History** at the bottom of the panel lists every change (sliders, settings, brush
strokes); click one to go back to it, or use Undo / Redo (Ctrl+Z, Ctrl+Y). Closing or
reloading the page with work that hasn't been downloaded asks first.

Both the web app and the `.exe` are in **Thai and English**: the **TH | EN** switch at
the top right changes every label, hint and message, and the browser remembers the
choice. The first visit follows the browser's language. The words are in
`web/js/i18n.js`, which both pages share.

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
