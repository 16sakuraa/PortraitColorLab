# Skin Smoother

A small, self-contained face-skin smoothing tool. It reproduces the core effect
of portrait editors like DM Portrait using standard image processing — no dongle,
no external service, nothing taken from the original program.

## What it does

1. Finds the face and builds a soft **skin mask** (face oval minus eyes,
   eyebrows, mouth), so only skin is smoothed and edges stay sharp.
2. **Edge-preserving smoothing** flattens blotches and uneven tone.
3. **Texture retention** adds back a little of the original pore detail so the
   result doesn't look like plastic.
4. Blends the smoothed skin back through the feathered mask.

## Quick start (web UI)

Double-click **`run.bat`**. It installs anything missing, starts a small local
web server, and opens your browser. Drop in a photo, drag the sliders, and click
**Download**. Everything runs on your computer; no image is uploaded anywhere.

To start it manually instead:

```bash
python serve.py
```

## Install (manual)

```bash
python -m pip install -r requirements.txt
```

## Desktop app (alternative to the web UI)

```bash
python app.py
```

- **Open image…** load a photo.
- **Strength** how much to smooth.
- **Keep texture** higher = more natural pores, lower = glossier.
- Compare the Original (left) and Smoothed (right) panes live.
- **Save…** writes the full-resolution result.

## Share as a standalone app (no Python needed)

To give this to someone who doesn't have Python, build a single `.exe`:

1. Double-click **`build_exe.bat`** (or run `python -m PyInstaller ...` — the .bat
   does it for you).
2. Send them the one file it produces: **`dist\SkinSmoother.exe`** (~74 MB).

They just double-click it. It starts, waits a few seconds the first time, and
opens the app in their browser. The face model is bundled inside the .exe, so
nothing else is required.

Notes:
- On first run Windows may show a blue "Windows protected your PC" box because
  the file isn't code-signed. Click **More info -> Run anyway**. This is normal
  for unsigned apps.
- A console window stays open showing the address; closing it stops the app.

## Command line (batch)

```bash
python smoother.py input.jpg output.jpg --strength 0.6 --texture 0.35
```

Strength and texture are 0..1. If no output path is given, it writes
`input_smooth.jpg` next to the source.

## GPU acceleration (optional)

The smoothing filter can run on the GPU through OpenCL. In the web UI and the
desktop app there's a **Use GPU** checkbox; on the command line add `--gpu`.
It speeds up large images (roughly 2-3x in testing) and produces the same
result. If the machine has no OpenCL-capable GPU the option is disabled
automatically and everything runs on the CPU. Face detection always runs on the
CPU in this OpenCV build.

## Output quality

Results are saved like the original: same colour profile (e.g. Adobe RGB
from a DSLR), camera data (EXIF), print DPI and JPEG compression settings, so
colours match and the file is about the same size. (Older versions dropped the
colour profile, which made Adobe RGB photos look washed out, and re-saved at a
lower JPEG quality.) The web page downloads as `<original name>_retouched.jpg`;
PNGs stay PNG.

## Remove blemishes (learned model)

The **Remove blemishes** slider runs a small neural network that heals spots,
pimples, fine lines, under-eye darkness and stray hairs the way a professional
retoucher would. It was trained on the FFHQR dataset (see `../retouch_model`)
and runs on the CPU through OpenCV, so no GPU or extra install is needed
(about 0.3 s per face on a desktop CPU).

- It needs `models/blemish_ffhqr.onnx`. Without it the slider is greyed out.
- **Group photos:** every face is processed in its own crop; where people
  stand close and crops overlap, the corrections are blended (never doubled).
  Tiny faces (under ~50 px wide) are skipped.
- **No face** (landscapes, backdrops, products): the photo is left untouched.
- **Known quirk:** it learned to remove stray hairs, so thin lines touching
  the face (earphone cables, fine necklaces, glasses chains) may be faded.
  Lower the slider if that happens.
- It runs before smoothing, so the two combine: heal first, then smooth.
- Command line: `python smoother.py in.jpg out.jpg --blemish 1.0`
- **License:** FFHQR is CC BY-NC-SA 4.0 (non-commercial). Don't use this
  model for paid work; retrain on your own before/after photos for that.

## Face detection (optional but recommended)

Without a face model the tool falls back to a **skin-tone** mask, which smooths
skin-colored regions across the whole image. For proper face-aware masking, drop
OpenCV's YuNet model here:

```
skin_smoother/models/face_detection_yunet.onnx
```

`get_model.py` can fetch it (see that file). When the model is present the tool
uses it automatically.
