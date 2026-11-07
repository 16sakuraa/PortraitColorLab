# Skin Smoother

A small, self-contained face-skin smoothing tool. It reproduces the core effect
of portrait editors like DM Portrait using standard image processing — no dongle,
no external service, nothing taken from the original program.

## What it does

1. Finds the faces and builds a soft **skin mask** with a segmentation model, minus
   the eyes, eyebrows and lips, so only skin is treated and edges stay sharp.
2. **Removes blemishes:** finds small red marks and fills them from the skin around
   them.
3. **Edge-preserving smoothing** flattens blotches and uneven tone, sized to the
   face and reading only from skin.
4. **Keep texture** holds on to pore detail so the result doesn't look like plastic,
   and leaves real edges, lashes and stray hairs alone. **Add pores** can add fine
   synthetic pore texture on top.
5. Blends the result back through the soft mask.

Everything runs on your computer's CPU, with no GPU or service.

## Quick start (web UI)

Double-click **`run.bat`**. It installs anything missing, starts a small local
web server, and opens your browser. Drop in a photo, drag the sliders, and click
**Download**. Everything runs on your computer; no image is uploaded anywhere.

To start it manually instead:

```bash
python serve.py
```

### Fixing the mask by hand

If the automatic mask is wrong, press **Edit mask**. Skin that will be smoothed is
tinted over the photo, and you paint on it:

- **Protect** keeps an area exactly as it is: no smoothing, and no blemish fills.
- **Add skin** smooths an area the mask left out, such as skin inside an eye or lip
  protection zone. It doesn't look for blemishes there.

Scroll to zoom. Hold Space or the right mouse button to move around.

| Key | What it does |
|---|---|
| `[` and `]` | Brush size |
| `X` | Switch between Protect and Add skin |
| `M` | Show or hide the tint |
| `0` | Fit the whole photo |
| Ctrl+Z | Undo the last stroke |

Each stroke reprocesses the photo: about 1-2 s on a 20-megapixel file here, quicker
in the web app, which updates a preview. Strokes apply to the full-size download and
are cleared when you open another photo.

## Install (manual)

```bash
python -m pip install -r requirements.txt
```

## Desktop app (alternative to the web UI)

```bash
python app.py
```

- **Open image…** load a photo.
- **Remove blemishes** how much of each blemish fill to apply.
- **Strength** how much to smooth.
- **Keep texture** higher = more natural pores, lower = glossier.
- **Add pores** adds fine synthetic pore texture (0 = none).
- **Face only** leaves out the neck and other body skin.
- Compare the Original (left) and Smoothed (right) panes live.
- **Save…** writes the full-resolution result.

The brush for editing the mask isn't in this app: use the web UI for that.

## Share as a standalone app (no Python needed)

To give this to someone who doesn't have Python, build a single `.exe`:

1. Double-click **`build_exe.bat`** (or run `python -m PyInstaller ...` — the .bat
   does it for you).
2. Send them the one file it produces: **`dist\SkinSmoother.exe`** (~101 MB).

They just double-click it. It starts, waits a few seconds the first time, and
opens the app in their browser. The models are bundled inside the .exe, so nothing
else is required.

Notes:
- On first run Windows may show a blue "Windows protected your PC" box because
  the file isn't code-signed. Click **More info -> Run anyway**. This is normal
  for unsigned apps.
- A console window stays open showing the address; closing it stops the app.

## Command line (batch)

```bash
python smoother.py input.jpg output.jpg --blemish 1 --strength 0.6 --texture 0.6
```

`--blemish`, `--strength`, `--texture` and `--pores` are 0..1. `--face-only` leaves
out the neck and other body skin. If no output path is given, it writes
`input_smooth.jpg` next to the source.


## Output quality

Results are saved like the original: same colour profile (e.g. Adobe RGB
from a DSLR), camera data (EXIF), print DPI and JPEG compression settings, so
colours match and the file is about the same size. (Older versions dropped the
colour profile, which made Adobe RGB photos look washed out, and re-saved at a
lower JPEG quality.) The web page downloads as `<original name>_retouched.jpg`;
PNGs stay PNG.

## Remove blemishes

The **Remove blemishes** slider fills small red marks such as pimples and redness
with colour from the skin around them. It uses plain image processing, so it needs no
model and no training data. It runs before smoothing: heal first, then smooth.

- **Left alone on purpose:** anything also dark in the red channel, because redness
  stays bright there. That means moles, stubble and most brown marks. On darker skin,
  marks tend to be dark rather than red, so it removes much less there.
- **Only near faces:** arms and legs aren't searched.
- **No face:** nothing is removed.

### The optional network

An older method uses a small neural network trained on the FFHQR dataset (see
`../retouch_model`). It cleans more, including fine lines, under-eye darkness and
many freckles, but also changes moles, and most of the forehead and cheeks.

To use it, set `BLEMISH_ENGINE=network` before starting the app, or on the command
line add `--blemish-engine network`. It needs `models/blemish_ffhqr.onnx`.

**License:** FFHQR is CC BY-NC-SA 4.0 (non-commercial). Don't use the network for
paid work; retrain it on your own before/after photos for that.

## Skin mask

`models/selfie_multiclass_256x256.onnx` is Google's MediaPipe selfie segmentation
model, converted to ONNX (Apache 2.0, see `../THIRD_PARTY_NOTICES.md`). It separates
hair, face skin, body skin, clothes and accessories. It needs OpenCV 4.10 or newer.

- **Body skin** is kept only on a person with a detected face, and only near that
  face's skin colour, so skin-coloured walls and people standing apart stay out.
  Someone touching the subject counts as the same person.
- **Group photos:** faces close together share a run of the model, up to 8 runs. Other
  faces use the older oval mask, as do faces the model doesn't clearly see.
- **Not pixel-perfect:** Google says so too, so the edges are snapped to the photo.
  Google also reports lower accuracy on the darkest skin tones. Use **Edit mask** to
  correct anything it gets wrong.

## Face detection (optional but recommended)

Without a face model the tool falls back to a **skin-tone** mask, which smooths
skin-colored regions across the whole image. For proper face-aware masking, drop
OpenCV's YuNet model here:

```
skin_smoother/models/face_detection_yunet.onnx
```

`get_model.py` can fetch it (see that file). When the model is present the tool
uses it automatically.
