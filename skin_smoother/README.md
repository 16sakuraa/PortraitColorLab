# Skin Smoother

A small, self-contained face-skin smoothing tool. It reproduces the core effect
of portrait editors like DM Portrait using standard image processing — no dongle,
no external service, nothing taken from the original program.

## What it does

1. Finds the faces and builds a soft **skin mask** with a segmentation model, minus
   the eyes, eyebrows, lips and ears, so only skin is treated and edges stay sharp.
2. **Removes blemishes:** finds small red marks and fills them from the skin around
   them.
3. **Edge-preserving smoothing** flattens blotches and uneven tone, sized to the
   face and reading only from skin. **Lighting** set to hard light reaches the
   mottled light and shadow that flash leaves, and still keeps the face's shading.
4. **Keep texture** holds on to pore detail so the result doesn't look like plastic,
   and leaves real edges, lashes and stray hairs alone. **Reduce shine** tames the
   bright glints of oily skin. **Add pores** can add fine synthetic pore texture
   on top. **Print size** set to a small print keeps
   coarser grain, the kind that still shows in a 1-inch ID photo.
5. Blends the result back through the soft mask.

Everything runs on your computer's CPU, with no GPU or service. What AI does here (it
only finds faces and skin) and what it doesn't is in the main README, "AI in this app".

## Quick start (web UI)

Double-click **`run.bat`**. It installs anything missing, starts a small local
web server, and opens your browser. Drop in a photo, drag the sliders, and click
**Download**. Everything runs on your computer; no image is uploaded anywhere.

To start it manually instead:

```bash
python serve.py
```

The settings are in the panel on the right, in three steps. Click a step's tab, or
use Back and Next at the bottom of the panel:

1. **Skin:** Lighting, Print size and Face only, and the brush for the skin area
   (**Add skin**, **Remove skin**).
2. **Smooth:** Remove blemishes, Strength, Keep texture, Reduce shine, Add pores, and
   the **light brush**, **Nose line** and **Smooth again**.
3. **Finish:** **Undo smoothing**, **Restore**, **Heal** and **Remove spot**.

Everything is redone from the original photo each time, so you can go back to an
earlier step and change a setting without losing any brush work. New photo and
Download are at the top right; you can download at any step.

To compare closely, scroll on either photo: both zoom into the same spot. Drag to move
around, and double-click to see the whole photo again.

**History** at the bottom of the panel lists every change since the photo was opened:
each slider you let go, each setting, each brush stroke, and clearing the strokes.
Click any of them to go back to the photo as it was then; **Undo** and **Redo**
(Ctrl+Z, and Ctrl+Y or Ctrl+Shift+Z) step through the list one change at a time. After
going back, a new change replaces the ones below it. The list starts again with each
new photo.

Closing the browser tab or reloading the page while a photo is open asks first
("Leave site?"), so a retouch isn't lost by mistake. It doesn't ask when the photo as
it is was just downloaded. Browsers only ask once you've clicked or typed on the page.

The page is in Thai or English: press **TH** or **EN** at the top right. The choice is
remembered for next time (the first time, it follows the browser's language). The
words live in `web/js/i18n.js`, shared with the web app and bundled into the `.exe`.

### The brush

Press the brush button in the panel (its name changes with the step, for example
**Paint the skin area**). Skin that will be smoothed is tinted over the photo, and you
paint on it. The step tabs stay in view while you paint, and switching step switches
the brush's tools:

- **Add skin** smooths an area the mask left out, such as skin inside an eye or lip
  protection zone. It doesn't look for blemishes there.
- **Remove skin** (step 1) and **Restore** (step 3) are the same stroke: they bring
  back the original. At 100% Strength the area is kept exactly as it is (no
  smoothing, no blemish fills, no heals or light brush painted before it); lower
  Strength brings back part of it. Turn the tint off (`M`) to see the result.
- The **light brush** evens out light and shadow. Alt-click the skin whose colour you
  want (or press **Pick colour** and click it), then paint over blotches, dark circles,
  frown lines or the seam between light and shadow. Only the tone changes, so the
  skin's grain stays. It starts at 20% Strength and a fully soft edge: build it up
  gradually, and judge it from a distance.
- **Nose line** paints the same way along a straight line. Alt-click the brightest
  skin on the nose to pick the highlight colour (it's shared with the light brush),
  then click at the top of the nose bridge and at its tip. Make the brush as wide as
  the bridge. It starts at 25%. Esc cancels a line after its first click.
- **Smooth again** smooths a painted area once more, at full strength and with half
  the grain, so coarse grain, blotches or stubble shadow that are left become smooth
  and only the finest grain stays. It works even where the mask left the skin out;
  Strength says how much of it to use.
- **Undo smoothing** brings back the original's sharpness and texture but keeps the
  removed spots, heals and light brush: the last step of a retouch, to bring back
  what smoothing softened around the eyes, the nostrils, the jawline under the chin,
  or a mole. Strength works as for Restore.
- **Heal** covers a scar, a cut or a large spot. Alt-click clean skin to copy from
  (or press **Set source** and click it), then paint over the mark. The skin's
  detail is copied and its light and shading are matched to the skin around the
  stroke, so no seam shows. With **Aligned** on, the source keeps its distance from
  the brush from one stroke to the next; with it off, every stroke copies from the
  point you picked. Make the brush just cover the mark.
- **Remove spot** fills a spot from the skin around it in one click, for one the
  automatic blemish removal missed or a mole you want gone. It keeps the skin's fine
  grain. Make the brush a little bigger than the spot.

The step 2 and step 3 tools hide the tint while they're chosen. Heal and Remove spot
are applied before smoothing, so the patch is smoothed like the skin around it; Smooth
again after the smoothing, and the light brush and Nose line last. Each tool remembers
its own Strength and Softness.

Where strokes cross, the later one wins: a 40% Restore over a 100% one leaves 40%,
a Restore stroke over a heal, the light brush or Smooth again takes that share of it
back, Undo smoothing takes Smooth again back but keeps the light brush, and Undo
smoothing over a Restore brings the blemish fills back. Scroll to zoom.
Hold Space or the right mouse button to move around. Press **Done with brush** to go
back.

| Key | What it does |
|---|---|
| `[` and `]` | Brush size |
| `X` | Go through the step's brush tools |
| Alt+click | Heal: pick the skin to copy from. Light brush and Nose line: pick the colour |
| Esc | Cancel a nose line after its first click |
| `M` | Show or hide the tint |
| `0` | Fit the whole photo |
| Ctrl+Z | Undo the last change (any change, not only strokes) |
| Ctrl+Y or Ctrl+Shift+Z | Redo |

Each stroke reprocesses the photo. While you edit, both pages work on a preview at
most 1600 pixels on its long side, so a stroke comes back in a fraction of a second;
**Download** then processes the full-size photo (about a second on a 20-megapixel file
here). Zoomed in far, the preview looks softer than the download will. Strokes apply
to the full-size download and are cleared when you open another photo.

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
- **Reduce shine** tames bright glints on oily skin; dark pores and freckles stay
  (0 = off).
- **Add pores** adds fine synthetic pore texture (0 = none).
- **Lighting** "Hard or patchy (flash)" evens out mottled light and shadow more;
  keep "Soft (studio)" for softbox photos.
- **Print size** "Small print (1-inch ID)" keeps coarser grain for small prints; it
  looks coarser on screen and right in the print.
- **Face only** leaves out the neck and other body skin.
- Compare the Original (left) and Smoothed (right) panes live.
- **Save…** writes the full-resolution result.

The steps and the brush (skin area, light brush, Nose line, Smooth again, Undo
smoothing, Restore, Heal, Remove spot) aren't in this app: use the web UI for those.

## Share as a standalone app (no Python needed)

Download **`SkinSmoother.exe`** (about 96 MB) from the
[latest release](https://github.com/16sakuraa/PortraitColorLab/releases/latest)
and give that one file to anyone. They just double-click it. It starts, waits a few
seconds the first time, and opens the app in their browser. The models are bundled
inside the .exe, so nothing else is required.

Notes:
- Keep it in an ordinary folder such as Desktop or Documents, not Program Files,
  so it can update itself.
- On first run Windows may show a blue "Windows protected your PC" box because
  the file isn't code-signed. Click **More info -> Run anyway**. This is normal
  for unsigned apps.
- A console window stays open showing the address; closing it stops the app.

### Updates

From version 1.0.0, the .exe checks for a newer release each time it starts
(`updater.py`). If there is one, it downloads it in the background, checks its
size and SHA-256 checksum, and puts it in place of the old file. The page and the
console then say to close the console window and open the program again. Offline,
it just starts as usual.

- It only asks GitHub for the release list: no photos or personal details are sent.
- The checksums catch a damaged download. They're published in the same place as
  the .exe, so they can't protect against someone taking over the GitHub account;
  keep two-factor sign-in on for it.
- Copies from before version 1.0.0 have no updater: replace those by hand once.
- Set `SKINSMOOTHER_NO_UPDATE=1` to turn updates off.

### Releasing a new version

1. Bump `VERSION` in `version.py` and commit.
2. Tag that commit and push both:
   `git tag v1.0.1` then `git push origin main v1.0.1`.

GitHub Actions (`.github/workflows/exe-release.yml`) then runs the unit checks,
builds the .exe from the exact packages in `requirements-exe.txt`, tries the
built program (`exe_smoke.py`), and publishes it with its checksum file. Every copy
picks it up on its next start. A tag with a suffix, such as `v1.0.1-rc1`, makes a
pre-release that no copy picks up, for trying the build first.

### Building it yourself

Double-click **`build_exe.bat`**. It installs the pinned packages and writes
**`dist\SkinSmoother.exe`**. Check no Skin Smoother is running first. Its icon is
`icon.ico`, drawn by `make_icon.py` (run it again after changing the design).

## Command line (batch)

```bash
python smoother.py input.jpg output.jpg --blemish 1 --strength 0.6 --texture 0.6
```

`--blemish`, `--strength`, `--texture`, `--shine` and `--pores` are 0..1. `--print-size small`
keeps coarser grain for small prints such as ID photos. `--lighting hard` evens out
the mottled light and shadow of flash photos. `--face-only` leaves
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
  Google also reports lower accuracy on the darkest skin tones. Use the brush to
  correct anything it gets wrong.

`models/face_landmarks.onnx` is Google's MediaPipe Face Mesh V2 landmark model,
converted to ONNX (Apache 2.0, see `../THIRD_PARTY_NOTICES.md`). It outlines the eyes,
brows and lips, which smoothing leaves alone. A face it doesn't clearly see, such as a
strong profile, uses the older finder, as do all faces if the file is missing. Up to 30
faces use it, largest first.

Its outline of the face also keeps the ears out: the band just outside the outline,
from the top of the ear to the lobe. On a head turned away from the camera, the far
side's outline runs over the cheek and that ear is hidden, so only the near ear is cut.
Ears never cut into another face. A face without the outline keeps its ears: the face
detector's box stops well short of a turned face's cheek, so it can't say where the
ear starts.

## Face detection (optional but recommended)

Without a face model the tool falls back to a **skin-tone** mask, which smooths
skin-colored regions across the whole image. For proper face-aware masking, drop
OpenCV's YuNet model here:

```
skin_smoother/models/face_detection_yunet.onnx
```

`get_model.py` can fetch it (see that file). When the model is present the tool
uses it automatically.
