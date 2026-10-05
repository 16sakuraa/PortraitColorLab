# Retouch model (blemish removal)

Trains the small network behind SkinSmoother's **Remove blemishes** slider,
from before/after pairs of retouched faces.

Current training data: 3,000 pairs from **FFHQR** (professionally retouched
FFHQ faces, 1024x1024). License CC BY-NC-SA 4.0: **non-commercial only**.

## Pipeline

| Step | Command | What it does |
|---|---|---|
| 1 | `python download.py URL OUT --bytes N` | Resumable multi-connection download (the FFHQR server is slow per connection) |
| 2 | `python prepare_data.py extract-retouched` | Pulls complete images out of the partially downloaded FFHQR tar |
| 3 | `python prepare_data.py extract-original` | Pulls the matching originals out of the FFHQ shards |
| 4 | `python prepare_data.py index` | Verifies every pair (size, alignment) and records where the edits are |
| 5 | `python train.py --name run1 --steps 30000` | Trains on the RTX GPU; log, checkpoints and preview images in `runs/run1/` |
| 6 | `python export_onnx.py runs/run1/best.pt` | Exports `retouch.onnx`, checks OpenCV runs it, times it on CPU |
| 7 | `python demo.py runs/run1/retouch.onnx` | Before/after sheets through the real SkinSmoother code |
| 8 | copy `retouch.onnx` to `../skin_smoother/models/blemish_ffhqr.onnx` | Installs it in the app |

Use the venv: `.venv\Scripts\python.exe` (PyTorch with CUDA for the RTX 50-series).

## How it works

- `model.py`: 1.6 M-parameter U-Net that predicts a *correction* added to the
  photo (starts at zero = no change). Only uses layers OpenCV's DNN runs.
- `train.py`: FFHQR edits are sparse (median 4.6% of pixels), so crops are
  centred on edited areas 75% of the time and the loss counts edited pixels
  11x. Progress is measured on 100 held-out faces:
  - `fix_pct`: share of the retoucher's change reproduced inside edited areas
  - `out_err`: damage to pixels the retoucher left alone (0..255 scale)
- `check_alignment.py`: confirms SkinSmoother's face crop (from YuNet points)
  matches the FFHQ framing the model was trained on.

## Retraining on your own photos

Put pairs in `data/pairs/original/<id>.webp` (or change the extension in
`prepare_data.py` / `train.py`) and `data/pairs/retouched/<id>.png`. Pairs must be
pixel-aligned and FFHQ-framed 1024x1024 face crops: run `blemish.ffhq_quad` on
each raw photo and warp both raw and edited with the same transform. Then run
steps 4-8. Fine-tuning from the FFHQR model needs far fewer pairs and steps:
`python train.py --name mine --init runs/run1/best.pt --steps 5000 --lr 1e-4`.
