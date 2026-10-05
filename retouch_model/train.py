"""
Train RetouchNet on the FFHQR before/after pairs.

FFHQR edits are sparse (most pixels are untouched), so a plain pixel loss
rewards a model that changes nothing. Two things counter that:
  * crops are biased towards places the retoucher actually edited, and
  * the L1 loss is up-weighted inside the edited area.

Progress is judged on held-out faces at full 1024x1024 with two numbers,
always against the "do nothing" baseline:
  fix%      share of the retoucher's change the model reproduced inside the
            edited area (0% = did nothing, 100% = matches the retoucher)
  out_err   error on pixels the retoucher left alone (must stay ~0: this is
            how much the model damages good skin, hair, eyes, background)

Usage:
  python train.py --name smoke --limit 200 --steps 300        # quick check
  python train.py --name run1 --steps 20000                   # real run
"""

from __future__ import annotations
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import RetouchNet, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(HERE, "data", "pairs")
SIZE = 1024


def _paths(img_id: int) -> tuple[str, str]:
    return (os.path.join(PAIRS, "original", f"{img_id:05d}.webp"),
            os.path.join(PAIRS, "retouched", f"{img_id:05d}.png"))


def _load_pair(img_id: int) -> tuple[np.ndarray, np.ndarray]:
    po, pr = _paths(img_id)
    return cv2.imread(po, cv2.IMREAD_COLOR), cv2.imread(pr, cv2.IMREAD_COLOR)


def edit_mask(o: np.ndarray, r: np.ndarray, thr: int, grow: int = 7) -> np.ndarray:
    """1 where the retoucher changed the pixel (slightly grown), else 0. uint8."""
    a = np.abs(r.astype(np.int16) - o.astype(np.int16)).max(axis=-1) > thr
    m = a.astype(np.uint8)
    if grow > 1:
        m = cv2.dilate(m, np.ones((grow, grow), np.uint8))
    return m


class PairCrops(Dataset):
    """Each item = several crops from one image pair (decoding is the slow
    part, so every decoded pair is used more than once)."""

    def __init__(self, pairs, ids, crop, crops_per_image, p_edit, thr):
        self.by_id = {p["id"]: p for p in pairs}
        self.ids = list(ids)
        self.crop = crop
        self.cpi = crops_per_image
        self.p_edit = p_edit
        self.thr = thr

    def __len__(self):
        return len(self.ids)

    def _window(self, rng, cells, size):
        c = self.crop
        if cells and rng.random() < self.p_edit:
            cy, cx = cells[rng.integers(len(cells))]
            j = size // 3
            cy += int(rng.integers(-j, j + 1))
            cx += int(rng.integers(-j, j + 1))
        else:
            cy, cx = (int(v) for v in rng.integers(size // 2, SIZE - size // 2 + 1, size=2))
        y0 = int(np.clip(cy - size // 2, 0, SIZE - size))
        x0 = int(np.clip(cx - size // 2, 0, SIZE - size))
        return y0, x0

    def __getitem__(self, i):
        rng = np.random.default_rng()
        p = self.by_id[self.ids[i]]
        o, r = _load_pair(p["id"])
        xs, ys, ms = [], [], []
        c = self.crop
        for _ in range(self.cpi):
            # Scale jitter: real faces won't be aligned at exactly FFHQ scale.
            s = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25))))
            size = int(np.clip(round(c * s), 64, SIZE))
            y0, x0 = self._window(rng, p["cells"], size)
            oc = o[y0:y0 + size, x0:x0 + size]
            rc = r[y0:y0 + size, x0:x0 + size]
            if size != c:
                interp = cv2.INTER_AREA if size > c else cv2.INTER_CUBIC
                oc = cv2.resize(oc, (c, c), interpolation=interp)
                rc = cv2.resize(rc, (c, c), interpolation=interp)
            if rng.random() < 0.5:
                oc, rc = oc[:, ::-1], rc[:, ::-1]
            m = edit_mask(oc, rc, self.thr)
            # Same exposure / white-balance jitter on both sides.
            gain = (rng.uniform(0.85, 1.15) * rng.uniform(0.95, 1.05, size=3)).astype(np.float32)
            of = np.clip(oc.astype(np.float32) / 255.0 * gain, 0, 1)
            rf = np.clip(rc.astype(np.float32) / 255.0 * gain, 0, 1)
            xs.append(of.transpose(2, 0, 1))
            ys.append(rf.transpose(2, 0, 1))
            ms.append(m[None].astype(np.float32))
        return (torch.from_numpy(np.ascontiguousarray(np.stack(xs))),
                torch.from_numpy(np.ascontiguousarray(np.stack(ys))),
                torch.from_numpy(np.stack(ms)))


def _worker_init(_):
    cv2.setNumThreads(0)


def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bgr.astype(np.float32).transpose(2, 0, 1) / 255.0)[None]


@torch.no_grad()
def evaluate(model, val_pairs, thr, device):
    """Full-image metrics on held-out faces, model vs do-nothing baseline."""
    model.eval()
    in_err_id = in_err_m = out_err_m = 0.0
    n_in = n_out = 0.0
    for o, r in val_pairs:
        m = torch.from_numpy(edit_mask(o, r, thr)).to(device).float()[None, None]
        x = _to_tensor(o).to(device)
        y = _to_tensor(r).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out = out.float().clamp(0, 1)
        e_model = (out - y).abs().mean(1, keepdim=True)
        e_ident = (x - y).abs().mean(1, keepdim=True)
        in_err_id += float((e_ident * m).sum())
        in_err_m += float((e_model * m).sum())
        out_err_m += float((e_model * (1 - m)).sum())
        n_in += float(m.sum())
        n_out += float((1 - m).sum())
    model.train()
    fix = 1.0 - in_err_m / max(in_err_id, 1e-9)
    return {
        "fix_pct": 100.0 * fix,
        "in_err": 255.0 * in_err_m / max(n_in, 1),       # in 0..255 units
        "in_err_identity": 255.0 * in_err_id / max(n_in, 1),
        "out_err": 255.0 * out_err_m / max(n_out, 1),    # identity is exactly 0
    }


@torch.no_grad()
def save_samples(model, samples, path, device):
    """Rows of [original | model | retoucher | model change x4 | retoucher change x4]."""
    model.eval()
    rows = []
    for o, r, (y0, x0) in samples:
        x = _to_tensor(o).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out = (out.float().clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        sl = (slice(y0, y0 + 256), slice(x0, x0 + 256))
        oc, mc, rc = o[sl], out[sl], r[sl]
        dm = np.clip(128 + 4 * (mc.astype(np.int16) - oc), 0, 255).astype(np.uint8)
        dr = np.clip(128 + 4 * (rc.astype(np.int16) - oc), 0, 255).astype(np.uint8)
        rows.append(np.hstack([oc, mc, rc, dm, dr]))
    model.train()
    cv2.imwrite(path, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--images-per-batch", type=int, default=4)
    ap.add_argument("--crops-per-image", type=int, default=4)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--edit-weight", type=float, default=10.0,
                    help="extra loss weight inside the edited area")
    ap.add_argument("--p-edit", type=float, default=0.75,
                    help="share of crops centred on an edit")
    ap.add_argument("--limit", type=int, default=0, help="use only N train pairs")
    ap.add_argument("--val-images", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="continue this run from runs/<name>/last.pt")
    ap.add_argument("--init", help="start from another run's weights (fine-tuning), "
                                   "e.g. runs/run1/best.pt")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = "cuda"
    out_dir = os.path.join(HERE, "runs", args.name)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    with open(os.path.join(PAIRS, "index.json")) as fh:
        index = json.load(fh)
    thr = index["change_thr"]
    by_id = {p["id"]: p for p in index["pairs"]}
    train_ids = [i for i in index["train"] if by_id[i]["cells"]]
    if args.limit:
        train_ids = train_ids[:args.limit]
    val_ids = [i for i in index["val"] if by_id[i]["cells"]][:args.val_images]

    log(f"args {vars(args)}")
    log(f"train pairs {len(train_ids)}  val pairs {len(val_ids)}")
    val_pairs = [_load_pair(i) for i in val_ids]

    # Fixed preview crops: the 4 val faces with the most edits, centred on
    # their densest edit cell.
    by_edits = sorted(val_ids, key=lambda i: -len(by_id[i]["cells"]))[:4]
    samples = []
    for i in by_edits:
        o, r = val_pairs[val_ids.index(i)]
        cells = np.array(by_id[i]["cells"])
        cy, cx = np.median(cells, axis=0).astype(int)
        samples.append((o, r, (int(np.clip(cy - 128, 0, SIZE - 256)),
                               int(np.clip(cx - 128, 0, SIZE - 256)))))

    ds = PairCrops(index["pairs"], train_ids, args.crop, args.crops_per_image,
                   args.p_edit, thr)
    dl = DataLoader(ds, batch_size=args.images_per_batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0,
                    prefetch_factor=4 if args.workers > 0 else None,
                    worker_init_fn=_worker_init)

    model = RetouchNet().to(device).to(memory_format=torch.channels_last)
    log(f"model params {count_params(model)/1e6:.2f} M")
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
        log(f"initialised from {args.init}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = min(500, args.steps // 10)

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        t = (step - warm) / max(1, args.steps - warm)
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t)))

    step, best = 0, -1e9
    last_ckpt = os.path.join(out_dir, "last.pt")
    if args.resume and os.path.exists(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck.get("best", best)
        log(f"resumed at step {step}")

    base = evaluate(model, val_pairs, thr, device) if step == 0 else None
    if base:
        log(f"baseline (do nothing): {json.dumps({k: round(v, 3) for k, v in base.items()})}")

    t0, seen, run_loss = time.time(), 0, 0.0
    model.train()
    while step < args.steps:
        for x, y, m in dl:
            x = x.flatten(0, 1).to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.flatten(0, 1).to(device, non_blocking=True)
            m = m.flatten(0, 1).to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            w = 1.0 + args.edit_weight * m
            loss = ((out.float() - y).abs() * w).sum() / (w.sum() * 3)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            seen += x.shape[0]
            run_loss += float(loss)

            if step % 100 == 0:
                dt = time.time() - t0
                log(f"step {step}/{args.steps}  loss {run_loss/100:.5f}  "
                    f"lr {lr_at(step):.2e}  {seen/dt:.0f} crops/s")
                run_loss = 0.0

            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate(model, val_pairs, thr, device)
                log(f"eval step {step}: {json.dumps({k: round(v, 3) for k, v in ev.items()})}")
                save_samples(model, samples,
                             os.path.join(out_dir, "samples", f"step_{step:06d}.png"), device)
                # Best = most of the retoucher's change reproduced, as long as
                # good pixels are left essentially alone.
                score = ev["fix_pct"] - 20.0 * max(0.0, ev["out_err"] - 0.5)
                state = {"model": model.state_dict(), "opt": opt.state_dict(),
                         "step": step, "best": max(best, score), "eval": ev}
                torch.save(state, last_ckpt)
                if score > best:
                    best = score
                    torch.save({"model": model.state_dict(), "step": step, "eval": ev},
                               os.path.join(out_dir, "best.pt"))
                    log(f"  new best (score {score:.2f})")
            if step >= args.steps:
                break
    log(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
