"""
Export a trained RetouchNet to ONNX and check it runs in OpenCV on the CPU.

  python export_onnx.py runs/run1/best.pt            -> runs/run1/retouch.onnx
  python export_onnx.py --untrained                  -> pipeline test only

Checks done after export:
  * OpenCV DNN loads the file (what SkinSmoother uses, no extra install)
  * its output matches PyTorch
  * CPU time per 1024x1024 face, with all cores and with 4 threads
    (closer to an ordinary family PC)
"""

from __future__ import annotations
import argparse
import os
import time

import cv2
import numpy as np
import torch

from model import RetouchNet

SIZE = 1024


def export(model: torch.nn.Module, path: str) -> None:
    model.eval()
    x = torch.rand(1, 3, SIZE, SIZE)
    torch.onnx.export(model, (x,), path, input_names=["input"], output_names=["output"],
                      opset_version=17, dynamo=False)
    print(f"exported {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def check(model: torch.nn.Module, path: str, runs: int = 5) -> None:
    net = cv2.dnn.readNetFromONNX(path)
    rng = np.random.default_rng(0)
    img = rng.random((1, 3, SIZE, SIZE), dtype=np.float32)
    net.setInput(img)
    out_cv = net.forward()
    with torch.no_grad():
        out_pt = model(torch.from_numpy(img)).numpy()
    err = float(np.abs(out_cv - out_pt).max())
    print(f"OpenCV vs PyTorch max difference: {err:.2e}  ({'OK' if err < 1e-3 else 'MISMATCH'})")

    for threads in (cv2.getNumberOfCPUs(), 4):
        cv2.setNumThreads(threads)
        net.setInput(img)
        net.forward()  # warm-up
        t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            net.setInput(img)
            net.forward()
            t.append(time.perf_counter() - t0)
        print(f"CPU, {threads:2d} threads: {np.median(t):.2f} s per 1024x1024 face")
    cv2.setNumThreads(cv2.getNumberOfCPUs())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    model = RetouchNet()
    if a.untrained:
        # Random (non-zero) output layer so the comparison is meaningful.
        torch.nn.init.normal_(model.out.weight, std=0.01)
        out = a.out or os.path.join("runs", "untrained.onnx")
    else:
        ck = torch.load(a.checkpoint, map_location="cpu")
        model.load_state_dict(ck["model"])
        out = a.out or os.path.join(os.path.dirname(a.checkpoint), "retouch.onnx")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    export(model, out)
    check(model, out)


if __name__ == "__main__":
    main()
