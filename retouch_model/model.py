"""
Small residual U-Net for blemish removal.

  out = input + net(input)

The network only predicts the *correction*, starting from exactly zero (the
last conv is zero-initialised), so an untrained model leaves the photo
untouched and training only has to learn where and how to heal.

Deliberately built from layers that OpenCV's DNN module runs on the CPU:
Conv, ReLU, nearest-neighbour Resize, Concat, Add. No normalisation layers,
no dynamic shapes. Input/output: float32 BGR in [0, 1], NCHW.
"""

from __future__ import annotations
import torch
import torch.nn as nn


def _block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class _Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(cout + cskip, cout, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        return self.fuse(torch.cat([self.up(x), skip], dim=1))


class RetouchNet(nn.Module):
    def __init__(self, ch: tuple[int, ...] = (16, 32, 64, 128, 192)):
        super().__init__()
        self.enc = nn.ModuleList([_block(3, ch[0])])
        for i in range(1, len(ch)):
            self.enc.append(_block(ch[i - 1], ch[i], stride=2))
        self.dec = nn.ModuleList(
            _Up(ch[i], ch[i - 1], ch[i - 1]) for i in range(len(ch) - 1, 0, -1)
        )
        self.out = nn.Conv2d(ch[0], 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skips = []
        h = x
        for e in self.enc:
            h = e(h)
            skips.append(h)
        h = skips.pop()
        for d in self.dec:
            h = d(h, skips.pop())
        return x + self.out(h)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = RetouchNet()
    print(f"params: {count_params(m)/1e6:.2f} M")
    y = m(torch.rand(1, 3, 256, 256))
    print("out", tuple(y.shape))
