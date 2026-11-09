"""
Draws the app's icon, icon.ico (SkinSmoother.exe uses it, see SkinSmoother.spec): a
blue tile with a head-and-shoulders portrait and a sparkle. The bust stops short of the
tile's edge, so the tile keeps its outline on a white desktop. Each size is drawn on its
own, so the small ones stay sharp and leave out the second sparkle.

  python make_icon.py [preview.png]

The pages' favicon (an inline SVG in serve.py and web/index.html) draws the same shapes,
without the second sparkle, like the small sizes here; change both if the design changes.
"""

from __future__ import annotations
import os
import sys

from PIL import Image, ImageDraw

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
TOP, BOTTOM = (0x6A, 0x95, 0xF7), (0x3B, 0x63, 0xD6)   # the tile's gradient, around the pages' accent #4b7bec
SKIN = (255, 255, 255)
SPARK = (0xFF, 0xE7, 0x9E)

# Shapes in shares of the icon's width (0..1).
TILE = (0.03, 0.03, 0.97, 0.97, 0.22)          # left, top, right, bottom, corner radius
HEAD = (0.45, 0.385, 0.155)                     # centre x, y, radius
SHOULDERS = (0.45, 0.83, 0.30, 0.24)            # the top half of an ellipse: centre x, y (its flat base), radius x, y
SPARKS = ((0.775, 0.245, 0.135), (0.865, 0.475, 0.065))   # centre x, y, radius; the second only from 40 px


def _sparkle(cx: float, cy: float, r: float, steps: int = 24) -> list[tuple[float, float]]:
    """A four-point sparkle: quadratic curves between the tips, bent in towards the centre."""
    k = 0.12 * r
    tips = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    ctrl = [(cx + k, cy - k), (cx + k, cy + k), (cx - k, cy + k), (cx - k, cy - k)]
    pts = []
    for i in range(4):
        (x0, y0), (x2, y2), (x1, y1) = tips[i], tips[(i + 1) % 4], ctrl[i]
        for s in range(steps):
            t = s / steps
            pts.append(((1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t * t * x2,
                        (1 - t) ** 2 * y0 + 2 * (1 - t) * t * y1 + t * t * y2))
    return pts


def draw(size: int) -> Image.Image:
    ss = max(4, 1024 // size)          # supersampling
    n = size * ss
    px = lambda v: v * n               # noqa: E731

    tile = Image.new("L", (n, n), 0)
    l, t, r, b, rad = TILE
    ImageDraw.Draw(tile).rounded_rectangle((px(l), px(t), px(r), px(b)), radius=px(rad), fill=255)

    grad = Image.new("RGB", (1, n))
    for y in range(n):
        f = y / (n - 1)
        grad.putpixel((0, y), tuple(round(a + (c - a) * f) for a, c in zip(TOP, BOTTOM)))
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    img.paste(grad.resize((n, n)), (0, 0), tile)

    shapes = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(shapes)
    hx, hy, hr = HEAD
    d.ellipse((px(hx - hr), px(hy - hr), px(hx + hr), px(hy + hr)), fill=SKIN)
    sx, sy, srx, sry = SHOULDERS
    d.pieslice((px(sx - srx), px(sy - sry), px(sx + srx), px(sy + sry)), 180, 360, fill=SKIN)
    for i, (cx, cy, cr) in enumerate(SPARKS):
        if i and size < 40:
            continue
        d.polygon([(px(x), px(y)) for x, y in _sparkle(cx, cy, cr)], fill=SPARK)
    clipped = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    clipped.paste(shapes, (0, 0), Image.composite(shapes.getchannel("A"), Image.new("L", (n, n), 0), tile))
    img.alpha_composite(clipped)

    # Resize with premultiplied alpha, so the tile's edge doesn't pick up dark fringes.
    return img.convert("RGBa").resize((size, size), Image.LANCZOS).convert("RGBA")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    frames = [draw(s) for s in SIZES]
    big = frames[-1]
    big.save(os.path.join(here, "icon.ico"), sizes=[(s, s) for s in SIZES], append_images=frames[:-1])
    print("wrote icon.ico:", ", ".join(f"{s}px" for s in SIZES))
    if len(sys.argv) > 1:
        # A contact sheet of every size, for checking the small ones.
        w = sum(SIZES) + 10 * (len(SIZES) + 1)
        sheet = Image.new("RGBA", (w, 276), (240, 240, 240, 255))
        x = 10
        for s, f in zip(SIZES, frames):
            sheet.alpha_composite(f, (x, 10))
            x += s + 10
        sheet.save(sys.argv[1])
        print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()
