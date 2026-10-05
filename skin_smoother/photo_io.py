"""
Read and write photos without losing what the camera or editor stored in them.

OpenCV alone keeps only the pixels. Saving through it drops:
  * the colour profile (ICC): an Adobe RGB photo then looks washed out,
    because every viewer assumes plain sRGB
  * camera data (EXIF): date, camera, lens, settings
  * print resolution (DPI)
and re-compresses a JPEG at OpenCV's own quality with colour at half
resolution, so the file shrinks and fine detail is lost.

Here the profile, EXIF and DPI are carried over, and a JPEG is saved with the
original's own compression tables and colour sampling. The result matches the
original's quality and is about the same file size.
"""

from __future__ import annotations
import io
import os
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image, JpegImagePlugin

ORIENTATION = 0x0112


@dataclass
class PhotoInfo:
    format: str | None = None       # "JPEG", "PNG", ... of the original file
    icc: bytes | None = None        # colour profile
    exif: bytes | None = None       # camera data, orientation reset to upright
    dpi: tuple | None = None        # print resolution
    qtables: dict | None = None     # JPEG compression tables of the original
    subsampling: int | None = None  # JPEG colour sampling (0=4:4:4, 1=4:2:2, 2=4:2:0)


def decode(data: bytes) -> tuple[np.ndarray | None, PhotoInfo]:
    """Photo file bytes -> (BGR pixels, upright, and the file's PhotoInfo)."""
    # OpenCV applies the EXIF orientation, so the pixels come out upright.
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    info = PhotoInfo()
    try:
        im = Image.open(io.BytesIO(data))
        info.format = im.format
        info.icc = im.info.get("icc_profile") or None
        info.dpi = im.info.get("dpi")
        exif = im.getexif()
        if len(exif):
            # The pixels are already upright; keeping the old orientation tag
            # would make viewers rotate the result a second time.
            exif[ORIENTATION] = 1
            info.exif = exif.tobytes()
        if im.format == "JPEG":
            info.qtables = im.quantization
            info.subsampling = JpegImagePlugin.get_sampling(im)
    except Exception:
        pass  # pixels still decoded; just nothing extra to carry over
    return bgr, info


def encode(bgr: np.ndarray, info: PhotoInfo, fmt: str = "JPEG") -> bytes:
    """BGR pixels -> file bytes in `fmt`, with the original's profile, EXIF,
    DPI and (for JPEG) its compression settings."""
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    kw = {}
    if info.icc:
        kw["icc_profile"] = info.icc
    if info.exif:
        kw["exif"] = info.exif
    if info.dpi:
        kw["dpi"] = tuple(int(round(float(d))) for d in info.dpi)
    if fmt == "JPEG":
        if info.qtables and info.subsampling in (0, 1, 2):
            kw.update(qtables=info.qtables, subsampling=info.subsampling)
        else:
            # Not from a JPEG: high quality, full colour resolution.
            kw.update(quality=95, subsampling=0)
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


def output_format(info: PhotoInfo) -> str:
    """Keep PNGs as PNG (lossless); everything else is saved as JPEG."""
    return "PNG" if info.format == "PNG" else "JPEG"


_EXT_FORMAT = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".tif": "TIFF",
               ".tiff": "TIFF", ".bmp": "BMP", ".webp": "WEBP"}


def read(path: str) -> tuple[np.ndarray | None, PhotoInfo]:
    """Like cv2.imread, but works with non-ASCII Windows paths and also
    returns the file's PhotoInfo."""
    try:
        with open(path, "rb") as fh:
            return decode(fh.read())
    except OSError:
        return None, PhotoInfo()


def write(path: str, bgr: np.ndarray, info: PhotoInfo) -> bool:
    """Save with the original's profile/EXIF/DPI; format from the extension."""
    fmt = _EXT_FORMAT.get(os.path.splitext(path)[1].lower(), "JPEG")
    try:
        data = encode(bgr, info, fmt)
        with open(path, "wb") as fh:
            fh.write(data)
        return True
    except Exception:
        return False
