"""
Build test photos for checking colour / metadata handling (not committed).

  python web/tools/make_fixtures.py FACE.jpg|webp OUT_DIR

Writes OUT_DIR/dslr_adobe.jpg: a 3000x2000 JPEG like a DSLR file:
  * Adobe RGB (1998)-style ICC profile (so colour-managed decoding would
    change the numbers, unlike sRGB)
  * EXIF: camera make/model, Orientation = 6 (stored sideways), 300 DPI
  * an embedded EXIF thumbnail that is solid red, so it is obvious if an
    output still carries the original's thumbnail
  * quality 98, 4:4:4 colour sampling
and OUT_DIR/dslr_adobe.raw.png: the upright pixels as stored (no colour
conversion), for comparing what the browser decodes.
"""

from __future__ import annotations
import io
import os
import struct
import sys

import numpy as np
import cv2
from PIL import Image


def _s15(v: float) -> bytes:
    return struct.pack(">i", int(round(v * 65536)))


def adobe_rgb_icc() -> bytes:
    """Minimal ICC v2 RGB display profile with Adobe RGB (1998) primaries
    (D50-adapted colorants) and gamma 563/256."""
    def xyz(x, y, z):
        return b"XYZ " + b"\0" * 4 + _s15(x) + _s15(y) + _s15(z)
    desc_txt = b"Adobe RGB (1998) test\0"
    desc = (b"desc" + b"\0" * 4 + struct.pack(">I", len(desc_txt)) + desc_txt
            + b"\0" * 4 + b"\0" * 4 + b"\0" * 2 + b"\0" + b"\0" * 67)
    cprt = b"text" + b"\0" * 4 + b"No copyright, test profile\0"
    curv = b"curv" + b"\0" * 4 + struct.pack(">I", 1) + struct.pack(">H", 0x0233) + b"\0\0"
    tags = [(b"desc", desc), (b"cprt", cprt), (b"wtpt", xyz(0.9642, 1.0, 0.8249)),
            (b"rXYZ", xyz(0.60974, 0.31111, 0.01947)),
            (b"gXYZ", xyz(0.20528, 0.62567, 0.06087)),
            (b"bXYZ", xyz(0.14919, 0.06322, 0.74457)),
            (b"rTRC", curv), (b"gTRC", curv), (b"bTRC", curv)]
    table_len = 4 + 12 * len(tags)
    offset = 128 + table_len
    table, data = struct.pack(">I", len(tags)), b""
    for sig, blob in tags:
        while (offset + len(data)) % 4:
            data += b"\0"
        table += sig + struct.pack(">II", offset + len(data), len(blob))
        data += blob
    body = table + data
    size = 128 + len(body)
    header = (struct.pack(">I", size) + b"none" + bytes([2, 0x10, 0, 0]) + b"mntr"
              + b"RGB " + b"XYZ " + b"\0" * 12 + b"acsp" + b"MSFT" + b"\0" * 4
              + b"\0" * 4 + b"\0" * 4 + b"\0" * 8 + b"\0" * 4
              + _s15(0.9642) + _s15(1.0) + _s15(0.8249) + b"none" + b"\0" * 44)
    assert len(header) == 128
    return header + body


def exif_with_thumbnail(thumb_jpeg: bytes) -> bytes:
    """EXIF (little-endian TIFF) with IFD0 tags and an IFD1 JPEG thumbnail."""
    make, model = b"TestCam\0", b"Model X\0"
    ifd0 = [(0x010F, 2, len(make), make), (0x0110, 2, len(model), model),
            (0x0112, 3, 1, 6), (0x011A, 5, 1, (300, 1)), (0x011B, 5, 1, (300, 1)),
            (0x0128, 3, 1, 2)]
    ifd1_count = 3
    ifd0_off = 8
    ifd0_size = 2 + 12 * len(ifd0) + 4
    ifd1_off = ifd0_off + ifd0_size
    ifd1_size = 2 + 12 * ifd1_count + 4
    data_off = ifd1_off + ifd1_size
    data = b""

    def put(blob):
        nonlocal data
        off = data_off + len(data)
        data += blob + (b"\0" if len(blob) % 2 else b"")
        return off

    entries = b""
    for tag, typ, count, val in ifd0:
        if typ == 2:
            entries += struct.pack("<HHII", tag, typ, count, put(val))
        elif typ == 3:
            entries += struct.pack("<HHIHH", tag, typ, count, val, 0)
        elif typ == 5:
            entries += struct.pack("<HHII", tag, typ, count, put(struct.pack("<II", *val)))
    ifd0_bytes = struct.pack("<H", len(ifd0)) + entries + struct.pack("<I", ifd1_off)
    thumb_off = put(thumb_jpeg)
    ifd1_bytes = (struct.pack("<H", ifd1_count)
                  + struct.pack("<HHIHH", 0x0103, 3, 1, 6, 0)
                  + struct.pack("<HHII", 0x0201, 4, 1, thumb_off)
                  + struct.pack("<HHII", 0x0202, 4, 1, len(thumb_jpeg))
                  + struct.pack("<I", 0))
    tiff = b"II*\0" + struct.pack("<I", ifd0_off) + ifd0_bytes + ifd1_bytes + data
    return b"Exif\0\0" + tiff


def main(face_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    face = cv2.imread(face_path, cv2.IMREAD_COLOR)
    upright = np.full((3000, 2000, 3), (200, 120, 40), np.uint8)   # blue backdrop (BGR)
    f = cv2.resize(face, (1600, 1600), interpolation=cv2.INTER_CUBIC)
    upright[700:2300, 200:1800] = f
    stored = cv2.rotate(upright, cv2.ROTATE_90_COUNTERCLOCKWISE)  # camera held sideways

    red = io.BytesIO()
    Image.new("RGB", (160, 120), (255, 0, 0)).save(red, "JPEG", quality=80)
    buf = io.BytesIO()
    Image.fromarray(stored[..., ::-1]).save(
        buf, "JPEG", quality=98, subsampling=0, icc_profile=adobe_rgb_icc(),
        exif=exif_with_thumbnail(red.getvalue()), dpi=(300, 300))
    with open(os.path.join(out_dir, "dslr_adobe.jpg"), "wb") as fh:
        fh.write(buf.getvalue())
    cv2.imwrite(os.path.join(out_dir, "dslr_adobe.raw.png"), upright)
    print(f"wrote {out_dir}/dslr_adobe.jpg ({len(buf.getvalue())/1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
