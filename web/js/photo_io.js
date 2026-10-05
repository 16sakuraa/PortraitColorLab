// Read and write photos without losing what the camera or editor stored.
//
// Browsers normally colour-manage images when decoding (an Adobe RGB photo is
// converted to sRGB) and drop all metadata when saving. Here:
//   * pixels are decoded WITHOUT colour conversion (the stored numbers, as the
//     desktop app sees them), upright per the EXIF orientation;
//   * the result is saved as JPEG at the original's estimated quality and
//     colour sampling, and the original's colour profile (ICC), camera data
//     (EXIF), and print resolution (JFIF density) are put back;
//   * EXIF orientation is reset to "upright" (pixels are already rotated) and
//     the camera's embedded thumbnail is blanked, so file browsers don't show
//     the un-retouched photo as the preview.

const ENCODER = new URL('../lib/jsquash/encode.js', import.meta.url).href;

// ---------- decoding ----------

export async function decode(blob) {
  const bmp = await createImageBitmap(blob, {
    colorSpaceConversion: 'none', imageOrientation: 'from-image', premultiplyAlpha: 'none',
  });
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const g = canvas.getContext('2d', { willReadFrequently: true });
  g.drawImage(bmp, 0, 0);
  bmp.close();
  const image = g.getImageData(0, 0, canvas.width, canvas.height);
  const bytes = new Uint8Array(await blob.arrayBuffer());
  return { image, meta: readMeta(bytes) };
}

// ---------- reading the original's settings ----------

const STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
  14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109,
  103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95,
  98, 112, 100, 103, 99];

export function readMeta(bytes) {
  if (bytes[0] === 0x89 && bytes[1] === 0x50) return readPngMeta(bytes);
  const meta = { format: 'other', app0: null, exif: null, icc: [], quality: null, subsampling: null };
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) return meta;
  meta.format = 'jpeg';
  let p = 2;
  while (p + 4 <= bytes.length) {
    if (bytes[p] !== 0xff) break;
    const marker = bytes[p + 1];
    if (marker === 0xd8 || (marker >= 0xd0 && marker <= 0xd7) || marker === 0x01) { p += 2; continue; }
    const len = (bytes[p + 2] << 8) | bytes[p + 3];
    const seg = bytes.subarray(p, p + 2 + len);         // whole segment incl. marker
    const body = bytes.subarray(p + 4, p + 2 + len);
    if (marker === 0xe0 && _startsWith(body, 'JFIF\0')) meta.app0 = seg;
    else if (marker === 0xe1 && _startsWith(body, 'Exif\0\0')) meta.exif = seg;
    else if (marker === 0xe2 && _startsWith(body, 'ICC_PROFILE\0')) meta.icc.push(seg);
    else if (marker === 0xdb) _readDqt(body, meta);
    else if (marker === 0xc0 || marker === 0xc1 || marker === 0xc2) _readSof(body, meta);
    if (marker === 0xda) break;                           // start of scan: header done
    p += 2 + len;
  }
  meta.icc.sort((a, b) => a[4 + 12] - b[4 + 12]);         // chunk sequence number
  return meta;
}

function _startsWith(arr, str) {
  for (let i = 0; i < str.length; i++) if (arr[i] !== str.charCodeAt(i)) return false;
  return true;
}

function _readDqt(body, meta) {
  let q = 0;
  while (q < body.length) {
    const pq = body[q] >> 4, tq = body[q] & 15;
    const n = 64 * (pq ? 2 : 1);
    if (tq === 0 && meta.quality === null) {
      let sum = 0;
      for (let i = 0; i < 64; i++) sum += pq ? (body[q + 1 + 2 * i] << 8) | body[q + 2 + 2 * i] : body[q + 1 + i];
      const s = sum / STD_LUMA.reduce((a, b) => a + b) * 100;      // IJG scale factor
      meta.quality = Math.max(1, Math.min(100, Math.round(s <= 100 ? (200 - s) / 2 : 5000 / s)));
    }
    q += 1 + n;
  }
}

function _readSof(body, meta) {
  const nComp = body[5];
  if (nComp < 3) { meta.subsampling = '4:4:4'; return; }
  const y = body[7], c = body[10];                       // sampling factors (H<<4 | V)
  const ratio = ((y >> 4) / (c >> 4)) * 10 + (y & 15) / (c & 15);
  meta.subsampling = ratio === 11 ? '4:4:4' : ratio === 21 ? '4:2:2' : '4:2:0';
}

// ---------- EXIF patching ----------

// Copy of the APP1 segment with Orientation = 1 and the IFD1 thumbnail
// unlinked and blanked. Returns null if the EXIF can't be parsed safely.
export function patchExif(seg) {
  const out = seg.slice();
  const tiff = 10;                                        // FF E1 len(2) "Exif\0\0"
  const le = out[tiff] === 0x49;
  const dv = new DataView(out.buffer, out.byteOffset, out.byteLength);
  const u16 = (o) => dv.getUint16(tiff + o, le), u32 = (o) => dv.getUint32(tiff + o, le);
  try {
    const ifd0 = u32(4);
    const n0 = u16(ifd0);
    for (let i = 0; i < n0; i++) {
      const e = ifd0 + 2 + 12 * i;
      if (u16(e) === 0x0112) dv.setUint16(tiff + e + 8, 1, le);   // Orientation: upright
    }
    const nextPtr = ifd0 + 2 + 12 * n0;
    const ifd1 = u32(nextPtr);
    if (ifd1 && tiff + ifd1 + 2 <= out.length) {
      const n1 = u16(ifd1);
      let off = 0, len = 0;
      for (let i = 0; i < n1; i++) {
        const e = ifd1 + 2 + 12 * i;
        if (u16(e) === 0x0201) off = u32(e + 8);
        if (u16(e) === 0x0202) len = u32(e + 8);
      }
      if (off && len && tiff + off + len <= out.length) out.fill(0, tiff + off, tiff + off + len);
      dv.setUint32(tiff + nextPtr, 0, le);               // no IFD1 any more
    }
    return out;
  } catch {
    return null;
  }
}

// ---------- encoding ----------

let _encode = null;

export async function encodeJpeg(image, meta, qualityOverride = null) {
  if (!_encode) _encode = (await import(ENCODER)).default;
  const quality = qualityOverride ?? Math.max(90, meta.quality ?? 95);   // never below 90
  const sub = meta.subsampling === '4:2:0' ? 2 : 1;       // 4:2:2 is kept at full colour
  const encoded = new Uint8Array(await _encode(image, {
    quality, baseline: false, progressive: false, optimize_coding: true,
    quant_table: 0,                                        // standard tables, like cameras
    auto_subsample: false, chroma_subsample: sub,
    trellis_multipass: false, trellis_opt_zero: false, trellis_opt_table: false,
    separate_chroma_quality: false, chroma_quality: quality,
  }));
  return spliceJpeg(encoded, meta);
}

// Put the original's APP0 / EXIF / ICC segments into an encoded JPEG.
export function spliceJpeg(encoded, meta) {
  let p = 2;
  const keepApp0 = !meta.app0 && !meta.exif;               // plain file: keep encoder's JFIF
  const parts = [encoded.subarray(0, 2)];
  // Skip the encoder's own APP0 (JFIF) unless we keep it.
  if (encoded[p] === 0xff && encoded[p + 1] === 0xe0) {
    const len = (encoded[p + 2] << 8) | encoded[p + 3];
    if (keepApp0) parts.push(encoded.subarray(p, p + 2 + len));
    p += 2 + len;
  }
  if (meta.app0) parts.push(meta.app0);
  if (meta.exif) { const ex = patchExif(meta.exif); if (ex) parts.push(ex); }
  for (const seg of meta.icc) parts.push(seg);
  parts.push(encoded.subarray(p));
  return new Blob(parts, { type: 'image/jpeg' });
}

// ---------- PNG ----------

function readPngMeta(bytes) {
  const meta = { format: 'png', chunks: [] };
  let p = 8;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  while (p + 12 <= bytes.length) {
    const len = dv.getUint32(p);
    const type = String.fromCharCode(...bytes.subarray(p + 4, p + 8));
    if (type === 'iCCP' || type === 'pHYs' || type === 'eXIf') meta.chunks.push(bytes.subarray(p, p + 12 + len));
    if (type === 'IDAT' || type === 'IEND') break;
    p += 12 + len;
  }
  return meta;
}

export async function encodePng(image, meta) {
  const canvas = new OffscreenCanvas(image.width, image.height);
  canvas.getContext('2d').putImageData(image, 0, 0);
  const png = new Uint8Array(await (await canvas.convertToBlob({ type: 'image/png' })).arrayBuffer());
  if (!meta.chunks?.length) return new Blob([png], { type: 'image/png' });
  // Insert the original's iCCP / pHYs / eXIf right after IHDR, dropping any
  // colour chunks the browser wrote that would conflict with the profile.
  const dv = new DataView(png.buffer);
  const ihdrEnd = 8 + 12 + dv.getUint32(8);
  const parts = [png.subarray(0, ihdrEnd), ...meta.chunks];
  let p = ihdrEnd;
  while (p + 12 <= png.length) {
    const len = dv.getUint32(p);
    const type = String.fromCharCode(...png.subarray(p + 4, p + 8));
    if (!['iCCP', 'sRGB', 'gAMA', 'cHRM', 'pHYs', 'eXIf'].includes(type)) parts.push(png.subarray(p, p + 12 + len));
    p += 12 + len;
  }
  return new Blob(parts, { type: 'image/png' });
}

export async function encode(image, meta) {
  return meta.format === 'png'
    ? { blob: await encodePng(image, meta), ext: '.png' }
    : { blob: await encodeJpeg(image, meta), ext: '.jpg' };
}

// A quick on-screen copy that carries the original's colour profile, so the
// browser shows it with the right colours (e.g. an Adobe RGB photo).
export async function encodePreview(image, meta) {
  return meta.format === 'png' ? encodePng(image, meta) : encodeJpeg(image, meta, 90);
}
