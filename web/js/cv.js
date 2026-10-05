// Load OpenCV.js (a classic UMD script) from an ES module or module worker.

let ready = null;

export function loadCv(url = new URL('../lib/opencv/opencv.js', import.meta.url).href) {
  if (!ready) {
    ready = (async () => {
      if (!globalThis.cv?.Mat) {
        const code = await (await fetch(url)).text();
        (0, eval)(code);                       // UMD wrapper sets globalThis.cv
      }
      let cv = globalThis.cv;
      if (typeof cv?.then === 'function') cv = await cv;
      if (!cv.Mat) await new Promise((resolve) => { cv.onRuntimeInitialized = resolve; });
      globalThis.cv = cv;
      return cv;
    })();
  }
  return ready;
}

// A real (deep) copy. In OpenCV.js 5.0, Mat.clone() returns a Mat that
// shares memory with the original, so writing to the "clone" changes the
// source and deleting one frees the other. copyTo() does copy.
export function copyMat(cv, m) {
  const c = new cv.Mat();
  m.copyTo(c);
  return c;
}

// RGBA ImageData -> BGR cv.Mat (the desktop app works in BGR throughout).
export function matFromImage(cv, image) {
  const rgba = cv.matFromImageData(image);
  const bgr = new cv.Mat();
  cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR);
  rgba.delete();
  return bgr;
}

// BGR cv.Mat -> RGBA ImageData.
export function imageFromMat(cv, bgr) {
  const rgba = new cv.Mat();
  cv.cvtColor(bgr, rgba, cv.COLOR_BGR2RGBA);
  const image = new ImageData(new Uint8ClampedArray(rgba.data), rgba.cols, rgba.rows);
  rgba.delete();
  return image;
}
