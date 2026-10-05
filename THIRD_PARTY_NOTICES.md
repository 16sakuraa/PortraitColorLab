# Third-party components

## Models

| File | What | Licence |
|---|---|---|
| `*/models/blemish_ffhqr.onnx` | Blemish-removal network trained in `retouch_model/` on 3,000 image pairs from **FFHQR** (Shafaei, Little, Schmidt, *AutoRetouch: Automatic Professional Face Retouching*, WACV 2021), the retouched version of **FFHQ** (Karras, Laine, Aila, *A Style-Based Generator Architecture for GANs*, CVPR 2019, NVIDIA). | **CC BY-NC-SA 4.0** - non-commercial use only, share alike. Training data is not included in this repository. |
| `*/models/face_detection_yunet.onnx` | YuNet face detector from [OpenCV Zoo](https://github.com/opencv/opencv_zoo) / [libfacedetection](https://github.com/ShiqiYu/libfacedetection). The web copy has its input size made dynamic (no weights changed). | MIT |

## Libraries bundled with the web app (`web/lib/`)

| Library | Version | Licence |
|---|---|---|
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (`ort/`) | 1.30.0 | MIT (`web/lib/ort/LICENSE`) |
| [OpenCV.js](https://opencv.org/) via [@techstark/opencv-js](https://github.com/TechStark/opencv-js) (`opencv/`) | 5.0.0 | Apache 2.0 (`web/lib/opencv/LICENSE`) |
| [jSquash JPEG](https://github.com/jamsinclair/jSquash) (`jsquash/`), MozJPEG encoder compiled to WebAssembly | 1.6.0 | Apache 2.0 (`web/lib/jsquash/LICENSE`); MozJPEG / libjpeg-turbo: IJG and BSD-style licences |

## Desktop app dependencies (installed with pip, not bundled)

OpenCV (Apache 2.0), NumPy (BSD), Pillow (MIT-CMU). Training additionally
uses PyTorch (BSD) and ONNX (Apache 2.0).
