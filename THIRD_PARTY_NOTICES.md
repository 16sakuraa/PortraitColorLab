# Third-party components

## Models

| File | What | Licence |
|---|---|---|
| `*/models/blemish_ffhqr.onnx` | Blemish-removal network trained in `retouch_model/` on 3,000 image pairs from **FFHQR** (Shafaei, Little, Schmidt, *AutoRetouch: Automatic Professional Face Retouching*, WACV 2021), the retouched version of **FFHQ** (Karras, Laine, Aila, *A Style-Based Generator Architecture for GANs*, CVPR 2019, NVIDIA). | **CC BY-NC-SA 4.0** - non-commercial use only, share alike. Training data is not included in this repository. |
| `*/models/face_detection_yunet.onnx` | YuNet face detector from [OpenCV Zoo](https://github.com/opencv/opencv_zoo) / [libfacedetection](https://github.com/ShiqiYu/libfacedetection). The web copy has its input size made dynamic (no weights changed). | MIT |
| `*/models/selfie_multiclass_256x256.onnx` | Selfie multiclass segmentation model by Google ([MediaPipe image segmenter](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter), [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf)). Modified: converted from TFLite to ONNX with tf2onnx 1.16.1 at opset 13, with NCHW input and output, and three Resize nodes' coordinate mode changed from `tf_half_pixel_for_nn` to `asymmetric`, which picks the same pixels at these 2x scales. No weights changed. The output matches the original to within 0.0001. | Apache 2.0 (`web/models/LICENSE-selfie-multiclass`, `skin_smoother/models/LICENSE-selfie-multiclass`) |

## Libraries bundled with the web app (`web/lib/`)

| Library | Version | Licence |
|---|---|---|
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (`ort/`) | 1.30.0 | MIT (`web/lib/ort/LICENSE`) |
| [OpenCV.js](https://opencv.org/) via [@techstark/opencv-js](https://github.com/TechStark/opencv-js) (`opencv/`) | 5.0.0 | Apache 2.0 (`web/lib/opencv/LICENSE`) |
| [jSquash JPEG](https://github.com/jamsinclair/jSquash) (`jsquash/`), MozJPEG encoder compiled to WebAssembly | 1.6.0 | Apache 2.0 (`web/lib/jsquash/LICENSE`); MozJPEG / libjpeg-turbo: IJG and BSD-style licences |

## Desktop app dependencies (installed with pip, not bundled)

OpenCV (Apache 2.0), NumPy (BSD), Pillow (MIT-CMU). Training additionally
uses PyTorch (BSD) and ONNX (Apache 2.0).
