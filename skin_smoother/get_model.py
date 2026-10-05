"""
Download OpenCV's YuNet face-detection model into ./models so the smoother can
do proper face-aware masking.

The model is ~345 KB and comes from the official OpenCV Zoo:
  https://github.com/opencv/opencv_zoo

Run:  python get_model.py
"""

import os
import urllib.request

URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
       "face_detection_yunet/face_detection_yunet_2023mar.onnx")
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "models", "face_detection_yunet.onnx")


def main() -> int:
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"Downloading YuNet model…\n  from {URL}\n  to   {DEST}")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as e:
        print(f"Download failed: {e}")
        return 1
    size = os.path.getsize(DEST)
    print(f"Done. {size/1024:.0f} KB written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
