"""
Checks a built SkinSmoother.exe: it starts, serves its page, brush and version, and
retouches a photo with its bundled models. The release workflow runs it before
publishing anything.

  python skin_smoother/exe_smoke.py dist/SkinSmoother.exe [EXPECTED_VERSION]

Updates are switched off for the check, so it never downloads anything.
"""

from __future__ import annotations
import base64
import json
import os
import re
import subprocess
import sys
import threading
import urllib.request

import cv2
import numpy as np


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main() -> int:
    exe = os.path.abspath(sys.argv[1])
    want = sys.argv[2] if len(sys.argv) > 2 else None
    env = dict(os.environ, SKINSMOOTHER_NO_BROWSER="1", SKINSMOOTHER_NO_UPDATE="1")
    proc = subprocess.Popen([exe], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    found, lines = threading.Event(), []

    def read():
        for line in proc.stdout:
            lines.append(line.rstrip())
            if re.search(r"127\.0\.0\.1:\d+", line):
                found.set()

    threading.Thread(target=read, daemon=True).start()
    try:
        if not found.wait(240):
            print("\n".join(lines))
            raise SystemExit("the program didn't start")
        port = re.search(r"127\.0\.0\.1:(\d+)", "\n".join(lines)).group(1)
        base = f"http://127.0.0.1:{port}"
        print("\n".join(lines))

        page = _get(base + "/").decode("utf-8")
        assert "Skin Smoother" in page and "/mask_brush.js" in page and "__VERSION__" not in page, "the page"
        assert b"attachMaskBrush" in _get(base + "/mask_brush.js"), "the brush script"
        update = json.loads(_get(base + "/update"))
        assert update["state"] == "off", update
        if want:
            assert update["version"] == want, f"version {update['version']}, expected {want}"

        # A plain skin-coloured picture: no face in it, but every step still runs.
        ok, png = cv2.imencode(".png", np.full((300, 400, 3), (120, 150, 200), np.uint8))
        body = json.dumps({"image": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii"),
                           "strength": 0.6, "texture": 0.6, "blemish": 1.0, "pores": 0.3, "want_mask": True}).encode()
        req = urllib.request.Request(base + "/process", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            result = json.loads(r.read())
        assert "error" not in result, result.get("error")
        assert result["image"].startswith("data:image/") and "mask_image" in result, "a retouched picture and its mask"
        assert result["model"] and result["blemish_model"], "the bundled models"
        print(f"ok: version {update['version']}, page, brush, retouch {result['w']}x{result['h']}")
        return 0
    finally:
        # A one-file .exe runs as two processes; stop both.
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
