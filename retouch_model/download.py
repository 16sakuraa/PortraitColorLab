"""
Resumable, multi-connection downloader.

The FFHQR server is far away (~300 ms round trip), so one connection only
gets a few hundred KB/s. Splitting the file into byte ranges fetched in
parallel multiplies that. Each piece resumes where it stopped if a
connection drops, and the pieces are joined and size-checked at the end.

Usage:
  python download.py URL OUTPUT [--bytes N] [--segments 8]
    --bytes N   only fetch the first N bytes (used for the FFHQR subset)
"""

from __future__ import annotations
import argparse
import os
import sys
import threading
import time
import urllib.request

UA = {"User-Agent": "retouch-model-downloader/1.0"}


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def _fetch(url: str, path: str, start: int, end: int, errors: list) -> None:
    """Fetch bytes [start, end] into path, resuming from what is on disk."""
    tries = 0
    while True:
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if start + have > end:
            return
        headers = dict(UA, Range=f"bytes={start + have}-{end}")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "ab") as fh:
                if r.status != 206:
                    raise RuntimeError(f"server ignored the byte range (HTTP {r.status})")
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            tries = 0
        except Exception as e:  # dropped connection: back off and resume
            tries += 1
            if tries > 30:
                errors.append(f"{path}: {e}")
                return
            time.sleep(min(60, 3 * tries))


def download(url: str, out: str, nbytes: int | None = None, segments: int = 8) -> None:
    total = remote_size(url)
    end = (min(nbytes, total) if nbytes else total) - 1
    if os.path.exists(out) and os.path.getsize(out) == end + 1:
        print(f"already complete: {out}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    step = (end + 1 + segments - 1) // segments
    parts = []
    for k in range(segments):
        s, e = k * step, min(end, (k + 1) * step - 1)
        if s <= e:
            parts.append((f"{out}.part{k}", s, e))

    errors: list[str] = []
    threads = [threading.Thread(target=_fetch, args=(url, p, s, e, errors), daemon=True)
               for p, s, e in parts]
    for t in threads:
        t.start()

    want = end + 1
    t0 = time.time()
    last = (t0, sum(os.path.getsize(p) for p, _, _ in parts if os.path.exists(p)))
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        got = sum(os.path.getsize(p) for p, _, _ in parts if os.path.exists(p))
        now = time.time()
        rate = (got - last[1]) / max(1e-6, now - last[0])
        last = (now, got)
        eta = (want - got) / rate / 60 if rate > 0 else float("inf")
        print(f"  {got/1e9:6.2f} / {want/1e9:.2f} GB  {rate/1e6:5.2f} MB/s  ETA {eta:5.1f} min",
              flush=True)

    if errors:
        print("FAILED:\n  " + "\n  ".join(errors))
        sys.exit(1)
    for p, s, e in parts:
        if os.path.getsize(p) != e - s + 1:
            print(f"FAILED: {p} is {os.path.getsize(p)} bytes, expected {e - s + 1}")
            sys.exit(1)
    with open(out, "wb") as fo:
        for p, _, _ in parts:
            with open(p, "rb") as fi:
                while True:
                    b = fi.read(16 << 20)
                    if not b:
                        break
                    fo.write(b)
    for p, _, _ in parts:
        os.remove(p)
    assert os.path.getsize(out) == want
    print(f"done: {out} ({want/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--bytes", type=int, default=0)
    ap.add_argument("--segments", type=int, default=8)
    a = ap.parse_args()
    download(a.url, a.out, a.bytes or None, a.segments)
