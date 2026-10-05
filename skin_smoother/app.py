"""
Desktop app for face-aware skin smoothing.

Open a photo, drag the Strength slider, compare before/after, and save.
Full-resolution processing happens on Save; the preview uses a scaled copy so
the sliders stay responsive.

Run:  python app.py
"""

from __future__ import annotations
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

import smoother
import blemish
import photo_io

PREVIEW_MAX = 640  # longest edge of each preview pane


def to_photo(bgr) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Skin Smoother")
        self.geometry("1360x780")
        self.minsize(900, 600)

        self.full_bgr = None       # full-res original
        self.photo_info = photo_io.PhotoInfo()  # its colour profile, EXIF, JPEG settings
        self.preview_bgr = None    # scaled original for display
        self.preview_mask = None   # cached mask for the preview
        self.preview_corr = None   # cached blemish correction for the preview
        self.src_path = None
        self._after_id = None
        self._orig_photo = None
        self._smooth_photo = None

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        bar = ttk.Frame(self, padding=8)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Open image…", command=self.open_image).pack(side=tk.LEFT)
        self.save_btn = ttk.Button(bar, text="Save…", command=self.save_image,
                                   state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=(6, 16))

        ttk.Label(bar, text="Remove blemishes").pack(side=tk.LEFT)
        self.blemish = tk.DoubleVar(value=100 if blemish.have_model() else 0)
        s0 = ttk.Scale(bar, from_=0, to=100, variable=self.blemish,
                       command=self._on_slider, length=160)
        s0.pack(side=tk.LEFT, padx=(4, 16))
        if not blemish.have_model():
            s0.state(["disabled"])

        ttk.Label(bar, text="Strength").pack(side=tk.LEFT)
        self.strength = tk.DoubleVar(value=60)
        s1 = ttk.Scale(bar, from_=0, to=100, variable=self.strength,
                       command=self._on_slider, length=240)
        s1.pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(bar, text="Keep texture").pack(side=tk.LEFT)
        self.texture = tk.DoubleVar(value=35)
        s2 = ttk.Scale(bar, from_=0, to=100, variable=self.texture,
                       command=self._on_slider, length=200)
        s2.pack(side=tk.LEFT, padx=(4, 16))

        self.use_gpu = tk.BooleanVar(value=smoother.opencl_available())
        gpu_chk = ttk.Checkbutton(bar, text="Use GPU", variable=self.use_gpu,
                                  command=self._on_slider)
        gpu_chk.pack(side=tk.LEFT, padx=(0, 16))
        if not smoother.opencl_available():
            gpu_chk.state(["disabled"])

        self.status = ttk.Label(bar, text="Open an image to begin.")
        self.status.pack(side=tk.RIGHT)

        body = ttk.Frame(self, padding=8)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(body, text="Original")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        right = ttk.LabelFrame(body, text="Smoothed")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self.orig_label = ttk.Label(left, anchor="center")
        self.orig_label.pack(fill=tk.BOTH, expand=True)
        self.smooth_label = ttk.Label(right, anchor="center")
        self.smooth_label.pack(fill=tk.BOTH, expand=True)

    # ---------- actions ----------
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        img, info = photo_io.read(path)
        if img is None:
            messagebox.showerror("Error", f"Could not open:\n{path}")
            return

        self.full_bgr = img
        self.photo_info = info
        self.src_path = path

        # Build a scaled preview.
        h, w = img.shape[:2]
        scale = min(1.0, PREVIEW_MAX / max(h, w))
        if scale < 1.0:
            self.preview_bgr = cv2.resize(img, (int(w * scale), int(h * scale)),
                                          interpolation=cv2.INTER_AREA)
        else:
            self.preview_bgr = img.copy()

        self.status.config(text="Detecting face…")
        self.update_idletasks()
        self.preview_mask = smoother.build_skin_mask(self.preview_bgr)
        self.preview_corr, _ = blemish.correction(self.preview_bgr)
        covered = float(self.preview_mask.mean())
        if covered < 0.002:
            self.status.config(text="No face detected — using skin-tone areas.")
        else:
            self.status.config(text=f"{os.path.basename(path)}  ({w}×{h})")

        self._orig_photo = to_photo(self.preview_bgr)
        self.orig_label.config(image=self._orig_photo)
        self.save_btn.config(state=tk.NORMAL)
        self._render_preview()

    def _on_slider(self, _evt=None):
        if self.preview_bgr is None:
            return
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._after_id = self.after(90, self._render_preview)

    def _render_preview(self):
        self._after_id = None
        if self.preview_bgr is None:
            return
        s = self.strength.get() / 100.0
        t = self.texture.get() / 100.0
        smoother.set_opencl(self.use_gpu.get())
        base = blemish.apply(self.preview_bgr, self.preview_corr,
                             self.blemish.get() / 100.0)
        out = smoother.smooth_face(base, strength=s, texture=t,
                                   mask=self.preview_mask)
        self._smooth_photo = to_photo(out)
        self.smooth_label.config(image=self._smooth_photo)

    def save_image(self):
        if self.full_bgr is None:
            return
        stem, ext = os.path.splitext(self.src_path)
        default = os.path.basename(f"{stem}_smooth{ext or '.jpg'}")
        path = filedialog.asksaveasfilename(
            title="Save smoothed image",
            initialfile=default,
            defaultextension=ext or ".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return

        self.status.config(text="Processing full resolution…")
        self.update_idletasks()

        s = self.strength.get() / 100.0
        t = self.texture.get() / 100.0

        gpu = self.use_gpu.get()
        b = self.blemish.get() / 100.0
        info = self.photo_info

        def work():
            smoother.set_opencl(gpu)
            base, _ = blemish.remove_blemishes(self.full_bgr, amount=b)
            out = smoother.smooth_face(base, strength=s, texture=t)
            ok = photo_io.write(path, out, info)
            self.after(0, lambda: self._save_done(ok, path))

        threading.Thread(target=work, daemon=True).start()

    def _save_done(self, ok: bool, path: str):
        if ok:
            self.status.config(text=f"Saved: {os.path.basename(path)}")
        else:
            self.status.config(text="Save failed.")
            messagebox.showerror("Error", f"Could not save:\n{path}")


if __name__ == "__main__":
    App().mainloop()
