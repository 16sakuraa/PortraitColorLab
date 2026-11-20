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
import lip
import look
import photo_io

PREVIEW_MAX = 640  # longest edge of each preview pane
# Print size choices -> smoother.PRINT_GRAIN keys. A small print keeps coarser grain.
PRINT_SIZES = {"Large print or screen": "large", "Small print (1-inch ID)": "small"}
# Lighting choices -> smoother.LIGHTING keys. Hard light evens out mottled light and shadow more.
LIGHTINGS = {"Soft (studio)": "soft", "Hard or patchy (flash)": "hard"}
# Look choices -> look.LOOKS keys: settings chosen together. Custom is whatever the controls say.
LOOK_NAMES = {"Natural": "natural", "Studio ID photo": "studio"}
CUSTOM_LOOK = "Custom"


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
        self.preview_look = None   # cached (skin area, landmarks, faces' brightness, brighter skins' colours, outline)
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
        self.texture = tk.DoubleVar(value=60)
        s2 = ttk.Scale(bar, from_=0, to=100, variable=self.texture,
                       command=self._on_slider, length=200)
        s2.pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(bar, text="Reduce shine").pack(side=tk.LEFT)
        self.shine = tk.DoubleVar(value=0)
        s_shine = ttk.Scale(bar, from_=0, to=100, variable=self.shine,
                            command=self._on_slider, length=160)
        s_shine.pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(bar, text="Add pores").pack(side=tk.LEFT)
        self.pores = tk.DoubleVar(value=0)
        s3 = ttk.Scale(bar, from_=0, to=100, variable=self.pores,
                       command=self._on_slider, length=160)
        s3.pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(bar, text="Lighting").pack(side=tk.LEFT)
        self.lighting = tk.StringVar(value=next(iter(LIGHTINGS)))
        lights = ttk.Combobox(bar, textvariable=self.lighting, values=list(LIGHTINGS),
                              state="readonly", width=20)
        lights.bind("<<ComboboxSelected>>", self._on_slider)
        lights.pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(bar, text="Print size").pack(side=tk.LEFT)
        self.print_size = tk.StringVar(value=next(iter(PRINT_SIZES)))
        sizes = ttk.Combobox(bar, textvariable=self.print_size, values=list(PRINT_SIZES),
                             state="readonly", width=22)
        sizes.bind("<<ComboboxSelected>>", self._on_slider)
        sizes.pack(side=tk.LEFT, padx=(4, 16))

        self.face_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Face only", variable=self.face_only,
                        command=self._on_face_only).pack(side=tk.LEFT, padx=(0, 16))

        # A second row: the look, and the two finishing steps it also sets.
        more = ttk.Frame(self, padding=(8, 0, 8, 8))
        more.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(more, text="Look").pack(side=tk.LEFT)
        self.look_name = tk.StringVar(value=next(iter(LOOK_NAMES)))
        looks = ttk.Combobox(more, textvariable=self.look_name, values=list(LOOK_NAMES) + [CUSTOM_LOOK],
                             state="readonly", width=18)
        looks.bind("<<ComboboxSelected>>", self._on_look)
        looks.pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(more, text="Under-eyes").pack(side=tk.LEFT)
        self.eyes = tk.DoubleVar(value=0)
        ttk.Scale(more, from_=0, to=100, variable=self.eyes, command=self._on_slider,
                  length=160).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(more, text="Balance skin tone").pack(side=tk.LEFT)
        self.tone = tk.DoubleVar(value=0)
        ttk.Scale(more, from_=0, to=100, variable=self.tone, command=self._on_slider,
                  length=160).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(more, text="Straighten head").pack(side=tk.LEFT)
        self.level = tk.DoubleVar(value=0)
        ttk.Scale(more, from_=0, to=100, variable=self.level, command=self._on_slider,
                  length=140).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(more, text="Tidy stray hairs").pack(side=tk.LEFT)
        self.strays = tk.DoubleVar(value=0)
        ttk.Scale(more, from_=0, to=100, variable=self.strays, command=self._on_slider,
                  length=140).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(more, text="Thin moustache").pack(side=tk.LEFT)
        self.lip = tk.DoubleVar(value=0)
        ttk.Scale(more, from_=0, to=100, variable=self.lip, command=self._on_slider,
                  length=140).pack(side=tk.LEFT, padx=(4, 6))
        self.lip_full = tk.BooleanVar(value=False)      # a full moustache too: chosen for each photo
        ttk.Checkbutton(more, text="also a full one", variable=self.lip_full,
                        command=self._on_lip_full).pack(side=tk.LEFT, padx=(0, 16))

        self.status = ttk.Label(more, text="Open an image to begin.")
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
        self.lip_full.set(False)

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
        self.preview_faces = smoother._detect_faces(self.preview_bgr)
        self._analyse_preview()
        covered = float(self.preview_mask.mean())
        if covered < 0.002:
            self.status.config(text="No face detected — using skin-tone areas.")
        else:
            self.status.config(text=f"{os.path.basename(path)}  ({w}×{h})")

        self._orig_photo = to_photo(self.preview_bgr)
        self.orig_label.config(image=self._orig_photo)
        self.save_btn.config(state=tk.NORMAL)
        self._render_preview()

    def _analyse_preview(self):
        self.preview_mask, area, points, seg = smoother.skin_layers(self.preview_bgr, faces=self.preview_faces,
                                                                    face_only=self.face_only.get(), with_seg=True)
        self.preview_look = (area, points, look.skin_luma(self.preview_bgr, self.preview_mask, self.preview_faces),
                             look.bright_colours(self.preview_bgr, self.preview_mask, self.preview_faces),
                             look.person_of(seg))
        self.preview_corr, _ = blemish.correction(self.preview_bgr, self.preview_faces, self.preview_mask)

    def _on_lip_full(self):
        if self.lip_full.get() and self.lip.get() <= 0:
            self.lip.set(100)           # ticked with the slider at nothing: all of it
        self._on_slider()

    def _on_face_only(self):
        if self.preview_bgr is None:
            return
        self._analyse_preview()
        self._render_preview()

    def _grain(self) -> float:
        return smoother.PRINT_GRAIN[PRINT_SIZES[self.print_size.get()]]

    def _light(self) -> float:
        return smoother.LIGHTING[LIGHTINGS[self.lighting.get()]]

    def _settings(self) -> dict:
        """The settings a look sets, as they now are."""
        return {"strength": self.strength.get() / 100.0, "texture": self.texture.get() / 100.0,
                "lighting": LIGHTINGS[self.lighting.get()], "print": PRINT_SIZES[self.print_size.get()],
                "eyes": self.eyes.get() / 100.0, "tone": self.tone.get() / 100.0}

    def _on_look(self, _evt=None):
        values = look.LOOKS.get(LOOK_NAMES.get(self.look_name.get()))
        if values is not None:
            self.strength.set(values["strength"] * 100)
            self.texture.set(values["texture"] * 100)
            self.eyes.set(values["eyes"] * 100)
            self.tone.set(values["tone"] * 100)
            self.lighting.set(next(k for k, v in LIGHTINGS.items() if v == values["lighting"]))
            self.print_size.set(next(k for k, v in PRINT_SIZES.items() if v == values["print"]))
        self._on_slider()

    def _on_slider(self, _evt=None):
        # The look shown is the one the settings match, if any.
        now = self._settings()
        match = next((name for name, key in LOOK_NAMES.items()
                      if all(abs(now[k] - v) < 0.005 if isinstance(v, float) else now[k] == v
                             for k, v in look.LOOKS[key].items())), CUSTOM_LOOK)
        self.look_name.set(match)
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
        base = blemish.apply(self.preview_bgr, self.preview_corr,
                             self.blemish.get() / 100.0)
        base = lip.clear(base, self.preview_look[1], self.preview_mask, self.lip.get() / 100.0,
                         full=self.lip_full.get())
        out = smoother.smooth_face(base, strength=s, texture=t,
                                   mask=self.preview_mask, faces=self.preview_faces,
                                   pores=self.pores.get() / 100.0, grain=self._grain(),
                                   shine=self.shine.get() / 100.0, light=self._light())
        area, points, luma, bright, person = self.preview_look
        out = look.finish(out, self.preview_mask, area, points, luma,
                          self.eyes.get() / 100.0, self.tone.get() / 100.0,
                          self.shine.get() / 100.0, self.preview_faces, bright,
                          self.strays.get() / 100.0, person, None, self.level.get() / 100.0)
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
        p = self.pores.get() / 100.0
        g = self._grain()
        sh = self.shine.get() / 100.0
        li = self._light()
        eyes, tone, strays = self.eyes.get() / 100.0, self.tone.get() / 100.0, self.strays.get() / 100.0
        level = self.level.get() / 100.0
        moustache, full_moustache = self.lip.get() / 100.0, self.lip_full.get()

        b = self.blemish.get() / 100.0
        face_only = self.face_only.get()
        info = self.photo_info

        def work():
            faces = smoother._detect_faces(self.full_bgr)
            mask, area, points, seg = smoother.skin_layers(self.full_bgr, faces=faces, face_only=face_only, with_seg=True)
            corr, _ = blemish.correction(self.full_bgr, faces, mask)
            base = blemish.apply(self.full_bgr, corr, b)
            base = lip.clear(base, points, mask, moustache, full=full_moustache)
            out = smoother.smooth_face(base, strength=s, texture=t, mask=mask, faces=faces, pores=p, grain=g,
                                       shine=sh, light=li)
            out = look.finish(out, mask, area, points, look.skin_luma(self.full_bgr, mask, faces), eyes, tone,
                              sh, faces, look.bright_colours(self.full_bgr, mask, faces), strays, look.person_of(seg),
                              None, level)
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
