"""
Checks on the retouching code that need no photos or models.

  python skin_smoother/test_retouch.py
"""

from __future__ import annotations
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import urllib.error

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoother  # noqa: E402
import blemish  # noqa: E402
import blemish_classic  # noqa: E402
import chin  # noqa: E402
import heal  # noqa: E402
import lip  # noqa: E402
import look  # noqa: E402
import mask_edit  # noqa: E402
import skin_segment  # noqa: E402
import face_landmarks  # noqa: E402
import updater  # noqa: E402

FACE =np.array([[300, 200, 400, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9]], np.float32)


def skin_photo(h: int = 900, w: int = 1000, seed: int = 1) -> np.ndarray:
    """Soft shading, fine grain like pores, and a sharp dark edge."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tone = 165 + 25 * np.sin(xx / 170) * np.cos(yy / 210) + rng.normal(0, 3, (h, w))
    img = np.dstack([tone * 0.72, tone * 0.83, tone])
    img[:, 640:650] = 40
    return np.clip(img, 0, 255).astype(np.uint8)


def oval_mask(h: int = 900, w: int = 1000) -> np.ndarray:
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (500, 450), (260, 340), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 12)


def test_guided_filter_leaves_flat_input_alone():
    flat = np.full((64, 64), 0.4, np.float32)
    out = smoother.guided_filter(flat, flat, 5, 1e-3)
    assert np.abs(out - 0.4).max() < 1e-4


def test_guided_filter_keeps_edges_and_flattens_grain():
    rng = np.random.default_rng(2)
    step = np.zeros((128, 128), np.float32)
    step[:, 64:] = 0.8
    noisy = step + rng.normal(0, 0.01, step.shape).astype(np.float32)
    out = smoother.guided_filter(noisy, noisy, 6, 0.03 ** 2)
    assert out[:, 20:44].std() < noisy[:, 20:44].std() * 0.5, "grain should drop"
    assert out[:, 70:].mean() - out[:, :58].mean() > 0.78, "the edge should keep its height"


def test_strength_zero_returns_the_input():
    img = skin_photo()
    out = smoother.smooth_face(img, 0.0, 0.35, oval_mask(), FACE)
    assert np.array_equal(out, img)


def test_nothing_changes_outside_the_mask():
    img = skin_photo()
    mask = oval_mask()
    out = smoother.smooth_face(img, 1.0, 0.0, mask, FACE)
    assert np.array_equal(out[mask == 0], img[mask == 0])
    assert not np.array_equal(out, img), "skin inside the mask should change"


def test_more_strength_removes_more_grain():
    img = skin_photo()
    mask = oval_mask()
    def grain(out):
        y = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return (y - cv2.GaussianBlur(y, (0, 0), 3))[350:550, 350:550].std()
    weak = smoother.smooth_face(img, 0.3, 0.35, mask, FACE)
    strong = smoother.smooth_face(img, 1.0, 0.35, mask, FACE)
    assert grain(img) > grain(weak) > grain(strong)


def test_keep_texture_keeps_grain():
    img = skin_photo()
    mask = oval_mask()
    def grain(out):
        y = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return (y - cv2.GaussianBlur(y, (0, 0), 3))[350:550, 350:550].std()
    assert grain(smoother.smooth_face(img, 1.0, 1.0, mask, FACE)) \
        > grain(smoother.smooth_face(img, 1.0, 0.0, mask, FACE))


def test_large_scale_shading_survives():
    img = skin_photo()
    mask = oval_mask()
    out = smoother.smooth_face(img, 1.0, 0.0, mask, FACE)
    before = cv2.GaussianBlur(img, (0, 0), 25).astype(np.float32)[300:600, 350:620]
    after = cv2.GaussianBlur(out, (0, 0), 25).astype(np.float32)[300:600, 350:620]
    assert np.abs(before - after).max() < 2.5, "shading should stay within a couple of levels"


def test_strong_edge_inside_the_mask_is_kept():
    img = skin_photo()
    mask = oval_mask()
    out = smoother.smooth_face(img, 1.0, 0.0, mask, FACE)
    def edge_depth(a):
        y = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return y[400:500, 620:630].mean() - y[400:500, 642:648].mean()
    assert edge_depth(out) > 0.9 * edge_depth(img)


def test_a_backdrop_of_equal_brightness_does_not_tint_the_skin():
    # Skin and a blue backdrop with the same brightness, so no brightness edge between them.
    img = np.zeros((900, 1000, 3), np.uint8)
    img[:] = (120, 150, 190)
    brightness = float(cv2.cvtColor(img[:1, :1], cv2.COLOR_BGR2GRAY)[0, 0])
    img[:, 600:] = (255, 140, int(max(0.0, (brightness - 0.114 * 255 - 0.587 * 140) / 0.299)))
    mask = np.zeros((900, 1000), np.float32)
    mask[100:800, 100:600] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), 2)
    out = smoother.smooth_face(img, 1.0, 0.0, mask, FACE)
    before, after = img[200:700, 540:596].astype(np.float32), out[200:700, 540:596].astype(np.float32)
    assert np.abs(after - before).max() <= 3, "the skin beside the backdrop should keep its colour"


def test_strip_seams_do_not_show():
    img = skin_photo()
    mask = oval_mask()
    saved = smoother.STRIP_ROWS
    try:
        smoother.STRIP_ROWS = 10_000
        whole = smoother.smooth_face(img, 0.8, 0.35, mask, FACE)
        smoother.STRIP_ROWS = 64
        strips = smoother.smooth_face(img, 0.8, 0.35, mask, FACE)
    finally:
        smoother.STRIP_ROWS = saved
    assert np.abs(whole.astype(int) - strips.astype(int)).max() <= 1


def test_no_faces_still_works():
    img = skin_photo()
    out = smoother.smooth_face(img, 0.6, 0.35, oval_mask(), np.zeros((0, 15), np.float32))
    assert out.shape == img.shape and not np.array_equal(out, img)


def test_mixed_face_sizes_get_separate_regions():
    faces = np.array([[100, 100, 500, 600] + [0] * 11, [800, 300, 60, 70] + [0] * 11], np.float32)
    regions = smoother._scale_regions(faces, faces[:, 2].astype(np.float64), 1.0, 0, 0, (900, 1000))
    assert len(regions) == 2
    total = regions[0][1] + regions[1][1]
    assert np.abs(total - 1).max() < 1e-3, "weights should add up to 1"
    small, large = regions
    assert small[0] < large[0]
    assert large[1][450, 350] > 0.99 and small[1][335, 830] > 0.99


def pore_change(img, mask, pores=1.0, strength=0.8, face=FACE) -> np.ndarray:
    """What Add pores changes, in levels: float BGR."""
    without = smoother.smooth_face(img, strength, 0.6, mask, face)
    return smoother.smooth_face(img, strength, 0.6, mask, face, pores).astype(np.float32) - without


def test_pores_zero_adds_nothing():
    img = skin_photo()
    mask = oval_mask()
    assert np.array_equal(smoother.smooth_face(img, 0.6, 0.6, mask, FACE, 0.0),
                          smoother.smooth_face(img, 0.6, 0.6, mask, FACE))


def test_pores_are_repeatable():
    img = skin_photo()
    mask = oval_mask()
    assert np.array_equal(smoother.smooth_face(img, 0.6, 0.6, mask, FACE, 0.8),
                          smoother.smooth_face(img, 0.6, 0.6, mask, FACE, 0.8))


def test_pores_stay_in_the_mask():
    img = skin_photo()
    mask = oval_mask()
    change = pore_change(img, mask)
    assert (change[mask == 0] == 0).all()
    assert np.abs(change[mask > 0.99]).mean() > 0.3, "pores should show inside the mask"


def test_pores_change_brightness_only():
    change = pore_change(skin_photo(), oval_mask())
    assert np.abs(change[:, :, 0] - change[:, :, 2]).max() <= 2, "the rest is rounding"
    assert np.abs(change[:, :, 0] - change[:, :, 1]).max() <= 2, "the rest is rounding"


def test_pores_keep_average_brightness_and_lean_dark():
    change = pore_change(skin_photo(), oval_mask())[350:550, 350:550, 1]
    assert abs(change.mean()) < 0.1
    assert change.min() < -1.2 * change.max(), "pits should go deeper than the bumps rise"


def test_pores_skip_strong_edges():
    img = skin_photo()
    img[:, 470:472] = 40    # and a stray hair, thinner than the bar in skin_photo
    change = pore_change(img, oval_mask())
    assert np.abs(change[350:550, 638:642]).max() <= 1, "either side of a hard edge"
    assert np.abs(change[350:550, 470:472]).max() <= 1, "on a hair"
    assert np.abs(change[350:550, 350:450]).max() > 2, "but plain skin gets them"


def test_pore_strips_join_up():
    img = skin_photo()
    mask = oval_mask()
    saved = smoother.STRIP_ROWS
    try:
        smoother.STRIP_ROWS = 10_000
        whole = smoother.smooth_face(img, 0.8, 0.6, mask, FACE, 1.0)
        smoother.STRIP_ROWS = 64
        strips = smoother.smooth_face(img, 0.8, 0.6, mask, FACE, 1.0)
    finally:
        smoother.STRIP_ROWS = saved
    assert np.abs(whole.astype(int) - strips.astype(int)).max() <= 1


def test_pore_size_follows_the_face():
    def spacing(sigma):
        """Distance at which the pattern's autocorrelation first goes negative."""
        band = smoother._pore_band(0, 0, 512, 512, sigma)
        spectrum = np.abs(np.fft.fft2(band)) ** 2
        row = np.fft.ifft2(spectrum).real[0]
        return int(np.argmax(row < 0))
    assert spacing(0.0012 * 2000) > spacing(0.0012 * 900) > spacing(0.0012 * 500)


def test_pores_never_repeat():
    # Far from zero anywhere but at no shift: a tiled pattern would match itself.
    band = smoother._pore_band(1000, 1000, 1024, 1024, 1.0)
    spectrum = np.abs(np.fft.fft2(band)) ** 2
    corr = np.fft.ifft2(spectrum).real
    corr /= corr[0, 0]
    corr[:16, :16] = corr[:16, -16:] = corr[-16:, :16] = corr[-16:, -16:] = 0
    assert np.abs(corr).max() < 0.05


def test_small_faces_get_faint_pores():
    img = skin_photo()
    mask = oval_mask()
    tiny = FACE.copy()
    tiny[0, 2] = 60
    small = FACE.copy()
    small[0, 2] = 300
    assert np.array_equal(smoother.smooth_face(img, 0.8, 0.6, mask, tiny, 1.0),
                          smoother.smooth_face(img, 0.8, 0.6, mask, tiny)), "too small to show"
    def strength_of(face):
        return np.abs(pore_change(img, mask, face=face)[350:550, 350:550]).mean()
    assert strength_of(small) < strength_of(FACE)


def mottled_photo(h: int = 900, w: int = 1000, seed: int = 3) -> np.ndarray:
    """Skin whose grain is a few pixels across: grain a small print still shows."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.5)
    tone = 165 + 25 * np.sin(xx / 170) * np.cos(yy / 210) + grain * (6 / grain.std())
    return np.clip(np.dstack([tone * 0.72, tone * 0.83, tone]), 0, 255).astype(np.uint8)


def test_large_print_is_the_default():
    img = skin_photo()
    mask = oval_mask()
    plain = smoother.smooth_face(img, 0.8, 0.6, mask, FACE, 0.5)
    large = smoother.PRINT_GRAIN["large"]
    assert np.array_equal(smoother.smooth_face(img, 0.8, 0.6, mask, FACE, 0.5, grain=large), plain)
    assert np.array_equal(smoother.smooth_face(img, 0.8, 0.6, mask, FACE, 0.5, grain=0.1),
                          smoother.smooth_face(img, 0.8, 0.6, mask, FACE, 0.5, grain=smoother.MIN_GRAIN)), \
        "grain below MIN_GRAIN (used by Smooth again) counts as MIN_GRAIN"


def test_small_print_keeps_the_grain_the_print_shows():
    img = mottled_photo()
    mask = oval_mask()
    shrink = 400 / 154   # FACE is 400 px wide; a 1-inch ID print's face is about 154 px at 300 dpi
    a, b = int(350 / shrink), int(550 / shrink)
    def printed_grain(out):
        small = cv2.resize(out, None, fx=1 / shrink, fy=1 / shrink, interpolation=cv2.INTER_AREA)
        y = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return (y - cv2.GaussianBlur(y, (0, 0), 1.5))[a:b, a:b].std()
    large = printed_grain(smoother.smooth_face(img, 1.0, 0.6, mask, FACE))
    small = printed_grain(smoother.smooth_face(img, 1.0, 0.6, mask, FACE, grain=smoother.PRINT_GRAIN["small"]))
    assert small > large * 1.05, f"small print {small:.2f} vs large {large:.2f}"
    assert small < printed_grain(img), "the small print is still smoothed"


def test_small_print_makes_pores_coarser():
    img = skin_photo()
    mask = oval_mask()
    def coarse_share(grain):
        without = smoother.smooth_face(img, 0.8, 0.6, mask, FACE, 0.0, grain=grain).astype(np.float32)
        added = smoother.smooth_face(img, 0.8, 0.6, mask, FACE, 1.0, grain=grain).astype(np.float32) - without
        d = added.mean(axis=2)[350:550, 350:550]
        return cv2.GaussianBlur(d, (0, 0), 2).std() / d.std()
    assert coarse_share(smoother.PRINT_GRAIN["small"]) > coarse_share(1.0)


def shiny_photo(h: int = 900, w: int = 1000, seed: int = 4):
    """Skin from darker (left) to brighter (right), dotted with white specks and,
    as bright but coloured, reddish ones. Returns the photo and both specks' (y, x)."""
    rng = np.random.default_rng(seed)
    xx = np.mgrid[0:h, 0:w][1].astype(np.float32)
    tone = 120 + 70 * np.clip((xx - 240) / 520, 0, 1) + rng.normal(0, 2, (h, w))
    img = np.dstack([tone * 0.72, tone * 0.83, tone]).astype(np.float32)
    grid = np.array([(y, x) for y in range(150, 760, 9) for x in range(250, 760, 9)])
    white, red = grid[0::2], grid[1::2]
    img[white[:, 0], white[:, 1]] += 4.0
    img[red[:, 0], red[:, 1]] += np.array([2.5, 2.5, 7.5], np.float32)
    return np.clip(np.round(img), 0, 255).astype(np.uint8), white, red


def speck_left(out, specks, mask) -> float:
    """How far the specks on the bright side still stand out, in levels."""
    y = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lift = y - cv2.GaussianBlur(y, (0, 0), 3)
    on = specks[(specks[:, 1] > 620) & (mask[specks[:, 0], specks[:, 1]] > 0.99)]
    return float(lift[on[:, 0], on[:, 1]].mean())


def test_reduce_shine_is_off_by_default():
    img = skin_photo()
    mask = oval_mask()
    assert np.array_equal(smoother.smooth_face(img, 0.8, 0.6, mask, FACE, 0.5, shine=0.0),
                          smoother.smooth_face(img, 0.8, 0.6, mask, FACE, 0.5))


def test_reduce_shine_tames_white_specks_on_bright_skin():
    img, white, _ = shiny_photo()
    mask = oval_mask()
    before = speck_left(smoother.smooth_face(img, 1.0, 0.6, mask, FACE), white, mask)
    after = speck_left(smoother.smooth_face(img, 1.0, 0.6, mask, FACE, shine=1.0), white, mask)
    assert after < 0.75 * before, f"white specks {before:.2f} -> {after:.2f} levels"


def test_reduce_shine_leaves_coloured_specks_dark_skin_and_dark_pores():
    img, _, red = shiny_photo()
    mask = oval_mask()
    off = smoother.smooth_face(img, 1.0, 0.6, mask, FACE)
    on = smoother.smooth_face(img, 1.0, 0.6, mask, FACE, shine=1.0)
    assert speck_left(on, red, mask) > 0.9 * speck_left(off, red, mask), "coloured specks are not glints"
    sigma = float(FACE[0, 2]) * smoother.TUNE["split"]
    part, high = smoother._detail_strip(img, 0, img.shape[0], sigma)
    usual = smoother._skin_tone(img, mask, sigma) + smoother.TUNE["shine_lift"]
    duller = ((part - high) @ smoother._LUMA) < usual - 0.01
    assert duller[mask > 0.99].mean() > 0.3, "the photo should have plenty of skin at or below its usual tone"
    assert np.array_equal(on[duller], off[duller]), "skin no brighter than the face's usual tone has no shine"
    darker = (high @ smoother._LUMA) < 0
    assert np.array_equal(on[darker], off[darker]), "anything darker than its surroundings is left alone"


def mottled_light_photo(h: int = 900, w: int = 1000, seed: int = 5) -> np.ndarray:
    """Broad shading, mottled light and shadow a few millimetres across (hard
    light, at FACE's size), and fine grain."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    patches = cv2.GaussianBlur(n, (0, 0), 5) - cv2.GaussianBlur(n, (0, 0), 16)
    tone = 160 + 25 * np.sin(xx / 170) * np.cos(yy / 210) + patches * (6 / patches.std()) \
        + rng.normal(0, 1.5, (h, w))
    return np.clip(np.dstack([tone * 0.72, tone * 0.83, tone]), 0, 255).astype(np.uint8)


def test_soft_lighting_is_the_default():
    img = skin_photo()
    mask = oval_mask()
    assert np.array_equal(smoother.smooth_face(img, 0.8, 0.6, mask, FACE, light=smoother.LIGHTING["soft"]),
                          smoother.smooth_face(img, 0.8, 0.6, mask, FACE))


def test_hard_lighting_evens_mottled_light_and_keeps_shading():
    img = mottled_light_photo()
    mask = oval_mask()
    mm = float(FACE[0, 2]) / 150   # FACE is about 15 cm wide
    core = cv2.erode((mask > 0.99).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    def band(out, lo, hi):
        y = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return (cv2.GaussianBlur(y, (0, 0), lo * mm) - cv2.GaussianBlur(y, (0, 0), hi * mm))[core].std()
    soft = smoother.smooth_face(img, 1.0, 0.6, mask, FACE)
    hard = smoother.smooth_face(img, 1.0, 0.6, mask, FACE, light=smoother.LIGHTING["hard"])
    patches = band(img, 3, 10)
    assert band(hard, 3, 10) < 0.8 * band(soft, 3, 10) < 0.8 * patches, \
        f"patches kept: soft {band(soft, 3, 10) / patches:.2f}, hard {band(hard, 3, 10) / patches:.2f}"
    shading = band(img, 10, 60)
    assert band(hard, 10, 60) > 0.98 * shading, "the face's shading stays, so it doesn't look puffy"


def test_mixed_face_sizes_get_their_own_pores():
    img = skin_photo()
    mask = oval_mask()
    faces = np.array([[150, 150, 400, 600] + [0] * 11, [600, 300, 60, 70] + [0] * 11], np.float32)
    change = pore_change(img, mask, face=faces)
    assert np.abs(change[350:550, 250:450]).mean() > 0.3, "the large face gets pores"
    assert np.abs(change[325:345, 620:640]).mean() < 0.2, "the small face doesn't"


def stroke(mode, *points, size=0.04, soft=0.0):
    return {"mode": mode, "size": size, "soft": soft, "points": list(points)}


def edited_mask(strokes, base=0.0, w=400, h=300):
    mask = np.full((h, w), base, np.float32)
    return mask_edit.apply_to_mask(mask, mask_edit.region_of(mask_edit.layer_from(strokes, w, h), w, h))


def test_no_strokes_leave_the_mask_alone():
    mask = np.full((300, 400), 0.5, np.float32)
    region = mask_edit.region_of(mask_edit.layer_from([], 400, 300), 400, 300)
    assert region is None and mask_edit.apply_to_mask(mask, region) is mask


def test_protect_clears_the_mask_and_add_skin_fills_it():
    protected = edited_mask([stroke("protect", 0.5, 0.5)], base=1.0)
    assert protected[150, 200] < 0.02 and protected[20, 20] == 1.0
    added = edited_mask([stroke("skin", 0.5, 0.5)], base=0.0)
    assert added[150, 200] > 0.98 and added[20, 20] == 0.0


def test_restore_at_part_strength_brings_back_that_share():
    full = stroke("protect", 0.5, 0.5)
    assert np.array_equal(edited_mask([{**full, "amount": 1.0}], base=1.0), edited_mask([full], base=1.0)), \
        "a stroke without an amount is a full one"
    part = edited_mask([{**full, "amount": 0.4}], base=1.0)
    assert abs(part[150, 200] - 0.6) < 0.01 and part[20, 20] == 1.0, "40% restored leaves 60% of the smoothing"
    layer = mask_edit.layer_from([{**full, "amount": 0.4}], 400, 300)
    corr = (0, 0, np.ones((300, 400, 3), np.float32))
    _, _, out = mask_edit.apply_to_correction(corr, mask_edit.region_of(layer, 400, 300))
    assert abs(out[150, 200, 0] - 0.6) < 0.01, "and 60% of the spot fills"
    over = edited_mask([full, {**full, "amount": 0.4}], base=1.0)
    assert abs(over[150, 200] - 0.6) < 0.01, "a later stroke wins: 40% over 100% leaves 40%"
    added = edited_mask([{**stroke("skin", 0.5, 0.5), "amount": 0.5}], base=0.0)
    assert abs(added[150, 200] - 0.5) < 0.01, "Add skin at half strength"


def test_a_later_stroke_wins_where_they_cross():
    both = edited_mask([stroke("protect", 0.2, 0.5, 0.8, 0.5), stroke("skin", 0.5, 0.2, 0.5, 0.8)], base=0.5)
    assert both[150, 200] > 0.98, "the skin stroke came last"
    assert both[150, 100] < 0.02, "the protect stroke still holds elsewhere"
    undone = edited_mask([stroke("protect", 0.2, 0.5, 0.8, 0.5)], base=0.5)
    assert undone[150, 200] < 0.02


def test_a_soft_brush_has_a_soft_edge():
    hard = edited_mask([stroke("skin", 0.5, 0.5, size=0.1)])
    soft = edited_mask([stroke("skin", 0.5, 0.5, size=0.1, soft=1.0)])
    middling = lambda m: int(((m > 0.1) & (m < 0.9)).sum())
    assert middling(soft) > 3 * middling(hard)


def test_strokes_land_in_the_same_place_at_any_size():
    strokes = [stroke("skin", 0.3, 0.4, 0.7, 0.6, size=0.05, soft=0.3), stroke("protect", 0.5, 0.2, size=0.03)]
    small = edited_mask(strokes, w=800, h=600)
    large = edited_mask(strokes, w=6000, h=4500)     # past the layer's limit
    shrunk = cv2.resize(large, (800, 600), interpolation=cv2.INTER_AREA)
    assert np.abs(small - shrunk).mean() < 0.005, "one edit, on a preview and on an export"
    assert np.abs(small - shrunk).max() < 0.25


def test_protect_takes_spots_out_of_the_correction():
    values = np.full((100, 100, 3), 8.0, np.float32)
    corr = (150, 100, values)
    layer = mask_edit.layer_from([stroke("protect", 0.5, 0.5, size=0.1)], 400, 300)
    x0, y0, out = mask_edit.apply_to_correction(corr, mask_edit.region_of(layer, 400, 300))
    assert (x0, y0) == (150, 100)
    assert np.abs(out[50, 50]).max() < 0.2, "no fill under the stroke"
    assert out[2, 2, 0] == 8.0, "fills elsewhere are untouched"
    assert values[50, 50, 0] == 8.0, "the original is kept, so an undo can bring it back"


def test_add_skin_leaves_the_correction_alone():
    corr = (150, 100, np.full((100, 100, 3), 8.0, np.float32))
    layer = mask_edit.layer_from([stroke("skin", 0.5, 0.5, size=0.1)], 400, 300)
    _, _, out = mask_edit.apply_to_correction(corr, mask_edit.region_of(layer, 400, 300))
    assert (out == 8.0).all()


def fills_kept(strokes, w=400, h=300) -> float:
    """How much of a spot fill stays in the middle of the photo under these strokes."""
    corr = (0, 0, np.ones((h, w, 3), np.float32))
    _, fills = mask_edit.regions(strokes, w, h)
    return float(mask_edit.apply_to_correction(corr, fills)[2][h // 2, w // 2, 0])


def test_undo_smoothing_clears_the_mask_but_keeps_the_fixes():
    restore, unsmooth = stroke("protect", 0.5, 0.5), stroke("unsmooth", 0.5, 0.5)
    assert np.array_equal(edited_mask([unsmooth], base=1.0), edited_mask([restore], base=1.0)), "on the mask it's Restore"
    assert fills_kept([restore]) < 0.02 and fills_kept([unsmooth]) == 1.0, "but the spot fills stay"
    part = {**unsmooth, "amount": 0.4}
    assert abs(edited_mask([part], base=1.0)[150, 200] - 0.6) < 0.01 and fills_kept([part]) == 1.0, "at 40% too"


def test_undo_smoothing_and_restore_follow_the_later_stroke():
    restore, unsmooth = stroke("protect", 0.5, 0.5), stroke("unsmooth", 0.5, 0.5)
    assert fills_kept([restore, unsmooth]) > 0.98, "Undo smoothing over a Restore brings the fills back"
    assert fills_kept([restore, {**unsmooth, "amount": 0.4}]) > 0.98, "at any strength: it says keep the fixes"
    assert fills_kept([unsmooth, restore]) < 0.02, "a Restore over it takes them out again"
    assert fills_kept([unsmooth, stroke("skin", 0.5, 0.5)]) == 1.0


def test_without_undo_smoothing_the_fills_follow_the_mask_edit():
    strokes = [stroke("protect", 0.3, 0.4, 0.7, 0.6, soft=0.5), stroke("skin", 0.5, 0.2, 0.5, 0.8, soft=0.3)]
    assert mask_edit.layers_from(strokes, 400, 300)[1] is None
    region, fills = mask_edit.regions(strokes, 400, 300)
    assert fills is region, "one edit, as before Undo smoothing existed"


def test_a_heal_stays_under_undo_smoothing():
    _, scar, _ = scarred()
    over = stroke("unsmooth", 0.375, 0.5, 0.5, 0.517, 0.625, 0.533, size=60 / 800)
    assert np.array_equal(heal.apply(scar, [heal_stroke(), over]), heal.apply(scar, [heal_stroke()]))


def test_a_stroke_off_the_edge_of_the_photo_is_safe():
    assert edited_mask([stroke("skin", 0.0, 0.0, 1.0, 1.0, size=0.1, soft=0.5)]).shape == (300, 400)
    assert edited_mask([stroke("skin", 0.999, 0.999)])[299, 399] > 0.5


def test_brushed_skin_gets_smoothed_and_protected_skin_does_not():
    img = skin_photo()
    mask = np.zeros(img.shape[:2], np.float32)
    mask[300:600, 300:700] = 1.0
    h, w = img.shape[:2]
    # A patch outside the mask gets added, and one inside it gets protected.
    strokes = [stroke("skin", 0.2, 0.2, size=0.12), stroke("protect", 0.5, 0.5, size=0.1)]
    edited = mask_edit.apply_to_mask(mask, mask_edit.region_of(mask_edit.layer_from(strokes, w, h), w, h))
    out = smoother.smooth_face(img, 1.0, 0.0, edited, FACE)
    def grain(a, x, y):
        v = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        p = (v - cv2.GaussianBlur(v, (0, 0), 3))[y - 25:y + 25, x - 25:x + 25]
        return float(p.std())
    assert grain(out, 200, 180) < 0.7 * grain(img, 200, 180), "the added patch is smoothed"
    assert abs(grain(out, 500, 450) - grain(img, 500, 450)) < 0.2, "the protected patch isn't"
    assert grain(out, 380, 350) < 0.7 * grain(img, 380, 350), "the rest of the mask still is"


def scarred(h: int = 600, w: int = 800):
    """Shaded, grainy skin; the same with a flat dark scar across it; and where the scar is.
    Larger sizes show the same skin, with finer grain."""
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tone = 150 + 40 * (xx / w) + 10 * np.sin(yy / h * 6.67) + rng.normal(0, 6, (h, w))
    skin = np.clip(np.dstack([tone * 0.72, tone * 0.83, tone]), 0, 255).astype(np.uint8)
    scar = skin.copy()
    line = np.zeros((h, w), np.uint8)
    for a, colour in ((scar, (60, 70, 90)), (line, 255)):
        cv2.line(a, (round(0.375 * w), h // 2), (round(0.625 * w), round(0.533 * h)), colour, max(1, round(w * 14 / 800)))
    return skin, scar, line > 0


def heal_stroke(size=30 / 800, soft=0.3, offset=(-0.25, 0.25), **more):
    return {"mode": "heal", "size": size, "soft": soft, "points": [0.375, 0.5, 0.5, 0.517, 0.625, 0.533],
            "offset": list(offset), **more}


def grey(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)


def test_heal_and_spot_strokes_leave_the_mask_alone():
    image_strokes = [heal_stroke(), {"mode": "spot", "size": 0.05, "soft": 0.3, "points": [0.5, 0.5]}]
    assert mask_edit.region_of(mask_edit.layer_from(image_strokes, 400, 300), 400, 300) is None
    protect = [stroke("protect", 0.3, 0.3, size=0.1)]
    assert np.array_equal(mask_edit.layer_from(protect + image_strokes, 400, 300),
                          mask_edit.layer_from(protect, 400, 300)), "and don't count as Restore"


def test_without_heal_strokes_the_photo_is_left_alone():
    img = skin_photo()
    assert heal.apply(img, []) is img and heal.apply(img, None) is img
    assert heal.apply(img, [stroke("protect", 0.5, 0.5), stroke("skin", 0.2, 0.2)]) is img


def test_heal_copies_the_detail_and_matches_the_tone_around_it():
    skin, scar, inside = scarred()
    out = heal.apply(scar, [heal_stroke()])
    assert abs(grey(out)[inside].mean() - grey(skin)[inside].mean()) < 3, "the scar takes the tone around it"
    fine = lambda a: (grey(a) - cv2.GaussianBlur(grey(a), (0, 0), 3))[inside].std()
    assert abs(fine(out) - fine(skin)) < 0.15 * fine(skin), "and the source's grain"
    assert out[50, 50].tolist() == scar[50, 50].tolist(), "away from the stroke nothing changes"
    assert (scar[inside] == (60, 70, 90)).all(), "the input is left as it was"


def test_heal_at_part_strength_and_from_right_here():
    _, scar, inside = scarred()
    full = heal.apply(scar, [heal_stroke()]).astype(np.float32) - scar
    half = heal.apply(scar, [heal_stroke(amount=0.5)]).astype(np.float32) - scar
    assert abs(half[inside].mean() / full[inside].mean() - 0.5) < 0.02
    assert np.array_equal(heal.apply(scar, [heal_stroke(offset=(0, 0))]), scar), "no source, no change"


def test_restore_after_a_heal_takes_it_out_and_before_it_does_not():
    _, scar, inside = scarred()
    healed = heal.apply(scar, [heal_stroke()])
    restore = stroke("protect", 0.375, 0.5, 0.5, 0.517, 0.625, 0.533, size=60 / 800)
    assert np.array_equal(heal.apply(scar, [restore, heal_stroke()]), healed), "a later heal wins"
    assert np.array_equal(heal.apply(scar, [heal_stroke(), restore]), scar), "a later full Restore wins"
    part = heal.apply(scar, [heal_stroke(), {**restore, "amount": 0.4}]).astype(np.float32) - scar
    full = healed.astype(np.float32) - scar
    assert abs(part[inside].mean() / full[inside].mean() - 0.6) < 0.02, "40% Restore leaves 60% of the heal"
    skin_after = heal.apply(scar, [heal_stroke(), stroke("skin", 0.5, 0.5, size=0.2)])
    assert np.array_equal(skin_after, healed), "Add skin doesn't touch it"


def test_heals_land_in_the_same_place_at_any_size():
    _, small_scar, inside = scarred(600, 800)
    _, large_scar, _ = scarred(2400, 3200)
    small = heal.apply(small_scar, [heal_stroke()])
    large = heal.apply(large_scar, [heal_stroke()])     # a brush past the working grid's size
    shrunk = cv2.resize(large, (800, 600), interpolation=cv2.INTER_AREA)
    assert abs(grey(shrunk)[inside].mean() - grey(small)[inside].mean()) < 3, "one edit, on a preview and an export"
    assert grey(shrunk)[inside].mean() > 130, "the scar is gone at both sizes"


def test_a_heal_scribbled_over_a_broad_area_still_matches_the_tone():
    skin, scar, _ = scarred()
    zigzag = [v for i in range(9) for v in (0.3 + 0.05 * i, 0.35 if i % 2 else 0.65)]
    broad = {"mode": "heal", "size": 0.06, "soft": 0.2, "points": zigzag, "offset": [0.0, 0.3]}
    out = heal.apply(scar, [broad])
    middle = (slice(270, 330), slice(330, 470))       # far from the edge of all the strokes
    assert abs(grey(out)[middle].mean() - grey(skin)[middle].mean()) < 6
    assert grey(out)[middle].std() < 2 * grey(skin)[middle].std(), "no blotches where nothing outside is near"


def test_a_heal_reading_off_the_edge_of_the_photo_is_safe():
    _, scar, _ = scarred()
    corner = {"mode": "heal", "size": 0.1, "soft": 0.5, "points": [0.01, 0.01, 0.99, 0.99], "offset": [-0.9, 0.9]}
    out = heal.apply(scar, [corner, {**corner, "mode": "spot"}, stroke("protect", 1.0, 1.0)])
    assert out.shape == scar.shape and out.dtype == np.uint8


def test_spot_fills_a_mark_from_the_skin_around_it():
    skin, _, _ = scarred()
    spot = skin.copy()
    cv2.circle(spot, (400, 300), 8, (70, 80, 110), -1)
    inside = np.zeros(spot.shape[:2], np.uint8)
    cv2.circle(inside, (400, 300), 8, 255, -1)
    inside = inside > 0
    out = heal.apply(spot, [{"mode": "spot", "size": 24 / 800, "soft": 0.3, "points": [0.5, 0.5]}])
    assert abs(grey(out)[inside].mean() - grey(skin)[inside].mean()) < 4, "the spot takes the tone around it"
    ring = np.zeros(spot.shape[:2], np.uint8)
    cv2.circle(ring, (400, 300), 40, 255, -1)
    ring = (ring > 0) & ~inside
    assert np.abs(grey(out)[ring] - grey(spot)[ring]).mean() < 2, "the skin around it hardly changes"


def light_stroke(*points, pick=(0.8, 0.2), size=0.1, soft=0.3, **more):
    return {"mode": "light", "size": size, "soft": soft, "points": list(points), "pick": list(pick), **more}


def test_without_light_strokes_the_photo_is_left_alone():
    img = skin_photo()
    assert heal.apply_light(img, [], 3.0) is img and heal.apply_light(img, None, 3.0) is img
    assert heal.apply_light(img, [heal_stroke(), stroke("protect", 0.5, 0.5)], 3.0) is img


def test_light_paints_the_picked_tone_and_keeps_the_grain():
    skin, _, _ = scarred()
    patch = (slice(260, 340), slice(260, 340))              # well inside the stroke
    picked = grey(skin)[110:130, 630:650].mean()            # around the pick at (0.8, 0.2)
    fine = lambda a: (grey(a) - cv2.GaussianBlur(grey(a), (0, 0), 2))[patch].std()
    full = heal.apply_light(skin, [light_stroke(0.375, 0.5, size=0.2)], 3.0)
    assert abs(grey(full)[patch].mean() - picked) < 3, "the stroke takes the picked tone"
    assert abs(fine(full) - fine(skin)) < 0.1 * fine(skin), "and keeps the grain"
    part = heal.apply_light(skin, [light_stroke(0.375, 0.5, size=0.2, amount=0.2)], 3.0)
    moved = (grey(part)[patch].mean() - grey(skin)[patch].mean()) / (grey(full)[patch].mean() - grey(skin)[patch].mean())
    assert abs(moved - 0.2) < 0.03, "20% Strength goes a fifth of the way"
    assert full[20, 20].tolist() == skin[20, 20].tolist(), "away from the stroke nothing changes"


def test_light_paint_follows_the_later_stroke():
    skin, _, _ = scarred()
    light = light_stroke(0.375, 0.5)
    painted = heal.apply_light(skin, [light], 3.0)
    restore = stroke("protect", 0.375, 0.5, size=0.2)
    assert np.array_equal(heal.apply_light(skin, [light, restore], 3.0), skin), "a later Restore takes it back"
    assert np.array_equal(heal.apply_light(skin, [restore, light], 3.0), painted), "an earlier one doesn't limit it"
    kept = [light, stroke("unsmooth", 0.375, 0.5, size=0.2), stroke("skin", 0.375, 0.5, size=0.2)]
    assert np.array_equal(heal.apply_light(skin, kept, 3.0), painted), "Undo smoothing and Add skin keep it"


def test_light_and_heal_strokes_stay_in_their_own_pass():
    _, scar, _ = scarred()
    light = light_stroke(0.375, 0.5)
    assert heal.apply(scar, [light]) is scar, "the heal pass doesn't paint light"
    assert np.array_equal(heal.apply(scar, [heal_stroke(), light]), heal.apply(scar, [heal_stroke()])), \
        "and light strokes aren't Restore to it"
    assert heal.apply_light(scar, [heal_stroke()], 3.0) is scar
    assert mask_edit.region_of(mask_edit.layer_from([light], 800, 600), 800, 600) is None, "nor to the mask"
    assert np.array_equal(heal.apply_light(scar, [{**light, "pick": None}], 3.0), scar), "no pick, no paint"


def test_the_light_brush_keeps_the_grain_smoothing_keeps():
    one = smoother.detail_sigma((900, 1000), FACE, 1.0)
    assert one == max(smoother.MIN_SPLIT_SIGMA, 400 * smoother.TUNE["split"])
    assert abs(smoother.detail_sigma((900, 1000), FACE, 2.0) - 2 * one) < 1e-9, "Print size coarsens it"


def again_at_middle(strokes, w=400, h=300) -> float:
    """How much Smooth again these strokes leave in the middle of the photo."""
    again = mask_edit.edits(strokes, w, h)[2]
    if again is None:
        return 0.0
    x0, y0, edit = again
    return float(edit[h // 2 - y0, w // 2 - x0])


def test_smooth_again_has_its_own_layer_and_follows_the_later_stroke():
    again = stroke("resmooth", 0.5, 0.5)
    assert mask_edit.edits([stroke("protect", 0.2, 0.2)], 400, 300)[2] is None, "none without its strokes"
    assert mask_edit.region_of(mask_edit.layer_from([again], 400, 300), 400, 300) is None, "not on the mask"
    assert mask_edit.layers_from([again], 400, 300)[1] is None, "nor the fills"
    assert again_at_middle([again]) > 0.98
    assert abs(again_at_middle([{**again, "amount": 0.4}]) - 0.4) < 0.01
    assert again_at_middle([again, stroke("protect", 0.5, 0.5)]) < 0.02, "a later Restore takes it back"
    assert again_at_middle([again, stroke("unsmooth", 0.5, 0.5)]) < 0.02, "and Undo smoothing"
    assert again_at_middle([stroke("protect", 0.5, 0.5), again]) > 0.98, "an earlier one doesn't limit it"
    assert again_at_middle([again, stroke("skin", 0.5, 0.5)]) > 0.98, "Add skin leaves it"
    _, scar, _ = scarred()
    assert np.array_equal(heal.apply(scar, [heal_stroke(), again]), heal.apply(scar, [heal_stroke()])), \
        "and it isn't Restore to heals"


def test_smooth_again_smooths_coarse_grain_and_keeps_the_finest():
    rng = np.random.default_rng(5)
    h, w = 900, 1000
    coarse = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.0) * 25
    fine = rng.normal(0, 3, (h, w)).astype(np.float32)
    tone = 165 + coarse + fine
    img = np.clip(np.dstack([tone * 0.72, tone * 0.83, tone]), 0, 255).astype(np.uint8)
    mask = oval_mask()
    smoothed = smoother.smooth_face(img, 0.6, 0.6, mask, FACE)
    edit = np.ones((200, 200), np.float32)
    again = smoother.smooth_again(img, smoothed, (400, 350, edit), mask, FACE, 0.6)
    band = lambda a: (cv2.GaussianBlur(grey(a), (0, 0), 1.2) - cv2.GaussianBlur(grey(a), (0, 0), 4))[400:500, 450:550].std()
    finest = lambda a: (grey(a) - cv2.GaussianBlur(grey(a), (0, 0), 0.8))[400:500, 450:550].std()
    assert band(again) < 0.7 * band(smoothed), "coarse grain is smoothed more"
    assert finest(again) > 0.7 * finest(smoothed), "the finest grain stays"
    assert np.array_equal(again[:300], smoothed[:300]), "nothing changes away from the edit"
    half = smoother.smooth_again(img, smoothed, (400, 350, edit * 0.5), mask, FACE, 0.6)
    assert band(again) < band(half) < band(smoothed), "Strength blends it in"
    outside = smoother.smooth_again(img, img.copy(), (0, 0, np.ones((150, 150), np.float32)), mask, FACE, 0.6)
    assert not np.array_equal(outside[:150, :150], img[:150, :150]), "it smooths where the mask left skin out too"


def face_row(x, y, w, h, score=0.9) -> list[float]:
    """A row like YuNet's, with landmarks where a frontal face has them."""
    eye_y, nose_y, mouth_y = y + 0.4 * h, y + 0.6 * h, y + 0.78 * h
    return [x, y, w, h,
            x + 0.3 * w, eye_y, x + 0.7 * w, eye_y,
            x + 0.5 * w, nose_y,
            x + 0.35 * w, mouth_y, x + 0.65 * w, mouth_y, score]


def rows(*faces) -> np.ndarray:
    return np.array(faces, np.float32)


def test_a_face_found_twice_counts_once():
    big, off = face_row(200, 200, 500, 560), face_row(420, 260, 400, 520)
    kept = smoother._separate_faces(rows(off, big))
    assert len(kept) == 1 and kept[0][2] == 500, "the larger detection is the better fit"


def test_neighbouring_faces_stay_separate():
    left, right = face_row(100, 200, 300, 360), face_row(380, 200, 300, 360)
    assert len(smoother._separate_faces(rows(left, right))) == 2


def test_something_that_is_not_a_face_goes():
    real = face_row(100, 200, 300, 360)
    odd = face_row(600, 200, 300, 360)
    odd[8:10] = [odd[4] - 200, odd[5] - 200]    # nose nowhere near the eyes and mouth
    kept = smoother._separate_faces(rows(real, odd))
    assert len(kept) == 1 and kept[0][0] == 100


def test_a_lone_doubtful_face_is_kept():
    odd = face_row(600, 200, 300, 360)
    odd[8:10] = [odd[4] - 200, odd[5] - 200]
    assert len(smoother._separate_faces(rows(odd))) == 1
    assert len(smoother._separate_faces(rows(odd, odd))) == 1, "all of them doubtful: keep one"


def test_close_faces_share_a_crop_and_distant_ones_dont():
    close = rows(face_row(100, 100, 200, 240), face_row(320, 100, 200, 240))
    assert len(skin_segment._crops(close, 2000, 2000)) == 1
    distant = rows(face_row(100, 100, 200, 240), face_row(1500, 1500, 200, 240))
    assert len(skin_segment._crops(distant, 2000, 2000)) == 2


def test_a_crop_keeps_its_smallest_face_large_enough():
    mixed = rows(face_row(100, 100, 800, 900), face_row(900, 300, 100, 120))
    assert len(skin_segment._crops(mixed, 3000, 3000)) == 2, "the small face would be too small in one crop"


def test_crops_stay_inside_the_photo_and_are_largest_first():
    faces = rows(face_row(10, 10, 100, 120), face_row(800, 800, 300, 360))
    crops = skin_segment._crops(faces, 1000, 1000)
    for x0, y0, x1, y1, _ in crops:
        assert 0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000
    assert crops[0][4] == [1], "the largest face comes first"


def test_crop_edges_fade_except_at_the_photo_edge():
    r = skin_segment._ramp(100, 10.0, True, False)
    assert r[0] < 0.1 and r[50] == 1.0 and r[-1] == 1.0
    r = skin_segment._ramp(100, 10.0, True, True)
    assert r[0] < 0.1 and r[-1] < 0.1


def segments(face, body, person, work=None, covered=(0,)):
    """Scores for a 200x200 photo, at full resolution."""
    if work is None:
        work = np.full((200, 200, 3), (120, 150, 200), np.uint8)
    return skin_segment.Segments(0, 0, 200, 200, work, face.astype(np.float32), body.astype(np.float32),
                                 person.astype(np.float32), covered)


def test_body_skin_is_dropped_with_face_only():
    face, body = np.zeros((200, 200)), np.zeros((200, 200))
    face[20:100, 40:160] = 1
    body[100:190, 40:160] = 1
    faces = rows(face_row(40, 20, 120, 80))
    _, _, with_body = skin_segment.skin_area(segments(face, body, face + body), faces)
    _, _, without = skin_segment.skin_area(segments(face, body, face + body), faces, face_only=True)
    assert with_body.shape[0] > without.shape[0], "the neck's rows should only be there with body skin"


def test_body_skin_of_someone_else_is_dropped():
    face, body = np.zeros((200, 200)), np.zeros((200, 200))
    face[20:80, 20:80] = 1
    body[120:180, 120:180] = 1          # a separate person shape, with no detected face
    faces = rows(face_row(20, 20, 60, 60))
    x0, y0, area = skin_segment.skin_area(segments(face, body, face + body), faces)
    assert y0 + area.shape[0] < 120, "nothing from the other person"


def test_body_skin_of_another_colour_is_dropped():
    face, body = np.zeros((200, 200)), np.zeros((200, 200))
    face[20:100, 40:160] = 1
    body[100:190, 40:160] = 1
    work = np.full((200, 200, 3), (120, 150, 200), np.uint8)
    work[100:, :] = (200, 150, 60)      # blue, on the same connected shape
    faces = rows(face_row(40, 20, 120, 80))
    _, y0, area = skin_segment.skin_area(segments(face, body, face + body, work), faces)
    assert area[150 - y0:, 60:140].max() < 0.05 if y0 + area.shape[0] > 150 else True


def test_mask_snaps_to_the_photo_edge():
    # The model's edge is a soft ramp, off by a few pixels; the photo's is sharp.
    work = np.full((200, 200, 3), 60, np.uint8)
    work[:, 100:] = 210
    face = np.zeros((200, 200))
    face[:, 96:] = 1
    face = cv2.GaussianBlur(face.astype(np.float32), (0, 0), 4)
    faces = rows(face_row(0, 0, 400, 400))     # its width sets how far an edge can be snapped
    x0, y0, area = skin_segment.skin_area(segments(face, np.zeros((200, 200)), face, work), faces, face_only=True)
    row = np.zeros(200, np.float32)
    row[x0:x0 + area.shape[1]] = area[60 - y0]
    assert row[110] > 0.9 and row[90] < 0.1, "skin on the bright side only"
    assert row[103] - row[97] > 0.5, "and a sharper edge than the model's"


def test_older_mask_is_used_without_the_model():
    img = skin_photo()
    faces = rows(face_row(300, 200, 400, 500))
    saved = skin_segment.segment
    try:
        skin_segment.segment = lambda *_: None
        mask = smoother.build_skin_mask(img, faces=faces)
    finally:
        skin_segment.segment = saved
    assert mask[500, 500] > 0.5 and mask[50, 50] == 0


def test_a_face_the_model_misses_uses_the_older_mask():
    img = skin_photo()
    faces = rows(face_row(300, 200, 400, 500))
    empty = np.zeros(img.shape[:2], np.float32)
    seg = skin_segment.Segments(0, 0, img.shape[1], img.shape[0], img, empty, empty, empty, ())
    saved = skin_segment.segment
    try:
        skin_segment.segment = lambda *_: seg
        mask = smoother.build_skin_mask(img, faces=faces)
    finally:
        skin_segment.segment = saved
    assert mask[500, 500] > 0.5, "not a hole"


def test_a_model_that_fails_while_running_uses_the_older_mask():
    if not skin_segment.have_model():
        print("      (no segmentation model here; skipped)")
        return
    img = skin_photo()
    faces = rows(face_row(300, 200, 400, 500))
    saved = skin_segment._scores

    def fails(*_):
        raise cv2.error("this OpenCV can't run the model")

    try:
        skin_segment._scores = fails
        assert skin_segment.segment(img, faces) is None
        assert smoother.build_skin_mask(img, faces=faces)[500, 500] > 0.5, "not a crash, and not a hole"
    finally:
        skin_segment._scores = saved
        skin_segment._net_failed = False


def test_model_scores_add_up_and_cover_the_region():
    img = skin_photo()
    if not skin_segment.have_model():
        print("      (no segmentation model here; skipped)")
        return
    faces = rows(face_row(300, 200, 400, 500))
    seg = skin_segment.segment(img, faces)
    assert seg.face.shape == seg.work.shape[:2]
    assert seg.face.min() >= -1e-3 and (seg.face + seg.body).max() <= 1.001 and seg.person.max() <= 1.001
    probs = skin_segment._scores(skin_segment._get_net(), img)
    assert np.abs(probs.sum(axis=0) - 1).max() < 1e-4


def outline_points() -> np.ndarray:
    """Landmarks with each feature a circle of radius 30: eyes 200 px apart. The face's
    outline is an ellipse 500 px wide, its sides at x 154 and 646 level with the eyes."""
    pts = np.zeros((face_landmarks.POINTS, 2), np.float64)
    for ids, (cx, cy) in ((face_landmarks.EYE_R, (300, 300)), (face_landmarks.EYE_L, (500, 300)),
                          (face_landmarks.BROW_R, (300, 200)), (face_landmarks.BROW_L, (500, 200)),
                          (face_landmarks.LIPS, (400, 600))):
        for k, i in enumerate(ids):
            t = 2 * np.pi * k / len(ids)
            pts[i] = cx + 30 * np.cos(t), cy + 30 * np.sin(t)
    # The corners that set the eyes' distance, kept inside their circles.
    pts[list(face_landmarks.EYE_CORNERS_R)] = (270, 300), (330, 300)
    pts[list(face_landmarks.EYE_CORNERS_L)] = (470, 300), (530, 300)
    for k, i in enumerate(face_landmarks.OUTLINE):     # clockwise from the top
        t = 2 * np.pi * k / len(face_landmarks.OUTLINE)
        pts[i] = 400 + 250 * np.sin(t), 380 - 330 * np.cos(t)
    return pts


def turned_points() -> np.ndarray:
    """outline_points() turned to the person's right: their left side of the outline,
    on the right of the photo, runs over the cheek, short of the cheek's edge at x 646."""
    pts = outline_points()
    left = list(face_landmarks.OUTLINE_L)
    pts[left, 0] = 400 + 0.6 * (pts[left, 0] - 400)
    return pts


def painted(zones, w, h) -> np.ndarray:
    area = np.zeros((h, w), np.uint8)
    for m, origin in zones:
        smoother._paint(area, m, origin)
    return area


def test_outlines_cover_each_feature_and_stop_past_its_margin():
    area = painted(face_landmarks.zones(outline_points(), 800, 800), 800, 800)
    for x, y in ((300, 300), (500, 300), (300, 200), (500, 200), (400, 600)):
        assert area[y, x] == 255
    assert area[300, 300 + 30 + 3] == 255, "eyes reach 5% of their distance past the outline"
    assert area[300, 300 + 30 + 8] == 0
    assert area[600, 400 + 30 + 4] == 0, "lips reach 1% past theirs"
    assert area[450, 400] == 0, "the skin between them stays skin"


def test_outlines_at_the_photo_edge():
    pts = outline_points()
    area = painted(face_landmarks.zones(pts - (0, 590), 800, 800), 800, 800)    # lips near the top, eyes off it
    assert area[10, 400] == 255
    assert len(face_landmarks.zones(pts - (0, 2000), 800, 800)) == 0, "all of it off the photo"


class FakeLandmarkNet:
    def __init__(self):
        self.x = None
        self.points = np.zeros((1, 1, 1, face_landmarks.POINTS * 3), np.float32)

    def setInput(self, x):
        self.x = x

    def getUnconnectedOutLayersNames(self):
        return "points", "flag", "other"

    def forward(self, _names):
        return [self.points, np.full((1, 1), 5.0, np.float32), np.full((1, 1), 0.5, np.float32)]


def test_landmark_crop_and_points_share_one_mapping():
    img = np.zeros((800, 800, 3), np.uint8)
    cv2.circle(img, (440, 400), 3, (0, 0, 255), -1)     # 40 px right of the crop's centre
    net = FakeLandmarkNet()
    net.points[0, 0, 0, :2] = 128 + 100, 128            # point 0: 100 model pixels right of centre

    # Upright, one photo pixel per model pixel: the mark sits right of centre.
    pts, flag = face_landmarks._run(net, img, 400.0, 400.0, 256.0, 0.0)
    assert flag == 5.0, "the first single value"
    assert net.x.shape == (1, 3, 256, 256)
    assert net.x[0, 0, 128, 128 + 40] > 0.9 and net.x[0, 2, 128, 128 + 40] < 0.1, "red comes first"
    assert np.allclose(pts[0], (500, 400), atol=1e-6)

    # Turned 90 degrees: the crop's right is the photo's down, so the mark is above centre.
    pts, _ = face_landmarks._run(net, img, 400.0, 400.0, 256.0, np.pi / 2)
    assert net.x[0, 0, 128 - 40, 128] > 0.9
    assert np.allclose(pts[0], (400, 500), atol=1e-6)

    # A crop four times the model's size goes through the shrink.
    pts, _ = face_landmarks._run(net, img, 400.0, 400.0, 1024.0, 0.0)
    assert net.x[0, 0, 128, 128 + 10] > 0.3, "the mark, a quarter of the size"
    assert np.allclose(pts[0], (800, 400), atol=1e-6)

    assert face_landmarks._run(net, img, 5000.0, 400.0, 256.0, 0.0) is None, "misses the photo"


def with_landmark_stubs(fit):
    """landmarks() with this _fit and a net that's never touched."""
    saved = face_landmarks._fit, face_landmarks._get_net
    face_landmarks._fit, face_landmarks._get_net = fit, lambda: object()
    return saved


def test_a_weak_landmark_fit_uses_the_older_method():
    faces = rows(face_row(100, 100, 200, 200), face_row(500, 100, 200, 200))
    pts = outline_points()
    saved = with_landmark_stubs(lambda _n, _b, f: (pts, 1.0 if f[0] < 300 else -1.0))
    try:
        found = face_landmarks.landmarks(np.zeros((800, 800, 3), np.uint8), faces)
    finally:
        face_landmarks._fit, face_landmarks._get_net = saved
    assert found[0] is pts and found[1] is None


def test_the_smallest_faces_past_the_limit_use_the_older_method():
    n = face_landmarks.MAX_FACES + 2
    faces = rows(*[face_row(10, 10, 50 + i, 50 + i) for i in range(n)])
    pts = outline_points()
    saved = with_landmark_stubs(lambda *_: (pts, 5.0))
    try:
        found = face_landmarks.landmarks(np.zeros((800, 800, 3), np.uint8), faces)
    finally:
        face_landmarks._fit, face_landmarks._get_net = saved
    assert found[0] is None and found[1] is None and all(p is pts for p in found[2:])


def test_a_landmark_model_that_fails_while_running_uses_the_older_method():
    if not face_landmarks.have_model():
        print("      (no landmark model here; skipped)")
        return
    img = skin_photo()
    faces = rows(face_row(300, 200, 400, 500))
    saved = face_landmarks._run

    def fails(*_):
        raise cv2.error("this OpenCV can't run the model")

    try:
        face_landmarks._run = fails
        assert face_landmarks.landmarks(img, faces) == [None]
        assert smoother.build_skin_mask(img, faces=faces)[500, 500] > 0.5, "not a crash, and not a hole"
    finally:
        face_landmarks._run = saved
        face_landmarks._net_failed = False


def test_the_mask_uses_outlines_and_only_falls_back_without_them():
    img = skin_photo()
    faces = rows(face_row(150, 100, 500, 600))
    calls = []
    saved = face_landmarks.landmarks, smoother.lip_mask, face_landmarks.get_model_path

    def counted(*args):
        calls.append(1)
        return saved[1](*args)

    try:
        smoother.lip_mask = counted
        face_landmarks.landmarks = lambda _b, _f: [outline_points()]
        with_outlines = smoother.build_skin_mask(img, faces=faces)
        assert not calls and with_outlines[600, 400] == 0, "the outline, not the finder"

        face_landmarks.landmarks = lambda _b, _f: [None]
        smoother.build_skin_mask(img, faces=faces)
        assert len(calls) == 1, "a face without points uses the finder"

        calls.clear()
        face_landmarks.landmarks = saved[0]
        face_landmarks.get_model_path = lambda: None
        assert smoother.build_skin_mask(img, faces=faces)[150, 400] > 0.5, "no model file: still a mask"
        assert len(calls) == 1
    finally:
        face_landmarks.landmarks, smoother.lip_mask, face_landmarks.get_model_path = saved


def test_ear_zones_lie_beside_the_outline_from_ear_top_to_lobe():
    zones, inside = face_landmarks.ear_zones(outline_points(), 800, 800)
    ears = painted(zones, 800, 800)
    assert len(zones) == 2
    assert ears[322, 80] == 255 and ears[322, 720] == 255, "beside the outline, level with the eyes"
    assert ears[150, 200] == 255 and ears[455, 140] == 255, "up to the ear's top, down to its lobe"
    assert ears[322, 152] == 0 and ears[322, 170] == 0, "a gap, then the cheek"
    assert ears[100, 160] == 0 and ears[520, 100] == 0, "nothing above the ear or below the lobe"
    assert ears[322, 5] == 0, "nothing past the farthest an ear reaches"
    own = painted([inside], 800, 800)
    assert own[322, 170] == 255 and own[322, 140] == 0, "the inside is the outline"


def test_a_side_turned_away_gets_no_ear_zone():
    zones, inside = face_landmarks.ear_zones(turned_points(), 800, 800)
    ears = painted(zones, 800, 800)
    assert ears[322, 80] == 255, "the near side's ear"
    assert ears[:, 400:].max() == 0, "the far side: its outline is on the cheek and its ear is hidden"
    assert painted([inside], 800, 800)[322, 600] == 255, "the far cheek past its outline counts as inside"


def test_ear_zones_off_the_photo():
    assert face_landmarks.ear_zones(outline_points() - (0, 2000), 800, 800) == ([], None)
    zones, _ = face_landmarks.ear_zones(outline_points() - (100, 0), 800, 800)
    assert len(zones) == 2, "the right ear's zone partly off the photo"
    assert face_landmarks.ear_zones(np.zeros((face_landmarks.POINTS, 2)), 800, 800) == ([], None)


def all_skin(bgr, faces):
    """Segments saying the whole photo is face skin, for every face."""
    h, w = bgr.shape[:2]
    ones = np.ones((h, w), np.float32)
    return skin_segment.Segments(0, 0, w, h, bgr, ones, np.zeros((h, w), np.float32), ones, tuple(range(len(faces))))


def mask_with(img, faces, points) -> np.ndarray:
    """build_skin_mask (Face only) with every pixel skin and these landmarks per face."""
    saved = skin_segment.segment, face_landmarks.landmarks
    try:
        skin_segment.segment, face_landmarks.landmarks = all_skin, lambda _b, _f: points
        return smoother.build_skin_mask(img, faces=faces, face_only=True)
    finally:
        skin_segment.segment, face_landmarks.landmarks = saved


def test_ears_are_cut_from_the_mask_and_the_cheeks_are_not():
    mask = mask_with(skin_photo(800, 800), rows(face_row(150, 50, 500, 660)), [outline_points()])
    assert mask[322, 80] < 0.05 and mask[322, 720] < 0.05, "ears"
    assert mask[400, 200] > 0.95 and mask[400, 600] > 0.95, "cheeks"
    assert mask[560, 100] > 0.95 and mask[60, 100] > 0.95, "below the lobe and above the ear"


def test_an_ear_zone_never_cuts_another_face():
    img = skin_photo(800, 1300)
    faces = rows(face_row(150, 50, 500, 660), face_row(650, 50, 500, 660))
    mask = mask_with(img, faces, [outline_points(), outline_points() + (500, 0)])
    assert mask[322, 700] > 0.95, "the right face's cheek, in the left face's ear zone"
    assert mask[322, 600] > 0.95, "and the other way round"
    assert mask[322, 80] < 0.05 and mask[322, 1220] < 0.05, "the outer ears are still cut"

    # The far side of a turned face: its cheek reaches past its outline.
    turned = turned_points() + (500, 0)
    turned[:, 0] = 1800 - turned[:, 0]                  # mirrored, so its far side faces the left face
    mask = mask_with(img, faces, [outline_points(), turned])
    assert mask[322, 690] > 0.95, "its far cheek, past its outline, in the left face's ear zone"


def test_a_face_without_landmarks_keeps_its_ears_and_its_cheek():
    img = skin_photo(800, 1400)
    alone = mask_with(img[:, :800], rows(face_row(150, 50, 500, 660)), [None])
    assert alone[322, 80] > 0.95 and alone[322, 720] > 0.95, "nothing marks where its ears are"
    # Its box starts at x 800; a turned face's near cheek can reach well past it.
    faces = rows(face_row(150, 50, 500, 660), face_row(800, 50, 500, 660))
    mask = mask_with(img, faces, [outline_points(), None])
    assert mask[322, 700] > 0.95, "its cheek could reach into the left face's ear zone"
    assert mask[322, 80] < 0.05, "the left face's other ear is still cut"


def test_landmark_model_gives_one_answer_per_face():
    if not face_landmarks.have_model():
        print("      (no landmark model here; skipped)")
        return
    img = skin_photo()
    faces = rows(face_row(300, 200, 400, 500))
    found = face_landmarks.landmarks(img, faces)
    assert len(found) == 1 and (found[0] is None or found[0].shape == (face_landmarks.POINTS, 2))
    fit = face_landmarks._fit(face_landmarks._get_net(), img, faces[0])
    assert np.isfinite(fit[0]).all() and np.isfinite(fit[1])


def test_spots_are_only_looked_for_near_faces():
    img = skin_with(lambda a: darken_circle(a, 20, 880, 6, 30))     # FACE reaches x 50..950
    mask = np.ones(img.shape[:2], np.float32)
    corr, _ = blemish_classic.correction(img, FACE, mask)
    assert changed_at(corr, 20, 880) == 0


def skin_with(mark) -> np.ndarray:
    """skin_photo with something drawn on it. mark(img, level) draws in BGR."""
    img = skin_photo().astype(np.int32)
    mark(img)
    return np.clip(img, 0, 255).astype(np.uint8)


def darken_circle(img, cx, cy, r, by):
    """Dark in every colour, like a mole."""
    yy, xx = np.ogrid[0:img.shape[0], 0:img.shape[1]]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] -= by


def redden_circle(img, cx, cy, r, by):
    """Dark in blue and green only, like a red blemish."""
    yy, xx = np.ogrid[0:img.shape[0], 0:img.shape[1]]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r, :2] -= by


def changed_at(corr, x, y) -> float:
    """How much a correction changes the pixel at x, y."""
    if corr is None:
        return 0.0
    x0, y0, c = corr
    if not (0 <= y - y0 < c.shape[0] and 0 <= x - x0 < c.shape[1]):
        return 0.0
    return float(np.abs(c[y - y0, x - x0]).max())


def test_classic_removes_a_red_spot():
    img = skin_with(lambda a: redden_circle(a, 480, 420, 6, 30))
    corr, count = blemish_classic.correction(img, FACE, oval_mask())
    assert count >= 1
    healed = blemish.apply(img, corr, 1.0)
    def depth(a):
        g = a[:, :, 1].astype(np.float32)
        return g[405:435, 465:495].mean() - g[418:423, 478:483].mean()
    assert depth(healed) < 0.35 * depth(img), "the spot should nearly vanish"


def test_classic_leaves_a_mole_that_is_only_moderately_dark():
    # Under the darkness limit, but dark in red as well: not redness.
    img = skin_with(lambda a: darken_circle(a, 480, 420, 6, 45))
    corr, _ = blemish_classic.correction(img, FACE, oval_mask())
    assert changed_at(corr, 480, 420) < 3, "a half-removed mole leaves a smudge"


def test_classic_treats_dark_skin_the_same():
    # The same marks on much darker skin: the red limit is a share, not a fixed depth.
    dark = lambda a: a.__ifloordiv__(2)
    red = skin_with(lambda a: (dark(a), redden_circle(a, 480, 420, 6, 15)))
    assert blemish_classic.correction(red, FACE, oval_mask())[1] >= 1, "a red mark"
    mole = skin_with(lambda a: (dark(a), darken_circle(a, 480, 420, 6, 22)))
    corr, _ = blemish_classic.correction(mole, FACE, oval_mask())
    assert changed_at(corr, 480, 420) < 3, "a mole"


def test_classic_leaves_a_long_line():
    def line(a):
        a[420:423, 380:560] -= 30
    img = skin_with(line)
    corr, _ = blemish_classic.correction(img, FACE, oval_mask())
    assert changed_at(corr, 470, 421) < 3, "a hair or crease is not a spot"


def test_classic_leaves_a_very_dark_mole():
    img = skin_with(lambda a: darken_circle(a, 480, 420, 6, 110))
    corr, _ = blemish_classic.correction(img, FACE, oval_mask())
    assert changed_at(corr, 480, 420) < 3


def test_classic_ignores_shapes_too_large_to_be_a_spot():
    img = skin_with(lambda a: darken_circle(a, 480, 420, 24, 30))
    corr, _ = blemish_classic.correction(img, FACE, oval_mask())
    assert changed_at(corr, 480, 420) < 3


def test_classic_stays_inside_the_mask():
    img = skin_with(lambda a: darken_circle(a, 120, 120, 6, 30))
    corr, _ = blemish_classic.correction(img, FACE, oval_mask())
    assert changed_at(corr, 120, 120) == 0


def test_classic_without_faces_does_nothing():
    img = skin_photo()
    assert blemish_classic.correction(img, np.zeros((0, 15), np.float32), oval_mask()) == (None, 0)


def test_engine_switch_reaches_the_classic_detector():
    img = skin_with(lambda a: redden_circle(a, 480, 420, 6, 30))
    saved = blemish.ENGINE
    try:
        blemish.ENGINE = "classic"
        assert blemish.have_model()
        corr, count = blemish.correction(img, FACE, oval_mask())
        assert corr is not None and "spot" in blemish.count_text(count)
    finally:
        blemish.ENGINE = saved


# ---------- the .exe's updater ----------

TEST_LATEST = updater.TEST_PREFIX + "9/latest"
NEW_PROGRAM = b"MZ" + bytes(range(256)) * 40


class FakeWeb:
    """Stands in for updater._request: URL -> bytes, an exception to raise, or missing."""

    def __init__(self, pages):
        self.pages, self.asked = pages, []

    def __call__(self, url):
        self.asked.append(url)
        page = self.pages.get(url)
        if page is None:
            raise urllib.error.URLError("no connection")
        if isinstance(page, BaseException):
            raise page
        return io.BytesIO(page)


def fake_release(program: bytes, tag: str = "v99.0.0", base: str | None = None, digest: bool = True,
                 checksum: str | None = None, size: int | None = None) -> dict:
    """The pages of a release in GitHub's format, served from TEST_LATEST."""
    base = base or updater.TEST_PREFIX + "9/download/"
    real = hashlib.sha256(program).hexdigest()
    exe = {"name": "SkinSmoother.exe", "size": len(program) if size is None else size,
           "browser_download_url": base + "SkinSmoother.exe"}
    if digest:
        exe["digest"] = "sha256:" + real
    sums = {"name": "SkinSmoother.exe.sha256", "size": 84, "browser_download_url": base + "SkinSmoother.exe.sha256"}
    release = {"tag_name": tag, "prerelease": False, "draft": False, "assets": [exe, sums]}
    return {TEST_LATEST: json.dumps(release).encode(), exe["browser_download_url"]: program,
            sums["browser_download_url"]: f"{checksum or real}  SkinSmoother.exe\n".encode()}


@contextlib.contextmanager
def program_folder(pages: dict):
    """A folder holding a 'running' SkinSmoother.exe, with the web replaced by pages."""
    folder = tempfile.mkdtemp()
    exe = os.path.join(folder, "SkinSmoother.exe")
    with open(exe, "wb") as fh:
        fh.write(b"MZ the program that is running")
    saved = updater._request
    updater._request = FakeWeb(pages)
    try:
        yield exe, updater._request
    finally:
        updater._request = saved
        updater._set(state="off", latest=None, message="")
        shutil.rmtree(folder, ignore_errors=True)


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def unchanged(exe: str) -> bool:
    return (read_bytes(exe).startswith(b"MZ the program") and not os.path.exists(exe + ".download")
            and not os.path.exists(exe + ".old"))


def test_versions_compare_and_odd_tags_are_ignored():
    p = updater.parse_version
    assert p("v1.2.10") == (1, 2, 10) and p("1.2.3") == (1, 2, 3)
    assert p("v1.2") is None and p("v1.2.3-rc1") is None and p(None) is None and p("latest") is None
    assert p("v1.10.0") > p("v1.9.9")
    assert p(updater.VERSION) is not None, "version.py holds a plain x.y.z"


def test_a_newer_release_replaces_the_program_and_keeps_the_running_copy():
    with program_folder(fake_release(NEW_PROGRAM)) as (exe, _):
        s = updater.run(exe, TEST_LATEST)
        assert s["state"] == "ready" and s["latest"] == "99.0.0" and "open the program again" in s["message"], s
        assert read_bytes(exe) == NEW_PROGRAM and read_bytes(exe + ".old").startswith(b"MZ the program")
        assert not os.path.exists(exe + ".download")
        updater.cleanup(exe)        # the next start
        assert not os.path.exists(exe + ".old") and read_bytes(exe) == NEW_PROGRAM


def test_the_same_or_an_older_release_is_left_alone():
    for tag in ("v" + updater.VERSION, "v0.0.1", "v1.0", "nightly"):
        with program_folder(fake_release(NEW_PROGRAM, tag=tag)) as (exe, web):
            assert updater.run(exe, TEST_LATEST)["state"] == "current", tag
            assert web.asked == [TEST_LATEST] and unchanged(exe), tag


def test_a_pre_release_is_left_alone():
    pages = fake_release(NEW_PROGRAM)
    release = json.loads(pages[TEST_LATEST])
    release["prerelease"] = True
    pages[TEST_LATEST] = json.dumps(release).encode()
    with program_folder(pages) as (exe, _):
        assert updater.run(exe, TEST_LATEST)["state"] == "current" and unchanged(exe)


def test_no_release_or_no_connection_says_nothing():
    gone = urllib.error.HTTPError(TEST_LATEST, 404, "Not Found", None, None)
    limited = urllib.error.HTTPError(TEST_LATEST, 403, "rate limited", None, None)
    for page, state in ((gone, "current"), (limited, "offline"), (None, "offline"), (b"<html>busy</html>", "offline")):
        with program_folder({TEST_LATEST: page}) as (exe, _):
            s = updater.run(exe, TEST_LATEST)
            assert s["state"] == state and s["message"] == "" and unchanged(exe), (state, s)


def test_a_damaged_download_leaves_the_program_alone():
    cases = {
        "doesn't match its checksum": fake_release(NEW_PROGRAM, checksum="0" * 64, digest=False),
        "checksums don't agree": fake_release(NEW_PROGRAM, checksum="0" * 64),
        "stopped part-way": fake_release(NEW_PROGRAM, size=len(NEW_PROGRAM) + 10),
        "larger than expected": fake_release(NEW_PROGRAM, size=len(NEW_PROGRAM) - 10),
        "isn't a Windows program": fake_release(b"<html>" + NEW_PROGRAM),
    }
    for words, pages in cases.items():
        with program_folder(pages) as (exe, _):
            s = updater.run(exe, TEST_LATEST)
            assert s["state"] == "failed" and words in s["message"] and "try again" in s["message"], (words, s)
            assert unchanged(exe), words


def test_downloads_only_come_from_the_project_page():
    for base in ("https://example.com/", updater.TEST_PREFIX + "9/download/"):
        pages = fake_release(NEW_PROGRAM, base=base)
        pages[updater.LATEST_URL] = pages.pop(TEST_LATEST)       # GitHub's real address
        with program_folder(pages) as (exe, web):
            s = updater.run(exe)
            assert s["state"] == "failed" and "project's GitHub page" in s["message"], (base, s)
            assert web.asked == [updater.LATEST_URL] and unchanged(exe), "nothing else was fetched"


def test_a_busy_file_is_retried_and_a_failed_swap_puts_the_program_back():
    real_replace, real_remove, real_wait = os.replace, os.remove, updater.RETRY_FOR
    calls = []

    def busy_once(a, b):
        calls.append(a)
        if len(calls) == 1:
            raise PermissionError("being scanned")
        return real_replace(a, b)

    def refuse_new(a, b):
        if a.endswith(".download"):
            raise OSError("disk trouble")
        return real_replace(a, b)

    def old_copy_open(path):
        if path.endswith(".old"):
            raise PermissionError("in use")
        return real_remove(path)

    try:
        updater.RETRY_FOR = 0.3
        with program_folder(fake_release(NEW_PROGRAM)) as (exe, _):
            os.replace = busy_once
            assert updater.run(exe, TEST_LATEST)["state"] == "ready" and read_bytes(exe) == NEW_PROGRAM
        with program_folder(fake_release(NEW_PROGRAM)) as (exe, _):
            os.replace = refuse_new
            assert updater.run(exe, TEST_LATEST)["state"] == "failed" and unchanged(exe)
        os.replace = real_replace
        with program_folder(fake_release(NEW_PROGRAM)) as (exe, _):
            with open(exe + ".old", "wb") as fh:
                fh.write(b"MZ an older copy, still open")
            os.remove = old_copy_open
            s = updater.run(exe, TEST_LATEST)
            os.remove = real_remove
            assert s["state"] == "failed" and "still open" in s["message"], s
            assert read_bytes(exe).startswith(b"MZ the program") and not os.path.exists(exe + ".download")
    finally:
        os.replace, os.remove, updater.RETRY_FOR = real_replace, real_remove, real_wait


def test_the_updater_is_off_from_source_and_when_switched_off():
    updater._set(state="off")
    updater.start()
    assert updater.status()["state"] == "off", "running from source"
    had_frozen, frozen = hasattr(sys, "frozen"), getattr(sys, "frozen", None)
    had_switch = os.environ.get("SKINSMOOTHER_NO_UPDATE")
    try:
        sys.frozen = True
        os.environ["SKINSMOOTHER_NO_UPDATE"] = "1"
        updater.start()
        assert updater.status()["state"] == "off", "switched off"
    finally:
        if had_frozen:
            sys.frozen = frozen
        else:
            del sys.frozen
        if had_switch is None:
            os.environ.pop("SKINSMOOTHER_NO_UPDATE", None)
        else:
            os.environ["SKINSMOOTHER_NO_UPDATE"] = had_switch


# ---------- under-eyes and skin tone (look.py) ----------

def eye_points() -> np.ndarray:
    """outline_points() with real lower lids: each a shallow arc from the outer corner to
    the inner one, the eyes 200 px apart at y 300 and the chin at the outline's foot."""
    pts = outline_points()
    for lid, outer, inner in ((face_landmarks.LOWER_LID_R, 270, 330), (face_landmarks.LOWER_LID_L, 530, 470)):
        for k, i in enumerate(lid):
            u = k / (len(lid) - 1)
            pts[i] = outer + u * (inner - outer), 300 + 12 * np.sin(np.pi * u)
    return pts


def eye_photo(circle: float = 30.0) -> np.ndarray:
    """Even skin with grain, and a dark circle under the eye on the left of the photo."""
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:800, 0:800].astype(np.float32)
    tone = 170 + rng.normal(0, 3, (800, 800)) - circle * np.exp(-((xx - 300) ** 2 + (yy - 338) ** 2) / (2 * 15.0 ** 2))
    return np.clip(np.dstack([tone * 0.72, tone * 0.83, tone]), 0, 255).astype(np.uint8)


def test_under_eye_zones_lie_below_the_lids_and_follow_the_face():
    zones, (dx, dy), eye_d = face_landmarks.under_eye_zones(eye_points(), 800, 800)
    area = painted(zones, 800, 800)
    assert len(zones) == 2 and eye_d == 200 and abs(dx) < 1e-9 and dy == 200, "one eye distance down the face"
    assert area[340, 300] == 255 and area[340, 500] == 255, "under each eye"
    assert area[298, 300] == 0 and area[302, 275] == 0, "the eye itself and its lid stay out"
    assert area[340, 338] == 255 and area[340, 345] == 0, "a little past the inner corner, towards the nose"
    assert area[340, 265] == 255 and area[340, 258] == 0, "less past the outer one"
    assert area[372, 300] == 255 and area[385, 300] == 0, "down as far as dark circles reach"
    # The same face upside down: the zones are still on its chin's side.
    flipped = eye_points() * (1, -1) + (0, 800)
    zones, (dx, dy), _ = face_landmarks.under_eye_zones(flipped, 800, 800)
    assert dy == -200 and painted(zones, 800, 800)[460, 300] == 255
    assert face_landmarks.under_eye_zones(eye_points() - (0, 2000), 800, 800)[0] == [], "off the photo"


def test_look_is_off_by_default_and_without_landmarks():
    img, mask = eye_photo(), np.ones((800, 800), np.float32)
    assert look.finish(img, mask, mask, [eye_points()], 0.4) is img, "both steps are off unless asked for"
    assert look.under_eyes(img, [None], mask, 1.0) is img and look.under_eyes(img, [], mask, 1.0) is img
    assert look.under_eyes(img, [eye_points()], np.zeros((800, 800), np.float32), 1.0) is img, "no skin, no change"
    assert look.balance(img, mask, None, 1.0) is img, "no measured brightness, no change"


def test_under_eyes_evens_a_dark_circle_and_keeps_the_grain():
    img, mask = eye_photo(), np.ones((800, 800), np.float32)
    luma = lambda a: a[328:348, 290:310].astype(np.float32).mean()
    out = look.under_eyes(img, [eye_points()], mask, 1.0)
    half = look.under_eyes(img, [eye_points()], mask, 0.5)
    clean = float(img[328:348, 490:510].mean())
    gap, left, left_half = clean - luma(img), clean - luma(out), clean - luma(half)
    assert gap > 15 and left < 0.5 * gap, (gap, left)
    assert left < left_half < gap, "half the amount, less of it"
    green = lambda a: a[:, :, 1].astype(np.float32)
    fine = lambda a: (green(a) - cv2.GaussianBlur(green(a), (0, 0), 2))[320:360, 280:320].std()
    assert 0.9 < fine(out) / fine(img) < 1.1, "the grain stays"
    assert abs(float(out[328:348, 490:510].mean()) - clean) < 1.0, "even skin under the other eye stays as it is"
    assert np.array_equal(out[450:], img[450:]) and np.array_equal(out[:290], img[:290]), "nothing away from the eyes"
    assert np.array_equal(out[:, :200], img[:, :200])


def test_under_eyes_keeps_to_the_mask_and_is_safe_at_the_photo_edge():
    img = eye_photo()
    mask = np.ones((800, 800), np.float32)
    mask[:, :300] = 0                                       # a Restore stroke over half the zone
    out = look.under_eyes(img, [eye_points()], mask, 1.0)
    assert np.array_equal(out[:, :300], img[:, :300]) and not np.array_equal(out[:, 300:], img[:, 300:])
    low = eye_points() + (0, 470)                           # the cheeks are off the photo
    assert look.under_eyes(img, [low], np.ones((800, 800), np.float32), 1.0).shape == img.shape
    assert look.under_eyes(img, [eye_points() + (0, 1000)], np.ones((800, 800), np.float32), 1.0) is img


def test_skin_brightness_is_read_from_the_faces_skin():
    img = np.full((400, 600, 3), 200, np.uint8)
    img[100:300, 100:300] = (60, 120, 180)                  # the face
    mask = np.ones((400, 600), np.float32)
    mask[100:300, 100:150] = 0.5                            # its soft edge doesn't count
    faces = rows(face_row(100, 100, 200, 200))
    expected = (114 * 60 + 587 * 120 + 299 * 180) / 255000
    assert abs(look.skin_luma(img, mask, faces) - expected) < 1e-9
    assert abs(look.skin_luma(img, mask, None) - 200 / 255) < 1e-9, "without faces, all the skin in the mask"
    assert look.skin_luma(img, np.zeros((400, 600), np.float32), faces) is None
    assert look.skin_luma(img, mask, rows(face_row(700, 500, 50, 50))) is None, "a face off the photo"


def test_skin_tone_leaves_the_middle_range_alone_and_pulls_the_rest_part_way():
    t = look.BALANCE
    for luma in (t["centre"], t["centre"] - 0.9 * t["slack"], t["centre"] + 0.9 * t["slack"]):
        assert look.wanted_luma(luma, 1.0) == luma
    dark, bright = t["centre"] - t["slack"] - 0.1, t["centre"] + t["slack"] + 0.1
    assert abs(look.wanted_luma(dark, 1.0) - (dark + 0.1 * t["pull"])) < 1e-9
    assert abs(look.wanted_luma(bright, 1.0) - (bright - 0.1 * t["pull"])) < 1e-9
    assert abs(look.wanted_luma(dark, 0.5) - (dark + 0.05 * t["pull"])) < 1e-9, "half the amount, half the way"
    assert look.wanted_luma(dark, 0.0) == dark
    assert abs(look.wanted_luma(0.1, 1.0) - (0.1 + t["limit"])) < 1e-9, "never further than the limit"
    assert dark < look.wanted_luma(dark, 1.0) < t["centre"] - t["slack"], "never past the range's edge"


def test_skin_tone_moves_dark_skin_to_where_it_should_be_through_the_area_only():
    img = (skin_photo(600, 600).astype(np.float32) * 0.62).astype(np.uint8)      # dark skin
    img[:, 540:550] = 0
    img[10:20, 10:20] = 255
    area = np.zeros((600, 600), np.float32)
    area[:, :400] = 1.0
    area[:, 400:450] = 0.5
    faces = rows(face_row(50, 50, 300, 400))
    luma = look.skin_luma(img, area, faces)
    want = look.wanted_luma(luma, 1.0)
    assert luma < look.BALANCE["centre"] - look.BALANCE["slack"] and want > luma + 0.02
    out = look.balance(img, area, luma, 1.0)
    assert abs(look.skin_luma(out, area, faces) - want) < 0.006, (look.skin_luma(out, area, faces), want)
    assert np.array_equal(out[:, 450:], img[:, 450:]), "nothing outside the area"
    assert (out[10:20, 10:20] == 255).all(), "white stays white"
    part = float(out[:, 410:440, 2].astype(np.float32).mean() - img[:, 410:440, 2].mean())
    full = float(out[:, 300:390, 2].astype(np.float32).mean() - img[:, 300:390, 2].mean())
    assert 0.3 * full < part < 0.7 * full, "half as much where the area is half"
    ratio = lambda a: a[:, :400, 0].astype(np.float32).mean() / a[:, :400, 2].astype(np.float32).mean()
    assert ratio(out) > ratio(img), "brightening takes a little colour out, as a curve on the skin does"
    half = look.balance(img, area, luma, 0.5)
    assert luma < look.skin_luma(half, area, faces) < look.skin_luma(out, area, faces)
    black = np.zeros((50, 50, 3), np.uint8)
    assert not look.balance(black, np.ones((50, 50), np.float32), luma, 1.0).any(), "black stays black"
    # A bright face comes down, and one in the middle of the range isn't touched.
    light = np.clip(skin_photo(600, 600).astype(np.float32) * 1.22, 0, 255).astype(np.uint8)
    luma = look.skin_luma(light, area, faces)
    assert look.skin_luma(look.balance(light, area, luma, 1.0), area, faces) < luma - 0.02
    assert look.balance(img, area, look.BALANCE["centre"], 1.0) is img


def glossy_patch_photo() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Even skin with grain, a whitish shiny patch on the left and a blush (redder, no
    brighter) on the right: (photo, mask over the middle, faces)."""
    rng = np.random.default_rng(9)
    yy, xx = np.mgrid[0:600, 0:800].astype(np.float32)
    grain = rng.normal(0, 3, (600, 800))
    img = np.dstack([118 + grain, 138 + grain, 190 + grain])                 # B, G, R
    shine = np.exp(-((xx - 250) ** 2 + (yy - 300) ** 2) / (2 * 45.0 ** 2))
    img += shine[:, :, None] * np.array([95.0, 80.0, 50.0])                  # towards white
    blush = np.exp(-((xx - 560) ** 2 + (yy - 300) ** 2) / (2 * 45.0 ** 2))
    img += blush[:, :, None] * np.array([-18.0, -22.0, 30.0])                # redder, not brighter
    mask = np.zeros((600, 800), np.float32)
    mask[40:560, 60:740] = 1.0
    return np.clip(img, 0, 255).astype(np.uint8), mask, rows(face_row(100, 60, 600, 480))


def test_the_brighter_skins_colour_is_read_between_two_levels_of_brightness():
    img = np.zeros((100, 1000, 3), np.uint8)
    for i in range(1000):                              # a ramp: each column a little brighter
        img[:, i] = (i // 5, i // 5 + 20, i // 4)
    mask = np.ones((100, 1000), np.float32)
    b, g, r = look.bright_colour(img, mask, None)
    lo, hi = look.MATTE["from"], look.MATTE["to"]
    ordered = np.sort(img.reshape(-1, 3).astype(np.int32) @ np.array([114, 587, 299]))
    levels = img[0].astype(np.int32) @ np.array([114, 587, 299])
    band = img[0][(levels >= ordered[int(lo * 99999)]) & (levels <= ordered[int(hi * 99999)])]
    assert np.allclose([b, g, r], band.mean(axis=0) / 255.0, atol=1e-9)
    assert img[0, 690, 2] / 255 < r < img[0, 810, 2] / 255, "from the brighter skin, short of the brightest"
    assert look.bright_colour(img, np.zeros((100, 1000), np.float32), None) is None
    faces = rows(face_row(0, 0, 100, 100))
    assert look.bright_colour(img, mask, faces)[2] < 0.1, "only the faces' skin counts"


def test_reduce_shine_patches_are_off_by_default_and_need_the_colour():
    img, mask, faces = glossy_patch_photo()
    colour = look.bright_colour(img, mask, faces)
    assert look.matte(img, mask, faces, colour, 0.0) is img
    assert look.matte(img, mask, faces, None, 1.0) is img
    assert look.matte(img, np.zeros_like(mask), faces, colour, 1.0) is img
    assert look.finish(img, mask, mask, [], 0.5, faces=faces, bright=colour) is img, "every step off: the photo itself"


def test_reduce_shine_takes_a_large_patch_down_towards_the_skins_colour():
    img, mask, faces = glossy_patch_photo()
    colour = look.bright_colour(img, mask, faces)
    out = look.matte(img, mask, faces, colour, 1.0)
    half = look.matte(img, mask, faces, colour, 0.5)
    patch = lambda a: a[285:315, 235:265].astype(np.float32).reshape(-1, 3).mean(axis=0)
    ref = np.array(colour) * 255
    before, after, mid = patch(img) - ref, patch(out) - ref, patch(half) - ref
    assert (before > 30).all(), before
    cut = look.MATTE["cut"]
    assert np.allclose(after / before, 1 - cut, atol=0.06), ("the share above the colour that is left", after / before)
    assert np.allclose(mid / before, 1 - cut / 2, atol=0.06), "half the amount, half as much"
    ratio = lambda c: c[0] / c[2]
    assert abs(ratio(patch(out)) - ratio(ref)) < abs(ratio(patch(img)) - ratio(ref)), "its colour comes back, not grey"
    green = lambda a: a[:, :, 1].astype(np.float32)
    fine = lambda a: (green(a) - cv2.GaussianBlur(green(a), (0, 0), 2))[280:320, 230:270].std()
    assert 0.95 < fine(out) / fine(img) < 1.05, "the grain stays"


def test_reduce_shine_leaves_even_skin_a_blush_and_everything_outside_the_mask():
    img, mask, faces = glossy_patch_photo()
    out = look.matte(img, mask, faces, look.bright_colour(img, mask, faces), 1.0)
    change = np.abs(out.astype(int) - img)
    assert change[420:540, 100:700].max() <= 1, "even skin, no brighter than the colour"
    assert change[270:330, 530:590].mean() < 0.6, ("a blush is redder, not brighter", change[270:330, 530:590].mean())
    assert np.array_equal(out[:40], img[:40]) and np.array_equal(out[:, :60], img[:, :60]), "outside the mask"
    part = mask.copy()
    part[:, :250] = 0                                   # a Restore stroke over half the patch
    out = look.matte(img, part, faces, look.bright_colour(img, mask, faces), 1.0)
    assert np.array_equal(out[:, :250], img[:, :250]) and not np.array_equal(out[:, 250:320], img[:, 250:320])


def test_reduce_shine_judges_each_face_against_its_own_skin():
    """A fair face beside a larger dark one: with one colour for both, the fair face
    would sit above it everywhere and be darkened as if it were shine."""
    rng = np.random.default_rng(11)
    grain = rng.normal(0, 2, (500, 1000))
    img = np.zeros((500, 1000, 3), np.float32)
    img[:, :700] = (70, 90, 130)                       # a dark face, most of the photo
    img[:, 700:] = (150, 175, 225)                     # a fair one on the right
    yy, xx = np.mgrid[0:500, 0:1000].astype(np.float32)
    for cx in (350, 850):                              # the same shiny patch on each
        img += np.exp(-((xx - cx) ** 2 + (yy - 190) ** 2) / (2 * 40.0 ** 2))[:, :, None] * np.array([28.0, 26.0, 22.0])
    img = np.clip(img + grain[:, :, None], 0, 255).astype(np.uint8)
    mask = np.ones((500, 1000), np.float32)
    faces = rows(face_row(50, 0, 600, 500), face_row(720, 120, 260, 260))
    colours = look.bright_colours(img, mask, faces)
    assert len(colours) == 2 and colours[0][2] < 0.6 < colours[1][2], "one colour per face"
    assert look.bright_colours(img, mask, None)[0] is not None and len(look.bright_colours(img, mask, None)) == 1
    out = look.matte(img, mask, faces, colours, 1.0)
    change = out.astype(np.float32) - img
    even = lambda c: (float(np.abs(c[330:370, 100:600]).mean()), float(np.abs(c[330:370, 760:940]).mean()))
    assert max(even(change)) < 0.3, ("each face's even skin is left alone", even(change))
    dark, fair = float(change[180:200, 340:360].mean()), float(change[180:200, 840:860].mean())
    assert dark < -5 and fair < -5 and abs(dark - fair) < 3, ("the same patch comes down the same on both", dark, fair)
    pooled = look.matte(img, mask, faces, look.bright_colour(img, mask, faces), 1.0).astype(np.float32) - img
    assert float(pooled[330:370, 760:940].mean()) < -10, "one colour for both would darken the fair face"
    assert look.matte(img, mask, faces, [None, None], 1.0) is img


def stray_photo(busy: bool = False):
    """A dark head on a blue backdrop with a soft gradient, three thin stray hairs
    sticking out of it and a thick lock beside it: (photo, person region, faces)."""
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:700, 0:700].astype(np.float32)
    img = np.dstack([200 + 0.03 * yy, 110 + 0.02 * xx, 30 + 0 * xx]) + rng.normal(0, 1.2, (700, 700, 1))
    if busy:
        img += rng.normal(0, 22, (700, 700, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    head = ((xx - 350) / 150) ** 2 + ((yy - 380) / 200) ** 2 <= 1
    img[head] = (40, 45, 60)
    cv2.line(img, (350, 185), (340, 148), (35, 40, 50), 2, cv2.LINE_AA)       # strays: upwards,
    cv2.line(img, (480, 300), (522, 278), (35, 40, 50), 1, cv2.LINE_AA)       # to the right,
    cv2.line(img, (215, 330), (172, 318), (35, 40, 50), 2, cv2.LINE_AA)       # to the left
    cv2.circle(img, (212, 250), 14, (35, 40, 50), -1)                         # too thick to be a stray hair
    person = (0, 0, 700, 700, cv2.resize(head.astype(np.float32), (350, 350), interpolation=cv2.INTER_AREA))
    return img, person, rows(face_row(250, 280, 200, 250))


def test_tidy_stray_hairs_is_off_by_default_and_needs_the_outline_and_a_face():
    img, person, faces = stray_photo()
    assert look.tidy(img, person, faces, 0.0) is img
    assert look.tidy(img, None, faces, 1.0) is img, "no segmentation, nothing to go by"
    assert look.tidy(img, person, None, 1.0) is img
    mask = np.ones((700, 700), np.float32)
    assert look.finish(img, mask, mask, [], 0.5, faces=faces, person=person) is img, "every step off: the photo itself"
    assert look.person_of(None) is None


def test_tidy_stray_hairs_replaces_thin_strands_with_the_backdrop():
    img, person, faces = stray_photo()
    out = look.tidy(img, person, faces, 1.0)
    clean = stray_photo()[0].copy()
    yy, xx = np.mgrid[0:700, 0:700].astype(np.float32)
    backdrop = np.dstack([200 + 0.03 * yy, 110 + 0.02 * xx, 30 + 0 * xx])
    off = lambda a: np.abs(a.astype(np.float32) - backdrop).max(axis=2)
    for name, (ys, xs) in {"up": (slice(150, 172), slice(335, 352)), "right": (slice(276, 296), slice(494, 520)),
                           "left": (slice(314, 328), slice(176, 200))}.items():
        assert off(img)[ys, xs].max() > 60, name
        assert off(out)[ys, xs].max() < 12, (name, "the strand is gone", off(out)[ys, xs].max())
    assert off(out)[244:256, 206:218].min() > 60, "a thick lock is not a stray hair"
    head = ((xx - 350) / 150) ** 2 + ((yy - 380) / 200) ** 2 <= 1
    assert np.array_equal(out[head], img[head]), "the person is left alone"
    changed = np.abs(out.astype(int) - img).max(axis=2) > 2
    assert changed.mean() < 0.01, "only the strands change"
    assert not changed[620:].any() and not changed[:, :100].any(), "nothing far from the head or below it"
    half = look.tidy(img, person, faces, 0.5)
    assert 25 < off(half)[150:172, 335:352].max() < off(img)[150:172, 335:352].max() - 25, "half the amount, half gone"


def test_tidy_stray_hairs_leaves_a_busy_background_and_restored_areas():
    img, person, faces = stray_photo(busy=True)
    assert look.tidy(img, person, faces, 1.0) is img, "no plain backdrop to tidy against"
    img, person, faces = stray_photo()
    strokes = [{"id": 1, "mode": "protect", "size": 0.2, "soft": 0.1, "points": [0.49, 0.22, 0.5, 0.23]}]
    region, fills, _ = mask_edit.edits(strokes, 700, 700)
    out = look.tidy(img, person, faces, 1.0, fills)
    assert np.array_equal(out[152:170, 338:350], img[152:170, 338:350]), "Restore keeps the strand under it"
    assert not np.array_equal(out[276:296, 494:520], img[276:296, 494:520]), "the others still go"


def test_tidy_stray_hairs_leaves_thin_things_that_are_not_the_hairs_colour():
    """A white seam and a red wire beside the head are thin too, but they aren't the
    backdrop with dark hair mixed in."""
    img, person, faces = stray_photo()
    cv2.line(img, (520, 330), (545, 420), (245, 245, 245), 2, cv2.LINE_AA)       # a white line on the backdrop
    cv2.line(img, (170, 380), (200, 470), (40, 40, 230), 2, cv2.LINE_AA)         # a red one
    out = look.tidy(img, person, faces, 1.0)
    assert np.array_equal(out[335:415, 518:548], img[335:415, 518:548]), "the white line stays"
    assert np.array_equal(out[385:465, 168:202], img[385:465, 168:202]), "and the red one"
    assert not np.array_equal(out[150:172, 335:352], img[150:172, 335:352]), "the stray hairs still go"
    bald = (person[0], person[1], person[2], person[3], np.zeros_like(person[4]))
    assert look.tidy(img, bald, faces, 1.0) is img, "no hair to take the colour from, nothing done"


def test_tidy_stray_hairs_does_the_same_on_a_large_photo():
    img, person, faces = stray_photo()
    big = cv2.resize(img, (2800, 2800), interpolation=cv2.INTER_CUBIC)
    big_faces = faces.copy()
    big_faces[:, :14] *= 4
    out = look.tidy(big, (0, 0, 2800, 2800, person[4]), big_faces, 1.0)
    small = cv2.resize(out, (700, 700), interpolation=cv2.INTER_AREA).astype(np.float32)
    yy, xx = np.mgrid[0:700, 0:700].astype(np.float32)
    backdrop = np.dstack([200 + 0.03 * yy, 110 + 0.02 * xx, 30 + 0 * xx])
    assert np.abs(small - backdrop).max(axis=2)[150:172, 335:352].max() < 14, "found on a reduced copy, removed at full size"


def tilted_head(degrees: float = 5.0):
    """eye_points() turned by some degrees about the chin, and a photo with a white dot on
    each eye, one on the forehead and one on the collar: (photo, points, dot places)."""
    pts = eye_points()
    chin = pts[face_landmarks.CHIN].copy()
    t = np.radians(degrees)
    turn = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pts = (pts - chin) @ turn.T + chin
    rng = np.random.default_rng(2)
    img = np.clip(np.full((1000, 800, 3), (150, 120, 90), np.float32) + rng.normal(0, 2, (1000, 800, 1)), 0, 255).astype(np.uint8)
    eye_r, eye_l = pts[[33, 133]].mean(axis=0), pts[[263, 362]].mean(axis=0)
    forehead = (eye_r + eye_l) / 2 + (chin - (eye_r + eye_l) / 2) * -0.5
    collar = chin + np.array([10.0, 260.0])
    dots = {"eye_r": eye_r, "eye_l": eye_l, "forehead": forehead, "collar": collar}
    for x, y in dots.values():
        cv2.circle(img, (int(round(x)), int(round(y))), 6, (255, 255, 255), -1, cv2.LINE_AA)
    return img, pts, dots


def dot_near(img: np.ndarray, place, reach: int = 60) -> np.ndarray:
    """Where the white dot nearest a place is: the centre of the bright pixels around it."""
    x, y = int(round(place[0])), int(round(place[1]))
    box = img[y - reach:y + reach, x - reach:x + reach, 0].astype(np.float32)
    ys, xs = np.nonzero(box > 220)
    return np.array([xs.mean() + x - reach, ys.mean() + y - reach])


def test_straighten_head_is_off_by_default_and_needs_landmarks():
    img, pts, _ = tilted_head()
    assert look.straighten(img, [pts], 0.0) is img
    assert look.straighten(img, [None], 1.0) is img and look.straighten(img, [], 1.0) is img
    mask = np.ones(img.shape[:2], np.float32)
    assert look.finish(img, mask, mask, [pts], 0.5) is img, "every step off: the photo itself"
    assert abs(look.head_tilt(pts) - 5.0) < 1e-6 and abs(look.head_tilt(eye_points())) < 1e-9
    level, _, _ = tilted_head(0.0)
    assert look.straighten(level, [eye_points()], 1.0) is level, "a level head isn't touched"


def test_straighten_head_levels_the_eyes_and_leaves_the_shoulders():
    img, pts, dots = tilted_head(5.0)
    out = look.straighten(img, [pts], 1.0)
    r, l = dot_near(out, dots["eye_r"]), dot_near(out, dots["eye_l"])
    assert abs(l[1] - r[1]) < 1.0, ("the eye line is level", r, l)
    assert abs(np.hypot(*(l - r)) - 200) < 1.5, "and the eyes are as far apart as they were: turned, not stretched"
    half = look.straighten(img, [pts], 0.5)
    r2, l2 = dot_near(half, dots["eye_r"]), dot_near(half, dots["eye_l"])
    tilt = np.degrees(np.arctan2(l2[1] - r2[1], l2[0] - r2[0]))
    assert abs(tilt - 2.5) < 0.3, ("half the amount, half the tilt left", tilt)
    f0, f1 = dot_near(img, dots["forehead"]), dot_near(out, dots["forehead"])
    assert f1[0] < f0[0] - 10, "the top of the head moves most"
    assert np.array_equal(out[960:], img[960:]), "nothing below the neck (its line tilts with the face)"
    assert np.allclose(dot_near(out, dots["collar"]), dot_near(img, dots["collar"]), atol=0.01), "the collar stays"
    back = look.straighten(*tilted_head(-5.0)[:1], [tilted_head(-5.0)[1]], 1.0)
    rb, lb = dot_near(back, tilted_head(-5.0)[2]["eye_r"]), dot_near(back, tilted_head(-5.0)[2]["eye_l"])
    assert abs(lb[1] - rb[1]) < 1.0, "a tilt the other way is undone the other way"


def test_straighten_head_only_undoes_a_slip_and_is_safe_at_the_photo_edge():
    img, pts, dots = tilted_head(20.0)
    out = look.straighten(img, [pts], 1.0)
    r, l = dot_near(out, dots["eye_r"], 80), dot_near(out, dots["eye_l"], 80)
    left = np.degrees(np.arctan2(l[1] - r[1], l[0] - r[0]))
    assert abs(left - (20.0 - look.LEVEL["most"])) < 0.5, ("a pose is only turned by the most allowed", left)
    off = look.straighten(img, [pts + (0, -2000)], 1.0)
    assert off.shape == img.shape
    edge = look.straighten(img, [pts + (350, 0)], 1.0)
    assert edge.shape == img.shape and np.array_equal(edge[950:], img[950:])


def test_the_pages_and_the_window_have_the_same_looks():
    import re
    source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "js", "looks.js"),
                  encoding="utf-8").read()
    on_pages = {name: dict(re.findall(r"(\w+): '([\w.]+)'", body))
                for name, body in re.findall(r"^  (\w+): \{([^}]*)\},$", source, re.M)}
    assert set(on_pages) == set(look.LOOKS) == {"natural", "studio"}
    for name, values in look.LOOKS.items():
        as_shown = {k: v if isinstance(v, str) else f"{round(v * 100)}" for k, v in values.items()}
        assert on_pages[name] == as_shown, (name, on_pages[name], as_shown)
    assert all(v["lighting"] in smoother.LIGHTING and v["print"] in smoother.PRINT_GRAIN for v in look.LOOKS.values())
    natural = look.LOOKS["natural"]
    assert natural["eyes"] == 0 and natural["tone"] == 0, "how the app opens: both steps off"


def test_the_skin_area_keeps_what_the_mask_cuts_and_the_landmarks_come_with_it():
    img, faces, points = skin_photo(800, 800), rows(face_row(150, 50, 500, 660)), [outline_points()]
    saved = skin_segment.segment, face_landmarks.landmarks
    try:
        skin_segment.segment, face_landmarks.landmarks = all_skin, lambda _b, _f: points
        mask, area, found = smoother.skin_layers(img, faces=faces, face_only=True)
        assert np.array_equal(mask, smoother.build_skin_mask(img, faces=faces, face_only=True))
    finally:
        skin_segment.segment, face_landmarks.landmarks = saved
    assert found is points
    assert mask[300, 300] < 0.05 and mask[322, 80] < 0.05, "an eye and an ear are out of the mask"
    assert area[300, 300] > 0.95 and area[322, 80] > 0.95, "and in the area"
    assert (area >= mask - 1e-6).all()
    plain = smoother.skin_layers(img, faces=np.zeros((0, 15), np.float32))
    assert plain[2] == [] and np.array_equal(plain[0], plain[1]), "without a face nothing is cut"


def test_the_skin_area_has_the_body_in_even_with_face_only():
    img, faces = skin_photo(800, 800), rows(face_row(150, 50, 500, 500))
    asked = []

    def model(seg, _faces, face_only=False):
        """A face's skin over the top of the photo, and the body's below it unless Face only."""
        asked.append(face_only)
        area = np.zeros((800, 800), np.float32)
        area[:(600 if face_only else 800)] = 1.0
        return 0, 0, area

    saved = skin_segment.segment, skin_segment.skin_area, face_landmarks.landmarks
    try:
        skin_segment.segment, skin_segment.skin_area = all_skin, model
        face_landmarks.landmarks = lambda _b, _f: [outline_points()]
        mask, area, _ = smoother.skin_layers(img, faces=faces, face_only=True)
        assert asked == [True, False]
        assert mask[700, 400] == 0 and area[700, 400] == 1, "the neck: not smoothed, but in the area"
        assert mask[300, 300] < 0.05 and area[300, 300] == 1, "an eye, as before"
        asked.clear()
        both, area, _ = smoother.skin_layers(img, faces=faces)
        assert asked == [False] and both[700, 400] == 1 and area[700, 400] == 1, "without Face only, one and the same"
    finally:
        skin_segment.segment, skin_segment.skin_area, face_landmarks.landmarks = saved


def test_a_restore_stroke_brings_the_original_back_under_both_steps():
    img = eye_photo()
    dark = (img.astype(np.float32) * 0.6).astype(np.uint8)
    mask = np.ones((800, 800), np.float32)
    strokes = [{"id": 1, "mode": "protect", "size": 0.2, "soft": 0.2, "points": [0.375, 0.42, 0.376, 0.43]},
               {"id": 2, "mode": "unsmooth", "size": 0.1, "soft": 0.2, "points": [0.625, 0.42, 0.626, 0.43]}]
    region, fills, _ = mask_edit.edits(strokes, 800, 800)
    edited, area = mask_edit.apply_to_mask(mask, region), mask_edit.apply_to_mask(mask, fills)
    luma = look.skin_luma(dark, mask, None)
    out = look.finish(dark, edited, area, [eye_points()], luma, 1.0, 1.0)
    assert not np.array_equal(out[700:], dark[700:]), "the skin tone moved elsewhere"
    assert edited[330:350, 290:310].max() < 1e-3, "well inside the Restore stroke"
    assert np.array_equal(out[330:350, 290:310], dark[330:350, 290:310]), "Restore: the photo as it was, under the eye too"
    undone = out[320:350, 485:515].astype(int) - dark[320:350, 485:515]
    assert undone.mean() > 5, "Undo smoothing keeps the skin tone, like the other fixes"
    alone = look.balance(dark, area, luma, 1.0)
    assert np.array_equal(out[320:350, 485:515], alone[320:350, 485:515]), "but not the under-eye step"


def i18n_strings():
    """The keys and words of web/js/i18n.js: {"en": {key: text}, "th": {...}}."""
    import re
    source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "js", "i18n.js"),
                  encoding="utf-8").read()
    en, th = source.split("\n  th: {", 1)
    pairs = lambda part: dict(re.findall(r"^\s*'([\w.]+)':\s*'((?:[^'\\]|\\.)*)',", part, re.M))
    return {"en": pairs(en.split("\n  en: {", 1)[1]), "th": pairs(th.split("\n};", 1)[0])}


def test_thai_and_english_have_the_same_words_and_every_word_used_exists():
    import re
    import serve
    words = i18n_strings()
    en, th = words["en"], words["th"]
    assert len(en) > 120, "the English table was read"
    assert set(th) <= set(en), sorted(set(th) - set(en))
    assert set(en) - set(th) == {"lang.th", "lang.en"}, sorted(set(en) - set(th) - {"lang.th", "lang.en"})
    fills = lambda text: sorted(re.findall(r"\{(\w+)\}", text))
    assert all(fills(en[k]) == fills(th[k]) for k in th), [k for k in th if fills(en[k]) != fills(th[k])]
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
    read = lambda *path: open(os.path.join(root, *path), encoding="utf-8").read()
    pages = read("index.html") + serve.PAGE
    scripts = read("js", "app.js") + read("js", "mask_brush.js") + read("js", "history.js") + serve.PAGE
    fields = re.findall(r"\{ id: '(\w+)', key: '([\w.]+)'", read("js", "history.js"))
    assert len(fields) > 8 and {key for _, key in fields} <= set(en), "History's names for the settings"
    assert all(f'id="{i}"' in serve.PAGE for i, _ in fields), "on the desktop app's page"
    desktop_only = set()                # settings the browser has no step for: none
    assert all(f'id="{i}"' in read("index.html") for i, _ in fields if i not in desktop_only), "and on the web page"
    assert not any(f'id="{i}"' in read("index.html") for i in desktop_only), "which has none of the desktop's own"
    assert {"lipfull", "chin"} <= {i for i, _ in fields}, "a full moustache's box and a chin beard's are settings on both"
    assert "chin: $('chin').checked" in serve.PAGE and 'payload.get("chin", False)' in open(serve.__file__, encoding="utf-8").read(), \
        "ticked, it reaches the step on the desktop"
    assert "chin: $('chin').checked" in read("js", "app.js") and "params.chin ? chin.clear(" in read("js", "pipeline.js"), \
        "and in the browser"
    used = set(re.findall(r'data-i18n(?:-html|-title|-alt|-aria)?="([\w.]+)"', pages))
    used |= set(re.findall(r"\bt\('([\w.]+)'", scripts))
    for m in ("protect", "unsmooth", "skin", "heal", "spot", "light", "line", "resmooth", "removeSkin"):
        used |= {f"mode.{m}.{field}" for field in ("label", "title", "strength", "hint")}
    for m in ("heal", "light", "line"):
        used |= {f"mode.{m}.{field}" for field in ("point", "pointTitle", "noPoint")}
    used |= {"mode.line.next"} | {f"step{n}.{f}" for n in (1, 2, 3) for f in ("title", "button", "about")}
    used |= {"engine.gpu", "engine.cpu", "summary.spots", "summary.spots1", "summary.healed", "summary.healed1"}
    used |= {f"progress.{p}" for p in ("tools", "models", "reading", "finding", "mask", "full", "saving")}
    used |= {"update.downloading", "update.ready", "update.failed"}
    assert len(used) > 120 and used <= set(en), sorted(used - set(en))
    assert b"attachLangSwitch" in serve._web_script("i18n.js"), "the desktop app serves the words too"
    for name, export in (("compare_zoom.js", b"attachCompareZoom"), ("history.js", b"attachHistory"),
                         ("looks.js", b"attachLook")):
        assert export in serve._web_script(name), f"and {name}"


def test_the_desktop_preview_scales_the_analysis_without_finding_anything_again():
    import serve
    img = np.full((2400, 3200, 3), 128, np.uint8)
    faces = FACE.copy()
    mask = np.ones((2400, 3200), np.float32)
    corr = (3100, 2300, np.full((100, 100, 3), 8.0, np.float32))    # reaching the photo's corner
    points = [np.array([[3200.0, 2400.0], [800.0, 400.0]]), None]
    full = (img, None, faces, mask, corr, 3, {"area": mask.copy(), "points": points, "luma": 0.5})
    small, info, f, m, c, count, extra = serve.scaled_analysis(full, 1600)
    assert small.shape == (1200, 1600, 3) and m.shape == (1200, 1600) and count == 3
    assert extra["area"].shape == (1200, 1600) and extra["luma"] == 0.5, "the skin area shrinks, the brightness is kept"
    assert np.allclose(extra["points"][0], [[1600, 1200], [400, 200]]) and extra["points"][1] is None
    assert points[0][0, 0] == 3200.0, "the full-size landmarks are left as they were"
    assert np.allclose(f[0, :4], [150, 100, 200, 250]) and f[0, 14] == np.float32(0.9), "positions scaled, score kept"
    assert np.array_equal(faces, FACE), "the full-size faces are left as they were"
    x0, y0, fills = c
    assert x0 + fills.shape[1] <= 1600 and y0 + fills.shape[0] <= 1200, "the fills fit the preview"
    assert abs(float(fills.mean()) - 8.0) < 1e-3
    assert blemish.apply(small, c, 1.0)[1190, 1590, 0] == 136, "and apply to it"
    assert serve.scaled_analysis(full, 4000) is full, "a photo already that small is used as it is"


def test_the_desktop_page_previews_while_editing_and_downloads_full_size():
    import base64
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    import serve
    ok, png = cv2.imencode(".png", skin_photo(1200, 2000))
    image = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def process(**extra):
        body = json.dumps({"image": image, "strength": 0.6, "texture": 0.6, "blemish": 1.0, **extra}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/process", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            result = json.loads(r.read())
        assert "error" not in result, result.get("error")
        return result

    picture = lambda url: cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8),
                                       cv2.IMREAD_COLOR)
    try:
        edit = process(preview=True, want_mask=True, want_before=True)
        assert edit["image"].startswith("data:image/jpeg") and picture(edit["image"]).shape == (960, 1600, 3)
        assert picture(edit["before_image"]).shape == (960, 1600, 3), "the Original at the same size"
        assert (edit["w"], edit["h"]) == (2000, 1200), "the status line gives the photo's own size"
        assert "mask_image" in edit
        assert "before_image" not in process(preview=True), "only when asked for"
        download = process()
        assert download["image"].startswith("data:image/png") and picture(download["image"]).shape == (1200, 2000, 3)
        assert "before_image" not in download and download["ext"] == ".png", "a PNG stays a PNG"
    finally:
        server.shutdown()
        server.server_close()
        serve._analysis_cache.clear()


def test_the_nose_is_found_from_the_detectors_points():
    face = np.array([face_row(300, 200, 400, 500)], np.float32)     # eyes at y 400, nose tip at (500, 500)
    nose = smoother._nose_map(face, 1.0, 0, 0, (900, 1000))
    assert nose[420:510, 485:516].min() > 0.99, "the ridge and the tip"
    assert nose[500, 465] > 0.99 and nose[500, 535] > 0.99 and nose[420, 465] < 0.5, "the wings, wider than the ridge"
    assert nose[400, 450] < 0.1 and nose[400, 550] < 0.1, "not the inner corners of the eyes"
    assert max(nose[520, 380], nose[520, 620], nose[300, 500], nose[590, 500]) < 0.01, "not the cheeks, forehead or lips"
    half = smoother._nose_map(face, 0.5, 100, 60, (420, 450))
    assert abs(float(half[(500 - 60) // 2, (500 - 100) // 2]) - 1.0) < 1e-3 and half[(520 - 60) // 2, (380 - 100) // 2] < 0.01, \
        "the same place on a reduced copy of a region"
    assert smoother._nose_map(FACE, 1.0, 0, 0, (900, 1000)) is None, "a row without points has no nose"


def test_hard_lighting_leaves_the_nose_at_soft_lighting():
    img = mottled_light_photo()
    mask = oval_mask()
    face = np.array([face_row(300, 200, 400, 500)], np.float32)
    hard = smoother.LIGHTING["hard"]
    soft = smoother.smooth_face(img, 1.0, 0.6, mask, face).astype(np.int16)
    out = smoother.smooth_face(img, 1.0, 0.6, mask, face, light=hard).astype(np.int16)
    nose_map = smoother._nose_map
    smoother._nose_map = lambda *a: None
    try:
        everywhere = smoother.smooth_face(img, 1.0, 0.6, mask, face, light=hard).astype(np.int16)
        assert np.array_equal(smoother.smooth_face(img, 1.0, 0.6, mask, face), soft), "soft light never asks where the nose is"
    finally:
        smoother._nose_map = nose_map
    nose = (slice(430, 505), slice(488, 513))
    cheek = (slice(480, 600), slice(340, 420))
    assert np.abs(everywhere[nose] - soft[nose]).mean() > 1.0, "the photo should be one hard light evens further"
    assert np.abs(out[nose] - soft[nose]).max() <= 1, "on the nose, as soft light"
    assert np.array_equal(out[cheek], everywhere[cheek]), "beside it, as hard light"
    assert np.array_equal(smoother.smooth_face(img, 1.0, 0.6, mask, FACE, light=hard),
                          smoother.smooth_face(img, 1.0, 0.6, mask, FACE, light=hard)), "the same every time"


def lip_face(moustache: float = 0.25, size: float = 1.0, seed: int = 3, reach: float = 1.2):
    """A drawn face around the mouth, 1000 x 900 at size 1: (photo, landmarks, smoothing mask, the
    moustache's own map 0..1). The skin between lip and nose is shaded as a bare lip is
    (lip.TABLE); moustache darkens a band above the lip by that share, with strands in it.
    reach: how far across the band goes (1 = the mouth's corners)."""
    h, w = round(900 * size), round(1000 * size)
    pts = np.zeros((face_landmarks.POINTS, 2), np.float64)
    place = {33: (380, 350), 133: (440, 350), 362: (560, 350), 263: (620, 350), 2: (500, 540), 205: (390, 500), 425: (610, 500),
             61: (430, 620), 185: (445, 612), 40: (462, 606), 39: (478, 602), 37: (490, 599), 0: (500, 600), 267: (510, 599),
             269: (522, 602), 270: (538, 606), 409: (555, 612), 291: (570, 620), 375: (555, 632), 321: (538, 640),
             405: (522, 646), 314: (510, 649), 17: (500, 650), 84: (490, 649), 181: (478, 646), 91: (462, 640), 146: (445, 632),
             13: (500, 623), 14: (500, 625),
             78: (436, 620), 191: (447, 621), 80: (462, 622), 81: (478, 623), 82: (490, 623), 312: (510, 623), 311: (522, 623),
             310: (538, 622), 415: (553, 621), 308: (564, 620)}
    for i, xy in place.items():
        pts[i] = np.array(xy, np.float64) * size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    frame = lip._Frame(pts)
    u, v = frame.uv(xx, yy)
    near = smoother._smoothstep(2.6 - np.abs(u), 0.0, 0.4) * smoother._smoothstep(v + 0.9, 0.0, 0.3) * smoother._smoothstep(1.3 - v, 0.0, 0.3)
    bare = 1.0 + near * (lip._table_at(u, v) / 0.97 - 1.0)
    band = smoother._smoothstep(reach - np.abs(u), 0.0, 0.15) * smoother._smoothstep(v, 0.0, 0.05) * smoother._smoothstep(0.42 - v, 0.0, 0.08)
    strands = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), max(0.6, 0.8 * size))
    hair = band * (1.0 + 0.5 * strands / float(strands.std()))
    tone = (0.72 + 0.06 * np.sin(xx / (170 * size))) * bare * (1.0 - moustache * np.clip(hair, 0.0, 2.0)) \
        + cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), max(0.5, 0.7 * size)) * 0.012
    img = np.clip(np.dstack([tone * 0.62, tone * 0.76, tone]) * 255.0, 0, 255).astype(np.uint8)
    lips = np.zeros((h, w), np.uint8)
    cv2.fillPoly(lips, [np.round(pts[list(lip.UPPER + lip.LOWER)]).astype(np.int32)], 255)
    img[lips > 0] = (90, 80, 190)
    mask = 1.0 - cv2.GaussianBlur(lips.astype(np.float32) / 255.0, (0, 0), 1.0)
    return img, pts, mask.astype(np.float32), band.astype(np.float32)


def darkness(img, bare, where) -> float:
    """How much darker img is than the same face with a bare lip, over the moustache's band, as a share."""
    y = lambda a: cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 4.0)
    return float(((y(bare) - y(img)) / y(bare))[where].mean())


def test_no_moustache_step_without_an_amount_or_a_face():
    img, pts, mask, band = lip_face()
    assert lip.clear(img, [pts], mask, 0.0) is img
    assert lip.clear(img, [], mask, 1.0) is img and lip.clear(img, [None], mask, 1.0) is img
    tiny = pts * (lip.MIN_EYES * 0.9 / 180.0)
    assert lip.clear(img, [tiny], mask, 1.0) is img, "a face too small to judge is left alone"


def test_a_thin_moustache_is_replaced_not_faded():
    img, pts, mask, band = lip_face()
    bare = lip_face(0.0)[0]
    where = band > 0.9
    before = darkness(img, bare, where)
    out = lip.clear(img, [pts], mask, 1.0)
    after = darkness(out, bare, where)
    assert before > 0.15, f"the drawn moustache should be dark: {before:.3f}"
    assert abs(after) < 0.2 * before, f"left of the moustache: {after:.3f} of {before:.3f}"
    grain = lambda a: (cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
                       - cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 3.0))
    cheek = np.zeros(band.shape, bool)
    cheek[440:560, 340:440] = True
    assert 0.5 * grain(img)[cheek].std() < grain(out)[where].std() < 2.0 * grain(img)[cheek].std() < grain(img)[where].std(), \
        "the new skin has a cheek's grain, not the strands' and not none"
    lips = np.zeros(band.shape, np.uint8)
    cv2.fillPoly(lips, [np.round(pts[list(lip.UPPER + lip.LOWER)]).astype(np.int32)], 255)
    inner = cv2.erode(lips, np.ones((7, 7), np.uint8)) > 0
    assert np.array_equal(out[inner], img[inner]), "the lips themselves are not skin"
    far = np.ones(band.shape, bool)
    far[430:760, 250:750] = False
    assert np.array_equal(out[far], img[far]), "nothing changes away from the mouth"
    blurred = lambda a: cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 4.0)
    assert float((blurred(out) - blurred(img)).min()) > -2.0, "nothing is made darker"


def test_a_bare_lip_and_a_full_moustache_are_left_alone():
    bare, pts, mask, band = lip_face(0.0)
    assert lip.clear(bare, [pts], mask, 1.0) is bare, "a bare lip has nothing to remove"
    full = lip_face(0.6)[0]
    assert lip.clear(full, [pts], mask, 1.0) is full, "a full moustache is kept whole"
    forced = lip.clear(full, [pts], mask, 1.0, force=True)
    assert forced is not full, "unless the question is skipped (the parity check)"


def test_a_full_moustache_goes_when_it_is_asked_for():
    img, pts, mask, band = lip_face(0.6)
    bare = lip_face(0.0)[0]
    where = band > 0.9
    before = darkness(img, bare, where)
    out = lip.clear(img, [pts], mask, 1.0, full=True)
    assert out is not img
    after = darkness(out, bare, where)
    assert before > 0.4, f"the drawn moustache should be dense: {before:.3f}"
    assert abs(after) < 0.1 * before, f"left of the moustache: {after:.3f} of {before:.3f}"
    grey = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grain = lambda a: grey(a) - cv2.GaussianBlur(grey(a), (0, 0), 3.0)
    cheek = np.zeros(band.shape, bool)
    cheek[440:560, 340:440] = True
    assert 0.5 * grain(img)[cheek].std() < grain(out)[where].std() < 2.0 * grain(img)[cheek].std() < grain(img)[where].std(), \
        "the new skin has a cheek's grain, not the strands' and not none"
    lips = np.zeros(band.shape, np.uint8)
    cv2.fillPoly(lips, [np.round(pts[list(lip.UPPER + lip.LOWER)]).astype(np.int32)], 255)
    inner = cv2.erode(lips, np.ones((15, 15), np.uint8)) > 0
    assert np.array_equal(out[inner], img[inner]), "the lips themselves stay"
    far = np.ones(band.shape, bool)
    far[430:800, 200:800] = False
    assert np.array_equal(out[far], img[far]), "nothing changes away from the mouth"
    assert np.array_equal(out, lip.clear(img, [pts], mask, 1.0, full=True)), "the same every time"


def test_asking_for_a_full_moustache_changes_nothing_for_a_thin_one_or_a_bare_lip():
    thin, pts, mask, band = lip_face()
    assert np.array_equal(lip.clear(thin, [pts], mask, 1.0, full=True), lip.clear(thin, [pts], mask, 1.0)), "a thin one goes the thin way"
    bare = lip_face(0.0)[0]
    assert lip.clear(bare, [pts], mask, 1.0, full=True) is bare, "a bare lip has nothing to remove"
    full = lip_face(0.6)[0]
    assert lip.clear(full, [pts], mask, 0.0, full=True) is full, "no amount, no change"


def test_a_full_moustache_is_followed_to_its_ends():
    img, pts, mask, band = lip_face(0.6, reach=1.9)             # past where the thin moustache's zone fades out
    bare = lip_face(0.0)[0]
    u, v = lip._Frame(pts).uv(*np.meshgrid(np.arange(1000.0), np.arange(900.0)))
    ends = (band > 0.9) & (np.abs(u) > 1.6) & (np.abs(u) < 1.8)
    before = darkness(img, bare, ends)
    thin_way = darkness(lip.clear(img, [pts], mask, 1.0, force=True), bare, ends)
    after = darkness(lip.clear(img, [pts], mask, 1.0, full=True), bare, ends)
    assert thin_way > 0.4 * before, f"the fixed zone leaves the ends: {thin_way:.3f} of {before:.3f}"
    assert abs(after) < 0.15 * before, f"left of the ends: {after:.3f} of {before:.3f}"


def test_a_full_moustache_keeps_the_guards_and_the_brush():
    img, pts, mask, band = lip_face(0.6)
    turned = pts.copy()
    turned[2, 0] += 0.1 * 180
    assert lip.clear(img, [turned], mask, 1.0, full=True) is img, "a turned head is left alone"
    edit = -np.ones((300, 250), np.float32)
    out = lip.clear(img, [pts], mask, 1.0, (250, 480, edit), full=True)       # Restore over the photo's left half of the mouth
    assert np.array_equal(out[480:780, 250:500], img[480:780, 250:500]), "restored: as it was"
    bare = lip_face(0.0)[0]
    right = (band > 0.9) & (np.arange(band.shape[1])[None, :] > 520)
    assert abs(darkness(out, bare, right)) < 0.15 * darkness(img, bare, right), "the other half still goes"
    where = band > 0.9
    half = darkness(lip.clear(img, [pts], mask, 0.5, full=True), bare, where)
    assert 0.35 * darkness(img, bare, where) < half < 0.65 * darkness(img, bare, where), "the amount is a share"


def test_a_full_moustache_goes_at_other_sizes():
    for size in (0.5, 2.0):
        img, pts, mask, band = lip_face(0.6, size)
        bare = lip_face(0.0, size)[0]
        where = band > 0.9
        before = darkness(img, bare, where)
        after = darkness(lip.clear(img, [pts], mask, 1.0, full=True), bare, where)
        assert abs(after) < 0.15 * before, f"at {size}x, left of the moustache: {after:.3f} of {before:.3f}"


def test_a_full_moustache_leaves_a_philtrum():
    img, pts, mask, band = lip_face(0.6)
    out = lip.clear(img, [pts], mask, 1.0, full=True)
    assert out is not img
    y = cv2.GaussianBlur(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2.0)
    u, v = lip._Frame(pts).uv(*np.meshgrid(np.arange(1000.0), np.arange(900.0)))
    rows = (v > 0.3) & (v < 0.7)
    middle, beside = float(y[rows & (np.abs(u) < 0.05)].mean()), float(y[rows & (np.abs(u) > 0.3) & (np.abs(u) < 0.4)].mean())
    drawn = cv2.GaussianBlur(cv2.cvtColor(lip_face(0.0)[0], cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2.0)
    flat = float(drawn[rows & (np.abs(u) < 0.05)].mean()) / float(drawn[rows & (np.abs(u) > 0.3) & (np.abs(u) < 0.4)].mean())
    assert 0.03 < flat - middle / beside < 0.14, f"a groove down the middle of the new lip: {middle / beside:.3f} against {flat:.3f} on the drawn bare lip"
    low = (v > 0.15) & (v < 0.35) & (np.abs(u) < 0.05)
    assert float(y[low].mean()) < middle - 1.0, "deepest just above the lip"
    rows_, cols_ = lip.PHILTRUM.shape
    assert (rows_, cols_) == (12, 13) and abs(lip.PHILTRUM_U * (cols_ - 1) - 0.6) < 1e-9 and abs(lip.PHILTRUM_V * (rows_ - 1) - 1.1) < 1e-9
    assert float(lip.PHILTRUM[1:8, 0].max()) < -3.0 and int(lip.PHILTRUM[:, 0].argmin()) in (2, 3), "the groove: down the middle, deepest just above the lip"
    assert float(np.abs(lip.PHILTRUM[1:9, 9:]).max()) < 1.0, "none of it by 0.45 from the middle"
    assert float(lip.PHILTRUM[10, 4:8].min()) > 2.0 > float(lip.PHILTRUM[10, 0]), "under the nose, light either side of the groove"
    assert lip.FULL["philtrum_light"] >= lip.FULL["philtrum"] >= 1.0
    assert max(abs(m) for m in lip._Frame(pts).mid) < 1e-6, "down the lip's own middle (here the mouth's)"
    aside = pts.copy()
    aside[[0, 2]] += (14.0, 0.0)                                # the lip's top point and the nose's base 0.2 (u) to one side
    assert all(abs(m - 0.2) < 1e-6 for m in lip._Frame(aside).mid)


def test_a_full_moustache_leaves_clear_skin_beside_it_alone():
    img, pts, mask, band = lip_face(0.6)
    out = lip.clear(img, [pts], mask, 1.0, full=True)
    hair = (band > 0.02).astype(np.uint8)
    near = cv2.dilate(hair, np.ones((3, 3), np.uint8), iterations=10) > 0        # within 0.055 eye distances of the moustache
    ring = near & ~(cv2.dilate(hair, np.ones((3, 3), np.uint8), iterations=4) > 0) & (mask > 0.99)
    moved = np.abs(out.astype(np.int16) - img).max(axis=2)
    assert ring.sum() > 2000 and float((moved[ring] <= 1).mean()) > 0.75, \
        f"skin with no hair on it, 4 to 10 px from the moustache, is mostly as it was (0.66 of it before this was built): {float((moved[ring] <= 1).mean()):.2f} of it"
    assert float((moved[band > 0.9] > 6).mean()) > 0.5, "while the moustache itself is replaced"


def test_the_moustache_step_is_for_a_face_straight_on_with_its_mouth_closed():
    img, pts, mask, band = lip_face()
    assert lip.clear(img, [pts], mask, 1.0) is not img
    turned = pts.copy()
    turned[2, 0] += 0.1 * 180                       # the nose's base a tenth of the eyes' distance to one side
    assert lip.clear(img, [turned], mask, 1.0) is img, "a turned head"
    open_mouth = pts.copy()
    open_mouth[14, 1] += 0.15 * 180
    assert lip.clear(img, [open_mouth], mask, 1.0) is img, "an open mouth"
    smiling = pts.copy()
    for i, x in ((61, 405), (291, 595)):
        smiling[i, 0] = x                           # the mouth's corners pulled wide
    assert lip.clear(img, [smiling], mask, 1.0) is img, "a broad smile"
    assert lip.clear(img, [turned], mask, 1.0, force=True) is not img, "the parity check skips the question"


def test_the_moustache_amount_is_a_share():
    img, pts, mask, band = lip_face()
    bare = lip_face(0.0)[0]
    where = band > 0.9
    before = darkness(img, bare, where)
    half = darkness(lip.clear(img, [pts], mask, 0.5), bare, where)
    assert 0.35 * before < half < 0.65 * before, f"half the amount: {half:.3f} of {before:.3f}"
    assert np.array_equal(lip.clear(img, [pts], mask, 3.0), lip.clear(img, [pts], mask, 1.0)), "no more than all of it"


def test_restore_keeps_the_moustache_where_it_was_painted():
    img, pts, mask, band = lip_face()
    edit = -np.ones((300, 250), np.float32)
    keep = (250, 480, edit)                         # the photo's left half of the mouth
    out = lip.clear(img, [pts], mask, 1.0, keep)
    assert np.array_equal(out[480:780, 250:500], img[480:780, 250:500]), "restored: as it was"
    bare = lip_face(0.0)[0]
    right = (band > 0.9) & (np.arange(band.shape[1])[None, :] > 520)
    assert abs(darkness(out, bare, right)) < 0.2 * darkness(img, bare, right), "the other half still goes"


def test_the_moustache_step_holds_at_other_sizes():
    for size in (0.5, 2.0):                         # 90 px between the eyes (worked at its own size) and 360 (on the grid)
        img, pts, mask, band = lip_face(size=size)
        bare = lip_face(0.0, size)[0]
        where = band > 0.9
        before = darkness(img, bare, where)
        after = darkness(lip.clear(img, [pts], mask, 1.0), bare, where)
        assert abs(after) < 0.25 * before, f"at {size}x, left of the moustache: {after:.3f} of {before:.3f}"
        assert lip.clear(bare, [pts], mask, 1.0) is bare, f"at {size}x a bare lip is still bare"


def test_moustache_helpers():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 1001)
    for q in (10.0, 25.0, 90.0):
        assert abs(lip._percentile(values, q) - float(np.percentile(values, q))) < 1e-9
    assert lip._percentile(np.array([]), 50.0) == 0.0
    rows, cols = lip.TABLE.shape
    u, v = np.meshgrid(np.arange(cols) * lip.TABLE_U, np.arange(rows) * lip.TABLE_V)
    assert np.allclose(lip._table_at(u, v), lip.TABLE, atol=1e-6) and np.allclose(lip._table_at(-u, v), lip.TABLE, atol=1e-6)
    assert np.allclose(lip._table_at(np.array([9.0]), np.array([-3.0])), lip.TABLE[0, -1]), "held at its edges"
    seen = rng.permutation(np.concatenate([np.full(300, 0.2), np.full(700, -0.3)]))   # most points under the line: hair
    a, b = lip._fit(seen, np.linspace(-1, 1, 1000))
    assert abs(a - 0.2) < 0.1 and abs(b) < 0.05, f"the fit rests on the upper points: {a:.3f} {b:+.3f}"
    corners = np.array([(0.0, 0.0), (10.0, 5.0), (20.0, 5.0), (30.0, 0.0)])
    curve = lip._spline(corners)
    assert len(curve) == 3 * 8 + 1 and all(np.abs(curve - c).sum(axis=1).min() < 1e-9 for c in corners), "the curve goes through its points"
    assert float(np.abs(np.diff(curve, axis=0)).max()) < 2.0, "in small steps"
    within = np.zeros((20, 30), bool)
    within[5, 2:12] = within[5, 14:28] = True               # two runs with a gap between them
    seed = np.zeros((20, 30), bool)
    seed[5, 3] = True
    got = lip._grown(seed, within)
    assert got[5, 2:12].all() and not got[5, 14:28].any() and got.sum() == 10, "spread through what it touches, no further"
    data = np.zeros((40, 40, 3), np.float32)
    data[:, :20] = 1.0
    known = np.zeros((40, 40), np.float32)
    known[:, :5] = known[:, 35:] = 1.0                      # known at both sides: 1 on the left, 0 on the right
    carried = lip._fill_in(data, known, (0.4, 0.1), 40.0, 0.02, 0.2)[:, :, 0]
    assert carried[20, 2] > 0.9 and carried[20, 37] < 0.1 and 0.2 < carried[20, 20] < 0.8, "carried in from where it is known"


def chin_face(beard: float = 0.45, size: float = 1.0, seed: int = 5, wide: float = 0.7, edge: float = 7.0):
    """A drawn face from the mouth down, 1000 x 900 at size 1: (photo, landmarks, smoothing mask, the beard's
    own map 0..1). The chin is shaded as a bare chin is (chin.TABLE, so with the jaw's edge and the shadow
    under it); beard darkens strands that run down the chin, inside a band across the jaw's edge, by that
    share. wide: how far across the band goes (1 = under the mouth's corners). edge: the jaw's own sharp edge
    lies this many pixels (at size 1) above the landmarks' outline, as a photo's does."""
    h, w = round(900 * size), round(1000 * size)
    pts = np.zeros((face_landmarks.POINTS, 2), np.float64)
    place = {33: (380, 350), 133: (440, 350), 362: (560, 350), 263: (620, 350), 2: (500, 540), 205: (390, 500), 425: (610, 500),
             61: (430, 620), 185: (445, 612), 40: (462, 606), 39: (478, 602), 37: (490, 599), 0: (500, 600), 267: (510, 599),
             269: (522, 602), 270: (538, 606), 409: (555, 612), 291: (570, 620), 375: (555, 632), 321: (538, 640),
             405: (522, 646), 314: (510, 649), 17: (500, 650), 84: (490, 649), 181: (478, 646), 91: (462, 640), 146: (445, 632),
             13: (500, 623), 14: (500, 625),
             78: (436, 620), 191: (447, 621), 80: (462, 622), 81: (478, 623), 82: (490, 623), 312: (510, 623), 311: (522, 623),
             310: (538, 622), 415: (553, 621), 308: (564, 620),
             136: (385, 688), 150: (410, 705), 149: (435, 720), 176: (458, 731), 148: (480, 738), 152: (500, 740),
             377: (520, 738), 400: (542, 731), 378: (565, 720), 379: (590, 705), 365: (615, 688)}
    for i, xy in place.items():
        pts[i] = np.array(xy, np.float64) * size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    frame = chin._Frame(pts)
    u, v = frame.uv(xx, yy)
    under = ((v - 1.0) * (frame.jaw(u) - frame.top(u)) + edge * size).astype(np.float32)       # px below the jaw's own edge
    u, v = u.astype(np.float32), v.astype(np.float32)
    near = smoother._smoothstep(2.3 - np.abs(u), 0.0, 0.5) * smoother._smoothstep(v + 0.3, 0.0, 0.3)
    bare = (1.0 + near * (chin._table_at(u, v) / 0.9 - 1.0)) * (1.0 - 0.14 * near * smoother._smoothstep(under, -1.5 * size, 1.5 * size))
    band = smoother._smoothstep(wide - np.abs(u), 0.0, 0.15) * smoother._smoothstep(v, 0.5, 0.6) * smoother._smoothstep(1.2 - v, 0.0, 0.1)
    lines = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=max(0.6, 0.9 * size), sigmaY=7.0 * size)
    strands = (lines > 0.9 * float(lines.std())).astype(np.float32)
    tone = (0.72 + 0.04 * np.sin(xx / (170 * size))) * bare * (1.0 - beard * band * strands) \
        + cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), max(0.5, 0.7 * size)) * 0.012
    img = np.clip(np.dstack([tone * 0.62, tone * 0.76, tone]) * 255.0, 0, 255).astype(np.uint8)
    lips = np.zeros((h, w), np.uint8)
    cv2.fillPoly(lips, [np.round(pts[list(lip.UPPER + lip.LOWER)]).astype(np.int32)], 255)
    img[lips > 0] = (90, 80, 190)
    mask = 1.0 - cv2.GaussianBlur(lips.astype(np.float32) / 255.0, (0, 0), 1.0)
    return img, pts, mask.astype(np.float32), band.astype(np.float32)


def test_no_chin_step_without_an_amount_or_a_face():
    img, pts, mask, band = chin_face()
    assert chin.clear(img, [pts], mask, 0.0) is img
    assert chin.clear(img, [], mask, 1.0) is img and chin.clear(img, [None], mask, 1.0) is img
    tiny = pts * (lip.MIN_EYES * 0.9 / 180.0)
    assert chin.clear(img, [tiny], mask, 1.0) is img, "a face too small to judge is left alone"


def test_a_chin_beard_is_replaced_and_the_rest_stays():
    img, pts, mask, band = chin_face()
    bare = chin_face(0.0)[0]
    where = band > 0.9
    before = darkness(img, bare, where)
    out = chin.clear(img, [pts], mask, 1.0)
    after = darkness(out, bare, where)
    assert before > 0.06, f"the drawn beard should be dark: {before:.3f}"
    assert abs(after) < 0.25 * before, f"left of the beard: {after:.3f} of {before:.3f}"
    grain = lambda a: (cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
                       - cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 3.0))
    cheeks = np.zeros(img.shape[:2], bool)
    cheeks[480:520, 370:410] = cheeks[480:520, 590:630] = True
    new, own = float(grain(out)[where].std()), float(grain(bare)[cheeks].std())
    assert 0.5 * own < new < 2.0 * own, f"the new skin has the cheeks' grain, neither none nor the strands': {new:.2f} against {own:.2f}"
    lips = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(lips, [np.round(pts[list(lip.UPPER + lip.LOWER)]).astype(np.int32)], 255)
    inside = cv2.erode(lips, np.ones((9, 9), np.uint8)) > 0
    assert np.array_equal(out[inside], img[inside]), "the lips are not touched"
    far = cv2.erode((band < 1e-6).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    assert np.abs(out[far].astype(np.int16) - img[far]).max() <= 2, "nor is the skin away from the beard"
    assert not np.array_equal(out, img) and np.array_equal(out, chin.clear(img, [pts], mask, 1.0)), "the same every time"


def test_a_bare_chin_is_left_as_it_is():
    for seed in (5, 6, 7):
        bare, pts, mask, band = chin_face(0.0, seed=seed)
        assert chin.clear(bare, [pts], mask, 1.0) is bare, "a bare chin has nothing to remove"


def test_the_jaw_keeps_its_edge_under_the_beard():
    img, pts, mask, band = chin_face()
    bare = chin_face(0.0)[0]
    out = chin.clear(img, [pts], mask, 1.0)
    y = lambda a: cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2.0)
    column = lambda a: y(a)[690:790, 480:520].mean(axis=1)             # down the middle of the chin, across the jaw's edge
    step = lambda c_: float((c_[:-8] - c_[8:]).max())                  # the most brightness lost over 8 rows
    place = lambda c_: int(np.argmax(c_[:-8] - c_[8:]))
    was, now = column(bare), column(out)
    assert step(now) > 0.7 * step(was), f"the edge is still an edge: {step(now):.1f} against {step(was):.1f}"
    assert abs(place(now) - place(was)) <= 3, f"and where it was: {place(now)} against {place(was)}"
    seen = {}
    assert chin._clear_face(img.copy(), pts, mask, 1.0, None, False, seen)
    assert seen["edge"]["moved"] and abs(seen["edge"]["middle"] * 180.0 + 7.0) < 2.0,         f"the frame was moved onto the photo's own edge, 7 px above the outline: {seen['edge']}"


def test_a_beard_across_the_jaw_is_left_alone():
    img, pts, mask, band = chin_face(wide=2.2)                         # strands on past the bounds on both sides
    assert chin.clear(img, [pts], mask, 1.0) is img, "a clean patch cut out of it would show"
    assert chin.clear(img, [pts], mask, 1.0, force=True) is not img, "the parity check skips the question"


def test_the_chin_step_is_for_a_face_straight_on_with_its_mouth_closed():
    img, pts, mask, band = chin_face()
    assert chin.clear(img, [pts], mask, 1.0) is not img
    turned = pts.copy()
    turned[2] += (20.0, 0.0)                                           # the nose well to one side of the middle of the eyes
    assert chin.clear(img, [turned], mask, 1.0) is img, "a turned head"
    open_mouth = pts.copy()
    open_mouth[14] += (0.0, 25.0)
    assert chin.clear(img, [open_mouth], mask, 1.0) is img, "an open mouth"
    short = pts.copy()
    short[list(chin.JAW)] -= (0.0, 40.0)                               # the chin's point 0.28 eye distances under the lip: a head tipped back
    assert chin.clear(img, [short], mask, 1.0) is img, "a chin too short for its face"
    assert chin.clear(img, [turned], mask, 1.0, force=True) is not img, "the parity check skips the question"


def test_the_chin_amount_is_a_share_and_restore_keeps_the_beard():
    img, pts, mask, band = chin_face()
    bare = chin_face(0.0)[0]
    where = band > 0.9
    before = darkness(img, bare, where)
    half = darkness(chin.clear(img, [pts], mask, 0.5), bare, where)
    assert 0.3 * before < half < 0.7 * before, f"half of it: {half:.3f} of {before:.3f}"
    assert np.array_equal(chin.clear(img, [pts], mask, 3.0), chin.clear(img, [pts], mask, 1.0)), "no more than all of it"
    edit = np.full((400, 250), -1.0, np.float32)
    out = chin.clear(img, [pts], mask, 1.0, (250, 600, edit))          # Restore over the photo's left half of the chin
    left, right = where & (np.arange(1000)[None, :] < 480), where & (np.arange(1000)[None, :] > 520)
    assert np.array_equal(out[left], img[left]), "kept where it was painted"
    assert abs(darkness(out, bare, right)) < 0.3 * before, "and gone beside it"


def test_the_chin_step_holds_at_other_sizes():
    for size in (0.5, 2.0):
        img, pts, mask, band = chin_face(size=size)
        bare = chin_face(0.0, size=size)[0]
        where = band > 0.9
        before = darkness(img, bare, where)
        after = darkness(chin.clear(img, [pts], mask, 1.0), bare, where)
        assert abs(after) < 0.3 * before, f"at {size}x, left of the beard: {after:.3f} of {before:.3f}"
        assert chin.clear(bare, [pts], mask, 1.0) is bare, f"at {size}x a bare chin is still bare"


def test_the_chin_step_stays_out_without_a_jaw():
    img, pts, mask, band = chin_face()
    no_jaw = pts.copy()
    no_jaw[list(chin.JAW)] = 0.0                                        # landmarks that have no jaw (the older method's)
    assert not chin._Frame(no_jaw).ok
    assert chin.clear(img, [no_jaw], mask, 1.0) is img and chin.clear(img, [no_jaw], mask, 1.0, force=True) is img


def test_chin_helpers():
    rows, cols = chin.TABLE.shape
    u, v = np.meshgrid(np.arange(cols) * chin.TABLE_U, np.arange(rows) * chin.TABLE_V)
    assert np.allclose(chin._table_at(u, v), chin.TABLE, atol=1e-6) and np.allclose(chin._table_at(-u, v), chin.TABLE, atol=1e-6)
    assert np.allclose(chin._table_at(np.array([9.0]), np.array([-3.0])), chin.TABLE[0, -1]), "held at its edges"
    assert float(chin.TABLE[10:12, 0].max()) < float(chin.TABLE[4:6, 0].min()) - 0.2, "the jaw's shadow is darker than the chin"
    kernels = chin._line_kernels(6)
    assert len(kernels) == 5 and all(k.shape == (7, 7) and int(k.sum()) == 7 and (k.sum(axis=1) == 1).all() for k in kernels), "one pixel a row"
    dots = np.zeros((40, 40), np.uint8)
    dots[10:12, 10:12] = 1                                              # a pore
    dots[20:30, 25] = 1                                                 # a strand
    kept = np.zeros_like(dots)
    for k in kernels:
        kept |= cv2.morphologyEx(dots, cv2.MORPH_OPEN, k)
    assert kept[20:30, 25].all() and not kept[10:12, 10:12].any(), "a line stays, a dot goes"
    spot = np.zeros((41, 41), np.float32)
    spot[20, 20] = 1.0
    along = chin._blur2(spot, 4.0, 0.0)
    assert float(along[20].sum()) > 0.999 and float(np.abs(along[:20]).sum() + np.abs(along[21:]).sum()) == 0.0, "along its row only"
    img, pts, mask, band = chin_face()
    f = chin._Frame(pts)
    assert f.posed and abs(f.tall - 0.5) < 0.01
    us, vs = np.array([-1.2, -0.3, 0.0, 0.8, 1.4]), np.array([0.1, 0.6, 1.0, 1.3, 0.4])
    back = f.uv(*f.at(us, vs))
    assert np.allclose(back[0], us, atol=1e-6) and np.allclose(back[1], vs, atol=1e-6), "there and back"
    x, y = f.at(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    assert abs(x[0] - 500) < 1 and abs(y[0] - 650) < 1.5 and abs(y[1] - 740) < 1.5, "from the lip's edge to the jaw's"
    data = np.zeros((40, 60, 3), np.float32)
    data[20:] = 1.0                                                     # an edge along a row
    known = np.ones((40, 60), np.float32)
    known[:, 15:45] = 0.0                                               # not known in the middle
    carried = chin._fill_in(data, known, 40.0)[:, :, 0]
    assert carried[19, 30] < 0.02 and carried[20, 30] > 0.98, "carried in along the rows: the edge stays an edge"


if __name__ == "__main__":
    tests =[(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
