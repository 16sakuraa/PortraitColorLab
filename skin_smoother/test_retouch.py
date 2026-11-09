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
    assert np.array_equal(smoother.smooth_face(img, 0.8, 0.6, mask, FACE, 0.5, grain=0.3), plain), \
        "grain below 1 counts as 1"


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
