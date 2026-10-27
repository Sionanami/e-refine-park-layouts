#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build explicit site mask M from condition A only (never from B).

Full site = green fill (hole-filled) plus a thin red boundary band.
Purple roads are never part of M. Red detector is strict so purple is not
mistaken for the site outline.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _green_fill(a: np.ndarray) -> np.ndarray:
    # int16: uint8 (r+40) overflows and falsely flags near-white as green
    r = a[..., 0].astype(np.int16)
    g = a[..., 1].astype(np.int16)
    b = a[..., 2].astype(np.int16)
    return (g > 180) & (g > r + 40) & (g > b + 40)


def _red_border_strict(a: np.ndarray) -> np.ndarray:
    """Thin site outline: high R, low G/B (excludes purple roads)."""
    r = a[..., 0].astype(np.int16)
    g = a[..., 1].astype(np.int16)
    b = a[..., 2].astype(np.int16)
    return (r > 180) & (g < 120) & (b < 120) & (r > g + 60) & (r > b + 60)


def _purple_road(a: np.ndarray) -> np.ndarray:
    r = a[..., 0].astype(np.float32)
    g = a[..., 1].astype(np.float32)
    b = a[..., 2].astype(np.float32)
    white = (r > 220) & (g > 220) & (b > 220)
    return (r > 80) & (b > 80) & (g < r * 0.85) & (r + b > 200) & (~white)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub = mask[y0:y1, x0:x1]
    h, w = sub.shape
    seen = np.zeros_like(sub, dtype=bool)
    best = np.zeros_like(sub, dtype=bool)
    best_n = 0
    for y in range(h):
        for x in range(w):
            if not sub[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            comp = []
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and sub[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) > best_n:
                best_n = len(comp)
                best = np.zeros_like(sub, dtype=bool)
                for cy, cx in comp:
                    best[cy, cx] = True
    out = np.zeros_like(mask, dtype=bool)
    if best_n > 0:
        out[y0:y1, x0:x1] = best
        return out
    return mask


def site_mask_M_from_A(a_rgb: np.ndarray | Image.Image, border_dilate: int = 2) -> np.ndarray:
    """Return bool HxW mask. True = full site from A (not from B)."""
    if isinstance(a_rgb, Image.Image):
        a = np.array(a_rgb.convert("RGB"))
    else:
        a = a_rgb
    green = _green_fill(a)
    red = _red_border_strict(a)
    # close 1px gaps so anti-aliased green stays one component
    g_closed = np.array(Image.fromarray((green.astype(np.uint8) * 255)).filter(ImageFilter.MaxFilter(3))) > 0
    g_closed = np.array(Image.fromarray((g_closed.astype(np.uint8) * 255)).filter(ImageFilter.MinFilter(3))) > 0
    # main site = largest closed-green blob
    main = _largest_component(g_closed)
    # attach any original green pixels adjacent to main (recover from close erosion)
    main = main | (green & (np.array(Image.fromarray((main.astype(np.uint8) * 255)).filter(ImageFilter.MaxFilter(3))) > 0))
    if border_dilate > 0:
        k = border_dilate * 2 + 1
        band = np.array(Image.fromarray((main.astype(np.uint8) * 255)).filter(ImageFilter.MaxFilter(k))) > 0
    else:
        band = main
    # add only red outline that touches the dilated main site
    core = main | (red & band) | band
    # never claim purple roads as site; never drop main green
    core = (core & (~_purple_road(a))) | main
    return core


def M_to_float01(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.float32)


def pack_rgb_M(a_u8: np.ndarray | Image.Image, use_M: bool) -> np.ndarray:
    """HxWx4 float32 in [0,1]. use_M=False -> zero 4th channel (control)."""
    if isinstance(a_u8, Image.Image):
        a = np.array(a_u8.convert("RGB"))
    else:
        a = a_u8
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
    rgb = a.astype(np.float32) / 255.0
    m = M_to_float01(site_mask_M_from_A(a))
    if not use_M:
        m = np.zeros_like(m)
    return np.concatenate([rgb, m[..., None]], axis=-1).astype(np.float32)
