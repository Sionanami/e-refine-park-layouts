#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic site compositing for final product output.

Task: keep outside-site city roads/environment; generate park only inside site.

    Y = M * G + (1 - M) * C

If A is the retained basemap, C = A.

Red-line rule (fixed):
  REDLINE_PRESERVE_FROM_C
  - M = filled green site from A (largest component; purple roads excluded).
  - Strict red outline pixels are forced onto the context side: always taken from C.
  - Therefore final Y keeps known red boundary and all outside roads from C.

Diagnostics must still keep raw G (pre-compose). Outside fidelity of Y is
guaranteed by construction and must not be cited as model boundary control.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from mask_from_A import _green_fill, _largest_component, _purple_road, _red_border_strict


REDLINE_RULE = "REDLINE_PRESERVE_FROM_C"


def site_mask_M_compose(a_rgb: np.ndarray | Image.Image) -> np.ndarray:
    """Bool HxW. True = site interior to take from G. Red outline NOT included."""
    if isinstance(a_rgb, Image.Image):
        a = np.array(a_rgb.convert("RGB"))
    else:
        a = a_rgb
    green = _green_fill(a)
    g_closed = np.array(Image.fromarray((green.astype(np.uint8) * 255)).filter(ImageFilter.MaxFilter(3))) > 0
    g_closed = np.array(Image.fromarray((g_closed.astype(np.uint8) * 255)).filter(ImageFilter.MinFilter(3))) > 0
    main = _largest_component(g_closed)
    main = main | (
        green & (np.array(Image.fromarray((main.astype(np.uint8) * 255)).filter(ImageFilter.MaxFilter(3))) > 0)
    )
    # never claim purple roads
    return main & (~_purple_road(a))


def compose_Y(
    G: np.ndarray | Image.Image,
    C: np.ndarray | Image.Image,
    M: np.ndarray | None = None,
    redline_rule: str = REDLINE_RULE,
) -> tuple[np.ndarray, dict]:
    """Return uint8 Y and meta. M True = use G (inside)."""
    if isinstance(G, Image.Image):
        G = np.array(G.convert("RGB"))
    if isinstance(C, Image.Image):
        C = np.array(C.convert("RGB"))
    if G.shape != C.shape:
        raise ValueError(f"G/C shape mismatch {G.shape} vs {C.shape}")
    if M is None:
        M = site_mask_M_compose(C)
    if M.shape != G.shape[:2]:
        raise ValueError(f"M shape {M.shape} vs G {G.shape[:2]}")

    M_eff = M.astype(bool).copy()
    red = _red_border_strict(C)
    if redline_rule == "REDLINE_PRESERVE_FROM_C":
        M_eff = M_eff & (~red)
    elif redline_rule == "REDLINE_FROM_G":
        pass
    else:
        raise ValueError(redline_rule)

    Y = np.where(M_eff[..., None], G, C).astype(np.uint8)
    meta = {
        "formula": "Y = M_eff*G + (1-M_eff)*C",
        "C_source": "A (retained basemap)",
        "redline_rule": redline_rule,
        "M_frac": float(M.mean()),
        "M_eff_frac": float(M_eff.mean()),
        "red_frac": float(red.mean()),
    }
    return Y, meta


def outside_fidelity(Y: np.ndarray, C: np.ndarray, M_eff: np.ndarray) -> dict:
    """Verify outside (and redline-from-C) pixels match basemap exactly."""
    outside = ~M_eff
    if not outside.any():
        return {"n_outside": 0, "n_mismatch": 0, "exact": True, "mae_outside": 0.0}
    diff = np.abs(Y.astype(np.int16) - C.astype(np.int16)).sum(axis=-1)
    mism = outside & (diff > 0)
    return {
        "n_outside": int(outside.sum()),
        "n_mismatch": int(mism.sum()),
        "exact": int(mism.sum()) == 0,
        "mae_outside": float(diff[outside].mean() / 3.0),
    }


@dataclass
class ComposeBundle:
    M: np.ndarray
    M_eff: np.ndarray
    Y: np.ndarray
    meta: dict
    fidelity: dict


def compose_bundle(G, C, redline_rule: str = REDLINE_RULE) -> ComposeBundle:
    if isinstance(C, Image.Image):
        C_arr = np.array(C.convert("RGB"))
    else:
        C_arr = C
    M = site_mask_M_compose(C_arr)
    Y, meta = compose_Y(G, C_arr, M=M, redline_rule=redline_rule)
    red = _red_border_strict(C_arr)
    M_eff = M & (~red) if redline_rule == "REDLINE_PRESERVE_FROM_C" else M
    fid = outside_fidelity(Y, C_arr, M_eff)
    meta["outside_fidelity"] = fid
    return ComposeBundle(M=M, M_eff=M_eff, Y=Y, meta=meta, fidelity=fid)
