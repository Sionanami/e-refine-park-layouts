# -*- coding: utf-8 -*-
"""Semantic coverage / write-authority rules for R2 (analysis labels).

Priority (highest → lowest):
  1. Fixed zone + condition-A markers (black / urban purple / red)
  2. Building — top structural class; may cover road / pavement / water / plant / lawn;
               never covered by other R2 elements after approved mask is locked
  3. Road — second only to building; may cover pavement / water / plant / lawn;
            must not be damaged by pavement / water / plant / lawn
  4. Pavement — may cover plant / lawn; not road or building;
               with pavement_protect_water=True (主算法默认) must not cover water
  5. Water — softest planar; may cover plant / lawn only
  6. Plant / lawn — soft fill

Disappeared-pixel policy (contour / denoise shrink):
  Released pixels demote to **lawn** by default — never promote to road.
  Interior holes of an approved contour are filled back as that class
  (building holes → building; optional planar enclosed holes → pavement/water).
  Orphan exterior crumbs after shrink → lawn (soft sink), avoiding priority fights.

This is enforced as module-specific write masks + road/building locks, not a
single global argmax stack.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image

ORANGE = (241, 145, 73)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GREEN = (0, 152, 67)
LAWN = (0, 255, 0)


def _eq(arr: np.ndarray, rgb) -> np.ndarray:
    return np.all(arr == np.asarray(rgb, dtype=np.uint8), axis=2)


def class_masks(arr: np.ndarray, fixed: np.ndarray) -> Dict[str, np.ndarray]:
    return dict(
        fixed=fixed.astype(bool),
        building=_eq(arr, MAGENTA),
        road=_eq(arr, ORANGE),
        pavement=_eq(arr, YELLOW),
        water=_eq(arr, CYAN),
        plant=_eq(arr, GREEN),
        lawn=_eq(arr, LAWN),
    )


def protected_from_pavement(
    m: Dict[str, np.ndarray],
    *,
    protect_water: bool = False,
) -> np.ndarray:
    """Pavement must not cover: fixed, building, road; optionally water."""
    prot = m['fixed'] | m['building'] | m['road']
    if protect_water:
        prot = prot | m['water']
    return prot


def protected_from_water(m: Dict[str, np.ndarray]) -> np.ndarray:
    """Water must not cover: fixed, building, road, pavement."""
    return m['fixed'] | m['building'] | m['road'] | m['pavement']


def road_hard_blockers(m: Dict[str, np.ndarray]) -> list:
    """Roads must not enter fixed/building. Building outranks road."""
    return [m['fixed'], m['building']]


def vegetation_blockers(
    m: Dict[str, np.ndarray],
    *,
    protect_water: bool = False,
) -> np.ndarray:
    """Plant must not cover road/building/pavement/fixed; optionally water."""
    prot = m['fixed'] | m['building'] | m['road'] | m['pavement']
    if protect_water:
        prot = prot | m['water']
    return prot.astype(np.uint8)


def lawn_clump_blockers(m: Dict[str, np.ndarray]) -> np.ndarray:
    """Lawn clump must not cover road/building/pavement/fixed/plant."""
    return m['fixed'] | m['building'] | m['road'] | m['pavement'] | m['plant']


def apply_color_write_clipped(
    before: np.ndarray,
    proposed: np.ndarray,
    color_rgb,
    protected: np.ndarray,
) -> Image.Image:
    """Keep proposed colour only outside protected pixels; restore protected from before."""
    out = np.array(proposed, copy=True, dtype=np.uint8)
    prot = protected.astype(bool)
    out[prot] = before[prot]
    return Image.fromarray(out)


COVERAGE_RULES_TEXT = (
    "fixed/A > building > road > pavement > water > plant/lawn; "
    "shrink→lawn (never→road); building may cover road; "
    "road protected from pavement/water/vegetation"
)
