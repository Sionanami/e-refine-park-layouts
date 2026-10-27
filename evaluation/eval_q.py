# -*- coding: utf-8 -*-
"""Unified evaluation readout Q (eval-only; never fed into R2′).

Q keeps exact palette labels; non-palette pixels use unweighted nearest RGB L2.
"""
from __future__ import annotations

import numpy as np

from core.study_common import PALETTE, exact_labels, nearest_labels


def readout_Q(image: np.ndarray) -> np.ndarray:
    """Exact class where palette match; else unweighted nearest palette id."""
    exact = exact_labels(image)
    nearest = nearest_labels(image)
    return np.where(exact != 0, exact, nearest).astype(np.uint8)


def vlc_exact(image: np.ndarray, domain: np.ndarray) -> float:
    """Valid-label coverage: exact six-class RGB fraction inside D."""
    n = int(domain.sum())
    if n == 0:
        raise ValueError('Empty evaluation domain D')
    exact = exact_labels(image)
    return float(((exact != 0) & domain).sum() / n)


def palette_rgb_set():
    return {tuple(v) for v in PALETTE.values()}
