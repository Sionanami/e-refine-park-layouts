# -*- coding: utf-8 -*-
"""Small correctness checks for evaluation functions (not method ablations)."""
from __future__ import annotations

import numpy as np

from evaluation.eval_spatial import (
    clumpy_c4, mfr_from_sizes, geometric_holes,
    division_index, shape_am, _raster_edge_perimeter,
)


def check_clumpy_order() -> list:
    """Equal-area dispersed vs compact plant should order CLUMPY correctly."""
    D = np.ones((20, 20), dtype=bool)
    # Compact 4x4 block (A=16)
    compact = np.zeros((20, 20), dtype=bool)
    compact[8:12, 8:12] = True
    # Dispersed 16 isolated pixels
    disp = np.zeros((20, 20), dtype=bool)
    coords = [(1, 1), (1, 5), (1, 9), (1, 13), (1, 17),
              (5, 1), (5, 5), (5, 9), (5, 13), (5, 17),
              (9, 1), (9, 5), (9, 17), (13, 1), (13, 9), (17, 5)]
    for y, x in coords:
        disp[y, x] = True
    assert int(compact.sum()) == int(disp.sum()) == 16
    c_c = clumpy_c4(compact, D)
    c_d = clumpy_c4(disp, D)
    assert c_c['CLUMPY_P'] is not None and c_d['CLUMPY_P'] is not None
    assert c_c['CLUMPY_P'] > c_d['CLUMPY_P'], (c_c['CLUMPY_P'], c_d['CLUMPY_P'])
    # Empty / full NA
    assert clumpy_c4(np.zeros_like(D), D)['status'] == 'empty_class'
    assert clumpy_c4(D.copy(), D)['status'] == 'fills_domain'
    # Range
    for c in (c_c, c_d):
        assert -1.0 - 1e-9 <= c['CLUMPY_P'] <= 1.0 + 1e-9
    return ['clumpy_order_ok', f"compact={c_c['CLUMPY_P']:.4f}", f"disp={c_d['CLUMPY_P']:.4f}"]


def check_mfr_boundaries() -> list:
    sizes = np.array([1, 6, 7, 100], dtype=int)
    m = mfr_from_sizes(sizes, 6)
    assert m['N_micro'] == 2 and m['A_micro'] == 7
    # Denominator growth, micro area fixed
    sizes2 = np.array([1, 6, 7, 100, 200], dtype=int)
    m2 = mfr_from_sizes(sizes2, 6)
    assert m2['A_micro'] == m['A_micro']
    assert m2['MFR'] < m['MFR']
    empty = mfr_from_sizes(np.array([], dtype=int), 6)
    assert empty['status'] == 'empty_class' and empty['MFR'] is None
    return ['mfr_boundary_ok']


def check_holes_exclude_exterior() -> list:
    domain = np.zeros((10, 10), dtype=bool)
    domain[1:9, 1:9] = True
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    mask[4:6, 4:6] = False  # interior hole
    h = geometric_holes(mask, domain)
    assert h['NH'] == 1 and h['AH'] == 4, h
    # Exterior background not a hole
    mask2 = domain.copy()
    h2 = geometric_holes(mask2, domain)
    assert h2['NH'] == 0 and h2['AH'] == 0
    return ['holes_ok']


def check_division_shape() -> list:
    """Square patch: SHAPE_AM=1; absent class DIVISION=1; two equal patches lower DIVISION than one."""
    D = np.ones((20, 20), dtype=bool)
    sq = np.zeros((20, 20), dtype=bool)
    sq[4:12, 4:12] = True  # 8x8
    assert abs(_raster_edge_perimeter(sq) - 32.0) < 1e-9
    shp = shape_am(sq)
    assert shp['SHAPE_AM'] is not None and abs(shp['SHAPE_AM'] - 1.0) < 1e-6, shp
    div1 = division_index(np.array([64], dtype=int), 400)
    assert abs(div1['DIVISION'] - (1.0 - (64 / 400) ** 2)) < 1e-9
    div_abs = division_index(np.array([], dtype=int), 400)
    assert div_abs['absent'] and abs(div_abs['DIVISION'] - 1.0) < 1e-15
    two = np.zeros((20, 20), dtype=bool)
    two[2:6, 2:6] = True
    two[12:16, 12:16] = True
    from scipy import ndimage as ndi
    lab, n = ndi.label(two, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(lab.ravel())[1:]
    div2 = division_index(sizes, 400)
    # same total area 32 vs one blob 32: two patches → higher DIVISION
    one = division_index(np.array([32], dtype=int), 400)
    assert div2['DIVISION'] > one['DIVISION']
    return [
        'division_shape_ok',
        f"shape_am_square={shp['SHAPE_AM']:.6f}",
        f"div_one={one['DIVISION']:.6f}",
        f"div_two={div2['DIVISION']:.6f}",
    ]


def run_all_checks() -> dict:
    msgs = []
    msgs += check_clumpy_order()
    msgs += check_mfr_boundaries()
    msgs += check_holes_exclude_exterior()
    msgs += check_division_shape()
    return dict(ok=True, messages=msgs)
