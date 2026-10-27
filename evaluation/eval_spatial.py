# -*- coding: utf-8 -*-
"""Spatial structure metrics for manuscript evaluation (road / area / green)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import perimeter_crofton
from skimage.morphology import skeletonize

K8 = np.ones((3, 3), dtype=np.uint8)
K4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

# Class ids (study_common)
ID_BUILDING, ID_GREEN, ID_PAVEMENT, ID_WATER, ID_ROAD, ID_PLANT = 1, 2, 3, 4, 5, 6

# Effective micro-fragment area thresholds (px²), from frozen call sites / parameters.
# Comparison: area <= t_c (matches remove_noise_crumbs / plant noise branch).
MFR_THRESHOLDS = {
    'pavement': dict(t_c=10, source='refine.regularize_surface_rgb pavement_crumb_area_px default',
                     param='pavement_crumb_area_px', op='<='),
    'building': dict(t_c=16, source='refine.building_settings max_crumb_area_px default',
                     param='max_crumb_area_px', op='<='),
    'water': dict(t_c=16, source='refine.regularize_surface_rgb water_crumb_area_px default',
                  param='water_crumb_area_px', op='<='),
    'plant': dict(t_c=6, source='parameters.json r2.vegetation.noise_area_max_px',
                  param='noise_area_max_px', op='<='),
}


def _component_sizes(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask, structure=K8)
    if n == 0:
        return np.array([], dtype=int)
    return np.bincount(lab.ravel())[1:]


def road_metrics(Q: np.ndarray, domain: np.ndarray) -> Dict[str, Any]:
    road = (Q == ID_ROAD) & domain
    out: Dict[str, Any] = dict(road_px=int(road.sum()), LCR=None, RER=None,
                               LCR_status='ok', RER_status='ok')
    if not road.any():
        out['LCR_status'] = 'no_road'
        out['RER_status'] = 'no_road'
        return out
    sizes = _component_sizes(road)
    out['LCR'] = float(sizes.max() / road.sum())
    sk = skeletonize(road, method='zhang')
    sk_px = int(sk.sum())
    if sk_px == 0:
        out['RER_status'] = 'no_skeleton'
        return out
    # Isolated single-pixel skeleton: still defined; note degenerate
    degree = ndi.convolve(sk.astype(np.int16), K8, mode='constant') - sk
    endpoints = int(((degree == 1) & sk).sum())
    out['RER'] = float(endpoints / sk_px)
    out['skeleton_px'] = sk_px
    out['endpoint_px'] = endpoints
    if sk_px <= 2:
        out['RER_status'] = 'degenerate_skeleton'
    return out


def patch_stats(mask: np.ndarray) -> Dict[str, Any]:
    """NP / MCS / MaxCS over all 8-connected components (no area filter)."""
    sizes = _component_sizes(mask)
    np_c = int(len(sizes))
    if np_c == 0:
        return dict(NP=0, MCS=None, MaxCS=None, A=0, sizes=sizes)
    A = int(sizes.sum())
    return dict(NP=np_c, MCS=float(A / np_c), MaxCS=int(sizes.max()), A=A, sizes=sizes)


def mfr_from_sizes(sizes: np.ndarray, t_c: float) -> Dict[str, Any]:
    if len(sizes) == 0:
        return dict(MFR=None, N_micro=None, A_micro=None, A=0, status='empty_class')
    A = int(sizes.sum())
    micro = sizes[sizes <= float(t_c)]
    A_micro = int(micro.sum())
    return dict(
        MFR=float(A_micro / A) if A > 0 else None,
        N_micro=int(len(micro)),
        A_micro=A_micro,
        A=A,
        t_c=float(t_c),
        status='ok',
    )


def _bbox_pad(mask: np.ndarray, pad: int = 2) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    return y0, y1, x0, x1


def obc_area_weighted(mask: np.ndarray) -> Optional[float]:
    """Area-weighted mean of P^2/(4πA) on hole-filled measurement copies (Crofton-4)."""
    lab, n = ndi.label(mask, structure=K8)
    if n == 0:
        return None
    num = 0.0
    den = 0.0
    for k in range(1, n + 1):
        comp = lab == k
        box = _bbox_pad(comp, pad=2)
        if box is None:
            continue
        y0, y1, x0, x1 = box
        win = np.zeros((y1 - y0 + 2, x1 - x0 + 2), dtype=bool)  # zero pad border
        win[1:-1, 1:-1] = comp[y0:y1, x0:x1]
        filled = ndi.binary_fill_holes(win, structure=K4)
        A_k = float(filled.sum())
        if A_k <= 0:
            continue
        P_k = float(perimeter_crofton(filled, directions=4))
        obc_k = (P_k * P_k) / (4.0 * np.pi * A_k)
        num += obc_k * A_k
        den += A_k
    if den <= 0:
        return None
    return float(num / den)


def geometric_holes(mask: np.ndarray, domain: np.ndarray) -> Dict[str, Any]:
    """Background-4 holes fully inside D, on class mask (no nested double-count)."""
    if not mask.any():
        return dict(NH=None, AH=None, status='empty_class')
    # Work only inside domain: outside-domain treated as foreground barrier so
    # exterior background is not counted as a hole.
    work = mask & domain
    if not work.any():
        return dict(NH=None, AH=None, status='empty_in_domain')
    # Background candidates: non-class inside D
    bg = (~work) & domain
    # Holes = bg CCs not connected (4) to domain boundary through bg∪(~domain)
    # Equivalent: fill holes of work within a padded crop; difference ∩ domain.
    filled = ndi.binary_fill_holes(work, structure=K4)
    hole_pix = filled & ~work & domain
    if not hole_pix.any():
        return dict(NH=0, AH=0, status='ok', hole_areas=[])
    lab, n = ndi.label(hole_pix, structure=K4)
    areas = [int((lab == i).sum()) for i in range(1, n + 1)]
    return dict(NH=int(n), AH=int(sum(areas)), status='ok', hole_areas=areas)


def _raster_edge_perimeter(comp: np.ndarray) -> float:
    """4-orthogonal raster edge perimeter of a patch (includes hole & crop edges)."""
    # Count directed edges from foreground to non-foreground (incl. OOB as non-fg).
    m = comp.astype(bool)
    p = 0
    p += int((m[:, :-1] & ~m[:, 1:]).sum())   # right edge of left pixel
    p += int((m[:, 1:] & ~m[:, :-1]).sum())   # left edge of right pixel
    p += int((m[:-1, :] & ~m[1:, :]).sum())   # bottom edge of upper pixel
    p += int((m[1:, :] & ~m[:-1, :]).sum())   # top edge of lower pixel
    # Image border: each border foreground pixel contributes one edge outward
    p += int(m[0, :].sum()) + int(m[-1, :].sum()) + int(m[:, 0].sum()) + int(m[:, -1].sum())
    return float(p)


def division_index(sizes: np.ndarray, A_D: int) -> Dict[str, Any]:
    """FRAGSTATS class-level DIVISION: 1 - sum (a_k / A_D)^2. Absent class → 1 + flag."""
    A_D = int(A_D)
    if A_D <= 0:
        return dict(DIVISION=None, status='empty_domain', absent=True)
    if len(sizes) == 0:
        return dict(DIVISION=1.0, status='absent', absent=True)
    s = sizes.astype(np.float64)
    return dict(
        DIVISION=float(1.0 - float(np.sum((s / float(A_D)) ** 2))),
        status='ok',
        absent=False,
    )


def shape_am(mask: np.ndarray) -> Dict[str, Any]:
    """Area-weighted SHAPE index; perimeter counts hole & domain-crop edges."""
    lab, n = ndi.label(mask, structure=K8)
    if n == 0:
        return dict(SHAPE_AM=None, status='empty_class', n_patches=0)
    A_c = float(mask.sum())
    if A_c <= 0:
        return dict(SHAPE_AM=None, status='empty_class', n_patches=0)
    acc = 0.0
    for k in range(1, n + 1):
        comp = lab == k
        a_k = float(comp.sum())
        if a_k <= 0:
            continue
        p_k = _raster_edge_perimeter(comp)
        shape_k = p_k / (4.0 * np.sqrt(a_k))
        acc += (a_k / A_c) * shape_k
    return dict(SHAPE_AM=float(acc), status='ok', n_patches=int(n))


def area_element_metrics(Q: np.ndarray, domain: np.ndarray, class_id: int,
                         class_name: str) -> Dict[str, Any]:
    mask = (Q == class_id) & domain
    base = patch_stats(mask)
    thr = MFR_THRESHOLDS.get(class_name)
    A_D = int(domain.sum())
    div = division_index(base['sizes'], A_D)
    shp = shape_am(mask)
    out: Dict[str, Any] = dict(
        class_name=class_name, class_id=class_id,
        NP_c=base['NP'], MCS_c=base['MCS'], MaxCS_c=base['MaxCS'], A_c=base['A'],
        DIVISION_c=div['DIVISION'], DIVISION_status=div['status'],
        DIVISION_absent=bool(div.get('absent')),
        SHAPE_AM_c=shp['SHAPE_AM'], SHAPE_status=shp['status'],
    )
    if thr is None:
        out.update(MFR_c=None, N_micro=None, A_micro=None, t_c=None,
                   MFR_status='no_threshold')
    else:
        mfr = mfr_from_sizes(base['sizes'], thr['t_c'])
        out.update(
            MFR_c=mfr['MFR'], N_micro=mfr['N_micro'], A_micro=mfr['A_micro'],
            t_c=thr['t_c'], MFR_status=mfr['status'],
            t_source=thr['source'], t_param=thr['param'], t_op=thr['op'],
        )
    if base['NP'] == 0:
        out['OBC_c'] = None
        out['OBC_status'] = 'empty_class'
    else:
        out['OBC_c'] = obc_area_weighted(mask)
        out['OBC_status'] = 'ok'
    holes = geometric_holes(mask, domain)
    out['NH_c'] = holes['NH']
    out['AH_c'] = holes['AH']
    out['hole_status'] = holes['status']
    return out


def clumpy_c4(plant_mask: np.ndarray, domain: np.ndarray) -> Dict[str, Any]:
    """FRAGSTATS C4 clumpiness without extra same-class border.

    p = A/|D|; g_PP = directed 4-neighbour plant–plant adjacencies;
    s_P = 4A (includes edges to other classes and outside D);
    G = g_PP / s_P.
    """
    m = plant_mask & domain
    A = int(m.sum())
    D = int(domain.sum())
    out: Dict[str, Any] = dict(CLUMPY_P=None, p_P=None, G_P=None, g_PP=None,
                               s_P=None, A_P=A, D=D, status='ok')
    if A == 0:
        out['status'] = 'empty_class'
        return out
    if A == D:
        out['status'] = 'fills_domain'
        out['p_P'] = 1.0
        return out
    # Directed 4-neighbour counts inside array; outside D counts as non-plant edge
    # via s_P=4A definition (edges to ~plant, including out-of-D).
    up = m[:-1, :] & m[1:, :]
    down = m[1:, :] & m[:-1, :]
    left = m[:, :-1] & m[:, 1:]
    right = m[:, 1:] & m[:, :-1]
    g_PP = int(up.sum() + down.sum() + left.sum() + right.sum())
    s_P = 4 * A
    if s_P <= 0:
        out['status'] = 'degenerate'
        return out
    G = g_PP / float(s_P)
    p = A / float(D)
    out.update(p_P=p, G_P=G, g_PP=g_PP, s_P=s_P)
    if abs(1.0 - p) < 1e-15:
        out['status'] = 'fills_domain'
        return out
    if G < p and p < 0.5:
        out['CLUMPY_P'] = float((G - p) / p)
    else:
        out['CLUMPY_P'] = float((G - p) / (1.0 - p))
    return out


def green_space_metrics(Q: np.ndarray, domain: np.ndarray) -> Dict[str, Any]:
    lawn = (Q == ID_GREEN) & domain
    plant = (Q == ID_PLANT) & domain
    lawn_s = patch_stats(lawn)
    plant_s = patch_stats(plant)
    thr = MFR_THRESHOLDS['plant']
    mfr = mfr_from_sizes(plant_s['sizes'], thr['t_c'])
    clumpy = clumpy_c4(plant, domain)
    return dict(
        NGC=lawn_s['NP'],
        MCS_lawn=lawn_s['MCS'],
        MaxCS_lawn=lawn_s['MaxCS'],
        A_lawn=lawn_s['A'],
        NP_P=plant_s['NP'],
        MCS_P=plant_s['MCS'],
        MaxCS_P=plant_s['MaxCS'],
        A_P=plant_s['A'],
        MFR_P=mfr['MFR'],
        N_micro_P=mfr['N_micro'],
        A_micro_P=mfr['A_micro'],
        t_P=thr['t_c'],
        CLUMPY_P=clumpy['CLUMPY_P'],
        clumpy_p_P=clumpy['p_P'],
        clumpy_G_P=clumpy['G_P'],
        clumpy_g_PP=clumpy['g_PP'],
        clumpy_s_P=clumpy['s_P'],
        clumpy_status=clumpy['status'],
    )


def tcr_from_tree_groups(tree_groups: Optional[Sequence[dict]],
                         plant_mask: np.ndarray) -> Dict[str, Any]:
    """Group-internal contact ratio from native object table + plant support."""
    if not tree_groups:
        return dict(TCR=None, n_multi_groups=0, n_single_groups=0,
                    status='no_object_table')
    multi = []
    n_single = 0
    for g in tree_groups:
        centres = g.get('centres') or []
        n_g = len(centres)
        if n_g < 2:
            if n_g == 1:
                n_single += 1
            continue
        multi.append(centres)
    if not multi and n_single == 0:
        return dict(TCR=None, n_multi_groups=0, n_single_groups=0,
                    status='no_groups')
    if not multi:
        return dict(TCR=None, n_multi_groups=0, n_single_groups=n_single,
                    status='no_multi_symbol_groups')

    sum_k = 0
    sum_n = 0
    h, w = plant_mask.shape
    for centres in multi:
        n_g = len(centres)
        sum_n += n_g
        # Build support masks I_i
        supports = []
        for c in centres:
            cy, cx = int(c.get('y', -1)), int(c.get('x', -1))
            r = max(1, int(round(float(c.get('r', 2)))))
            if not (0 <= cy < h and 0 <= cx < w):
                supports.append(None)
                continue
            yy, xx = np.ogrid[:h, :w]
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            inter = disk & plant_mask
            supports.append(inter if inter.any() else None)
        # Nodes = non-empty supports; edge if overlap or 8-adjacent contact
        nodes = [i for i, s in enumerate(supports) if s is not None]
        if not nodes:
            # k_g = 0; all planned symbols still in denominator
            continue
        # BFS connectivity
        adj = {i: [] for i in nodes}
        for a_i, i in enumerate(nodes):
            for j in nodes[a_i + 1:]:
                si, sj = supports[i], supports[j]
                if np.any(si & sj):
                    adj[i].append(j)
                    adj[j].append(i)
                    continue
                dil = ndi.binary_dilation(si, structure=K8)
                if np.any(dil & sj):
                    adj[i].append(j)
                    adj[j].append(i)
        seen = set()
        best = 0
        for start in nodes:
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            comp = 0
            while stack:
                u = stack.pop()
                comp += 1
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            best = max(best, comp)
        sum_k += best
    tcr = float(sum_k / sum_n) if sum_n else None
    return dict(TCR=tcr, n_multi_groups=len(multi), n_single_groups=n_single,
                status='ok', sum_k=sum_k, sum_n=sum_n)


def composition_ratios(Q: np.ndarray, domain: np.ndarray) -> Dict[str, float]:
    n = int(domain.sum())
    # GSR=green/lawn, PAR=plant, BCR=building, WBR=water, RAR=road, PVR=pavement
    mapping = dict(GSR=ID_GREEN, PAR=ID_PLANT, BCR=ID_BUILDING,
                   WBR=ID_WATER, RAR=ID_ROAD, PVR=ID_PAVEMENT)
    out = {name: float(((Q == kid) & domain).sum() / n) for name, kid in mapping.items()}
    out['SUM'] = float(sum(out[k] for k in mapping))
    return out
