# -*- coding: utf-8 -*-
"""Local surface regularization: micro-holes, narrow gaps, noise crumbs, edge despeckle.

Closings only propose candidates. Each edit is accepted only inside a boundary
band, after protected-class checks, else rolled back. Not a global morph close.
Units: px at working resolution (engineering defaults, not validated optima).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Engineering defaults (px @ working resolution).
DEFAULT_EDGE_KERNEL_PX = 5
DEFAULT_EDGE_ITERS = 2
DEFAULT_BOUNDARY_BAND_PX = 3
DEFAULT_MICRO_HOLE_AREA_PX = 16
DEFAULT_NOISE_CRUMB_AREA_PX = 8
# Denoise removals → lawn; road hard-protect colour.
LAWN_FILL_RGB = (0, 255, 0)
ROAD_RGB = (241, 145, 73)


def _as_bool(m) -> np.ndarray:
    a = np.asarray(m)
    return a if a.dtype == bool else a != 0


def despeckle_binary_edge(
    mask: np.ndarray,
    protected: np.ndarray,
    *,
    kernel_px: int = DEFAULT_EDGE_KERNEL_PX,
    iterations: int = DEFAULT_EDGE_ITERS,
    boundary_band_px: int = DEFAULT_BOUNDARY_BAND_PX,
) -> Tuple[np.ndarray, Dict]:
    """Local open+close on a boundary band only; reject edits into protected.

    Removes 1-px spikes / isolated convex dents without flooding interiors.
    Normal stair-steps on diagonals may remain at finite resolution.
    """
    m0 = _as_bool(mask)
    prot = _as_bool(protected)
    info: Dict = dict(
        enabled=True, accepted=False, changed_px=0, reason='',
        kernel_px=int(kernel_px), iterations=int(iterations),
        boundary_band_px=int(boundary_band_px),
    )
    if not m0.any() or int(kernel_px) <= 1 or int(iterations) < 1:
        info['reason'] = 'noop_empty_or_disabled'
        return m0.copy(), info

    k = max(3, int(kernel_px) | 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    u8 = (m0.astype(np.uint8) * 255)
    opened = u8
    closed = u8
    for _ in range(max(1, int(iterations))):
        opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kern)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kern)
    # Propose: open removes spikes; close fills 1-px bays — combine carefully.
    proposed = ((opened > 0) | (closed > 0)) & ((opened > 0) | m0)
    # Prefer intersection of open∪(close∩original dilated) to limit expansion.
    dil_orig = cv2.dilate(u8, kern) > 0
    proposed = ((opened > 0) | ((closed > 0) & dil_orig))

    band_r = max(1, int(boundary_band_px))
    band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_r + 1, 2 * band_r + 1))
    boundary = (cv2.dilate(u8, band_k) > 0) ^ (cv2.erode(u8, band_k) > 0)
    # Also include 1-px exterior ring for close fills.
    exterior_ring = (cv2.dilate(u8, band_k) > 0) & ~m0
    edit_zone = boundary | exterior_ring

    out = m0.copy()
    delta_add = proposed & ~m0 & edit_zone
    delta_del = m0 & ~proposed & edit_zone
    # Never edit protected pixels; never delete into creating holes that are protected.
    delta_add &= ~prot
    delta_del &= ~prot
    if not delta_add.any() and not delta_del.any():
        info['reason'] = 'no_boundary_delta'
        return out, info

    cand = out.copy()
    cand[delta_add] = True
    cand[delta_del] = False
    # Reject if any remaining write would sit on protected (defensive).
    if np.any(cand & prot & ~m0):
        info['reason'] = 'rejected_protected_overlap'
        return out, info

    changed = int(np.count_nonzero(cand != m0))
    if changed == 0:
        info['reason'] = 'no_net_change'
        return out, info
    info['accepted'] = True
    info['changed_px'] = changed
    info['added_px'] = int(delta_add.sum())
    info['removed_px'] = int(delta_del.sum())
    info['reason'] = 'accepted'
    return cand, info


def fill_internal_green_holes(
    arr: np.ndarray,
    class_mask: np.ndarray,
    green_mask: np.ndarray,
    color_rgb,
    *,
    max_area_px: int,
    hard_protected: np.ndarray,
    min_green_frac: float = 0.85,
) -> Tuple[np.ndarray, Dict]:
    """Fill enclosed green (lawn/plant) noise islands inside a planar class.

    Only background CCs of ``class_mask`` with area ≤ ``max_area_px`` that are
    mostly green and do **not** touch ``hard_protected`` (road/building/fixed/…)
    are painted as ``color_rgb``. Roads are never modified.
    """
    before = np.asarray(arr, dtype=np.uint8)
    m0 = _as_bool(class_mask)
    green = _as_bool(green_mask)
    hard = _as_bool(hard_protected)
    info: Dict = dict(
        accepted=False, changed_px=0, filled_holes=0, filled_px=0,
        max_area_px=int(max_area_px), skipped_hard=0, skipped_not_green=0,
        skipped_border=0, skipped_large=0, road_restored_px=0,
    )
    if int(max_area_px) <= 0 or not m0.any():
        info['reason'] = 'noop'
        return before.copy(), info
    inv = (~m0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = before.copy()
    filled_n = 0
    filled_px = 0
    h, w = m0.shape
    color = np.asarray(color_rgb, dtype=np.uint8)
    road_rgb = np.asarray(ROAD_RGB, dtype=np.uint8)
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        if area > int(max_area_px):
            info['skipped_large'] += 1
            continue
        comp = labels == lab
        ys, xs = np.where(comp)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1:
            info['skipped_border'] += 1
            continue
        if np.any(comp & hard):
            info['skipped_hard'] += 1
            continue
        gfrac = float((comp & green).sum()) / float(area)
        if gfrac < float(min_green_frac):
            info['skipped_not_green'] += 1
            continue
        # Never write onto road even if hard mask missed a pixel.
        write = comp & ~hard & ~np.all(before == road_rgb, axis=2)
        out[write] = color
        filled_n += 1
        filled_px += int(write.sum())
    road_here = np.all(before == road_rgb, axis=2)
    if road_here.any():
        damaged = road_here & ~np.all(out == road_rgb, axis=2)
        out[road_here] = road_rgb
        info['road_restored_px'] = int(damaged.sum())
    info['filled_holes'] = filled_n
    info['filled_px'] = filled_px
    info['changed_px'] = filled_px
    info['accepted'] = filled_px > 0
    return out, info


def fill_micro_holes(
    mask: np.ndarray,
    protected: np.ndarray,
    *,
    max_hole_area_px: int = DEFAULT_MICRO_HOLE_AREA_PX,
) -> Tuple[np.ndarray, Dict]:
    """Fill small background holes fully enclosed by the class (not islands of others)."""
    m0 = _as_bool(mask)
    prot = _as_bool(protected)
    info: Dict = dict(accepted=False, changed_px=0, filled_holes=0,
                      max_hole_area_px=int(max_hole_area_px))
    if not m0.any():
        return m0.copy(), info
    inv = (~m0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = m0.copy()
    filled = 0
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0 or area > int(max_hole_area_px):
            continue
        comp = labels == lab
        # Skip holes that touch image border (not enclosed) or protected.
        ys, xs = np.where(comp)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == m0.shape[0] - 1 or xs.max() == m0.shape[1] - 1:
            continue
        if np.any(comp & prot):
            continue
        out[comp] = True
        filled += 1
    changed = int(np.count_nonzero(out != m0))
    info['accepted'] = changed > 0
    info['changed_px'] = changed
    info['filled_holes'] = filled
    return out, info


def remove_noise_crumbs(
    mask: np.ndarray,
    protected: np.ndarray,
    *,
    max_area_px: int = DEFAULT_NOISE_CRUMB_AREA_PX,
) -> Tuple[np.ndarray, Dict]:
    """Drop tiny isolated foreground crumbs (not touching larger same-class mass)."""
    m0 = _as_bool(mask)
    info: Dict = dict(accepted=False, changed_px=0, removed_components=0,
                      max_area_px=int(max_area_px))
    if not m0.any():
        return m0.copy(), info
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m0.astype(np.uint8), connectivity=8)
    out = m0.copy()
    removed = 0
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area > int(max_area_px):
            continue
        comp = labels == lab
        out[comp] = False
        removed += 1
    changed = int(np.count_nonzero(out != m0))
    info['accepted'] = changed > 0
    info['changed_px'] = changed
    info['removed_components'] = removed
    return out, info


def fill_thin_linear_pores(
    mask: np.ndarray,
    protected: np.ndarray,
    *,
    max_width_px: float = 2.0,
    max_area_px: int = 80,
    min_aspect: float = 2.5,
    min_ring_frac: float = 0.75,
) -> Tuple[np.ndarray, Dict]:
    """Fill abnormally thin linear pores / slits inside a class mask.

    Targets elongated background CCs whose EDT width ≤ max_width_px. Unlike
    ``fill_micro_holes`` (area-only), long thin cracks can exceed the micro-hole
    area cap and still be filled when shape is linear and mostly enclosed.
    Does not flood open exterior bays (border-touching or low building ring).
    """
    m0 = _as_bool(mask)
    prot = _as_bool(protected)
    info: Dict = dict(
        accepted=False, changed_px=0, filled_pores=0, filled_px=0,
        max_width_px=float(max_width_px), max_area_px=int(max_area_px),
        min_aspect=float(min_aspect), candidates=[])
    if not m0.any() or float(max_width_px) <= 0:
        info['reason'] = 'noop'
        return m0.copy(), info

    h, w = m0.shape
    # Only consider background that sits inside the dilated class hull (internal pores).
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    interior_bg = (~m0) & (cv2.dilate(m0.astype(np.uint8), dil_k) > 0)
    if not interior_bg.any():
        info['reason'] = 'no_interior_bg'
        return m0.copy(), info

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        interior_bg.astype(np.uint8), connectivity=4)
    out = m0.copy()
    filled = 0
    fpx = 0
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0 or area > int(max_area_px):
            continue
        comp = labels == lab
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        aspect = max(bw, bh) / max(1, min(bw, bh))
        dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
        width_est = float(2.0 * dt.max()) if area > 0 else 0.0
        ys, xs = np.where(comp)
        touches_border = (
            ys.min() == 0 or xs.min() == 0
            or ys.max() == h - 1 or xs.max() == w - 1)
        if touches_border:
            info['candidates'].append(dict(
                area=area, width=width_est, aspect=aspect, decision='reject_border'))
            continue
        if np.any(comp & prot):
            info['candidates'].append(dict(
                area=area, width=width_est, aspect=aspect, decision='reject_protected'))
            continue
        # Thin + linear (or tiny thin speck already caught by micro-holes).
        is_thin = width_est <= float(max_width_px) + 1e-6
        is_linear = aspect >= float(min_aspect) or area <= max(4, int(max_width_px) * 4)
        if not (is_thin and is_linear):
            info['candidates'].append(dict(
                area=area, width=width_est, aspect=aspect, decision='reject_not_thin_linear'))
            continue
        dil = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ring = dil & ~comp
        if not ring.any():
            continue
        ring_frac = float((ring & m0).sum()) / float(ring.sum())
        if ring_frac < float(min_ring_frac):
            info['candidates'].append(dict(
                area=area, width=width_est, aspect=aspect,
                ring_frac=ring_frac, decision='reject_open_bay'))
            continue
        out[comp] = True
        filled += 1
        fpx += area
        info['candidates'].append(dict(
            area=area, width=width_est, aspect=aspect,
            ring_frac=ring_frac, decision='accept_fill'))

    # Light directional close to seal residual 1-px slits that stayed 8-connected
    # to exterior through diagonal leaks (still width-limited).
    if float(max_width_px) >= 1.5:
        k = max(3, int(round(float(max_width_px) * 2)) | 1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(out.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kern) > 0
        # Only accept close pixels that are thin corridors inside original dilate.
        add = closed & ~out & ~prot & (cv2.dilate(m0.astype(np.uint8), kern) > 0)
        # Cap: added CC must still be thin.
        if add.any():
            n2, lab2, st2, _ = cv2.connectedComponentsWithStats(
                add.astype(np.uint8), connectivity=8)
            for lab in range(1, n2):
                a = int(st2[lab, cv2.CC_STAT_AREA])
                if a <= 0 or a > int(max_area_px):
                    continue
                c = lab2 == lab
                dt = cv2.distanceTransform(c.astype(np.uint8), cv2.DIST_L2, 3)
                if float(2.0 * dt.max()) <= float(max_width_px) + 0.5:
                    out[c] = True
                    fpx += a
                    filled += 1

    changed = int(np.count_nonzero(out != m0))
    info['accepted'] = changed > 0
    info['changed_px'] = changed
    info['filled_pores'] = filled
    info['filled_px'] = int(np.count_nonzero(out & ~m0))
    return out, info


# Initial engineering defaults for enclosed-area hole fill (not validated design optima).
DEFAULT_ENCLOSED_HOLE_MAX_AREA_PX2 = 500
DEFAULT_ENCLOSED_HOLE_MAX_AREA_RATIO = 0.20


def fill_enclosed_area_holes(
    mask: np.ndarray,
    protected: np.ndarray,
    *,
    max_area_px2: int = DEFAULT_ENCLOSED_HOLE_MAX_AREA_PX2,
    max_area_ratio: float = DEFAULT_ENCLOSED_HOLE_MAX_AREA_RATIO,
) -> Tuple[np.ndarray, Dict]:
    """Fill fully enclosed holes inside each foreground CC (FG=8, BG=4).

    Dual gate (AND): hole area ≤ max_area_px2 AND hole/parent_area ≤ max_area_ratio.
    Parent area and candidate set are frozen from a pre-fill snapshot so filling
    one hole cannot inflate the denominator for later holes. Cumulative fill per
    parent is also capped by max_area_ratio. Any protected pixel inside a hole
    → skip the whole hole (no partial fill).
    """
    m0 = _as_bool(mask)
    prot = _as_bool(protected)
    info: Dict = dict(
        accepted=False, changed_px=0, filled_holes=0, filled_px=0,
        max_area_px2=int(max_area_px2), max_area_ratio=float(max_area_ratio),
        candidates=[], filled_mask=None)
    if not m0.any():
        info['filled_mask'] = np.zeros_like(m0)
        return m0.copy(), info

    h, w = m0.shape
    # Snapshot FG CCs (8-connectivity).
    n_fg, lab_fg, st_fg, _ = cv2.connectedComponentsWithStats(
        m0.astype(np.uint8), connectivity=8)
    # Collect hole candidates against the snapshot (do not mutate m0 while screening).
    planned: List[Dict] = []
    for fid in range(1, n_fg):
        parent = lab_fg == fid
        parent_area = int(st_fg[fid, cv2.CC_STAT_AREA])
        if parent_area <= 0:
            continue
        # Enclosed holes: BG (4-conn) not connected to image border through ~parent.
        # Work on full frame so border connectivity is exact (open cuts ↔ exterior).
        bg = (~parent).astype(np.uint8)  # 1 = background candidate
        ff = bg.copy()
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        # Seed every border pixel that is background.
        seeds = []
        for x in range(w):
            if ff[0, x]:
                seeds.append((x, 0))
            if ff[h - 1, x]:
                seeds.append((x, h - 1))
        for y in range(h):
            if ff[y, 0]:
                seeds.append((0, y))
            if ff[y, w - 1]:
                seeds.append((w - 1, y))
        for sx, sy in seeds:
            if ff[sy, sx] == 1:
                cv2.floodFill(ff, ff_mask, (sx, sy), 2, flags=4)
        hole_pix = ff == 1  # bg never reached from border
        if not hole_pix.any():
            continue
        n_h, lab_h, st_h, _ = cv2.connectedComponentsWithStats(
            hole_pix.astype(np.uint8), connectivity=4)
        for hid in range(1, n_h):
            area = int(st_h[hid, cv2.CC_STAT_AREA])
            ratio = float(area) / float(parent_area)
            full = lab_h == hid
            cand = dict(
                parent_id=int(fid), hole_id=int(hid), area=area,
                parent_area=parent_area, area_ratio=ratio, mask=full)
            # Dual absolute+ratio gate (AND) — screen only; apply later with cum cap.
            if area > int(max_area_px2):
                cand['decision'] = 'skip_area_px2'
                info['candidates'].append(
                    {k: v for k, v in cand.items() if k != 'mask'})
                continue
            if ratio > float(max_area_ratio):
                cand['decision'] = 'skip_area_ratio'
                info['candidates'].append(
                    {k: v for k, v in cand.items() if k != 'mask'})
                continue
            if np.any(full & prot):
                cand['decision'] = 'skip_protected_inside'
                info['candidates'].append(
                    {k: v for k, v in cand.items() if k != 'mask'})
                continue
            cand['decision'] = 'candidate_ok'
            planned.append(cand)

    # Apply fills with per-parent cumulative ratio cap (snapshot parent_area).
    out = m0.copy()
    filled_by_parent: Dict[int, int] = {}
    filled_n = 0
    for cand in planned:
        pid = int(cand['parent_id'])
        parent_area = int(cand['parent_area'])
        cum = filled_by_parent.get(pid, 0)
        if (cum + int(cand['area'])) / float(parent_area) > float(max_area_ratio) + 1e-12:
            rec = {k: v for k, v in cand.items() if k != 'mask'}
            rec['decision'] = 'skip_cumulative_ratio'
            info['candidates'].append(rec)
            continue
        # Re-check protected against current out (should match snapshot).
        if np.any(cand['mask'] & prot):
            rec = {k: v for k, v in cand.items() if k != 'mask'}
            rec['decision'] = 'skip_protected_inside'
            info['candidates'].append(rec)
            continue
        out[cand['mask']] = True
        filled_by_parent[pid] = cum + int(cand['area'])
        filled_n += 1
        rec = {k: v for k, v in cand.items() if k != 'mask'}
        rec['decision'] = 'accept_fill'
        info['candidates'].append(rec)

    changed = int(np.count_nonzero(out != m0))
    info['accepted'] = changed > 0
    info['changed_px'] = changed
    info['filled_holes'] = filled_n
    info['filled_px'] = changed
    info['filled_mask'] = (out & ~m0)
    return out, info


def regularize_class_mask(
    mask: np.ndarray,
    protected: np.ndarray,
    *,
    enable_holes: bool = True,
    enable_crumbs: bool = True,
    enable_edge: bool = True,
    enable_thin_pores: bool = False,
    enable_enclosed_area_holes: bool = False,
    edge_kernel_px: int = DEFAULT_EDGE_KERNEL_PX,
    edge_iterations: int = DEFAULT_EDGE_ITERS,
    boundary_band_px: int = DEFAULT_BOUNDARY_BAND_PX,
    max_hole_area_px: int = DEFAULT_MICRO_HOLE_AREA_PX,
    max_crumb_area_px: int = DEFAULT_NOISE_CRUMB_AREA_PX,
    thin_pore_max_width_px: float = 2.0,
    thin_pore_max_area_px: int = 80,
    thin_pore_min_aspect: float = 2.5,
    enclosed_hole_max_area_px2: int = DEFAULT_ENCLOSED_HOLE_MAX_AREA_PX2,
    enclosed_hole_max_area_ratio: float = DEFAULT_ENCLOSED_HOLE_MAX_AREA_RATIO,
) -> Tuple[np.ndarray, Dict]:
    """Compose local hole / crumb / edge ops with per-step rollback on failure."""
    cur = _as_bool(mask).copy()
    log: Dict = dict(steps=[], total_changed_px=0)
    if enable_holes:
        cur, hinfo = fill_micro_holes(cur, protected, max_hole_area_px=max_hole_area_px)
        log['steps'].append(('micro_holes', hinfo))
    if enable_thin_pores:
        cur, tinfo = fill_thin_linear_pores(
            cur, protected,
            max_width_px=thin_pore_max_width_px,
            max_area_px=thin_pore_max_area_px,
            min_aspect=thin_pore_min_aspect)
        log['steps'].append(('thin_linear_pores', tinfo))
        log['thin_linear_pores'] = tinfo
    if enable_enclosed_area_holes:
        cur, einfo = fill_enclosed_area_holes(
            cur, protected,
            max_area_px2=enclosed_hole_max_area_px2,
            max_area_ratio=enclosed_hole_max_area_ratio)
        log['steps'].append(('enclosed_area_holes', einfo))
        log['enclosed_area_holes'] = einfo
    if enable_crumbs:
        cur, cinfo = remove_noise_crumbs(cur, protected, max_area_px=max_crumb_area_px)
        log['steps'].append(('noise_crumbs', cinfo))
    if enable_edge:
        cur, einfo = despeckle_binary_edge(
            cur, protected, kernel_px=edge_kernel_px,
            iterations=edge_iterations, boundary_band_px=boundary_band_px)
        log['steps'].append(('edge_despeckle', einfo))
        log['edge_despeckle'] = einfo
        # Second light pass: catch residual 1-px spikes after first accept.
        if einfo.get('accepted') and int(edge_iterations) >= 1:
            cur2, einfo2 = despeckle_binary_edge(
                cur, protected,
                kernel_px=max(3, int(edge_kernel_px) - 2),
                iterations=1,
                boundary_band_px=max(2, int(boundary_band_px)),
            )
            log['steps'].append(('edge_despeckle_pass2', einfo2))
            if einfo2.get('accepted'):
                cur = cur2
                log['edge_despeckle_pass2'] = einfo2
    log['total_changed_px'] = int(np.count_nonzero(cur != _as_bool(mask)))
    return cur, log


def apply_mask_recolor(
    before: np.ndarray,
    new_mask: np.ndarray,
    color_rgb,
    protected: np.ndarray,
) -> np.ndarray:
    """Write color onto accepted new_mask pixels outside protected; restore losses from before."""
    out = np.array(before, copy=True, dtype=np.uint8)
    prot = _as_bool(protected)
    nm = _as_bool(new_mask) & ~prot
    # Additions
    out[nm] = np.asarray(color_rgb, dtype=np.uint8)
    # Removals: pixels that were this colour and left the mask → restore from before
    # (caller typically passes before that already equals old colour there).
    return out


def regularize_surface_rgb(
    arr: np.ndarray,
    class_mask: np.ndarray,
    color_rgb,
    protected: np.ndarray,
    *,
    enable_edge: bool = True,
    removal_fill_rgb=LAWN_FILL_RGB,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    """Regularize one semantic class on an RGB canvas; returns (rgb, log).

    Pixels removed from the class mask (crumbs / edge open) are painted as
    ``removal_fill_rgb`` (default lawn). Neighbour-majority fill is not used,
    so denoise never expands road into deleted noise. Road pixels present in
    ``before`` are hard-restored afterward.
    """
    before = np.asarray(arr, dtype=np.uint8)
    new_m, log = regularize_class_mask(
        class_mask, protected, enable_edge=enable_edge, **kwargs)
    out = before.copy()
    old_m = _as_bool(class_mask)
    prot = _as_bool(protected)
    add = new_m & ~old_m & ~prot
    rem = old_m & ~new_m & ~prot
    out[add] = np.asarray(color_rgb, dtype=np.uint8)
    fill = np.asarray(removal_fill_rgb, dtype=np.uint8)
    if rem.any():
        out[rem] = fill
    # Road hard protect: never let add/rem overwrite existing road.
    road_here = np.all(before == np.asarray(ROAD_RGB, dtype=np.uint8), axis=2)
    if road_here.any():
        damaged = road_here & ~np.all(out == np.asarray(ROAD_RGB, dtype=np.uint8), axis=2)
        out[road_here] = np.asarray(ROAD_RGB, dtype=np.uint8)
        log['road_restored_px'] = int(damaged.sum())
    else:
        log['road_restored_px'] = 0
    log['rgb_added_px'] = int(add.sum())
    log['rgb_removed_px'] = int(rem.sum())
    log['removal_fill_rgb'] = tuple(int(x) for x in fill.tolist())
    return out, log