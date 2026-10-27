# -*- coding: utf-8 -*-
"""Local building repair: enclosed micro-holes, edge despeckle, abnormal external crumbs.

Not MBB/convex hull. Courtyards that open to non-building or exceed max hole area
are left. Returns approved building mask for downstream protection.

Optional hybrid contour pass: keep near-straight edges as polylines; smooth only
curved spans (no global morph that rounds every corner).
Units: px / px² at working resolution (engineering defaults).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .surface import despeckle_binary_edge
from .building_ortho import (
    DEFAULT_ANGLE_TOL_DEG,
    DEFAULT_AREA_CHANGE_TOL,
    DEFAULT_COMPLEX_ISOPERIMETRIC_Q,
    DEFAULT_COMPLEX_PRECLOSE_PX,
    DEFAULT_COMPLEX_SOLIDITY_MAX,
    DEFAULT_CURVE_EDGE_FRAC,
    DEFAULT_DP_EPSILON_FRAC,
    DEFAULT_ENABLE_ROTATE_ORTHO,
    DEFAULT_FALLBACK_MODE,
    DEFAULT_FORCE_ORTHO_SOLIDITY,
    DEFAULT_MAX_ASPECT_RATIO,
    DEFAULT_MIN_BUILDING_AREA,
    DEFAULT_MIN_EDGE_LEN,
    DEFAULT_MULTI_AXIS_SPREAD_DEG,
    DEFAULT_NEAR_RECT_RATIO,
    DEFAULT_NEAR_RECT_MAX_AREA_PX,
    DEFAULT_CONCAVE_SOLIDITY_MIN,
    DEFAULT_CONCAVE_DEFECT_DEPTH_FRAC,
    DEFAULT_STAGE2_ANGLE_DEG,
    DEFAULT_STAGE2_MAX_EDGE_PX,
    regularize_building_components,
)

# Engineering default at 512 working res; ring_frac gate avoids open courtyards.
DEFAULT_MAX_HOLE_AREA_PX = 400
DEFAULT_CRUMB_AREA_PX = 10
DEFAULT_EDGE_KERNEL_PX = 5
DEFAULT_BOUNDARY_BAND_PX = 3


def _as_bool(m) -> np.ndarray:
    a = np.asarray(m)
    return a if a.dtype == bool else a != 0


def _poly_line_max_dev(pts: np.ndarray) -> float:
    """Max orthogonal deviation of intermediate points from chord endpoints."""
    if len(pts) < 3:
        return 0.0
    p0 = pts[0].astype(np.float64)
    p1 = pts[-1].astype(np.float64)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return float(np.max(np.linalg.norm(pts.astype(np.float64) - p0, axis=1)))
    vn = v / L
    rel = pts[1:-1].astype(np.float64) - p0
    # 2D cross magnitude = abs(relx*vny - rely*vnx)
    cross = np.abs(rel[:, 0] * vn[1] - rel[:, 1] * vn[0])
    return float(cross.max()) if len(cross) else 0.0


def _moving_average_open(pts: np.ndarray, win: int = 5) -> np.ndarray:
    """Open-chain moving-average; endpoints preserved."""
    n = len(pts)
    if n < 3 or win < 3:
        return pts.copy()
    w = min(win | 1, n if n % 2 == 1 else n - 1)
    if w < 3:
        return pts.copy()
    r = w // 2
    out = pts.astype(np.float64).copy()
    src = pts.astype(np.float64)
    for i in range(1, n - 1):
        lo = max(0, i - r)
        hi = min(n, i + r + 1)
        out[i] = src[lo:hi].mean(axis=0)
    out[0] = src[0]
    out[-1] = src[-1]
    return out


def _chaikin_open(pts: np.ndarray, iters: int = 1) -> np.ndarray:
    """Chaikin corner-cutting on an open polyline (keeps endpoints)."""
    cur = pts.astype(np.float64)
    for _ in range(max(0, int(iters))):
        if len(cur) < 2:
            break
        nxt = [cur[0]]
        for i in range(len(cur) - 1):
            p, q = cur[i], cur[i + 1]
            nxt.append(0.75 * p + 0.25 * q)
            nxt.append(0.25 * p + 0.75 * q)
        nxt.append(cur[-1])
        cur = np.asarray(nxt, dtype=np.float64)
    return cur


def _resample_polyline(pts: np.ndarray, step: float = 1.0) -> np.ndarray:
    if len(pts) < 2:
        return pts.copy()
    d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total < step:
        return pts.copy()
    samples = np.arange(0.0, total, step)
    if samples[-1] < total:
        samples = np.concatenate([samples, [total]])
    x = np.interp(samples, s, pts[:, 0])
    y = np.interp(samples, s, pts[:, 1])
    return np.stack([x, y], axis=1)


def _dp_open_chain(chain: np.ndarray, eps_px: float) -> np.ndarray:
    """Douglas-Peucker on an open polyline (keep endpoints)."""
    pts = np.asarray(chain, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 3 or float(eps_px) <= 0:
        return np.asarray(chain, dtype=np.float64).reshape(-1, 2)
    approx = cv2.approxPolyDP(pts, float(eps_px), False)
    out = approx.reshape(-1, 2).astype(np.float64)
    if len(out) < 2:
        return np.asarray(chain, dtype=np.float64).reshape(-1, 2)
    out[0] = chain[0]
    out[-1] = chain[-1]
    return out


def _hybrid_reshape_contour(
    contour: np.ndarray,
    *,
    straight_dev_px: float = 1.5,
    corner_angle_deg: float = 35.0,
    curve_smooth_iters: int = 2,
    min_straight_len: float = 6.0,
    pre_dp_eps_px: float = 1.5,
    curve_dp_eps_px: float = 1.5,
    curve_ma_win: int = 7,
) -> np.ndarray:
    """Rewrite one OpenCV contour: straighten linear spans, smooth curved spans.

    Contour shape (N,1,2) int → float points → int for fillPoly.
    Light DP first merges pixel stair-steps before corner split / Chaikin.
    """
    cnt0 = np.asarray(contour, dtype=np.float32)
    if cnt0.ndim == 2:
        cnt0 = cnt0.reshape(-1, 1, 2)
    if len(cnt0) < 8:
        return contour

    # Mild closed DP: kill 1px stairs while keeping macro silhouette.
    peri = float(cv2.arcLength(cnt0, True))
    pre_eps = max(float(pre_dp_eps_px), 0.004 * peri)
    approx = cv2.approxPolyDP(cnt0, pre_eps, True)
    if len(approx) >= 8:
        pts = approx.reshape(-1, 2).astype(np.float64)
    else:
        pts = cnt0.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 8:
        return contour

    # Local turning angle at each vertex (degrees).
    angles = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            angles[i] = 0.0
            continue
        cos_t = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        turn = abs(180.0 - float(np.degrees(np.arccos(cos_t))))
        angles[i] = turn

    is_corner = angles >= float(corner_angle_deg)
    corner_idx = np.where(is_corner)[0].tolist()
    if len(corner_idx) < 2:
        corner_idx = [0, n // 4, n // 2, (3 * n) // 4]

    corner_idx = sorted(set(int(i) % n for i in corner_idx))
    segs: List[np.ndarray] = []
    for i, c0 in enumerate(corner_idx):
        c1 = corner_idx[(i + 1) % len(corner_idx)]
        if c1 > c0:
            chain = pts[c0:c1 + 1]
        else:
            chain = np.concatenate([pts[c0:], pts[:c1 + 1]], axis=0)
        if len(chain) < 2:
            continue
        length = float(np.sqrt(((chain[1:] - chain[:-1]) ** 2).sum(axis=1)).sum())
        dev = _poly_line_max_dev(chain)
        if length >= float(min_straight_len) and dev <= float(straight_dev_px):
            segs.append(np.vstack([chain[0], chain[-1]]))
        else:
            # Curved: DP denoise → Chaikin → moving average; keep corner ends.
            sm = _dp_open_chain(chain, curve_dp_eps_px)
            sm = _chaikin_open(sm, iters=max(1, int(curve_smooth_iters)))
            win = max(3, int(curve_ma_win) | 1)  # odd window
            if len(sm) >= win:
                sm = _moving_average_open(sm, win=win)
            sm[0] = chain[0]
            sm[-1] = chain[-1]
            sm = _resample_polyline(sm, step=1.0)
            sm[0] = chain[0]
            sm[-1] = chain[-1]
            segs.append(sm)

    if not segs:
        return contour
    out_pts = [segs[0][0]]
    for seg in segs:
        for p in seg[1:]:
            if np.linalg.norm(p - out_pts[-1]) > 0.25:
                out_pts.append(p)
    arr = np.asarray(out_pts, dtype=np.float64)
    if np.linalg.norm(arr[0] - arr[-1]) > 0.5:
        arr = np.vstack([arr, arr[0]])
    return np.round(arr).astype(np.int32).reshape(-1, 1, 2)


def hybrid_smooth_building_edges(
    building: np.ndarray,
    fixed: np.ndarray,
    *,
    straight_dev_px: float = 1.5,
    corner_angle_deg: float = 35.0,
    curve_smooth_iters: int = 2,
    boundary_band_px: int = 4,
    min_contour_area: int = 40,
    pre_dp_eps_px: float = 1.5,
    curve_dp_eps_px: float = 1.5,
    curve_ma_win: int = 7,
) -> Tuple[np.ndarray, Dict]:
    """Per-component contour hybrid smooth; edits restricted to a boundary band."""
    b0 = _as_bool(building)
    fix = _as_bool(fixed)
    info: Dict = dict(
        accepted=False, changed_px=0, n_components=0, n_contours=0,
        straight_dev_px=float(straight_dev_px),
        corner_angle_deg=float(corner_angle_deg),
        curve_smooth_iters=int(curve_smooth_iters),
        boundary_band_px=int(boundary_band_px),
        pre_dp_eps_px=float(pre_dp_eps_px))
    if not b0.any():
        return b0.copy(), info

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        b0.astype(np.uint8), connectivity=8)
    rebuilt = np.zeros_like(b0)
    n_cont = 0
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < int(min_contour_area):
            rebuilt |= labels == lab
            continue
        comp = (labels == lab).astype(np.uint8) * 255
        contours, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not contours:
            rebuilt |= labels == lab
            continue
        canvas = np.zeros_like(comp)
        order = list(range(len(contours)))
        if hier is not None:
            order.sort(key=lambda i: 1 if int(hier[0, i, 3]) >= 0 else 0)
        for ci in order:
            cnt = contours[ci]
            is_hole = (
                hier is not None and hier.shape[1] > ci and int(hier[0, ci, 3]) >= 0)
            if len(cnt) < 8:
                cv2.drawContours(
                    canvas, [cnt], -1, 0 if is_hole else 255, thickness=-1)
                continue
            new_cnt = _hybrid_reshape_contour(
                cnt,
                straight_dev_px=straight_dev_px,
                corner_angle_deg=corner_angle_deg,
                curve_smooth_iters=curve_smooth_iters,
                pre_dp_eps_px=pre_dp_eps_px,
                curve_dp_eps_px=curve_dp_eps_px,
                curve_ma_win=curve_ma_win)
            cv2.drawContours(
                canvas, [new_cnt], -1, 0 if is_hole else 255, thickness=-1)
            n_cont += 1
        rebuilt |= canvas > 0

    # Wider editable band so Chaikin displacement is not clipped away.
    band_r = max(1, int(boundary_band_px))
    band_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * band_r + 1, 2 * band_r + 1))
    u8 = b0.astype(np.uint8) * 255
    dil = cv2.dilate(u8, band_k) > 0
    ero = cv2.erode(u8, band_k) > 0
    edit_zone = dil ^ ero  # full annulus (interior + exterior ring)

    out = b0.copy()
    add = rebuilt & ~b0 & edit_zone & ~fix
    rem = b0 & ~rebuilt & edit_zone & ~fix
    out[add] = True
    out[rem] = False
    out |= b0 & fix

    changed = int(np.count_nonzero(out != b0))
    info['accepted'] = changed > 0
    info['changed_px'] = changed
    info['added_px'] = int(add.sum())
    info['removed_px'] = int(rem.sum())
    info['n_components'] = int(num - 1)
    info['n_contours'] = n_cont
    return out, info


def fill_enclosed_building_holes(
    building: np.ndarray,
    fixed: np.ndarray,
    *,
    max_hole_area_px: int = DEFAULT_MAX_HOLE_AREA_PX,
) -> Tuple[np.ndarray, Dict]:
    """Fill background CCs fully enclosed by building and not touching fixed/border."""
    b0 = _as_bool(building)
    fix = _as_bool(fixed)
    info: Dict = dict(accepted=False, filled_holes=0, filled_px=0,
                      max_hole_area_px=int(max_hole_area_px), candidates=[])
    if not b0.any():
        return b0.copy(), info
    h, w = b0.shape
    inv = (~b0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = b0.copy()
    filled = 0
    px = 0
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0 or area > int(max_hole_area_px):
            continue
        comp = labels == lab
        ys, xs = np.where(comp)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1:
            info['candidates'].append(dict(area=area, decision='reject_border'))
            continue
        if np.any(comp & fix):
            info['candidates'].append(dict(area=area, decision='reject_fixed'))
            continue
        # Must be surrounded mostly by building (enclosed hole, not open court finger).
        dil = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ring = dil & ~comp
        if not ring.any():
            continue
        b_frac = float((ring & b0).sum()) / float(ring.sum())
        if b_frac < 0.85:
            info['candidates'].append(dict(
                area=area, decision='reject_open_court', building_ring_frac=b_frac))
            continue
        out[comp] = True
        filled += 1
        px += area
        info['candidates'].append(dict(
            area=area, decision='accept_fill', building_ring_frac=b_frac))
    info['accepted'] = filled > 0
    info['filled_holes'] = filled
    info['filled_px'] = px
    return out, info


def remove_building_crumbs(
    building: np.ndarray,
    fixed: np.ndarray,
    *,
    max_area_px: int = DEFAULT_CRUMB_AREA_PX,
) -> Tuple[np.ndarray, Dict]:
    """Drop tiny isolated building CCs not adjacent to a larger building mass."""
    b0 = _as_bool(building)
    info: Dict = dict(accepted=False, removed_components=0, removed_px=0,
                      max_area_px=int(max_area_px))
    if not b0.any():
        return b0.copy(), info
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        b0.astype(np.uint8), connectivity=8)
    out = b0.copy()
    removed = 0
    rpx = 0
    large = np.zeros_like(b0)
    for lab in range(1, num):
        if int(stats[lab, cv2.CC_STAT_AREA]) > int(max_area_px):
            large |= labels == lab
    dil_large = cv2.dilate(large.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area > int(max_area_px):
            continue
        comp = labels == lab
        if np.any(comp & dil_large):
            continue  # attached / near main mass — keep as possible annex
        if np.any(comp & _as_bool(fixed)):
            continue
        out[comp] = False
        removed += 1
        rpx += area
    info['accepted'] = removed > 0
    info['removed_components'] = removed
    info['removed_px'] = rpx
    return out, info


def repair_building_mask(
    building: np.ndarray,
    fixed: np.ndarray,
    *,
    enable_holes: bool = True,
    enable_crumbs: bool = True,
    enable_edge: bool = True,
    enable_hybrid_curve_smooth: bool = False,
    enable_conditional_ortho: bool = False,
    max_hole_area_px: int = DEFAULT_MAX_HOLE_AREA_PX,
    max_crumb_area_px: int = DEFAULT_CRUMB_AREA_PX,
    edge_kernel_px: int = DEFAULT_EDGE_KERNEL_PX,
    boundary_band_px: int = DEFAULT_BOUNDARY_BAND_PX,
    straight_dev_px: float = 1.5,
    corner_angle_deg: float = 35.0,
    curve_smooth_iters: int = 2,
    hybrid_band_px: Optional[int] = None,
    hybrid_pre_dp_eps_px: float = 1.5,
    hybrid_curve_dp_eps_px: float = 1.5,
    hybrid_curve_ma_win: int = 7,
    angle_tol_deg: float = DEFAULT_ANGLE_TOL_DEG,
    near_rect_ratio: float = DEFAULT_NEAR_RECT_RATIO,
    area_change_tol: float = DEFAULT_AREA_CHANGE_TOL,
    min_building_area: int = DEFAULT_MIN_BUILDING_AREA,
    min_edge_len: float = DEFAULT_MIN_EDGE_LEN,
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO,
    dp_epsilon_frac: float = DEFAULT_DP_EPSILON_FRAC,
    multi_axis_spread_deg: float = DEFAULT_MULTI_AXIS_SPREAD_DEG,
    curve_edge_frac: float = DEFAULT_CURVE_EDGE_FRAC,
    fallback_mode: str = DEFAULT_FALLBACK_MODE,
    near_rect_max_area_px: float = DEFAULT_NEAR_RECT_MAX_AREA_PX,
    concave_solidity_min: float = DEFAULT_CONCAVE_SOLIDITY_MIN,
    concave_defect_depth_frac: float = DEFAULT_CONCAVE_DEFECT_DEPTH_FRAC,
    force_ortho_solidity: float = DEFAULT_FORCE_ORTHO_SOLIDITY,
    enable_rotate_ortho: bool = DEFAULT_ENABLE_ROTATE_ORTHO,
    stage2_angle_deg: float = DEFAULT_STAGE2_ANGLE_DEG,
    stage2_max_edge_px: float = DEFAULT_STAGE2_MAX_EDGE_PX,
    enable_ortho_debug_viz: bool = False,
    ortho_debug_dir: Optional[str] = None,
    ortho_debug_tag: str = 'building_ortho',
    complex_isoperimetric_q_max: float = DEFAULT_COMPLEX_ISOPERIMETRIC_Q,
    complex_solidity_max: float = DEFAULT_COMPLEX_SOLIDITY_MAX,
    complex_preclose_px: int = DEFAULT_COMPLEX_PRECLOSE_PX,
) -> Tuple[np.ndarray, Dict]:
    """Local repair (+ optional balanced conditional orthogonalization).

    When ``enable_conditional_ortho`` is True:
      holes → crumbs → light edge despeckle → optional mild close (fragment mend)
      → per-CC ortho branches (complex contours → CURVE_EXEMPT/hybrid) →
      hybrid_smooth **only** on CURVE_OR_MULTI_AXIS_EXEMPT components.
    FALLBACK never MABR.
    """
    cur = _as_bool(building).copy()
    log: Dict = dict(steps=[], total_changed_px=0)
    if enable_holes:
        cur, hinfo = fill_enclosed_building_holes(
            cur, fixed, max_hole_area_px=max_hole_area_px)
        log['steps'].append(('holes', hinfo))
    if enable_crumbs:
        cur, cinfo = remove_building_crumbs(
            cur, fixed, max_area_px=max_crumb_area_px)
        log['steps'].append(('crumbs', cinfo))
    if enable_edge:
        # Soften despeckle when ortho is on: avoid bridging steps into rectangles.
        if enable_conditional_ortho:
            k_edge = min(int(edge_kernel_px), 3)
            iters = 1
            cur, einfo = despeckle_binary_edge(
                cur, fixed, kernel_px=k_edge, iterations=iters,
                boundary_band_px=min(int(boundary_band_px), 2))
            log['steps'].append(('edge_despeckle_light', einfo))
            log['edge_despeckle'] = einfo
        else:
            cur, einfo = despeckle_binary_edge(
                cur, fixed, kernel_px=edge_kernel_px, iterations=2,
                boundary_band_px=boundary_band_px)
            log['steps'].append(('edge_despeckle', einfo))
            log['edge_despeckle'] = einfo
            cur2, einfo2 = despeckle_binary_edge(
                cur, fixed, kernel_px=max(3, int(edge_kernel_px) - 2), iterations=1,
                boundary_band_px=boundary_band_px)
            log['steps'].append(('edge_despeckle_pass2', einfo2))
            if einfo2.get('accepted'):
                cur = cur2

    # Mild close: reconnect fragmented building silhouettes before branching.
    if enable_conditional_ortho and int(complex_preclose_px) >= 3:
        k = int(complex_preclose_px) | 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(
            cur.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kern) > 0
        closed &= ~_as_bool(fixed)
        changed = int(np.count_nonzero(closed != cur))
        cur = closed
        log['steps'].append(('complex_preclose', dict(
            accepted=changed > 0, changed_px=changed, kernel_px=k)))

    if enable_conditional_ortho:
        cur, olog = regularize_building_components(
            cur, fixed,
            enable_ortho_debug_viz=bool(enable_ortho_debug_viz),
            debug_dir=ortho_debug_dir,
            debug_tag=ortho_debug_tag,
            angle_tol_deg=float(angle_tol_deg),
            near_rect_ratio=float(near_rect_ratio),
            area_change_tol=float(area_change_tol),
            min_building_area=int(min_building_area),
            min_edge_len=float(min_edge_len),
            max_aspect_ratio=float(max_aspect_ratio),
            dp_epsilon_frac=float(dp_epsilon_frac),
            multi_axis_spread_deg=float(multi_axis_spread_deg),
            curve_edge_frac=float(curve_edge_frac),
            fallback_mode='original',  # harden: never MABR on FALLBACK
            near_rect_max_area_px=float(near_rect_max_area_px),
            concave_solidity_min=float(concave_solidity_min),
            concave_defect_depth_frac=float(concave_defect_depth_frac),
            force_ortho_solidity=float(force_ortho_solidity),
            enable_rotate_ortho=bool(enable_rotate_ortho),
            stage2_angle_deg=float(stage2_angle_deg),
            stage2_max_edge_px=float(stage2_max_edge_px),
            complex_isoperimetric_q_max=float(complex_isoperimetric_q_max),
            complex_solidity_max=float(complex_solidity_max),
        )
        log['steps'].append(('conditional_ortho', {
            k: v for k, v in olog.items() if k != 'needs_hybrid_mask'}))
        log['conditional_ortho'] = {
            k: v for k, v in olog.items() if k != 'needs_hybrid_mask'}
        log['branch_counts'] = olog.get('branch_counts', {})
        # Hybrid only on curve/multi-axis / complex-contour exempt CCs.
        if enable_hybrid_curve_smooth:
            need = _as_bool(olog.get('needs_hybrid_mask'))
            if need.any():
                before_h = cur.copy()
                band_r = max(1, int(
                    hybrid_band_px if hybrid_band_px is not None
                    else max(boundary_band_px, 4)))
                smooth_in = before_h & need
                other = before_h & ~need
                smoothed_ex, cinfo = hybrid_smooth_building_edges(
                    smooth_in, fixed,
                    straight_dev_px=float(straight_dev_px),
                    corner_angle_deg=float(corner_angle_deg),
                    curve_smooth_iters=int(curve_smooth_iters),
                    boundary_band_px=band_r,
                    pre_dp_eps_px=float(hybrid_pre_dp_eps_px),
                    curve_dp_eps_px=float(hybrid_curve_dp_eps_px),
                    curve_ma_win=int(hybrid_curve_ma_win),
                )
                band_k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * band_r + 1, 2 * band_r + 1))
                allow = cv2.dilate(need.astype(np.uint8), band_k) > 0
                delta_add = smoothed_ex & ~before_h & allow & ~_as_bool(fixed)
                delta_rem = before_h & ~smoothed_ex & ~other & allow & ~_as_bool(fixed)
                cur = before_h.copy()
                cur[delta_add] = True
                cur[delta_rem] = False
                cinfo = dict(cinfo)
                cinfo['restricted_to_exempt'] = True
                cinfo['exempt_px'] = int(need.sum())
                log['steps'].append(('hybrid_curve_smooth_exempt_only', cinfo))
                log['hybrid_curve_smooth'] = cinfo
    elif enable_hybrid_curve_smooth:
        # Legacy: global hybrid when ortho is off.
        cur, cinfo = hybrid_smooth_building_edges(
            cur, fixed,
            straight_dev_px=float(straight_dev_px),
            corner_angle_deg=float(corner_angle_deg),
            curve_smooth_iters=int(curve_smooth_iters),
            boundary_band_px=int(
                hybrid_band_px if hybrid_band_px is not None
                else max(boundary_band_px, 4)),
            pre_dp_eps_px=float(hybrid_pre_dp_eps_px),
            curve_dp_eps_px=float(hybrid_curve_dp_eps_px),
            curve_ma_win=int(hybrid_curve_ma_win),
        )
        log['steps'].append(('hybrid_curve_smooth', cinfo))
        log['hybrid_curve_smooth'] = cinfo

    log['total_changed_px'] = int(np.count_nonzero(cur != _as_bool(building)))
    log['approved'] = True
    return cur, log
