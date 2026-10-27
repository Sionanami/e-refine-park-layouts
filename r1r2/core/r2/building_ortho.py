# -*- coding: utf-8 -*-
"""Balanced ±10°~±15° structure-preserving building orthogonalization.

Branches per connected component:
  NEAR_RECT          — high rectangularity → MABR (mild; area-capped)
  ORTHO_PRESERVE     — line-dominant → rotate→axis-snap→unrotate (keeps L/step)
  CURVE_OR_MULTI_AXIS_EXEMPT — true arcs / conflicting axes → hybrid smooth
  FALLBACK           — area change / self-intersection → keep original contour

Uses OpenCV + NumPy only (no Shapely). Not the legacy AABB bbox path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

BRANCH_NEAR_RECT = 'NEAR_RECT'
BRANCH_ORTHO = 'ORTHO_PRESERVE'
BRANCH_EXEMPT = 'CURVE_OR_MULTI_AXIS_EXEMPT'
BRANCH_FALLBACK = 'FALLBACK'
BRANCH_SKIP_SMALL = 'SKIP_SMALL'

DEFAULT_ANGLE_TOL_DEG = 12.0
DEFAULT_NEAR_RECT_RATIO = 0.85
DEFAULT_AREA_CHANGE_TOL = 0.12
DEFAULT_MIN_BUILDING_AREA = 40
DEFAULT_MIN_EDGE_LEN = 6.0
DEFAULT_MAX_ASPECT_RATIO = 8.0
# Stronger DP so pixel stair-steps merge into macro edges before snap.
DEFAULT_DP_EPSILON_FRAC = 0.03
DEFAULT_DP_EPSILON_MIN_PX = 2.0
DEFAULT_MULTI_AXIS_SPREAD_DEG = 25.0
DEFAULT_CURVE_EDGE_FRAC = 0.35
DEFAULT_FALLBACK_MODE = 'original'  # FALLBACK never uses MABR (always original)
DEFAULT_NEAR_RECT_MAX_AREA_PX = 500
DEFAULT_CONCAVE_SOLIDITY_MIN = 0.92
DEFAULT_CONCAVE_DEFECT_DEPTH_FRAC = 0.08
# Compact shapes (solidity > this) must not be CURVE_EXEMPT — stair jaggies ≠ curves.
DEFAULT_FORCE_ORTHO_SOLIDITY = 0.85
# Stage-2 snap: short edges within this bias of 0/90 merge into ortho.
DEFAULT_STAGE2_ANGLE_DEG = 30.0
DEFAULT_STAGE2_MAX_EDGE_PX = 10.0
DEFAULT_ENABLE_ROTATE_ORTHO = True
DEFAULT_METRIC_MORPH_K = 3
# Contour complexity: isoperimetric quotient Q=4πA/P² (↓ = more twisted / fragmented).
# Complex buildings should skip aggressive ORTHO and use hybrid curve smooth.
DEFAULT_COMPLEX_ISOPERIMETRIC_Q = 0.35
DEFAULT_COMPLEX_SOLIDITY_MAX = 0.78
DEFAULT_COMPLEX_PRECLOSE_PX = 5


def contour_complexity_metrics(contour: np.ndarray) -> Dict[str, float]:
    """Measure outline complexity for adaptive building routing.

    - isoperimetric_q = 4πA/P² ∈ (0,1]: circle≈1, twisted/jagged ≪1
    - perimeter_over_sqrt_area = P/√A: higher → more complex boundary
    - solidity = A / convex_hull(A): lower → more concave / fragmented silhouette

    Note: 「周长/面积」对破碎轮廓通常偏高；识别复杂度请用 Q↓ 或 P/√A↑，
    不要用「周长/面积低」当复杂判据。
    """
    cnt = np.asarray(contour)
    area = float(abs(cv2.contourArea(cnt)))
    peri = float(cv2.arcLength(cnt, True))
    if area <= 1e-6 or peri <= 1e-6:
        return dict(
            area=area, perimeter=peri, isoperimetric_q=0.0,
            perimeter_over_sqrt_area=0.0, solidity=0.0, is_complex=1.0)
    q = float(4.0 * np.pi * area / (peri * peri))
    p_over = float(peri / np.sqrt(area))
    hull = cv2.convexHull(cnt)
    ha = float(abs(cv2.contourArea(hull)))
    solidity = float(area / ha) if ha > 1e-6 else 1.0
    return dict(
        area=area, perimeter=peri, isoperimetric_q=q,
        perimeter_over_sqrt_area=p_over, solidity=solidity, is_complex=0.0)


def is_complex_building_contour(
    contour: np.ndarray,
    *,
    isoperimetric_q_max: float = DEFAULT_COMPLEX_ISOPERIMETRIC_Q,
    solidity_max: float = DEFAULT_COMPLEX_SOLIDITY_MAX,
) -> Tuple[bool, Dict[str, float]]:
    """True when outline is too twisted/fragmented for ORTHO / NEAR_RECT."""
    m = contour_complexity_metrics(contour)
    complex_ = (
        m['isoperimetric_q'] < float(isoperimetric_q_max)
        or m['solidity'] < float(solidity_max)
    )
    m['is_complex'] = 1.0 if complex_ else 0.0
    m['isoperimetric_q_max'] = float(isoperimetric_q_max)
    m['solidity_max'] = float(solidity_max)
    return bool(complex_), m


def _as_bool(m) -> np.ndarray:
    a = np.asarray(m)
    return a if a.dtype == bool else a != 0


def _clamp_angle_tol(deg: float) -> float:
    return float(np.clip(float(deg), 10.0, 15.0))


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees (mod 180)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _edge_angle_deg(p0: np.ndarray, p1: np.ndarray) -> float:
    v = p1.astype(np.float64) - p0.astype(np.float64)
    return float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)


def _edge_length_stats(pts: np.ndarray) -> Dict[str, Any]:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 2:
        return dict(
            n_vertices=n, edge_lengths=[], edge_len_mean=0.0,
            edge_len_median=0.0, n_edges_lt6=0, n_edges_lt10=0)
    lens = [float(np.linalg.norm(pts[(i + 1) % n] - pts[i])) for i in range(n)]
    arr = np.asarray(lens, dtype=np.float64)
    return dict(
        n_vertices=int(n),
        edge_lengths=[round(x, 2) for x in lens],
        edge_len_mean=float(arr.mean()) if len(arr) else 0.0,
        edge_len_median=float(np.median(arr)) if len(arr) else 0.0,
        n_edges_lt6=int((arr < 6.0).sum()),
        n_edges_lt10=int((arr < 10.0).sum()),
    )


def estimate_mabr(contour: np.ndarray) -> Dict[str, Any]:
    """Minimum-area bounding rect from OpenCV; contour shape (N,1,2) or (N,2)."""
    cnt = np.asarray(contour, dtype=np.float32)
    if cnt.ndim == 2:
        cnt = cnt.reshape(-1, 1, 2)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle = angle + 90.0
    box = cv2.boxPoints(rect)
    area = float(max(w, 1e-6) * max(h, 1e-6))
    return dict(
        center=(float(cx), float(cy)),
        width=float(w),
        height=float(h),
        angle_deg=float(angle % 180.0),
        area=area,
        box=box.astype(np.float64),
        aspect=float(w / max(h, 1e-6)),
        rect=rect,
    )


def contour_area(contour: np.ndarray) -> float:
    cnt = np.asarray(contour, dtype=np.float32)
    if cnt.ndim == 2:
        cnt = cnt.reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(cnt)))


def rectangularity(contour: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    mabr = estimate_mabr(contour)
    ca = contour_area(contour)
    ratio = ca / mabr['area'] if mabr['area'] > 0 else 0.0
    return float(ratio), mabr


def douglas_peucker(
    contour: np.ndarray,
    epsilon_frac: float,
    *,
    epsilon_min_px: float = DEFAULT_DP_EPSILON_MIN_PX,
) -> np.ndarray:
    """DP simplify; eps = max(epsilon_min_px, epsilon_frac * arcLength)."""
    cnt = np.asarray(contour, dtype=np.float32)
    if cnt.ndim == 2:
        cnt = cnt.reshape(-1, 1, 2)
    peri = float(cv2.arcLength(cnt, True))
    eps = max(float(epsilon_min_px), float(epsilon_frac) * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) < 3:
        return cnt
    return approx


def _segments_from_poly(pts: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(pts)
    return [(pts[i], pts[(i + 1) % n]) for i in range(n)]


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c, eps: float = 1e-6) -> bool:
    return (min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps)


def _segments_intersect(p1, q1, p2, q2) -> bool:
    o1 = _orient(p1, q1, p2)
    o2 = _orient(p1, q1, q2)
    o3 = _orient(p2, q2, p1)
    o4 = _orient(p2, q2, q1)
    for a, b in ((p1, p2), (p1, q2), (q1, p2), (q1, q2)):
        if np.linalg.norm(a - b) < 1e-6:
            return False
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) < 1e-8 and _on_segment(p1, q1, p2):
        return True
    if abs(o2) < 1e-8 and _on_segment(p1, q1, q2):
        return True
    if abs(o3) < 1e-8 and _on_segment(p2, q2, p1):
        return True
    if abs(o4) < 1e-8 and _on_segment(p2, q2, q1):
        return True
    return False


def polygon_self_intersects(pts: np.ndarray) -> bool:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 4:
        return False
    segs = _segments_from_poly(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            if _segments_intersect(segs[i][0], segs[i][1], segs[j][0], segs[j][1]):
                return True
    return False


def _line_intersection(p, ang_deg, q, ang2_deg) -> Optional[np.ndarray]:
    a = np.radians(ang_deg)
    b = np.radians(ang2_deg)
    d1 = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
    d2 = np.array([np.cos(b), np.sin(b)], dtype=np.float64)
    A = np.column_stack([d1, -d2])
    if abs(np.linalg.det(A)) < 1e-8:
        return None
    t = np.linalg.solve(A, q.astype(np.float64) - p.astype(np.float64))
    return p.astype(np.float64) + t[0] * d1


def _snap_angle_to_ortho(edge_ang: float, main_ang: float, tol: float) -> Optional[float]:
    cands = [main_ang % 180.0, (main_ang + 90.0) % 180.0]
    best = None
    best_d = 1e9
    for c in cands:
        d = _angle_diff_deg(edge_ang, c)
        if d < best_d:
            best_d = d
            best = c
    if best_d <= tol:
        return best
    return None


def _nearest_ortho_angle(edge_ang: float, main_ang: float) -> Tuple[float, float]:
    cands = [main_ang % 180.0, (main_ang + 90.0) % 180.0]
    best = cands[0]
    best_d = 1e9
    for c in cands:
        d = _angle_diff_deg(edge_ang, c)
        if d < best_d:
            best_d = d
            best = c
    return best, best_d


def _merge_short_edges(pts: np.ndarray, min_len: float) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 4:
        return pts
    changed = True
    cur = pts.copy()
    guard = 0
    while changed and len(cur) >= 4 and guard < 256:
        guard += 1
        changed = False
        n = len(cur)
        lengths = [float(np.linalg.norm(cur[(i + 1) % n] - cur[i])) for i in range(n)]
        i_min = int(np.argmin(lengths))
        if lengths[i_min] >= float(min_len):
            break
        drop = (i_min + 1) % n
        cur = np.delete(cur, drop, axis=0)
        changed = True
    return cur


def _two_stage_snap_angles(
    pts: np.ndarray,
    main_ang: float,
    *,
    angle_tol_deg: float,
    stage2_angle_deg: float = DEFAULT_STAGE2_ANGLE_DEG,
    stage2_max_edge_px: float = DEFAULT_STAGE2_MAX_EDGE_PX,
) -> Tuple[List[float], Dict[str, int]]:
    """Stage1: snap within ANGLE_TOL; Stage2: short edges within 30° → ortho."""
    n = len(pts)
    tol = _clamp_angle_tol(angle_tol_deg)
    snapped: List[float] = []
    stats = dict(n_snapped_s1=0, n_snapped_s2=0, n_kept=0)
    for i in range(n):
        p0, p1 = pts[i], pts[(i + 1) % n]
        a = _edge_angle_deg(p0, p1)
        L = float(np.linalg.norm(p1 - p0))
        s1 = _snap_angle_to_ortho(a, main_ang, tol)
        if s1 is not None:
            snapped.append(s1)
            stats['n_snapped_s1'] += 1
            continue
        ortho, bias = _nearest_ortho_angle(a, main_ang)
        if L < float(stage2_max_edge_px) and bias <= float(stage2_angle_deg):
            snapped.append(ortho)
            stats['n_snapped_s2'] += 1
        else:
            snapped.append(a)
            stats['n_kept'] += 1
    return snapped, stats


def _rebuild_from_snapped(
    pts: np.ndarray,
    snapped: List[float],
    min_edge_len: float,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    info: Dict[str, Any] = dict(self_intersect=False)
    n = len(pts)
    new_pts = []
    for i in range(n):
        j = (i + 1) % n
        mid_i = 0.5 * (pts[i] + pts[j])
        k = (j + 1) % n
        mid_j = 0.5 * (pts[j] + pts[k])
        hit = _line_intersection(mid_i, snapped[i], mid_j, snapped[j])
        if hit is None:
            new_pts.append(pts[j].copy())
        else:
            new_pts.append(hit)
    out = np.asarray(new_pts, dtype=np.float64)
    out = _merge_short_edges(out, min_edge_len)
    if len(out) < 3:
        info['reason'] = 'collapsed'
        return None, info
    if polygon_self_intersects(out):
        info['self_intersect'] = True
        info['reason'] = 'self_intersect'
        return None, info
    return out, info


def _rotate_points(pts: np.ndarray, center: Tuple[float, float], ang_deg: float) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(center, dtype=np.float64)
    a = np.radians(ang_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float64)
    return (pts - c) @ R.T + c


def ortho_preserve_polygon(
    contour: np.ndarray,
    main_ang: float,
    *,
    angle_tol_deg: float,
    dp_epsilon_frac: float,
    min_edge_len: float,
    enable_rotate_ortho: bool = DEFAULT_ENABLE_ROTATE_ORTHO,
    stage2_angle_deg: float = DEFAULT_STAGE2_ANGLE_DEG,
    stage2_max_edge_px: float = DEFAULT_STAGE2_MAX_EDGE_PX,
    mabr: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """DP → (optional rotate) → two-stage ortho snap → rebuild → merge short edges."""
    info: Dict[str, Any] = dict(
        n_snapped=0, n_kept=0, self_intersect=False,
        rotate_ortho=bool(enable_rotate_ortho))
    approx = douglas_peucker(contour, dp_epsilon_frac)
    pts0 = approx.reshape(-1, 2).astype(np.float64)
    info['n_vertices_after_dp'] = int(len(pts0))
    info['edges_before'] = _edge_length_stats(pts0)
    if len(pts0) < 4:
        info['reason'] = 'too_few_vertices'
        return None, info

    if mabr is None:
        mabr = estimate_mabr(contour)
    center = mabr['center']
    theta = float(mabr['angle_deg'])

    if enable_rotate_ortho:
        # Align long axis to +X, snap to axis 0°/90°, then rotate back.
        pts = _rotate_points(pts0, center, -theta)
        snap_main = 0.0
        info['rotate_theta_deg'] = theta
    else:
        pts = pts0
        snap_main = float(main_ang)

    snapped, sstats = _two_stage_snap_angles(
        pts, snap_main,
        angle_tol_deg=angle_tol_deg,
        stage2_angle_deg=stage2_angle_deg,
        stage2_max_edge_px=stage2_max_edge_px)
    info.update(sstats)
    info['n_snapped'] = int(sstats['n_snapped_s1'] + sstats['n_snapped_s2'])
    info['n_kept'] = int(sstats['n_kept'])

    rebuilt, rinfo = _rebuild_from_snapped(pts, snapped, min_edge_len)
    info.update({k: v for k, v in rinfo.items() if k not in info or v})
    if rebuilt is None:
        return None, info

    if enable_rotate_ortho:
        rebuilt = _rotate_points(rebuilt, center, theta)

    info['edges_after'] = _edge_length_stats(rebuilt)
    info['n_vertices_after_ortho'] = int(len(rebuilt))
    return rebuilt.reshape(-1, 1, 2).astype(np.float32), info


def significant_concavity(
    contour: np.ndarray,
    *,
    solidity_min: float = DEFAULT_CONCAVE_SOLIDITY_MIN,
    defect_depth_frac: float = DEFAULT_CONCAVE_DEFECT_DEPTH_FRAC,
) -> Tuple[bool, Dict[str, Any]]:
    """Detect L / step / notch via solidity + max convexity-defect depth."""
    cnt = np.asarray(contour, dtype=np.float32)
    if cnt.ndim == 2:
        cnt = cnt.reshape(-1, 1, 2)
    ca = float(abs(cv2.contourArea(cnt)))
    hull = cv2.convexHull(cnt)
    ha = float(abs(cv2.contourArea(hull)))
    solidity = (ca / ha) if ha > 0 else 1.0
    info: Dict[str, Any] = dict(solidity=solidity, max_defect_depth=0.0)
    concave = solidity < float(solidity_min)
    try:
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is not None and len(cnt) >= 4 and len(hull_idx) >= 3:
            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is not None:
                depths = [float(defects[i, 0, 3]) / 256.0 for i in range(defects.shape[0])]
                max_d = max(depths) if depths else 0.0
                info['max_defect_depth'] = max_d
                thresh = float(defect_depth_frac) * max(1.0, float(np.sqrt(ca)))
                if max_d >= thresh:
                    concave = True
    except cv2.error:
        pass
    info['significant_concavity'] = bool(concave)
    return bool(concave), info


def _prep_contour_for_edge_metrics(
    contour: np.ndarray,
    *,
    dp_epsilon_frac: float,
    morph_k: int = DEFAULT_METRIC_MORPH_K,
) -> np.ndarray:
    """Open/close (k) then DP so stair jaggies are not scored as curve edges."""
    cnt = np.asarray(contour, dtype=np.float32)
    if cnt.ndim == 2:
        cnt = cnt.reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(cnt)
    pad = max(4, int(morph_k) + 2)
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    shifted = cnt.copy()
    shifted[:, 0, 0] = cnt[:, 0, 0] - x + pad
    shifted[:, 0, 1] = cnt[:, 0, 1] - y + pad
    cv2.fillPoly(canvas, [np.round(shifted).astype(np.int32)], 255)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(morph_k), int(morph_k)))
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, k)
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k)
    cs, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return douglas_peucker(cnt, dp_epsilon_frac)
    c2 = max(cs, key=cv2.contourArea).astype(np.float32)
    c2[:, 0, 0] = c2[:, 0, 0] + x - pad
    c2[:, 0, 1] = c2[:, 0, 1] + y - pad
    return douglas_peucker(c2, dp_epsilon_frac)


def classify_building_branch(
    contour: np.ndarray,
    *,
    near_rect_ratio: float,
    angle_tol_deg: float,
    max_aspect_ratio: float,
    multi_axis_spread_deg: float,
    curve_edge_frac: float,
    dp_epsilon_frac: float,
    original_area: Optional[float] = None,
    near_rect_max_area_px: float = DEFAULT_NEAR_RECT_MAX_AREA_PX,
    concave_solidity_min: float = DEFAULT_CONCAVE_SOLIDITY_MIN,
    concave_defect_depth_frac: float = DEFAULT_CONCAVE_DEFECT_DEPTH_FRAC,
    force_ortho_solidity: float = DEFAULT_FORCE_ORTHO_SOLIDITY,
    complex_isoperimetric_q_max: float = DEFAULT_COMPLEX_ISOPERIMETRIC_Q,
    complex_solidity_max: float = DEFAULT_COMPLEX_SOLIDITY_MAX,
) -> Tuple[str, Dict[str, Any]]:
    """Branch on **original** contour geometry (no morph-close before concavity)."""
    ratio, mabr = rectangularity(contour)
    area = float(original_area) if original_area is not None else contour_area(contour)
    concave, cinfo = significant_concavity(
        contour,
        solidity_min=concave_solidity_min,
        defect_depth_frac=concave_defect_depth_frac)
    meta: Dict[str, Any] = dict(
        rectangularity=ratio, mabr=mabr,
        main_orientation=mabr['angle_deg'], aspect=mabr['aspect'],
        original_area=area, **cinfo)
    meta['edges_raw'] = _edge_length_stats(
        np.asarray(contour, dtype=np.float64).reshape(-1, 2))

    # Twisted / fragmented silhouettes → hybrid curve path (skip ORTHO/NEAR_RECT).
    complex_, cpx = is_complex_building_contour(
        contour,
        isoperimetric_q_max=complex_isoperimetric_q_max,
        solidity_max=complex_solidity_max)
    meta['complexity'] = cpx
    if complex_:
        meta['reason'] = 'complex_contour_hybrid'
        return BRANCH_EXEMPT, meta

    if mabr['aspect'] > float(max_aspect_ratio):
        meta['reason'] = 'extreme_aspect'
        return BRANCH_EXEMPT, meta

    # Edge metrics on morph+DP prep (avoid stair-step → false curve).
    approx = _prep_contour_for_edge_metrics(contour, dp_epsilon_frac=dp_epsilon_frac)
    pts = approx.reshape(-1, 2).astype(np.float64)
    meta['edges_metric_prep'] = _edge_length_stats(pts)
    n = len(pts)
    if n < 4:
        meta['reason'] = 'too_few_vertices'
        return BRANCH_EXEMPT, meta

    tol = _clamp_angle_tol(angle_tol_deg)
    main = mabr['angle_deg']
    edge_lens = []
    near_ortho_len = 0.0
    far_angs = []
    for i in range(n):
        p0, p1 = pts[i], pts[(i + 1) % n]
        L = float(np.linalg.norm(p1 - p0))
        a = _edge_angle_deg(p0, p1)
        edge_lens.append(L)
        if _snap_angle_to_ortho(a, main, tol) is not None:
            near_ortho_len += L
        else:
            far_angs.append(a)
    total = sum(edge_lens) + 1e-9
    near_frac = near_ortho_len / total
    curve_frac = 1.0 - near_frac
    meta['near_ortho_edge_frac'] = near_frac
    meta['curve_edge_frac'] = curve_frac

    solidity = float(cinfo.get('solidity') or 0.0)
    force_ortho = solidity > float(force_ortho_solidity)
    meta['force_ortho_by_solidity'] = force_ortho

    # Curve / multi-axis exempt — blocked when solidity is high (compact footprint).
    if not force_ortho:
        if far_angs and curve_frac >= float(curve_edge_frac):
            spreads = [_angle_diff_deg(a, main) for a in far_angs]
            spreads2 = [_angle_diff_deg(a, main + 90.0) for a in far_angs]
            min_spread = [min(s, t) for s, t in zip(spreads, spreads2)]
            if float(np.median(min_spread)) >= float(multi_axis_spread_deg) * 0.5:
                meta['reason'] = 'curve_or_multi_axis'
                return BRANCH_EXEMPT, meta
        if near_frac < (1.0 - float(curve_edge_frac)):
            meta['reason'] = 'not_line_dominant'
            return BRANCH_EXEMPT, meta
    else:
        meta['exempt_blocked'] = 'solidity_gt_force_ortho'

    # Tight NEAR_RECT: high rectangularity AND small AND not concave.
    if ratio >= float(near_rect_ratio):
        if concave:
            meta['reason'] = 'concave_block_near_rect'
            return BRANCH_ORTHO, meta
        if area > float(near_rect_max_area_px):
            meta['reason'] = 'large_area_block_near_rect'
            return BRANCH_ORTHO, meta
        meta['reason'] = 'near_rect_ok'
        return BRANCH_NEAR_RECT, meta

    meta['reason'] = 'force_ortho_solidity' if force_ortho else 'ORTHO_PRESERVE'
    return BRANCH_ORTHO, meta


def _fill_poly_mask(shape: Tuple[int, int], pts: np.ndarray) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    p = np.round(np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)).astype(np.int32)
    if len(p) >= 3:
        cv2.fillPoly(canvas, [p], 255)
    return canvas > 0


def _mabr_mask(shape: Tuple[int, int], mabr: Dict[str, Any]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    box = np.round(mabr['box']).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [box], 255)
    return canvas > 0


def regularize_one_component(
    comp_mask: np.ndarray,
    *,
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
    complex_isoperimetric_q_max: float = DEFAULT_COMPLEX_ISOPERIMETRIC_Q,
    complex_solidity_max: float = DEFAULT_COMPLEX_SOLIDITY_MAX,
    forbid_other_buildings: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Regularize a single building CC; branch on original geometry (no close)."""
    m0 = _as_bool(comp_mask)
    area0 = int(m0.sum())
    attrs: Dict[str, Any] = dict(
        area_before=area0, area_after=area0, branch_used=BRANCH_SKIP_SMALL,
        fallback_used=False, main_orientation=None,
        angle_tol=_clamp_angle_tol(angle_tol_deg), rectangularity=None,
        select_reason='', fallback_reason='')
    if area0 < int(min_building_area):
        attrs['select_reason'] = 'skip_small'
        return m0.copy(), attrs

    contours, _ = cv2.findContours(
        m0.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        attrs['select_reason'] = 'no_contour'
        return m0.copy(), attrs
    contour = max(contours, key=cv2.contourArea)
    attrs['n_vertices_before'] = int(len(contour))
    attrs['edges_before'] = _edge_length_stats(
        np.asarray(contour, dtype=np.float64).reshape(-1, 2))

    branch, meta = classify_building_branch(
        contour,
        near_rect_ratio=near_rect_ratio,
        angle_tol_deg=angle_tol_deg,
        max_aspect_ratio=max_aspect_ratio,
        multi_axis_spread_deg=multi_axis_spread_deg,
        curve_edge_frac=curve_edge_frac,
        dp_epsilon_frac=dp_epsilon_frac,
        original_area=float(area0),
        near_rect_max_area_px=near_rect_max_area_px,
        concave_solidity_min=concave_solidity_min,
        concave_defect_depth_frac=concave_defect_depth_frac,
        force_ortho_solidity=force_ortho_solidity,
        complex_isoperimetric_q_max=complex_isoperimetric_q_max,
        complex_solidity_max=complex_solidity_max,
    )
    attrs['rectangularity'] = meta.get('rectangularity')
    attrs['main_orientation'] = meta.get('main_orientation')
    attrs['branch_used'] = branch
    attrs['select_reason'] = str(meta.get('reason') or branch)
    attrs['solidity'] = meta.get('solidity')
    attrs['significant_concavity'] = meta.get('significant_concavity')
    attrs['force_ortho_by_solidity'] = meta.get('force_ortho_by_solidity')
    attrs['complexity'] = meta.get('complexity')
    attrs.update({k: meta[k] for k in (
        'near_ortho_edge_frac', 'curve_edge_frac', 'aspect',
        'max_defect_depth', 'edges_metric_prep') if k in meta})

    shape = m0.shape
    mabr = meta['mabr']
    other = _as_bool(forbid_other_buildings) if forbid_other_buildings is not None else None

    def _clip_write(mask: np.ndarray) -> np.ndarray:
        out = _as_bool(mask).copy()
        if other is not None:
            out &= ~other
        return out

    if branch == BRANCH_EXEMPT:
        attrs['needs_hybrid'] = True
        attrs['n_vertices_after'] = attrs['n_vertices_before']
        attrs['edges_after'] = attrs['edges_before']
        return m0.copy(), attrs

    if branch == BRANCH_NEAR_RECT:
        out = _clip_write(_mabr_mask(shape, mabr))
        # Mild expand only: keep original core, allow MABR fill of micro notches.
        out |= m0
        area1 = int(out.sum())
        attrs['area_after'] = area1
        attrs['n_vertices_after'] = 4
        attrs['edges_after'] = _edge_length_stats(mabr['box'])
        if area0 > 0 and abs(area1 - area0) / float(area0) > float(area_change_tol):
            attrs['fallback_used'] = True
            attrs['branch_used'] = BRANCH_FALLBACK
            attrs['fallback_from'] = BRANCH_NEAR_RECT
            attrs['fallback_reason'] = 'area_change'
            attrs['area_after'] = area0
            attrs['n_vertices_after'] = attrs['n_vertices_before']
            attrs['edges_after'] = attrs['edges_before']
            return m0.copy(), attrs
        return out, attrs

    # ORTHO_PRESERVE: replace footprint with snapped polygon (do NOT |= m0,
    # else stair-step pixels remain and XOR stays tiny).
    poly, oinfo = ortho_preserve_polygon(
        contour, mabr['angle_deg'],
        angle_tol_deg=angle_tol_deg,
        dp_epsilon_frac=dp_epsilon_frac,
        min_edge_len=min_edge_len,
        enable_rotate_ortho=enable_rotate_ortho,
        stage2_angle_deg=stage2_angle_deg,
        stage2_max_edge_px=stage2_max_edge_px,
        mabr=mabr)
    attrs['ortho_info'] = oinfo
    if poly is None:
        attrs['fallback_used'] = True
        attrs['branch_used'] = BRANCH_FALLBACK
        attrs['fallback_from'] = BRANCH_ORTHO
        attrs['fallback_reason'] = str(oinfo.get('reason') or 'ortho_failed')
        attrs['n_vertices_after'] = attrs['n_vertices_before']
        attrs['edges_after'] = attrs['edges_before']
        return m0.copy(), attrs

    # Replace footprint with snapped polygon (do NOT |= m0 — that kept stair jags).
    out = _clip_write(_fill_poly_mask(shape, poly))
    area1 = int(out.sum())
    attrs['area_after'] = area1
    attrs['n_vertices_after'] = int(oinfo.get('n_vertices_after_ortho') or len(poly))
    attrs['edges_after'] = oinfo.get('edges_after') or _edge_length_stats(poly)
    if area0 > 0 and abs(area1 - area0) / float(area0) > float(area_change_tol):
        attrs['fallback_used'] = True
        attrs['branch_used'] = BRANCH_FALLBACK
        attrs['fallback_from'] = BRANCH_ORTHO
        attrs['fallback_reason'] = 'area_change'
        attrs['area_after'] = area0
        attrs['n_vertices_after'] = attrs['n_vertices_before']
        attrs['edges_after'] = attrs['edges_before']
        return m0.copy(), attrs
    if not out.any():
        attrs['fallback_used'] = True
        attrs['branch_used'] = BRANCH_FALLBACK
        attrs['fallback_from'] = BRANCH_ORTHO
        attrs['fallback_reason'] = 'empty_mask'
        attrs['area_after'] = area0
        return m0.copy(), attrs
    return out, attrs


def regularize_building_components(
    building: np.ndarray,
    fixed: np.ndarray,
    *,
    enable_ortho_debug_viz: bool = False,
    debug_dir: Optional[str] = None,
    debug_tag: str = 'building_ortho',
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Per-CC conditional ortho; fixed pixels preserved from input."""
    b0 = _as_bool(building)
    fix = _as_bool(fixed)
    log: Dict[str, Any] = dict(
        accepted=False, n_components=0, components=[],
        branch_counts={}, needs_hybrid_mask=np.zeros_like(b0),
        changed_px=0)
    if not b0.any():
        return b0.copy(), log

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        b0.astype(np.uint8), connectivity=8)
    out = np.zeros_like(b0)
    hybrid_need = np.zeros_like(b0)
    counts: Dict[str, int] = {}
    viz = None
    if enable_ortho_debug_viz:
        viz = np.zeros((b0.shape[0], b0.shape[1], 3), dtype=np.uint8)
        viz[b0] = (80, 80, 80)

    for lab in range(1, num):
        comp = labels == lab
        other = b0 & ~comp
        reg, attrs = regularize_one_component(
            comp, forbid_other_buildings=other, **kwargs)
        attrs['building_id'] = int(lab)
        log['components'].append(attrs)
        br = attrs.get('branch_used', '?')
        counts[br] = counts.get(br, 0) + 1
        out |= (reg & ~fix)
        if attrs.get('needs_hybrid'):
            hybrid_need |= comp
        if viz is not None:
            color = {
                BRANCH_NEAR_RECT: (0, 255, 0),
                BRANCH_ORTHO: (0, 200, 255),
                BRANCH_EXEMPT: (255, 128, 0),
                BRANCH_FALLBACK: (0, 0, 255),
                BRANCH_SKIP_SMALL: (128, 128, 128),
            }.get(br, (255, 255, 255))
            viz[reg] = color
            ys, xs = np.where(reg)
            if len(ys):
                cy, cx = int(ys.mean()), int(xs.mean())
                cv2.putText(
                    viz, f'{lab}:{br[:6]}', (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    out |= b0 & fix
    log['n_components'] = int(num - 1)
    log['branch_counts'] = counts
    log['needs_hybrid_mask'] = hybrid_need
    log['changed_px'] = int(np.count_nonzero(out != b0))
    log['accepted'] = log['changed_px'] > 0 or bool(counts)

    if viz is not None and debug_dir:
        from pathlib import Path
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f'{debug_tag}_branches.png'), viz)
        log['debug_viz'] = str(d / f'{debug_tag}_branches.png')
        lines = [
            'building_id\tarea_before\tarea_after\trectangularity\tbranch\t'
            'select_reason\tfallback\tfallback_reason\tsolidity\tconcavity\t'
            'n_vert_before\tn_vert_after\tedge_mean_before\tedge_mean_after\t'
            'n_edges_lt6_before\tn_edges_lt6_after'
        ]
        for c in log['components']:
            eb = c.get('edges_before') or {}
            ea = c.get('edges_after') or {}
            lines.append(
                f"{c.get('building_id')}\t{c.get('area_before')}\t{c.get('area_after')}\t"
                f"{c.get('rectangularity')}\t{c.get('branch_used')}\t"
                f"{c.get('select_reason')}\t{c.get('fallback_used')}\t"
                f"{c.get('fallback_reason')}\t{c.get('solidity')}\t"
                f"{c.get('significant_concavity')}\t"
                f"{c.get('n_vertices_before')}\t{c.get('n_vertices_after')}\t"
                f"{eb.get('edge_len_mean')}\t{ea.get('edge_len_mean')}\t"
                f"{eb.get('n_edges_lt6')}\t{ea.get('n_edges_lt6')}"
            )
            # Console-style summary line for diagnostics consumers.
            print(
                f"[ortho] id={c.get('building_id')} area {c.get('area_before')}→"
                f"{c.get('area_after')} branch={c.get('branch_used')} "
                f"reason={c.get('select_reason')} "
                f"verts {c.get('n_vertices_before')}→{c.get('n_vertices_after')} "
                f"edge_mean {eb.get('edge_len_mean')}→{ea.get('edge_len_mean')} "
                f"lt6 {eb.get('n_edges_lt6')}→{ea.get('n_edges_lt6')}",
                flush=True)
        (d / f'{debug_tag}_branches.tsv').write_text(
            '\n'.join(lines) + '\n', encoding='utf-8')
    return out, log
