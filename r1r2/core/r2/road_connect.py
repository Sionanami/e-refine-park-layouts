# -*- coding: utf-8 -*-
"""R2 road repair: local gap closing + endpoint links; vegetation optional soft region.

Preserves existing R1 road width. New links only are locally thickened.
Hard barriers are whatever the caller passes in blocking_masks (refine: fixed +
building only; water is coverable and must NOT be injected as hard by default).
Vegetation is soft when allow_cross_vegetation=True and hard when False.

Endpoint access status (local to the endpoint, not whole CC):
  CONNECTED_PAVEMENT | CONNECTED_ROAD | PENDING | UNKNOWN
Building-proxy (EB) candidates are skipped when status is CONNECTED_PAVEMENT.

Engineering defaults (not claimed as validation-set optima):
  direction_tolerance_deg=45, branch_trace_px=8, raw_support_max_l2=80.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

ORANGE_DEFAULT = (241, 145, 73)

# Engineering defaults — units: px or degrees as named. Not tuned on eval sets.
_DEFAULT_DIRECTION_TOL_DEG = 45.0
_DEFAULT_BRANCH_TRACE_PX = 8
_DEFAULT_RAW_SUPPORT_MAX_L2 = 80.0
_DEFAULT_MIN_HALFWIDTH = 1.0
_DEFAULT_MAX_HALFWIDTH = 6.0
_DEFAULT_PAVEMENT_CONTACT_RADIUS_PX = 2

# Endpoint access states (shared by EE/ET/EB within one repair call).
STATUS_CONNECTED_PAVEMENT = 'CONNECTED_PAVEMENT'
STATUS_CONNECTED_ROAD = 'CONNECTED_ROAD'
STATUS_PENDING = 'PENDING'
STATUS_UNKNOWN = 'UNKNOWN'
STATUS_ACCESS_STUB = 'ACCESS_STUB'  # small patch already on pav/bld/water — no road–road link

# Pinch closure vs long false mesh (px @ working res).
_PINCH_GAP_PX = 10.0          # near gaps always eligible (incl. large↔large); v8b freeze
_MEDIUM_GAP_PX = 16.0         # stub↔main only
_LARGE_CC_EE_CAP_PX = 12.0    # two large CCs: do not EE beyond pinch+ε
_ACCESS_STUB_AREA_PX = 80     # small isolated access patch


def _combine_blocks(shape, blocking_masks: Optional[Sequence]) -> np.ndarray:
    block = np.zeros(shape, dtype=bool)
    if not blocking_masks:
        return block
    for m in blocking_masks:
        if m is None:
            continue
        arr = np.asarray(m)
        block |= (arr.astype(bool) if arr.dtype == bool else arr != 0)
    return block


def _endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Degree-1 skeleton pixels; out-of-image treated as zero (no reflect pad)."""
    sk = (skeleton > 0).astype(np.uint8)
    if not sk.any():
        return np.zeros((0, 2), dtype=int)
    # BORDER_CONSTANT=0 so edge pixels are not falsely given reflected neighbours.
    padded = cv2.copyMakeBorder(sk, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    deg = np.zeros_like(sk, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            deg += padded[1 + dy:1 + dy + sk.shape[0], 1 + dx:1 + dx + sk.shape[1]]
    ys, xs = np.where((sk > 0) & (deg == 1))
    return np.stack([ys, xs], axis=1) if len(ys) else np.zeros((0, 2), dtype=int)


def _pinch_face_endpoints(
    skeleton: np.ndarray,
    labels: np.ndarray,
    *,
    max_pinch_px: float = _PINCH_GAP_PX,
) -> np.ndarray:
    """Extra endpoints at skeleton faces where two CCs approach within max_pinch.

    Degree-1 tips often sit far from a mid-segment pinch; without these faces,
    near gaps across plant/lawn never enter the candidate list.
    """
    sk = skeleton > 0
    if not sk.any() or max_pinch_px <= 0:
        return np.zeros((0, 2), dtype=int)
    n = int(labels.max()) if labels is not None else 0
    if n < 2:
        return np.zeros((0, 2), dtype=int)
    pts_by: Dict[int, np.ndarray] = {}
    for lab in range(1, n + 1):
        comp = (labels == lab) & sk
        if not comp.any():
            continue
        er = cv2.erode(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        border = comp & ~er
        if not border.any():
            border = comp
        ys, xs = np.where(border)
        if len(ys) > 60:
            idx = np.linspace(0, len(ys) - 1, 60).astype(int)
            ys, xs = ys[idx], xs[idx]
        pts_by[lab] = np.column_stack([ys, xs])
    faces = []
    labs = sorted(pts_by.keys())
    for ai, a in enumerate(labs):
        ta = cKDTree(pts_by[a].astype(np.float64))
        for b in labs[ai + 1:]:
            d, ix = ta.query(pts_by[b].astype(np.float64), k=1)
            d = np.asarray(d, dtype=float)
            if d.size == 0:
                continue
            kmin = int(np.argmin(d))
            if float(d[kmin]) < 1.0 or float(d[kmin]) > float(max_pinch_px):
                continue
            yb, xb = int(pts_by[b][kmin, 0]), int(pts_by[b][kmin, 1])
            ya, xa = int(pts_by[a][int(ix[kmin]), 0]), int(pts_by[a][int(ix[kmin]), 1])
            link = np.array([yb - ya, xb - xa], dtype=np.float64)
            ln = float(np.linalg.norm(link))
            if ln < 1e-6:
                continue
            link_u = link / ln
            # Skip lateral bridges between parallel elongated roads.
            ax_a = _principal_axis_yx((labels == a) & sk)
            ax_b = _principal_axis_yx((labels == b) & sk)
            if _is_lateral_parallel_bridge(ax_a, ax_b, link_u):
                continue
            faces.append((ya, xa))
            faces.append((yb, xb))
    if not faces:
        return np.zeros((0, 2), dtype=int)
    return np.unique(np.asarray(faces, dtype=int), axis=0)


def _principal_axis_yx(mask: np.ndarray) -> Optional[np.ndarray]:
    """Unit principal axis (dy, dx) of a binary blob; None if too small."""
    ys, xs = np.where(mask)
    if len(ys) < 8:
        return None
    pts = np.column_stack([ys.astype(np.float64), xs.astype(np.float64)])
    pts -= pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    axis = vt[0]
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return None
    return axis / n


def _link_aligns_axis(axis: Optional[np.ndarray], link_u: np.ndarray, *, tol_deg: float = 40.0) -> bool:
    """True if link is roughly along the blob axis (not a lateral bridge)."""
    if axis is None:
        return True
    ang = _angle_deg(axis, link_u)
    # axis is undirected
    ang = min(ang, 180.0 - ang)
    return ang <= float(tol_deg)


def _is_lateral_parallel_bridge(
    ax_a: Optional[np.ndarray],
    ax_b: Optional[np.ndarray],
    link_u: np.ndarray,
    *,
    perp_deg: float = 50.0,
) -> bool:
    """True only when link is nearly perpendicular to *both* elongated axes."""
    def _perp(axis):
        if axis is None:
            return False
        ang = _angle_deg(axis, link_u)
        ang = min(ang, 180.0 - ang)
        return ang >= float(perp_deg)

    return _perp(ax_a) and _perp(ax_b)


def _cc_access_stub_labs(
    skeleton_labels: np.ndarray,
    seed_road: np.ndarray,
    *,
    pavement: Optional[np.ndarray],
    building: Optional[np.ndarray],
    water: Optional[np.ndarray],
    max_area_px: int = _ACCESS_STUB_AREA_PX,
) -> Dict[int, str]:
    """Small thick-road CCs that already touch pav/building/water → skip road–road.

    Area/contact use ``seed_road`` (full width). Skeleton labels are mapped by
    overlap so thin skeletons still inherit access-stub status.
    """
    skip: Dict[int, str] = {}
    seed = np.asarray(seed_road, dtype=bool)
    if not seed.any():
        return skip
    pav = np.asarray(pavement, dtype=bool) if pavement is not None else None
    bld = np.asarray(building, dtype=bool) if building is not None else None
    wat = np.asarray(water, dtype=bool) if water is not None else None
    n_seed, seed_lab = cv2.connectedComponents(seed.astype(np.uint8), connectivity=8)
    kern = np.ones((3, 3), np.uint8)
    seed_skip_comps = set()
    for lab in range(1, n_seed):
        comp = seed_lab == lab
        area = int(comp.sum())
        if area <= 0 or area > int(max_area_px):
            continue
        dil = cv2.dilate(comp.astype(np.uint8), kern) > 0
        ring = dil & ~comp
        reasons = []
        if pav is not None and np.any(ring & pav):
            reasons.append('pavement')
        if bld is not None and np.any(ring & bld):
            reasons.append('building')
        if wat is not None and np.any(ring & wat):
            reasons.append('water')
        if reasons:
            seed_skip_comps.add(lab)
            # Map seed CC → overlapping skeleton label(s)
            sk_labs = skeleton_labels[comp]
            for sl in np.unique(sk_labs):
                sl = int(sl)
                if sl > 0:
                    skip[sl] = '+'.join(reasons)
    return skip


def _local_pavement_contact(
    y: int,
    x: int,
    pavement: Optional[np.ndarray],
    *,
    radius: int = _DEFAULT_PAVEMENT_CONTACT_RADIUS_PX,
) -> Tuple[bool, str]:
    """True if this endpoint has stable local pavement contact.

    Whole-component distant pavement does NOT count. A single diagonal-only
    touch is treated as unstable (not connected).
    """
    if pavement is None:
        return False, 'no_pavement_mask'
    pav = np.asarray(pavement, dtype=bool)
    if pav.shape != (pav.shape[0], pav.shape[1]) or y < 0 or x < 0:
        return False, 'bad_pavement_mask'
    h, w = pav.shape
    if y >= h or x >= w:
        return False, 'oob'
    r = max(0, int(radius))
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    local = pav[y0:y1, x0:x1]
    if not local.any():
        return False, 'none_in_radius'
    # Prefer orthogonal (4-neighbour) contact at the endpoint itself.
    ortho = 0
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy, xx = y + dy, x + dx
        if 0 <= yy < h and 0 <= xx < w and pav[yy, xx]:
            ortho += 1
    if ortho >= 1:
        return True, f'ortho_contact_{ortho}'
    # Diagonals alone: require ≥2 distinct diagonal pavement neighbours.
    diag = 0
    for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        yy, xx = y + dy, x + dx
        if 0 <= yy < h and 0 <= xx < w and pav[yy, xx]:
            diag += 1
    if diag >= 2:
        return True, f'multi_diag_contact_{diag}'
    if diag == 1:
        return False, 'single_diag_only'
    # Pavement in radius but not adjacent — not a stable access at this endpoint.
    return False, 'pavement_nearby_not_adjacent'


def classify_endpoint_status(
    y: int,
    x: int,
    *,
    pavement: Optional[np.ndarray] = None,
    forced: Optional[str] = None,
    pavement_radius: int = _DEFAULT_PAVEMENT_CONTACT_RADIUS_PX,
) -> Tuple[str, str]:
    """Return (status, reason). Local pavement wins over PENDING."""
    if forced in (
        STATUS_CONNECTED_PAVEMENT, STATUS_CONNECTED_ROAD,
        STATUS_PENDING, STATUS_UNKNOWN,
    ):
        return forced, 'forced'
    ok, note = _local_pavement_contact(y, x, pavement, radius=pavement_radius)
    if ok:
        return STATUS_CONNECTED_PAVEMENT, note
    if pavement is None:
        return STATUS_UNKNOWN, note
    return STATUS_PENDING, note


def _building_exterior_targets(building: np.ndarray, hard: np.ndarray) -> np.ndarray:
    """Pixels adjacent (4-neigh) to building that are not hard/building (proxy door)."""
    b = np.asarray(building, dtype=bool)
    if not b.any():
        return np.zeros((0, 2), dtype=int)
    h, w = b.shape
    kern = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    dil = cv2.dilate(b.astype(np.uint8), kern) > 0
    exterior = dil & ~b & ~np.asarray(hard, dtype=bool)
    ys, xs = np.where(exterior)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=int)
    return np.stack([ys, xs], axis=1)


def _neighbors8(sk: np.ndarray, y: int, x: int) -> List[Tuple[int, int]]:
    h, w = sk.shape
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and sk[yy, xx]:
                out.append((yy, xx))
    return out


def _branch_outward_direction(
    skeleton: np.ndarray,
    y: int,
    x: int,
    *,
    trace_px: int = _DEFAULT_BRANCH_TRACE_PX,
) -> Tuple[Optional[np.ndarray], str]:
    """Trace inward along the unique branch; return outward unit vector (row, col).

    Reliability: 'ok' | 'uncertain_short' | 'uncertain_junction' | 'none'
    """
    sk = skeleton > 0
    if not sk[y, x]:
        return None, 'none'
    nbrs = _neighbors8(sk, y, x)
    if len(nbrs) != 1:
        return None, 'uncertain_junction' if len(nbrs) > 1 else 'none'
    path = [(y, x)]
    prev = (y, x)
    cur = nbrs[0]
    for _ in range(max(1, int(trace_px))):
        path.append(cur)
        nbs = [p for p in _neighbors8(sk, cur[0], cur[1]) if p != prev]
        if len(nbs) != 1:
            break
        prev, cur = cur, nbs[0]
    if len(path) < 3:
        return None, 'uncertain_short'
    # Inward vector: endpoint → interior; outward is the reverse.
    ey, ex = path[0]
    iy, ix = path[-1]
    vec = np.array([ey - iy, ex - ix], dtype=np.float64)
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return None, 'uncertain_short'
    return vec / n, 'ok'


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = float(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1, 1))
    return float(np.degrees(np.arccos(c)))


def _draw_line_mask(shape: Tuple[int, int], y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 1, 1, lineType=cv2.LINE_8)
    return m > 0


def regularize_road_width(
    road: np.ndarray,
    skeleton: np.ndarray,
    hard: np.ndarray,
    *,
    half_min_px: float = 1.0,
    half_max_px: float = 4.0,
    width_mode: str = 'median',
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Replace each road component with one half-width.

    median: h = clip(median of positive distance-to-complement on that
    component's skeleton, [half_min_px, half_max_px]). The painted band is
    distance-to-skeleton <= h+0.35, then a 3x3 closing, minus hard blockers.
    Incoming width is not kept. local_edt is not the reported setting.
    """
    base = np.asarray(road, dtype=bool)
    sk_all = np.asarray(skeleton, dtype=bool) & base
    blocked = np.asarray(hard, dtype=bool)
    mode = str(width_mode or 'median').lower()
    info: Dict[str, Any] = dict(components=0, median_halfwidth_px=[], width_mode=mode)
    if not base.any():
        return base.copy(), info
    if not sk_all.any():
        sk_all = skeletonize(base).astype(bool)
    out = np.zeros_like(base)
    dt = cv2.distanceTransform(base.astype(np.uint8), cv2.DIST_L2, 5)
    num, labels = cv2.connectedComponents(base.astype(np.uint8), connectivity=8)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for lab in range(1, num):
        comp_sk = sk_all & (labels == lab)
        if not comp_sk.any():
            out |= (labels == lab) & ~blocked
            continue
        vals = dt[comp_sk]
        vals = vals[vals > 0]
        hw_med = float(np.median(vals)) if vals.size else float(half_min_px)
        hw_med = float(np.clip(hw_med, float(half_min_px), float(half_max_px)))
        info['median_halfwidth_px'].append(round(hw_med, 3))
        if mode == 'local_edt':
            band = np.zeros(comp_sk.shape, dtype=np.uint8)
            ys, xs = np.where(comp_sk)
            # Subsample dense skeletons for speed while keeping thickness.
            step = 1 if len(ys) < 2500 else 2
            for y, x in zip(ys[::step], xs[::step]):
                r = float(np.clip(float(dt[y, x]), float(half_min_px), float(half_max_px)))
                cv2.circle(band, (int(x), int(y)), max(1, int(round(r))), 1, -1)
            band = band.astype(bool) | comp_sk
        else:
            inv = np.where(comp_sk, np.uint8(0), np.uint8(255))
            dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
            band = (dist <= hw_med + 0.35) | comp_sk
        band = cv2.morphologyEx(band.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kern) > 0
        band &= ~blocked
        band |= comp_sk & ~blocked
        out |= band
        info['components'] += 1
    return out, info


def _skeleton_degree(skeleton: np.ndarray) -> np.ndarray:
    sk = (skeleton > 0).astype(np.uint8)
    if not sk.any():
        return np.zeros(sk.shape, dtype=np.uint8)
    padded = cv2.copyMakeBorder(sk, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    deg = np.zeros_like(sk, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            deg += padded[1 + dy:1 + dy + sk.shape[0], 1 + dx:1 + dx + sk.shape[1]]
    return deg


def prune_attached_spurs(
    road: np.ndarray,
    *,
    max_len_px: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """Trim short claws still attached to the main skeleton (opt-in).

    From each degree-1 tip, walk inward until a junction (degree ≥ 3) or the
    walk exceeds ``max_len_px``. If a junction is reached within that budget,
    the tip path is deleted. ``max_len_px<=0`` disables.
    """
    m0 = np.asarray(road, dtype=bool)
    info: Dict[str, Any] = dict(
        accepted=False, removed_spurs=0, removed_px=0, max_len_px=int(max_len_px))
    limit = int(max_len_px)
    if limit <= 0 or not m0.any():
        return m0.copy(), info
    sk = skeletonize(m0).astype(bool)
    if not sk.any():
        return m0.copy(), info
    deg = _skeleton_degree(sk)
    tips = np.argwhere((sk > 0) & (deg == 1))
    if len(tips) == 0:
        return m0.copy(), info
    h, w = sk.shape
    drop = np.zeros_like(sk)
    removed = 0
    for y0, x0 in tips:
        y, x = int(y0), int(x0)
        path = [(y, x)]
        prev = None
        hit_junction = False
        for _ in range(limit):
            nbrs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if not sk[ny, nx]:
                        continue
                    if prev is not None and (ny, nx) == prev:
                        continue
                    nbrs.append((ny, nx))
            if not nbrs:
                break
            if len(nbrs) > 1 and int(deg[y, x]) >= 3:
                hit_junction = True
                path = path[:-1]
                break
            nxt = nbrs[0] if len(nbrs) == 1 else min(
                nbrs, key=lambda p: (p[0] - y) ** 2 + (p[1] - x) ** 2)
            prev = (y, x)
            y, x = nxt
            path.append((y, x))
            if int(deg[y, x]) >= 3:
                hit_junction = True
                path = path[:-1]
                break
        if not hit_junction or not path:
            continue
        for py, px in path:
            drop[py, px] = True
        removed += 1
    if not drop.any():
        return m0.copy(), info
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    claw = cv2.dilate(drop.astype(np.uint8), kern) > 0
    claw &= ~((deg >= 3) & sk)
    out = m0 & ~claw
    out |= sk & ~drop
    info['accepted'] = True
    info['removed_spurs'] = removed
    info['removed_px'] = int((m0 & ~out).sum())
    return out, info


def _thicken_mask(center: np.ndarray, half_width: float) -> np.ndarray:
    if not center.any() or half_width <= 0.5:
        return center.copy()
    r = max(1, int(round(half_width)))
    k = 2 * r + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(center.astype(np.uint8), kern) > 0


def _band_hits_hard(center: np.ndarray, half_width: float, hard: np.ndarray) -> bool:
    """True if centerline or restored-width band intersects hard barriers."""
    if not center.any():
        return False
    if np.any(center & hard):
        return True
    if half_width > 0.5:
        band = _thicken_mask(center, half_width)
        if np.any(band & hard):
            return True
    return False


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[rb] = ra
        return True

    def same(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


def _estimate_link_halfwidth(
    seed_road: np.ndarray,
    skeleton: np.ndarray,
    y0: int, x0: int, y1: int, x1: int,
    *,
    fallback: float,
    min_hw: float = _DEFAULT_MIN_HALFWIDTH,
    max_hw: float = _DEFAULT_MAX_HALFWIDTH,
) -> float:
    """Estimate half-width from EDT of seed road near endpoints, away from junctions."""
    if not seed_road.any():
        return float(np.clip(fallback, min_hw, max_hw))
    # Distance to background inside seed road = local radius proxy.
    # EDT of ~seed gives distance to nearest non-road; on road pixels this is inscribed radius.
    radius = distance_transform_edt(seed_road)
    sk = skeleton > 0
    samples = []
    for (ey, ex) in ((y0, x0), (y1, x1)):
        # Step a few pixels inward along skeleton from endpoint.
        nbrs = _neighbors8(sk, ey, ex)
        if len(nbrs) != 1:
            if seed_road[ey, ex]:
                samples.append(float(radius[ey, ex]))
            continue
        prev = (ey, ex)
        cur = nbrs[0]
        for step in range(6):
            deg = len(_neighbors8(sk, cur[0], cur[1]))
            if deg != 2:
                break  # stop before junction
            if seed_road[cur]:
                samples.append(float(radius[cur]))
            nbs = [p for p in _neighbors8(sk, cur[0], cur[1]) if p != prev]
            if len(nbs) != 1:
                break
            prev, cur = cur, nbs[0]
            if step >= 2 and samples:
                break
    if not samples:
        return float(np.clip(fallback, min_hw, max_hw))
    hw = float(np.median(samples))
    return float(np.clip(hw, min_hw, max_hw))


def _raw_support_score(
    guide_rgb: Optional[np.ndarray],
    center: np.ndarray,
    orange_rgb: Sequence[int],
    max_l2: float,
) -> Tuple[Optional[float], str]:
    """Mean L2 distance of guide pixels on centerline to orange; lower = stronger support.

    This is a colour support score, NOT a calibrated road probability.
    """
    if guide_rgb is None:
        return None, 'no_raw_guide'
    if not center.any():
        return None, 'empty_path'
    pix = guide_rgb[center].astype(np.float64)
    target = np.asarray(orange_rgb, dtype=np.float64)
    d = np.linalg.norm(pix - target, axis=1)
    mean_d = float(d.mean())
    # Map to [0,1] support: 1 when mean_d=0, 0 when mean_d>=max_l2
    support = float(np.clip(1.0 - mean_d / max(max_l2, 1e-6), 0.0, 1.0))
    return support, f'mean_l2={mean_d:.2f}'


def _safe_morph_close(
    seed: np.ndarray,
    hard: np.ndarray,
    morph_close_px: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Close tiny gaps; drop bridge pixels that intersect hard or float after clip.

    Returns (new_seed, added_mask).
    """
    if not morph_close_px or morph_close_px <= 1 or not seed.any():
        return seed.copy(), np.zeros_like(seed, dtype=bool)
    k = int(morph_close_px) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(seed.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0
    # Never keep hard-barrier pixels.
    candidate = (seed | closed) & ~hard
    # Keep only pixels reachable from the original seed (drops mid-barrier stubs).
    work = candidate.astype(np.uint8)
    num, labels = cv2.connectedComponents(work, connectivity=8)
    keep = np.zeros_like(seed, dtype=bool)
    seed_labels = set(int(x) for x in np.unique(labels[seed & (labels > 0)]))
    for lab in seed_labels:
        keep |= labels == lab
    added = keep & ~seed
    # Reject added bridges whose centerline (approx: added pixels) still touch hard
    # (should be empty after &~hard) or that only exist as dangling stubs — already handled.
    return keep, added


def _link_endpoints(
    skeleton: np.ndarray,
    seed_road: np.ndarray,
    *,
    max_gap: float,
    max_links_per_endpoint: int,
    hard: np.ndarray,
    direction_tolerance_deg: float,
    branch_trace_px: int,
    enable_t_junction: bool,
    fallback_halfwidth: float,
    guide_rgb: Optional[np.ndarray],
    orange_rgb: Sequence[int],
    raw_support_max_l2: float,
    diagnostics: Optional[List[Dict[str, Any]]],
    reject_third_road_contact: bool = True,
    pavement_mask: Optional[np.ndarray] = None,
    building_mask: Optional[np.ndarray] = None,
    water_mask: Optional[np.ndarray] = None,
    enable_building_proxy: bool = True,
    pavement_contact_radius_px: int = _DEFAULT_PAVEMENT_CONTACT_RADIUS_PX,
    access_stub_area_px: int = _ACCESS_STUB_AREA_PX,
    large_cc_ee_cap_px: float = _LARGE_CC_EE_CAP_PX,
    medium_gap_px: float = _MEDIUM_GAP_PX,
    pinch_gap_px: float = _PINCH_GAP_PX,
    large_cc_area_px: int = 160,
) -> Tuple[np.ndarray, np.ndarray]:
    """Endpoint–endpoint / segment / building-proxy links with commit semantics.

    Returns (updated_skeleton, new_link_centerline_mask).
    EB (building-proxy) is skipped when the *local* endpoint status is
    CONNECTED_PAVEMENT. Status is shared across EE/ET/EB in this call.
    Small ACCESS_STUB CCs (touch pav/building/water) skip road–road links
    when ``access_stub_area_px`` > 0.
    """
    out = (skeleton > 0).astype(np.uint8).copy()
    new_links_mask = np.zeros_like(out, dtype=bool)
    tip_ends = _endpoints(out)
    num, labels = cv2.connectedComponents(out, connectivity=8)
    if num < 1 or max_gap <= 0:
        return out, new_links_mask

    stub_area_lim = int(access_stub_area_px)
    large_ee_cap = float(large_cc_ee_cap_px)
    med_gap = float(medium_gap_px)
    pinch_lim = float(pinch_gap_px)
    large_area = int(large_cc_area_px)

    # Face endpoints at near pinches (fixes "so close but no tip on skeleton").
    face_ends = _pinch_face_endpoints(out, labels, max_pinch_px=pinch_lim)
    if len(tip_ends) and len(face_ends):
        ends = np.unique(np.vstack([tip_ends, face_ends]), axis=0)
    elif len(face_ends):
        ends = face_ends
    else:
        ends = tip_ends
    if len(ends) < 1:
        return out, new_links_mask

    # Areas + access-stub CCs (do not chase other roads).
    cc_area = {lab: int((labels == lab).sum()) for lab in range(1, num)}
    access_skip = (
        _cc_access_stub_labs(
            labels, seed_road, pavement=pavement_mask, building=building_mask,
            water=water_mask, max_area_px=stub_area_lim)
        if stub_area_lim > 0 else {}
    )
    # Pixel mask — label IDs become stale after merges / CC renumbering.
    access_stub_pix = np.zeros(out.shape, dtype=bool)
    for lab, why in access_skip.items():
        access_stub_pix |= labels == int(lab)
    if diagnostics is not None and access_skip:
        for lab, why in access_skip.items():
            diagnostics.append(dict(
                type='CC_ACCESS_STUB_SKIP', component=lab, area=cc_area.get(lab, 0),
                accepted=False, reason=f'skip_road_links_touch_{why}',
                note='small_patch_already_on_access'))

    uf = _UnionFind(max(num, 1))
    new_link_count = np.zeros(len(ends), dtype=int)
    end_labs = np.array([int(labels[y, x]) for y, x in ends], dtype=int)

    end_status: List[str] = []
    end_status_note: List[str] = []
    for i, (y, x) in enumerate(ends):
        lab = int(end_labs[i])
        if access_stub_pix[int(y), int(x)] or lab in access_skip:
            end_status.append(STATUS_ACCESS_STUB)
            end_status_note.append(
                f"touch_{access_skip.get(lab, 'pix')}")
            continue
        st, note = classify_endpoint_status(
            int(y), int(x), pavement=pavement_mask,
            pavement_radius=pavement_contact_radius_px)
        end_status.append(st)
        end_status_note.append(note)

    dirs = []
    dir_rel = []
    for y, x in ends:
        d, rel = _branch_outward_direction(out, int(y), int(x), trace_px=branch_trace_px)
        dirs.append(d)
        dir_rel.append(rel)

    tree = cKDTree(ends.astype(np.float64)) if len(ends) else None
    sk_pts = np.column_stack(np.where(out > 0))
    sk_tree = cKDTree(sk_pts.astype(np.float64)) if len(sk_pts) else None
    end_set = {(int(ends[k, 0]), int(ends[k, 1])) for k in range(len(ends))}

    bldg = np.asarray(building_mask, dtype=bool) if building_mask is not None else None
    b_targets = (
        _building_exterior_targets(bldg, hard)
        if (enable_building_proxy and bldg is not None and bldg.any())
        else np.zeros((0, 2), dtype=int))
    b_tree = cKDTree(b_targets.astype(np.float64)) if len(b_targets) else None

    # Build deterministic candidate list: (dist, type_ord, i, j_or_-1, ty, tx, cand_type)
    # type_ord: 0=EE, 1=ET, 2=EB so road–road preferred before building proxy.
    candidates = []
    used_pair_keys = set()

    for i, (y, x) in enumerate(ends):
        lab = int(end_labs[i])
        if lab == 0:
            continue
        # Access stubs (v8b): no EE/ET/EB toward other roads / buildings.
        if end_status[i] == STATUS_ACCESS_STUB or access_stub_pix[int(y), int(x)]:
            if diagnostics is not None:
                diagnostics.append(dict(
                    candidate_id=0, type='EE_SKIP', y0=int(y), x0=int(x),
                    accepted=False, reason='skip_access_stub_no_road_link',
                    endpoint_status=end_status[i],
                    endpoint_status_note=end_status_note[i],
                    component=lab, area=cc_area.get(lab, 0)))
            continue
        area_i = cc_area.get(lab, 0)
        if tree is not None:
            idxs = tree.query_ball_point([float(y), float(x)], r=float(max_gap) + 1e-9)
            for j in idxs:
                if j == i:
                    continue
                if (end_status[j] == STATUS_ACCESS_STUB
                        or access_stub_pix[int(ends[j, 0]), int(ends[j, 1])]):
                    continue
                y2, x2 = int(ends[j, 0]), int(ends[j, 1])
                lab2 = int(end_labs[j])
                d = float(np.hypot(y - y2, x - x2))
                if d < 1.0 or d > float(max_gap):
                    continue
                area_j = cc_area.get(lab2, 0)
                # Far false mesh: two large CCs only within pinch cap.
                if area_i >= large_area and area_j >= large_area and d > large_ee_cap:
                    continue
                a, b = (i, j) if i < j else (j, i)
                key = ('EE', a, b)
                if key in used_pair_keys:
                    continue
                used_pair_keys.add(key)
                candidates.append((d, 0, i, j, y2, x2, 'EE'))

        if enable_t_junction and sk_tree is not None:
            et_cap = float(max_gap)
            if area_i >= large_area:
                et_cap = min(et_cap, med_gap)
            near = sk_tree.query_ball_point([float(y), float(x)], r=et_cap + 1e-9)
            best = None  # (d, ty, tx, tlab)
            for si in near:
                ty, tx = int(sk_pts[si, 0]), int(sk_pts[si, 1])
                d = float(np.hypot(y - ty, x - tx))
                if d < 2.0 or d > et_cap:
                    continue
                tlab = int(labels[ty, tx])
                if tlab == 0 or tlab == lab or access_stub_pix[ty, tx]:
                    continue
                if (ty, tx) in end_set:
                    continue
                if len(_neighbors8(out > 0, ty, tx)) < 2:
                    continue
                if best is None or d < best[0] or (d == best[0] and (ty, tx) < (best[1], best[2])):
                    best = (d, ty, tx, tlab)
            if best is not None:
                d, ty, tx, _tlab = best
                key = ('ET', i, ty, tx)
                if key not in used_pair_keys:
                    used_pair_keys.add(key)
                    candidates.append((d, 1, i, -1, ty, tx, 'ET'))

        # Building-proxy (EB): only if local endpoint is NOT already on pavement.
        if b_tree is not None and end_status[i] != STATUS_CONNECTED_PAVEMENT:
            near_b = b_tree.query_ball_point([float(y), float(x)], r=float(max_gap) + 1e-9)
            best_b = None
            for bi in near_b:
                ty, tx = int(b_targets[bi, 0]), int(b_targets[bi, 1])
                d = float(np.hypot(y - ty, x - tx))
                if d < 1.0 or d > float(max_gap):
                    continue
                if best_b is None or d < best_b[0] or (d == best_b[0] and (ty, tx) < (best_b[1], best_b[2])):
                    best_b = (d, ty, tx)
            if best_b is not None:
                d, ty, tx = best_b
                key = ('EB', i, ty, tx)
                if key not in used_pair_keys:
                    used_pair_keys.add(key)
                    candidates.append((d, 2, i, -1, ty, tx, 'EB'))
        elif b_tree is not None and end_status[i] == STATUS_CONNECTED_PAVEMENT:
            # Record skip so dual callers cannot re-introduce a building stub.
            if diagnostics is not None:
                diagnostics.append(dict(
                    candidate_id=0, type='EB_SKIP',
                    y0=int(y), x0=int(x), y1=-1, x1=-1,
                    accepted=False,
                    reason='skip_building_proxy_already_connected_pavement',
                    endpoint_status=end_status[i],
                    endpoint_status_note=end_status_note[i],
                ))

    # Deterministic sort: distance, type, endpoint index, target coords
    candidates.sort(key=lambda c: (c[0], c[1], c[2], c[4], c[5], c[6]))

    cand_id = 0
    for dist, _type_ord, i, j, ty, tx, ctype in candidates:
        cand_id += 1
        y, x = int(ends[i, 0]), int(ends[i, 1])
        lab = int(end_labs[i])
        if ctype == 'EE':
            lab2 = int(end_labs[j])
            target_lab = lab2
        elif ctype == 'ET':
            lab2 = int(labels[ty, tx])
            target_lab = lab2
        else:  # EB — exterior proxy, not a road component
            lab2 = 0
            target_lab = 0

        rec: Dict[str, Any] = dict(
            candidate_id=cand_id,
            type=ctype,
            y0=y, x0=x, y1=int(ty), x1=int(tx),
            component_0=lab, component_1=target_lab,
            distance_px=float(dist),
            direction_reliability_0=dir_rel[i],
            direction_reliability_1=dir_rel[j] if ctype == 'EE' and j >= 0 else 'n/a',
            proposed_new_px=0,
            committed_new_px=0,
            deleted_original_road_px=0,
            accepted=False,
            reason='',
            touches_target=False,
            merged_components=False,
            still_valid_after_paint=False,
            endpoint_status=end_status[i],
            endpoint_status_note=end_status_note[i],
        )

        def _reject(reason: str):
            rec['accepted'] = False
            rec['reason'] = reason
            if diagnostics is not None:
                diagnostics.append(rec)

        # Re-check pavement gate for EB (status may have changed after prior accepts).
        if ctype == 'EB':
            if end_status[i] == STATUS_CONNECTED_PAVEMENT:
                _reject('skip_building_proxy_already_connected_pavement')
                continue
            # Refresh local pavement in case a prior link changed neighbourhood.
            st_now, note_now = classify_endpoint_status(
                y, x, pavement=pavement_mask,
                pavement_radius=pavement_contact_radius_px)
            end_status[i], end_status_note[i] = st_now, note_now
            rec['endpoint_status'] = st_now
            rec['endpoint_status_note'] = note_now
            if st_now == STATUS_CONNECTED_PAVEMENT:
                _reject('skip_building_proxy_already_connected_pavement')
                continue

        # Link budget: count NEW links on BOTH ends (EE updates both).
        if new_link_count[i] >= max_links_per_endpoint:
            _reject('max_new_links_source')
            continue
        if ctype == 'EE' and new_link_count[j] >= max_links_per_endpoint:
            _reject('max_new_links_target')
            continue

        # Same component after prior unions → redundant (road–road only)
        if ctype != 'EB' and lab > 0 and target_lab > 0 and uf.same(lab, target_lab):
            _reject('redundant_same_component')
            continue

        # Direction checks
        dir0 = dirs[i]
        link_vec = np.array([ty - y, tx - x], dtype=np.float64)
        link_n = np.linalg.norm(link_vec)
        if link_n < 1e-6:
            _reject('zero_length')
            continue
        link_u = link_vec / link_n

        if dir0 is None or dir_rel[i] != 'ok':
            _reject(f'direction_uncertain_source:{dir_rel[i]}')
            continue
        ang0 = _angle_deg(dir0, link_u)
        rec['direction_angle_0_deg'] = ang0
        if ang0 > float(direction_tolerance_deg):
            _reject('direction_not_facing_target')
            continue

        if ctype == 'EE':
            dir1 = dirs[j]
            if dir1 is None or dir_rel[j] != 'ok':
                _reject(f'direction_uncertain_target:{dir_rel[j]}')
                continue
            # Target outward should face back toward source ≈ opposite of link_u
            ang1 = _angle_deg(dir1, -link_u)
            rec['direction_angle_1_deg'] = ang1
            if ang1 > float(direction_tolerance_deg):
                _reject('endpoints_not_mutual')
                continue
        # ET/EB: only source facing target.

        half_w = _estimate_link_halfwidth(
            seed_road, out, y, x, int(ty), int(tx), fallback=fallback_halfwidth)
        rec['local_halfwidth_px'] = half_w

        # Propose on copies — never leave half links.
        center = _draw_line_mask(out.shape, y, x, int(ty), int(tx))
        # New pixels only (exclude existing skeleton)
        proposed_new = center & ~(out > 0)
        rec['proposed_new_px'] = int(proposed_new.sum())

        if _band_hits_hard(center, half_w, hard):
            _reject('hard_obstacle_on_path_or_band')
            continue

        # Full width band vs existing skeleton: reject accidental third-road contact.
        band = _thicken_mask(center, half_w)
        if reject_third_road_contact and ctype != 'EB':
            touch = band & (out > 0)
            if touch.any():
                touched = set(int(v) for v in np.unique(labels[touch]) if int(v) != 0)
                allowed_roots = set()
                if lab > 0:
                    allowed_roots.add(uf.find(lab))
                if target_lab > 0:
                    allowed_roots.add(uf.find(target_lab))
                foreign = [L for L in touched if uf.find(L) not in allowed_roots]
                rec['touched_components'] = sorted(touched)
                rec['foreign_components'] = foreign
                if foreign:
                    _reject('third_road_contact_on_band')
                    continue

        support, support_note = _raw_support_score(
            guide_rgb, center, orange_rgb, raw_support_max_l2)
        rec['raw_support_score'] = support
        rec['raw_support_note'] = support_note
        # RAW is optional evidence only; absence does not force accept/reject.

        # Commit: draw centerline, update counts and UF (atomic — no half links).
        before = out.copy()
        cv2.line(out, (int(x), int(y)), (int(tx), int(ty)), 1, 1, lineType=cv2.LINE_8)
        touches = bool(out[int(ty), int(tx)] > 0) and bool(out[y, x] > 0)
        if not np.all(out[center] > 0):
            out[:] = before
            _reject('paint_incomplete_rollback')
            continue
        if not touches:
            out[:] = before
            _reject('did_not_reach_target_rollback')
            continue

        committed = (out > 0) & ~(before > 0)
        if np.any(committed & hard):
            out[:] = before
            _reject('hard_after_paint_rollback')
            continue
        # Re-check third contact on the committed link band only (not whole network).
        if reject_third_road_contact and ctype != 'EB':
            band2 = _thicken_mask(center | committed, half_w)
            touch2 = band2 & (before > 0)
            if touch2.any():
                touched2 = set(int(v) for v in np.unique(labels[touch2]) if int(v) != 0)
                allowed_roots = set()
                if lab > 0:
                    allowed_roots.add(uf.find(lab))
                if target_lab > 0:
                    allowed_roots.add(uf.find(target_lab))
                foreign2 = [L for L in touched2 if uf.find(L) not in allowed_roots]
                if foreign2:
                    rec['foreign_components'] = foreign2
                    out[:] = before
                    _reject('third_road_contact_after_paint_rollback')
                    continue

        merged = False
        if ctype != 'EB' and lab > 0 and target_lab > 0:
            merged = uf.union(lab, target_lab)
        new_link_count[i] += 1
        if ctype == 'EE' and j >= 0:
            new_link_count[j] += 1
        new_links_mask |= committed

        # Update shared endpoint status immediately (no stale state for later EB).
        if ctype in ('EE', 'ET'):
            end_status[i] = STATUS_CONNECTED_ROAD
            end_status_note[i] = f'linked_{ctype}'
            if ctype == 'EE' and j >= 0:
                end_status[j] = STATUS_CONNECTED_ROAD
                end_status_note[j] = f'linked_{ctype}'
        elif ctype == 'EB':
            end_status[i] = STATUS_CONNECTED_ROAD
            end_status_note[i] = 'linked_EB_building_proxy'

        rec['accepted'] = True
        rec['reason'] = 'accepted'
        rec['committed_new_px'] = int(committed.sum())
        rec['touches_target'] = True
        rec['merged_components'] = bool(merged)
        rec['still_valid_after_paint'] = True
        rec['new_links_at_source'] = int(new_link_count[i])
        rec['endpoint_status_after'] = end_status[i]
        if ctype == 'EE' and j >= 0:
            rec['new_links_at_target'] = int(new_link_count[j])
        if diagnostics is not None:
            diagnostics.append(rec)

    def _try_commit_ee(i, j, ty, tx, *, tag: str, half_w_fb=None, skip_dir=False):
        """Commit one EE link if hard/third-road OK. Mutates out/UF/status."""
        y, x = int(ends[i, 0]), int(ends[i, 1])
        lab = int(end_labs[i])
        lab2 = int(end_labs[j])
        if lab == 0 or lab2 == 0 or uf.same(lab, lab2):
            return False
        if new_link_count[i] >= max_links_per_endpoint or new_link_count[j] >= max_links_per_endpoint:
            return False
        half_w = half_w_fb if half_w_fb is not None else _estimate_link_halfwidth(
            seed_road, out, y, x, int(ty), int(tx), fallback=fallback_halfwidth)
        center = _draw_line_mask(out.shape, y, x, int(ty), int(tx))
        if _band_hits_hard(center, half_w, hard):
            if diagnostics is not None:
                diagnostics.append(dict(
                    type=tag, y0=y, x0=x, y1=int(ty), x1=int(tx),
                    accepted=False, reason='hard_obstacle_on_path_or_band',
                    note=tag))
            return False
        band = _thicken_mask(center, half_w)
        if reject_third_road_contact:
            touch = band & (out > 0)
            if touch.any():
                touched = set(int(v) for v in np.unique(labels[touch]) if int(v) != 0)
                allowed = {uf.find(lab), uf.find(lab2)}
                foreign = [L for L in touched if uf.find(L) not in allowed]
                if foreign:
                    if diagnostics is not None:
                        diagnostics.append(dict(
                            type=tag, y0=y, x0=x, y1=int(ty), x1=int(tx),
                            accepted=False, reason='third_road_contact_on_band',
                            note=tag))
                    return False
        before = out.copy()
        cv2.line(out, (int(x), int(y)), (int(tx), int(ty)), 1, 1, lineType=cv2.LINE_8)
        if not np.all(out[center] > 0) or not (out[int(ty), int(tx)] > 0):
            out[:] = before
            return False
        committed = (out > 0) & ~(before > 0)
        if np.any(committed & hard):
            out[:] = before
            return False
        uf.union(lab, lab2)
        new_link_count[i] += 1
        new_link_count[j] += 1
        nonlocal new_links_mask
        new_links_mask |= committed
        end_status[i] = STATUS_CONNECTED_ROAD
        end_status[j] = STATUS_CONNECTED_ROAD
        end_status_note[i] = f'linked_{tag}'
        end_status_note[j] = f'linked_{tag}'
        if diagnostics is not None:
            diagnostics.append(dict(
                candidate_id=0, type=tag,
                y0=y, x0=x, y1=int(ty), x1=int(tx),
                accepted=True, reason='accepted',
                note=tag, skip_dir=bool(skip_dir),
                committed_new_px=int(committed.sum()),
                local_halfwidth_px=float(half_w),
            ))
        return True

    # Soft EE retry: relax angle; also allow uncertain_short for short gaps.
    soft_tol = float(direction_tolerance_deg) * 1.5
    short_gap_px = min(med_gap, float(max_gap))
    pinch_gap_px = min(pinch_lim, float(max_gap))
    pending_idx = [
        i for i in range(len(ends))
        if end_status[i] == STATUS_PENDING and new_link_count[i] < max_links_per_endpoint
        and not access_stub_pix[int(ends[i, 0]), int(ends[i, 1])]
    ]
    for i in pending_idx:
        if end_status[i] != STATUS_PENDING:
            continue
        y, x = int(ends[i, 0]), int(ends[i, 1])
        lab = int(end_labs[i])
        if lab == 0 or tree is None:
            continue
        dir0 = dirs[i]
        rel0 = dir_rel[i]
        idxs = tree.query_ball_point([float(y), float(x)], r=float(max_gap) + 1e-9)
        best = None  # (d, j, ty, tx, skip_dir)
        for j in idxs:
            if j <= i:
                continue
            if new_link_count[j] >= max_links_per_endpoint:
                continue
            y2, x2 = int(ends[j, 0]), int(ends[j, 1])
            lab2 = int(end_labs[j])
            if lab2 == 0 or uf.same(lab, lab2):
                continue
            d = float(np.hypot(y - y2, x - x2))
            if d < 1.0 or d > float(max_gap):
                continue
            link_u = np.array([y2 - y, x2 - x], dtype=np.float64)
            ln = np.linalg.norm(link_u)
            if ln < 1e-6:
                continue
            link_u /= ln
            rel1 = dir_rel[j]
            dir1 = dirs[j]
            if (end_status[j] == STATUS_ACCESS_STUB
                    or access_stub_pix[int(ends[j, 0]), int(ends[j, 1])]):
                continue
            aa, ab = cc_area.get(lab, 0), cc_area.get(lab2, 0)
            # Two large CCs: soft only inside pinch + axis-aligned (not parallel lateral).
            if aa >= large_area and ab >= large_area:
                if d > pinch_gap_px:
                    continue
                ax0 = _principal_axis_yx(labels == lab)
                ax1 = _principal_axis_yx(labels == lab2)
                if _is_lateral_parallel_bridge(ax0, ax1, link_u):
                    continue
            if d <= pinch_gap_px:
                skip_dir = True
            elif rel0 != 'ok' or rel1 != 'ok':
                if d > short_gap_px:
                    continue
                skip_dir = True
            else:
                skip_dir = False
                ang0 = _angle_deg(dir0, link_u)
                ang1 = _angle_deg(dir1, -link_u)
                if ang0 > soft_tol or ang1 > soft_tol:
                    continue
            if best is None or d < best[0]:
                best = (d, j, y2, x2, skip_dir)
        if best is None:
            continue
        _d, j, ty, tx, skip_dir = best
        _try_commit_ee(i, j, ty, tx, tag='EE_SOFT', skip_dir=skip_dir)

    # Component nearest-point bridge for remaining short gaps (direction-agnostic).
    # Targets clear breaks where skeleton ends are noisy / non-facing.
    road_bin = out > 0
    if road_bin.any():
        n_cc, cc_lab = cv2.connectedComponents(road_bin.astype(np.uint8), connectivity=8)
        # Sample boundary pixels per CC for nearest search (v8b: cap 80).
        cc_pts: Dict[int, np.ndarray] = {}
        for lab in range(1, n_cc):
            comp = cc_lab == lab
            if not comp.any():
                continue
            er = cv2.erode(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            border = comp & ~er
            if not border.any():
                border = comp
            ys, xs = np.where(border)
            if len(ys) > 80:
                idx = np.linspace(0, len(ys) - 1, 80).astype(int)
                ys, xs = ys[idx], xs[idx]
            cc_pts[lab] = np.column_stack([ys, xs])
        # Recompute access stubs on *current* CC ids (labels after prior unions).
        access_skip_now = (
            _cc_access_stub_labs(
                cc_lab, seed_road, pavement=pavement_mask, building=building_mask,
                water=water_mask, max_area_px=stub_area_lim)
            if stub_area_lim > 0 else {}
        )
        gap_area = {lab: int((cc_lab == lab).sum()) for lab in cc_pts}
        # When access-stub skip is off (lim<=0), do not use the
        # "neither-is-stub" medium-gap reject (would reject everything).
        stub_area_max = stub_area_lim if stub_area_lim > 0 else 10 ** 9
        large_area_min = large_area
        labs_list = sorted(cc_pts.keys())
        bridges = 0
        max_bridges = 60
        for ai, a in enumerate(labs_list):
            if bridges >= max_bridges:
                break
            if a not in cc_pts:
                continue
            # Access-stub CCs: do not bridge to other roads (v8b).
            if a in access_skip_now:
                continue
            ta = cKDTree(cc_pts[a].astype(np.float64))
            for b in labs_list[ai + 1:]:
                if bridges >= max_bridges:
                    break
                if b in access_skip_now:
                    continue
                aa, ab = gap_area.get(a, 0), gap_area.get(b, 0)
                tb = cc_pts[b]
                d, ix = ta.query(tb.astype(np.float64), k=1)
                d = np.asarray(d, dtype=float)
                if d.size == 0:
                    continue
                kmin = int(np.argmin(d))
                dist = float(d[kmin])
                if dist < 1.0:
                    continue
                # Near pinch: allow large↔large end-to-end; not lateral parallel.
                if dist <= pinch_gap_px:
                    pass
                elif dist <= short_gap_px:
                    if aa > stub_area_max and ab > stub_area_max:
                        if diagnostics is not None:
                            diagnostics.append(dict(
                                type='EE_GAP', accepted=False,
                                reason='reject_neither_stub_medium_gap',
                                area_0=aa, area_1=ab, distance_px=dist,
                                note='avoid_spurious_mesh'))
                        continue
                else:
                    continue
                y2, x2 = int(tb[kmin, 0]), int(tb[kmin, 1])
                y1, x1 = int(cc_pts[a][int(ix[kmin]), 0]), int(cc_pts[a][int(ix[kmin]), 1])
                if out[y1, x1] == 0 or out[y2, x2] == 0:
                    continue
                link = np.array([y2 - y1, x2 - x1], dtype=np.float64)
                ln = float(np.linalg.norm(link))
                if ln < 1e-6:
                    continue
                link_u = link / ln
                ax_a = _principal_axis_yx(cc_lab == a)
                ax_b = _principal_axis_yx(cc_lab == b)
                if _is_lateral_parallel_bridge(ax_a, ax_b, link_u):
                    if diagnostics is not None:
                        diagnostics.append(dict(
                            type='EE_GAP', y0=y1, x0=x1, y1=y2, x1=x2,
                            accepted=False, reason='reject_lateral_parallel_bridge',
                            distance_px=dist, note='parallel_roads'))
                    continue
                n2, lab2m = cv2.connectedComponents((out > 0).astype(np.uint8), connectivity=8)
                if lab2m[y1, x1] == lab2m[y2, x2]:
                    continue
                half_w = float(max(1.0, fallback_halfwidth))
                center = _draw_line_mask(out.shape, y1, x1, y2, x2)
                if _band_hits_hard(center, half_w, hard):
                    if diagnostics is not None:
                        diagnostics.append(dict(
                            type='EE_GAP', y0=y1, x0=x1, y1=y2, x1=x2,
                            accepted=False, reason='hard_obstacle_on_path_or_band',
                            distance_px=dist, note='component_short_gap'))
                    continue
                band = _thicken_mask(center, half_w)
                if reject_third_road_contact:
                    touch = band & (out > 0)
                    if touch.any():
                        touched = set(int(v) for v in np.unique(lab2m[touch]) if int(v) != 0)
                        allowed = {int(lab2m[y1, x1]), int(lab2m[y2, x2])}
                        if any(t not in allowed for t in touched):
                            if diagnostics is not None:
                                diagnostics.append(dict(
                                    type='EE_GAP', y0=y1, x0=x1, y1=y2, x1=x2,
                                    accepted=False, reason='third_road_contact_on_band',
                                    distance_px=dist, note='component_short_gap'))
                            continue
                before = out.copy()
                cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 1, 1, lineType=cv2.LINE_8)
                if not (out[y2, x2] > 0 and out[y1, x1] > 0):
                    out[:] = before
                    continue
                committed = (out > 0) & ~(before > 0)
                if np.any(committed & hard):
                    out[:] = before
                    continue
                new_links_mask |= committed
                bridges += 1
                if diagnostics is not None:
                    diagnostics.append(dict(
                        type='EE_GAP', y0=y1, x0=x1, y1=y2, x1=x2,
                        accepted=True, reason='accepted',
                        distance_px=dist, note='stub_to_main_short_gap',
                        area_0=aa, area_1=ab,
                        committed_new_px=int(committed.sum()),
                        local_halfwidth_px=half_w,
                    ))

    return out, new_links_mask


def prune_tiny_road_components(
    road: np.ndarray,
    *,
    min_area_px: int = 14,
    max_span_px: int = 10,
) -> Tuple[np.ndarray, Dict]:
    """Drop abnormal micro road crumbs / short isolated spurs (not main network).

    Removes CCs with area ≤ min_area_px, or area ≤ 2*min and bbox span ≤ max_span.
    Does not delete large legitimate dead-ends.
    """
    m0 = np.asarray(road, dtype=bool)
    info: Dict[str, Any] = dict(
        accepted=False, removed_components=0, removed_px=0,
        min_area_px=int(min_area_px), max_span_px=int(max_span_px))
    if not m0.any():
        return m0.copy(), info
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        m0.astype(np.uint8), connectivity=8)
    out = m0.copy()
    removed = 0
    rpx = 0
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        span = max(w, h)
        drop = area <= int(min_area_px) or (
            area <= int(min_area_px) * 2 and span <= int(max_span_px))
        if not drop:
            continue
        out[labels == lab] = False
        removed += 1
        rpx += area
    info['accepted'] = removed > 0
    info['removed_components'] = removed
    info['removed_px'] = rpx
    return out, info


def repair_roads_cross_vegetation(
    canvas_img,
    source_img,
    orange_rgb=ORANGE_DEFAULT,
    tolerance: int = 30,
    max_width: int = 2,
    blocking_masks=None,
    *,
    morph_close_px: int = 5,
    connect_max_gap_px: float = 40.0,
    max_links_per_endpoint: int = 2,
    allow_cross_vegetation: bool = True,
    vegetation_masks=None,
    guide_rgb=None,
    direction_tolerance_deg: float = _DEFAULT_DIRECTION_TOL_DEG,
    branch_trace_px: int = _DEFAULT_BRANCH_TRACE_PX,
    enable_t_junction: bool = True,
    raw_support_max_l2: float = _DEFAULT_RAW_SUPPORT_MAX_L2,
    reject_third_road_contact: bool = True,
    pavement_mask=None,
    building_mask=None,
    water_mask=None,
    enable_building_proxy: bool = True,
    pavement_contact_radius_px: int = _DEFAULT_PAVEMENT_CONTACT_RADIUS_PX,
    prune_min_area_px: int = 14,
    prune_max_span_px: int = 10,
    prune_spur_max_px: int = 0,
    enable_width_regularization: bool = False,
    width_half_min_px: float = 1.0,
    width_half_max_px: float = 4.0,
    width_mode: str = 'median',
    access_stub_area_px: int = _ACCESS_STUB_AREA_PX,
    large_cc_ee_cap_px: float = _LARGE_CC_EE_CAP_PX,
    medium_gap_px: float = _MEDIUM_GAP_PX,
    pinch_gap_px: float = _PINCH_GAP_PX,
    large_cc_area_px: int = 160,
    diagnostics: Optional[list] = None,
):
    """Paste + safe-close + endpoint-connect roads.

    Parameters
    ----------
    source_img :
        Discrete R1 (or equivalent) used for orange label extraction.
    max_width :
        Legacy name kept for API compatibility. **Not** a whole-map morphological
        close. Used only as fallback half-width (px) when EDT cannot estimate the
        width of a *new* link. Prefer thinking of it as new_link_fallback_halfwidth.
    blocking_masks :
        Hard barriers from caller (refine: fixed + building). Water must not be
        passed here under current coverage rules unless explicitly testing blockers.
    vegetation_masks :
        Plant/lawn masks. Hard when allow_cross_vegetation is False; ignored as
        barriers when True (roads may cross vegetation).
    pavement_mask :
        Current pavement for local endpoint access status. Roads may contact /
        locally overlay pavement; whole pavement is not converted to road.
    building_mask :
        Building footprint for EB proxy targets (exterior adjacent pixels only).
        Full road band still cannot cover building (hard).
    enable_building_proxy :
        If True, generate EB candidates for PENDING endpoints; skipped when the
        endpoint already has CONNECTED_PAVEMENT status.
    guide_rgb :
        Optional RAW RGB for colour support scoring along candidates. Not a
        calibrated probability and never substituted by layout-B / human labels.
    diagnostics :
        If a list is provided, per-candidate decision records are appended.
    """
    canvas = np.asarray(canvas_img, dtype=np.uint8)
    source = np.asarray(source_img, dtype=np.uint8)
    if canvas.ndim != 3 or source.ndim != 3:
        raise ValueError('RGB images required')
    # Defensive copies — never mutate caller arrays in-place.
    canvas_work = canvas.copy()
    source_work = source  # read-only use

    lower = np.array([max(0, int(c) - int(tolerance)) for c in orange_rgb], dtype=np.uint8)
    upper = np.array([min(255, int(c) + int(tolerance)) for c in orange_rgb], dtype=np.uint8)
    mask = cv2.inRange(source_work, lower, upper) > 0

    hard = _combine_blocks(mask.shape, blocking_masks)
    if not allow_cross_vegetation:
        hard = hard | _combine_blocks(mask.shape, vegetation_masks)

    guide = None
    if guide_rgb is not None:
        guide = np.asarray(guide_rgb, dtype=np.uint8)
        if guide.shape[:2] != mask.shape:
            raise ValueError('guide_rgb spatial size must match source_img')

    summary = {
        'allow_cross_vegetation': bool(allow_cross_vegetation),
        'morph_close_added_px': 0,
        'endpoint_link_added_px': 0,
        'original_road_px': 0,
        'final_road_px': 0,
        'deleted_original_road_px': 0,
        'legacy_max_width_px_role': (
            'fallback half-width for new links only; does not thicken whole network'
        ),
        'unimplemented': [
            # True labelled entrances are unavailable; EB uses exterior proxy pixels.
            # Full stepwise turning-cost search with heading state is not implemented;
            # only endpoint direction gating is applied.
            'route_through_array_turning_cost',
            'deep_road_mapper_trained_prior',
        ],
        'building_proxy': 'EB exterior-adjacent; skipped if CONNECTED_PAVEMENT',
        'pavement_contact_radius_px': int(pavement_contact_radius_px),
        'enable_building_proxy': bool(enable_building_proxy),
    }

    # Prefer explicit building_mask; else derive from blocking if a single mask
    # looks building-sized — refine always passes building_mask explicitly.
    b_mask = None
    if building_mask is not None:
        b_mask = np.asarray(building_mask, dtype=bool)
    pav_mask = None if pavement_mask is None else np.asarray(pavement_mask, dtype=bool)
    wat_mask = None if water_mask is None else np.asarray(water_mask, dtype=bool)

    seed = mask & ~hard
    original_seed = seed.copy()
    summary['original_road_px'] = int(original_seed.sum())

    seed, added_close = _safe_morph_close(seed, hard, int(morph_close_px))
    summary['morph_close_added_px'] = int(added_close.sum())

    sk = skeletonize(seed).astype(np.uint8)
    link_diags: List[Dict[str, Any]] = []
    sk, new_link_sk = _link_endpoints(
        sk,
        seed_road=original_seed,
        max_gap=float(connect_max_gap_px),
        max_links_per_endpoint=int(max_links_per_endpoint),
        hard=hard,
        direction_tolerance_deg=float(direction_tolerance_deg),
        branch_trace_px=int(branch_trace_px),
        enable_t_junction=bool(enable_t_junction),
        fallback_halfwidth=float(max(1, int(max_width))),
        guide_rgb=guide,
        orange_rgb=orange_rgb,
        raw_support_max_l2=float(raw_support_max_l2),
        reject_third_road_contact=bool(reject_third_road_contact),
        pavement_mask=pav_mask,
        building_mask=b_mask,
        water_mask=wat_mask,
        enable_building_proxy=bool(enable_building_proxy),
        pavement_contact_radius_px=int(pavement_contact_radius_px),
        access_stub_area_px=int(access_stub_area_px),
        large_cc_ee_cap_px=float(large_cc_ee_cap_px),
        medium_gap_px=float(medium_gap_px),
        pinch_gap_px=float(pinch_gap_px),
        large_cc_area_px=int(large_cc_area_px),
        diagnostics=link_diags,
    )

    # Width: keep original R1/seed roads; thicken ONLY new link centerlines locally.
    road = original_seed.copy()
    # Morph-close additions keep their closed geometry (already at seed resolution).
    road |= (seed & ~original_seed)

    if new_link_sk.any():
        # Per-link thickening already validated; apply a uniform EDT-based band
        # using fallback clamped width on the union of new centerlines.
        # Estimate median half-width from endpoints of accepted links.
        half_ws = [
            float(d['local_halfwidth_px'])
            for d in link_diags
            if d.get('accepted') and 'local_halfwidth_px' in d
        ]
        hw = float(np.median(half_ws)) if half_ws else float(max(1, int(max_width)))
        band = _thicken_mask(new_link_sk, hw)
        if np.any(band & hard):
            # Strip hard; if a committed link's band is ruined, mark diagnostics.
            band &= ~hard
            for d in link_diags:
                if not d.get('accepted'):
                    continue
                cy, cx = d['y0'], d['x0']
                ty, tx = d['y1'], d['x1']
                center = _draw_line_mask(road.shape, cy, cx, ty, tx)
                if not np.any(band & center) or not (
                    band[ty, tx] or road[ty, tx] or (sk[ty, tx] > 0)
                ):
                    d['still_valid_after_paint'] = False
                    d['reason'] = 'width_band_lost_target_contact'
        road |= band
        # Ensure 1px centerlines remain even if band was clipped.
        road |= new_link_sk & ~hard

    road &= ~hard
    if bool(enable_width_regularization):
        road, winfo = regularize_road_width(
            road, sk > 0, hard,
            half_min_px=float(width_half_min_px),
            half_max_px=float(width_half_max_px),
            width_mode=str(width_mode),
        )
        summary['width_regularization'] = winfo
    else:
        summary['width_regularization'] = dict(enabled=False)
    # Keep surviving original seed only when width rebuild is off.
    if not enable_width_regularization:
        road |= original_seed & ~hard
    road, prune_info = prune_tiny_road_components(
        road,
        min_area_px=int(prune_min_area_px),
        max_span_px=int(prune_max_span_px))
    summary['tiny_road_prune'] = prune_info
    road, spur_info = prune_attached_spurs(
        road, max_len_px=int(prune_spur_max_px))
    summary['attached_spur_prune'] = spur_info
    # Re-attach substantial original seed pieces that prune might have dropped
    # only if they were large enough — prune already skipped those.

    deleted = int((original_seed & ~hard & ~road).sum())
    summary['deleted_original_road_px'] = deleted
    summary['endpoint_link_added_px'] = int((new_link_sk & ~hard).sum())
    summary['final_road_px'] = int(road.sum())
    summary['candidates'] = link_diags
    summary['accepted_links'] = sum(1 for d in link_diags if d.get('accepted'))
    summary['rejected_links'] = sum(1 for d in link_diags if not d.get('accepted'))

    # Final validity: accepted EE/ET still connect endpoints on final road mask.
    for d in link_diags:
        if not d.get('accepted'):
            continue
        y0, x0, y1, x1 = d['y0'], d['x0'], d['y1'], d['x1']
        if not (road[y0, x0] and road[y1, x1]):
            d['still_valid_after_paint'] = False
            d['reason'] = 'lost_after_final_road_compose'
            continue
        num_f, lab_f = cv2.connectedComponents((road > 0).astype(np.uint8), connectivity=8)
        if lab_f[y0, x0] == 0 or lab_f[y0, x0] != lab_f[y1, x1]:
            d['still_valid_after_paint'] = False
            d['reason'] = 'components_split_after_final_compose'
        else:
            d['still_valid_after_paint'] = True

    out = canvas_work.copy()
    # Clear pruned road crumbs on canvas (don't leave orange ghosts).
    prev_road = np.all(out == np.asarray(orange_rgb, dtype=np.uint8), axis=2)
    rem = prev_road & ~road & ~hard
    if rem.any():
        # Fill pruned crumbs with local non-road majority (prefer lawn).
        lawn = np.array([0, 255, 0], dtype=np.uint8)
        for y, x in zip(*np.where(rem)):
            y0, y1 = max(0, y - 1), min(out.shape[0], y + 2)
            x0, x1 = max(0, x - 1), min(out.shape[1], x + 2)
            patch = out[y0:y1, x0:x1]
            is_r = np.all(patch == orange_rgb, axis=2)
            vals = patch[~is_r]
            if len(vals):
                uniq, counts = np.unique(vals.reshape(-1, 3), axis=0, return_counts=True)
                out[y, x] = uniq[int(np.argmax(counts))]
            else:
                out[y, x] = lawn
    out[road] = np.asarray(orange_rgb, dtype=np.uint8)

    if diagnostics is not None:
        diagnostics.append(summary)
        diagnostics.extend(link_diags)

    return Image.fromarray(out)
