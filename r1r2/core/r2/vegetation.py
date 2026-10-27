"""Scale-aware vegetation symbols without expanding domain or clearing plants to lawn.

Analysis labels stay discrete. Circles are inscribed into the *current* vegetation
work mask (green ∩ ~blockers). Uncovered original plant pixels are kept.
Morphological clump-close and clear-then-redraw are off unless explicitly enabled.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

PLANT_RGB = (0, 152, 67)
LAWN_RGB = (0, 255, 0)
Center = Tuple[int, int, float]  # y, x, radius


def estimate_orange_width(orange_mask) -> int:
    """Legacy orange-width heuristic (optional radius cap reference)."""
    orange = np.asarray(orange_mask, dtype=bool)
    if not orange.any():
        return 2
    u8 = (orange.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(u8, kernel, iterations=1)
    dilated = cv2.dilate(u8, kernel, iterations=1)
    width_estimate = (np.sum(dilated > 0) - np.sum(eroded > 0)) / max(1, int(orange.sum()))
    return int(max(2, min(width_estimate * 1.5, 2.5)))


def estimate_max_road_width_px(road_mask) -> float:
    """Widest road width (px) via EDT: diameter ≈ 2 * max distance-to-boundary."""
    road = _as_bool(road_mask)
    if not road.any():
        return 2.0
    dt = cv2.distanceTransform(road.astype(np.uint8), cv2.DIST_L2, 5)
    return float(max(2.0, 2.0 * float(dt.max())))


def _as_bool(mask) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype == bool:
        return arr
    return arr != 0


def select_centers_min_dist(
    mask: np.ndarray,
    *,
    d_min: float,
    radius_min: float,
    radius_max: Optional[float] = None,
    prefer_local_maxima: bool = True,
) -> List[Center]:
    """EDT circle centres (paper dynamic packing).

    Prefer local maxima of the distance field (structural cluster centres /
    max inscribed radii), then greedy accept with ||C_i-C_j||_2 >= d_min.
    Fallback: all pixels with dt >= radius_min ordered by dt descending.
    """
    work = _as_bool(mask)
    if not work.any() or d_min < 0 or radius_min < 0:
        return []
    dt = cv2.distanceTransform(work.astype(np.uint8), cv2.DIST_L2, 5)
    # Local maxima via morphological non-maximum suppression on EDT.
    if prefer_local_maxima:
        k = max(3, int(np.ceil(float(radius_min))) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dil = cv2.dilate(dt, kernel)
        peaks = (dt >= dil - 1e-6) & (dt >= float(radius_min)) & work
        pys, pxs = np.where(peaks)
        ays, axs = np.where(dt >= float(radius_min))
        if len(pys) == 0:
            ys, xs = ays, axs
        else:
            peak_key = pys.astype(np.int64) * work.shape[1] + pxs
            all_key = ays.astype(np.int64) * work.shape[1] + axs
            rest = ~np.isin(all_key, peak_key)
            ys = np.concatenate([pys, ays[rest]])
            xs = np.concatenate([pxs, axs[rest]])
    else:
        ys, xs = np.where(dt >= float(radius_min))
    if len(ys) == 0:
        return []
    order = np.argsort(-dt[ys, xs], kind='stable')
    centres: List[Center] = []
    d_min2 = float(d_min) * float(d_min)
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        r = float(dt[y, x])
        if radius_max is not None:
            r = min(r, float(radius_max))
        r = float(np.floor(r))
        if r < float(radius_min):
            continue
        if any((y - cy) * (y - cy) + (x - cx) * (x - cx) < d_min2 for cy, cx, _ in centres):
            continue
        centres.append((y, x, r))
    # Small blobs whose inscribed radius is below radius_min still become one tree.
    if not centres and work.any():
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        r = float(max(1, int(np.floor(float(dt[int(y), int(x)])))))
        if radius_max is not None:
            r = min(r, float(radius_max))
        if r >= 1.0:
            centres.append((int(y), int(x), r))
    return centres


def _edt_peak(src: np.ndarray) -> Center:
    """Strongest point of the distance transform and its inscribed radius."""
    dt = cv2.distanceTransform(src.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    r = float(max(1, int(np.floor(float(dt[y, x])))))
    return int(y), int(x), r


def manuscript_centres(
    comp: np.ndarray,
    *,
    multi: bool,
    dist_block: Optional[np.ndarray] = None,
    road_relax_px: float = 0.0,
    min_radius: float = 2.0,
    max_circles: int = 256,
) -> List[dict]:
    """Centres painted as vegetation.

    D is the distance to the component boundary. A centre is a local maximum
    and R = max(1, floor(D)). A smaller component keeps one centre. A large
    component keeps those centres, then repeats on the uncovered remainder.
    q is the distance from the centre to the nearest blocker, minus one pixel.
    If q < R+rho the drawn radius is min(R+rho, max(1, q)); otherwise it is R.
    """
    work = _as_bool(comp)
    if not work.any():
        return []
    ys, xs = np.where(work)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub = work[y0:y1, x0:x1].copy()
    floor_r = max(1.0, float(min_radius))

    def _peaks(src: np.ndarray) -> List[Center]:
        if not src.any():
            return []
        dt = cv2.distanceTransform(src.astype(np.uint8), cv2.DIST_L2, 5)
        if float(dt.max()) <= 0:
            return []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        peaks = (dt >= cv2.dilate(dt, kernel) - 1e-6) & src
        py, px = np.where(peaks)
        if len(py) == 0:
            y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
            py, px = np.array([y]), np.array([x])
        order = np.argsort(-dt[py, px], kind='stable')
        found: List[Center] = []
        for idx in order:
            y, x = int(py[idx]), int(px[idx])
            radius = float(max(1, int(np.floor(float(dt[y, x])))))
            found.append((y, x, radius))
        return found

    picked: List[Center] = []
    if not multi:
        batch = _peaks(sub)
        if batch:
            picked.append(batch[0])
    else:
        remain = sub
        for _ in range(int(max_circles)):
            if int(remain.sum()) < 4:
                break
            batch = _peaks(remain)
            if not batch:
                break
            added: List[Center] = []
            for y, x, radius in batch:
                if radius < floor_r and picked:
                    continue
                if any(
                    (y - cy) * (y - cy) + (x - cx) * (x - cx) < max(radius, cr) ** 2
                    for cy, cx, cr in picked
                ):
                    continue
                picked.append((y, x, radius))
                added.append((y, x, radius))
                if len(picked) >= int(max_circles):
                    break
            if not added:
                break
            for y, x, radius in added:
                remain[disk_mask_raw(remain.shape, y, x, radius)] = False
            if len(picked) >= int(max_circles):
                break
        if not picked:
            batch = _peaks(sub)
            if batch:
                picked.append(batch[0])

    relax = float(road_relax_px)
    out: List[dict] = []
    for y, x, radius in picked:
        gy, gx = int(y + y0), int(x + x0)
        r_eff = radius
        if dist_block is not None and 0 <= gy < dist_block.shape[0] and 0 <= gx < dist_block.shape[1]:
            room = float(dist_block[gy, gx]) - 1.0
            if room < radius + relax:
                r_eff = min(radius + relax, max(1.0, room))
        out.append(dict(
            y=gy, x=gx, r_inscribed=float(radius), r_eff=float(r_eff)))
    return out


def fill_enclosed_plant_holes(
    plant: np.ndarray,
    allowed: np.ndarray,
    *,
    max_hole_px: int,
) -> Tuple[np.ndarray, int]:
    """Fill small lawn pockets fully enclosed by plant.

    Only holes whose area is at most ``max_hole_px`` and that stay inside
    ``allowed`` (lawn ∪ plant, never road/building) are closed.
    """
    base = _as_bool(plant)
    room = _as_bool(allowed)
    limit = int(max_hole_px)
    if limit <= 0 or not base.any():
        return base.copy(), 0
    bg = (~base).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    out = base.copy()
    filled = 0
    for lab in range(1, int(num)):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0 or area > limit:
            continue
        hole = labels == lab
        if not np.all(room[hole]):
            continue
        # Touching the image border is an exterior bay, not an enclosed hole.
        ys, xs = np.where(hole)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == base.shape[0] - 1 or xs.max() == base.shape[1] - 1:
            continue
        out[hole] = True
        filled += area
    return out, filled


def smooth_tree_circle_edges(
    disks: np.ndarray,
    blocked: np.ndarray,
    lawn: np.ndarray,
    *,
    window: int = 5,
    passes: int = 1,
    band_px: int = 2,
    close_px: int = 0,
    min_area_px: int = 20,
    hole_max_px: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """Join nearby large crowns, fill enclosed gaps, then slide the outline.

    Crowns smaller than ``min_area_px`` stay exact disks and are excluded
    from close / hole-fill / contour smoothing. Larger crowns may be bridged
    by ``close_px``, filled up to ``hole_max_px``, then outline-averaged with
    ``window`` / ``passes``. Expansion stays inside ``band_px`` and on lawn.
    """
    base = _as_bool(disks) & ~_as_bool(blocked)
    prot = _as_bool(blocked)
    yard = _as_bool(lawn)
    allowed = (yard | base) & ~prot
    info: Dict = dict(
        enabled=True, changed_px=0, smoothed_components=0, skipped_small=0,
        window=int(window), passes=int(passes), band_px=int(band_px),
        close_px=int(close_px), min_area_px=int(min_area_px),
        hole_max_px=int(hole_max_px), hole_filled_px=0)
    if not base.any():
        info['reason'] = 'empty'
        return base.copy(), info
    floor_area = max(1, int(min_area_px))
    # Protect small crowns first: they stay exact disks and are not closed
    # into neighbouring trees.
    num0, lab0 = cv2.connectedComponents(base.astype(np.uint8), connectivity=8)
    keep_single = np.zeros(base.shape, dtype=bool)
    large_seed = np.zeros(base.shape, dtype=bool)
    for lab in range(1, int(num0)):
        comp = lab0 == lab
        if int(comp.sum()) < floor_area:
            keep_single[comp] = True
            info['skipped_small'] += 1
        else:
            large_seed[comp] = True
    work = large_seed
    gap = int(close_px)
    if gap >= 1 and work.any():
        k = gap * 2 + 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(
            work.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kern) > 0
        # Close may only grow over lawn, and must not swallow protected singles.
        work = closed & allowed & ~keep_single
        work |= large_seed
    hole_cap = int(hole_max_px)
    if hole_cap <= 0 and gap >= 1:
        hole_cap = max(16, gap * gap)
    if hole_cap > 0 and work.any():
        work, hole_px = fill_enclosed_plant_holes(
            work, allowed & ~keep_single, max_hole_px=hole_cap)
        info['hole_filled_px'] += int(hole_px)
    win = max(3, int(window) | 1)
    half = win // 2
    weights = np.concatenate([
        np.arange(1, half + 1),
        np.array([half + 1]),
        np.arange(half, 0, -1),
    ]).astype(np.float64)
    weights /= weights.sum()
    n_pass = max(1, int(passes))
    band = max(1, int(band_px))
    band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 2 + 1, band * 2 + 1))
    out = work.copy()
    u8_all = np.ascontiguousarray(work.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(u8_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = abs(float(cv2.contourArea(cnt)))
        if area < floor_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        pad = band + 2
        y0 = max(0, y - pad)
        x0 = max(0, x - pad)
        y1 = min(base.shape[0], y + bh + pad)
        x1 = min(base.shape[1], x + bw + pad)
        sub = u8_all[y0:y1, x0:x1]
        local = cnt.reshape(-1, 2).astype(np.float64)
        local[:, 0] -= x0
        local[:, 1] -= y0
        canvas = np.zeros(sub.shape, dtype=np.uint8)
        if len(local) < win + 2:
            shifted = cnt.copy()
            shifted[:, 0, 0] -= x0
            shifted[:, 0, 1] -= y0
            cv2.drawContours(canvas, [shifted], -1, 255, thickness=-1)
        else:
            sm = local
            for _ in range(n_pass):
                pad_pts = np.concatenate([sm[-half:], sm, sm[:half]], axis=0)
                xs = np.convolve(pad_pts[:, 0], weights, mode='valid')
                ys = np.convolve(pad_pts[:, 1], weights, mode='valid')
                sm = np.stack([xs, ys], axis=1)
            poly = np.round(sm).astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(canvas, [poly], -1, 255, thickness=-1)
        filled = canvas > 0
        room = (cv2.dilate(sub, band_k) > 0) & allowed[y0:y1, x0:x1] & ~keep_single[y0:y1, x0:x1]
        out[y0:y1, x0:x1] |= filled & room
        info['smoothed_components'] += 1
    if hole_cap > 0 and out.any():
        out, hole_px = fill_enclosed_plant_holes(
            out, allowed & ~keep_single, max_hole_px=hole_cap)
        info['hole_filled_px'] += int(hole_px)
    if gap >= 1 and out.any():
        k = max(3, (gap | 1))
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(
            out.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kern) > 0
        out = (closed & allowed & ~keep_single) | out
    out |= keep_single
    out &= ~prot
    info['changed_px'] = int(np.count_nonzero(out != base))
    info['reason'] = 'accepted' if info['changed_px'] else 'no_net_change'
    return out, info


def outward_edge_circles(comp: np.ndarray, radius: float = 4.0) -> List[Center]:
    """One overlapping ring of circles along the outer contour.

    Centres sit on the boundary, one radius apart, so neighbouring disks
    overlap and the added rim is a sequence of circular arcs.
    """
    work = _as_bool(comp)
    rad = max(2, int(round(float(radius))))
    step = max(1, rad)
    u8 = np.ascontiguousarray(work.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    centres: List[Center] = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) == 0:
            continue
        for i in range(0, len(pts), step):
            x = int(pts[i, 0])
            y = int(pts[i, 1])
            centres.append((y, x, float(rad)))
    return centres


def dynamic_pack_component(
    mask: np.ndarray,
    *,
    multi: bool,
    min_radius: float = 2.0,
    max_circles: int = 96,
) -> List[Center]:
    """Unused by the reported vegetation step. Centres come from manuscript_centres.

    D is the Euclidean distance to the vegetation boundary. The centre is
    a maximum of D and the radius is max(1, floor(D)).
    A small region keeps that one tree. A large region repeats on the
    uncovered remainder (dynamic packing) while the next crown is still
    at least 3 px, so the patch becomes a cluster of decreasing circles
    instead of a refilled silhouette. A large region that is itself
    thinner than 3 px is a dispersed row: local maxima at least 8 px
    apart, not a solid strip. The caller draws Disk(centre, r_eff).
    """
    work = _as_bool(mask).copy()
    if not work.any():
        return []
    crown_r = max(3.0, float(min_radius))

    if not multi:
        return [_edt_peak(work)]

    dt0 = cv2.distanceTransform(work.astype(np.uint8), cv2.DIST_L2, 5)
    if float(dt0.max()) <= 0:
        return []
    if float(dt0.max()) < crown_r:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        peaks = (dt0 >= cv2.dilate(dt0, kernel) - 1e-6) & work
        ys, xs = np.where(peaks)
        if len(ys) == 0:
            return [_edt_peak(work)]
        order = np.argsort(-dt0[ys, xs], kind='stable')
        picked: List[Center] = []
        for idx in order:
            y, x = int(ys[idx]), int(xs[idx])
            r = float(max(1, int(np.floor(float(dt0[y, x])))))
            if r < max(1.0, float(min_radius)):
                continue
            if any((y - cy) ** 2 + (x - cx) ** 2 < 8.0 ** 2 for cy, cx, _cr in picked):
                continue
            picked.append((y, x, r))
            if len(picked) >= int(max_circles):
                break
        return picked or [_edt_peak(work)]

    centres: List[Center] = []
    remain = work
    for _ in range(int(max_circles)):
        if int(remain.sum()) < 4:
            break
        dt = cv2.distanceTransform(remain.astype(np.uint8), cv2.DIST_L2, 5)
        if float(dt.max()) < crown_r:
            break
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        r = float(max(1, int(np.floor(float(dt[y, x])))))
        if r < crown_r:
            break
        centres.append((int(y), int(x), r))
        remain[disk_mask_raw(remain.shape, int(y), int(x), r)] = False
    return centres


def inscribed_disk_mask(shape: Tuple[int, int], y: int, x: int, r: float, clip: np.ndarray) -> np.ndarray:
    """Pixels of the inscribed disk that also lie inside clip (strict mask containment)."""
    h, w = shape
    rad = max(0.0, float(r))
    if rad <= 0:
        out = np.zeros(shape, dtype=bool)
        if 0 <= y < h and 0 <= x < w and clip[y, x]:
            out[y, x] = True
        return out
    y0 = max(0, int(np.floor(y - rad)))
    y1 = min(h, int(np.ceil(y + rad)) + 1)
    x0 = max(0, int(np.floor(x - rad)))
    x1 = min(w, int(np.ceil(x + rad)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (yy - y) * (yy - y) + (xx - x) * (xx - x) <= rad * rad
    local = np.zeros(shape, dtype=bool)
    local[y0:y1, x0:x1] = disk
    return local & clip


def disk_mask_raw(shape: Tuple[int, int], y: int, x: int, r: float) -> np.ndarray:
    """Full disk without vegetation clip (used with road-relaxed soft ground)."""
    h, w = shape
    rad = max(0.0, float(r))
    out = np.zeros(shape, dtype=bool)
    if rad <= 0:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = True
        return out
    y0 = max(0, int(np.floor(y - rad)))
    y1 = min(h, int(np.ceil(y + rad)) + 1)
    x0 = max(0, int(np.floor(x - rad)))
    x1 = min(w, int(np.ceil(x + rad)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = (yy - y) * (yy - y) + (xx - x) * (xx - x) <= rad * rad
    return out


def soft_clip_for_disks(
    plant_mask: np.ndarray,
    blocking_mask: np.ndarray,
    *,
    road_relax_px: float = 0.0,
) -> np.ndarray:
    """Clip for circular cover: plant ∪ limited dilation, never into blockers.

    Paper: limited geometric relaxation near road edges while remaining
    topologically separated from the road system (``r_eff`` soft band).
    """
    plant = _as_bool(plant_mask)
    blocked = _as_bool(blocking_mask)
    soft = plant & ~blocked
    r = float(road_relax_px)
    if r >= 1.0 and plant.any():
        k = max(3, int(round(r)) * 2 + 1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dil = cv2.dilate(plant.astype(np.uint8), kern) > 0
        soft = (plant | dil) & ~blocked
    return soft


def _stamp_inscribed(shape: Tuple[int, int], centres: Sequence[Center], clip: np.ndarray) -> np.ndarray:
    cover = np.zeros(shape, dtype=bool)
    for y, x, r in centres:
        cover |= inscribed_disk_mask(shape, y, x, r, clip)
    return cover


def _stamp_relaxed(
    shape: Tuple[int, int],
    centres: Sequence[Center],
    plant_clip: np.ndarray,
    soft_clip: np.ndarray,
) -> np.ndarray:
    """Stamp disks: core inscribed in plant; rim may use soft_clip (road relax)."""
    cover = np.zeros(shape, dtype=bool)
    for y, x, r in centres:
        core = inscribed_disk_mask(shape, y, x, r, plant_clip)
        if soft_clip is plant_clip or np.array_equal(soft_clip, plant_clip):
            cover |= core
            continue
        full = disk_mask_raw(shape, y, x, r) & soft_clip
        cover |= core | full
    return cover


def clump_vegetation_mask(green_mask, blocking_mask, clump_close_px: float = 0.0) -> np.ndarray:
    """Optional morph-close. Default off: do not expand vegetation into other classes."""
    green = _as_bool(green_mask)
    blocked = _as_bool(blocking_mask)
    work = green & ~blocked
    if not work.any() or clump_close_px is None or float(clump_close_px) <= 1:
        return work
    k = int(round(float(clump_close_px))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(work.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0
    # Still cannot enter blockers (pavement/road/building/water/fixed).
    return closed & ~blocked


def _component_stats(mask: np.ndarray) -> List[Dict]:
    """Per-CC area / bbox aspect / compactness proxies (not ecological identity)."""
    m = _as_bool(mask).astype(np.uint8)
    if not m.any():
        return []
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        aspect = max(w, h) / max(1, min(w, h))
        # Compactness proxy: area / bbox area
        compact = area / max(1, w * h)
        out.append(dict(label=lab, area=area, width=w, height=h,
                        aspect=float(aspect), compactness=float(compact)))
    return out


def pack_vegetation_circles(
    green_mask,
    blocking_mask,
    *,
    min_center_dist_px: float = 16.0,
    large_radius_min_px: float = 8.0,
    small_min_center_dist_px: float = 8.0,
    small_radius_min_px: float = 2.0,
    radius_cap_px: Optional[float] = None,
    clump_close_px: float = 0.0,
    large_component_area_px: float = 40.0,
    noise_area_max_px: float = 6.0,
    thin_max_width_px: float = 2.0,
    noise_aspect_min: float = 3.0,
    noise_compact_max: float = 0.35,
    road_relax_px: float = 2.0,
) -> Dict:
    """Plant components become a union of inscribed circles.

    Micro noise and isolated thin strips become lawn. A component below
    large_component_area_px keeps one centre. A larger component repeats
    centres on the uncovered remainder. The drawn radius is the piecewise
    value from manuscript_centres. min_center_dist_px, large_radius_min_px
    and small_min_center_dist_px are not read here.
    """
    work = clump_vegetation_mask(green_mask, blocking_mask, clump_close_px)
    large_all: List[Center] = []
    small_all: List[Center] = []
    cover_large = np.zeros(work.shape, dtype=bool)
    cover_small = np.zeros(work.shape, dtype=bool)
    absorb_to_lawn = np.zeros(work.shape, dtype=bool)
    scale_log = []
    branch_counts = dict(noise=0, small_unit=0, cluster=0, thin_deleted=0)

    if not work.any():
        return dict(
            large_centres=[], small_centres=[],
            cover_large=cover_large, cover_small=cover_small,
            work_mask=work, remain_after_large_px=0, uncovered_after_both_px=0,
            clump_close_px=float(clump_close_px or 0.0),
            scale_log=scale_log,
            large_component_area_px=float(large_component_area_px),
            noise_area_max_px=float(noise_area_max_px),
            thin_max_width_px=float(thin_max_width_px),
            absorb_to_lawn=absorb_to_lawn,
            branch_counts=branch_counts,
        )

    num, labels = cv2.connectedComponents(work.astype(np.uint8), connectivity=8)
    thr = float(large_component_area_px)
    noise_thr = float(noise_area_max_px)
    thin_w = float(thin_max_width_px)
    aspect_min = float(noise_aspect_min)
    compact_max = float(noise_compact_max)
    # Dilated plant for adjacency of noise fragments to larger vegetation.
    plant_u8 = work.astype(np.uint8)
    dil_plant = cv2.dilate(plant_u8, np.ones((3, 3), np.uint8))
    blocked_cc = _as_bool(blocking_mask)
    dist_block = None
    if blocked_cc.any():
        dist_block = cv2.distanceTransform(
            np.where(blocked_cc, np.uint8(0), np.uint8(255)), cv2.DIST_L2, 5)

    def _stamp(centres, plant_clip):
        soft = soft_clip_for_disks(
            plant_clip, blocking_mask, road_relax_px=float(road_relax_px))
        return _stamp_relaxed(work.shape, centres, plant_clip, soft)

    for lab in range(1, num):
        comp = labels == lab
        area = int(comp.sum())
        ys, xs = np.where(comp)
        w = int(xs.max() - xs.min() + 1) if len(xs) else 0
        h = int(ys.max() - ys.min() + 1) if len(ys) else 0
        aspect = max(w, h) / max(1, min(w, h))
        compact = area / max(1, w * h)
        dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
        width_est = float(2.0 * dt.max()) if area > 0 else 0.0
        entry = dict(
            label=lab, area=area, aspect=float(aspect), compactness=float(compact),
            width_est=width_est, mode='', branch='', object_id=f'plant_cc_{lab}',
            object_type='')

        other = (dil_plant > 0) & ~comp & work
        touches_other = bool(np.any(comp & cv2.dilate(
            other.astype(np.uint8), np.ones((3, 3), np.uint8))))

        # Independent thin strip (any area below cluster split): delete to lawn.
        is_thin_independent = (
            (not touches_other) and area < thr and thin_w > 0
            and width_est <= thin_w and area > 0)

        # --- Branch 1: micro noise (area trigger + shape/adjacency) ---
        if area <= noise_thr or is_thin_independent:
            looks_noise = (
                area <= 2 or (compact < compact_max) or (aspect >= aspect_min)
                or touches_other or is_thin_independent)
            if looks_noise or is_thin_independent:
                entry['branch'] = 'noise'
                entry['object_type'] = (
                    'thin_independent' if is_thin_independent else 'micro_noise')
                if touches_other and not is_thin_independent:
                    entry['mode'] = 'noise_merge_into_plant'
                    entry['disposition'] = 'merge_adjacent_plant'
                else:
                    # Isolated crumb / thin strip → absorb to lawn (clears edge junk).
                    entry['mode'] = (
                        'thin_absorb_to_lawn' if is_thin_independent
                        else 'noise_absorb_to_lawn')
                    entry['disposition'] = 'absorb_lawn'
                    absorb_to_lawn |= comp
                    if is_thin_independent:
                        branch_counts['thin_deleted'] += 1
                branch_counts['noise'] += 1
                scale_log.append(entry)
                continue
            entry['mode'] = 'noise_ambiguous_single_tree'
            entry['branch'] = 'small_unit'
            entry['object_type'] = 'ambiguous_micro_or_planter'
            entry['disposition'] = 'single_circle'
            branch_counts['small_unit'] += 1
            paper = manuscript_centres(
                comp, multi=False, dist_block=dist_block,
                road_relax_px=float(road_relax_px))
            entry['paper_centres'] = paper
            entry['paper_n'] = len(paper)
            small = [(int(c['y']), int(c['x']), float(c['r_eff'])) for c in paper]
            small_all.extend(small)
            for c in paper:
                cover_small |= inscribed_disk_mask(
                    work.shape, int(c['y']), int(c['x']), float(c['r_inscribed']), comp)
            entry['n_centres'] = len(small)
            scale_log.append(entry)
            continue

        if area < thr:
            # Small region → one inscribed circle. Uncovered crown is dropped later.
            entry['mode'] = 'small_unit_single_tree'
            entry['branch'] = 'small_unit'
            entry['object_type'] = 'individual_tree'
            entry['disposition'] = 'single_circle'
            branch_counts['small_unit'] += 1
            paper = manuscript_centres(
                comp, multi=False, dist_block=dist_block,
                road_relax_px=float(road_relax_px))
            entry['paper_centres'] = paper
            entry['paper_n'] = len(paper)
            small = [(int(c['y']), int(c['x']), float(c['r_eff'])) for c in paper]
            small_all.extend(small)
            for c in paper:
                cover_small |= inscribed_disk_mask(
                    work.shape, int(c['y']), int(c['x']), float(c['r_inscribed']), comp)
            entry['n_centres'] = len(small)
        else:
            # Large region → every local-maximum disk, repeated on the
            # remainder so the group is filled by overlapping circles.
            entry['mode'] = 'cluster_circle_union'
            entry['branch'] = 'cluster'
            entry['object_type'] = 'cluster_planting_unit'
            entry['disposition'] = 'circle_union'
            branch_counts['cluster'] += 1
            paper = manuscript_centres(
                comp, multi=True, dist_block=dist_block,
                road_relax_px=float(road_relax_px),
                min_radius=float(small_radius_min_px))
            entry['paper_centres'] = paper
            entry['paper_n'] = len(paper)
            large = [(int(c['y']), int(c['x']), float(c['r_eff'])) for c in paper]
            large_all.extend(large)
            for c in paper:
                cover_large |= inscribed_disk_mask(
                    work.shape, int(c['y']), int(c['x']), float(c['r_inscribed']), comp)
            entry['n_large_centres'] = len(large)
            entry['n_small_centres'] = 0
            entry['centres'] = [
                dict(y=int(c['y']), x=int(c['x']),
                     r=float(c['r_inscribed']), r_eff=float(c['r_eff']), kind='cluster')
                for c in paper
            ]
        scale_log.append(entry)

    # Touching components are recorded only. Centres stay those already
    # chosen on each component; this step does not draw another set of disks.
    group_link_px = 4
    groups = []
    keep_for_group = work & ~absorb_to_lawn
    if keep_for_group.any():
        k = max(3, int(group_link_px) * 2 + 1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # Dilate only inside ~blocking already enforced by work; don't expand domain.
        dil = cv2.dilate(keep_for_group.astype(np.uint8), kern) > 0
        dil &= ~_as_bool(blocking_mask)
        gnum, glab = cv2.connectedComponents(dil.astype(np.uint8), connectivity=8)
        # Map original CC labels → group
        cc_to_group = {}
        for lab in range(1, num):
            comp = (labels == lab) & keep_for_group
            if not comp.any():
                continue
            gids = glab[comp]
            gids = gids[gids > 0]
            if len(gids) == 0:
                continue
            gid = int(np.bincount(gids).argmax())
            cc_to_group[lab] = gid
        from collections import defaultdict
        members = defaultdict(list)
        for lab, gid in cc_to_group.items():
            members[gid].append(lab)
        for gid, labs in members.items():
            union = np.zeros_like(work)
            for lab in labs:
                union |= labels == lab
            union &= keep_for_group
            area = int(union.sum())
            n_members = len(labs)
            gentry = dict(
                group_id=f'tree_group_{gid}', member_ccs=labs,
                n_members=n_members, area=area, mode='')
            if n_members >= 2:
                gentry['mode'] = 'grouped_already_packed'
                gentry['n_large_centres'] = 0
                gentry['n_small_centres'] = 0
                gentry['centres'] = []
            else:
                gentry['mode'] = 'singleton_keep'
            groups.append(gentry)
            for lab in labs:
                for e in scale_log:
                    if e.get('label') == lab:
                        e['group_id'] = gentry['group_id']
                        e['group_mode'] = gentry['mode']

    remain = work & ~cover_large & ~absorb_to_lawn
    return dict(
        large_centres=large_all,
        small_centres=small_all,
        cover_large=cover_large,
        cover_small=cover_small,
        work_mask=work,
        remain_after_large_px=int(remain.sum()),
        uncovered_after_both_px=int((remain & ~cover_small).sum()),
        clump_close_px=float(clump_close_px or 0.0),
        scale_log=scale_log,
        large_component_area_px=float(large_component_area_px),
        noise_area_max_px=float(noise_thr),
        absorb_to_lawn=absorb_to_lawn,
        branch_counts=branch_counts,
        tree_groups=groups,
    )


def estimate_isolated_plant_mean_diameter(
    plant_mask,
    lawn_mask,
    hard_mask,
    *,
    min_area_px: float = 6.0,
) -> Dict:
    """Mean crown diameter from isolated plant CCs that only touch lawn (not hard classes).

    A plant connected component is isolated if its 1-px exterior ring intersects
    lawn and/or plant only — never road / building / pavement / water / fixed.
    Patches with area ≤ ``min_area_px`` are ignored. Diameter = 2 · max EDT.
    """
    plant = _as_bool(plant_mask)
    lawn = _as_bool(lawn_mask)
    hard = _as_bool(hard_mask)
    empty = dict(
        mean_diameter_px=0.0, n_isolated=0, n_rejected_small=0, n_rejected_touch_hard=0,
        diameters=[],
    )
    if not plant.any():
        return empty
    num, labels = cv2.connectedComponents(plant.astype(np.uint8), connectivity=8)
    diameters: List[float] = []
    n_small = 0
    n_hard = 0
    kern = np.ones((3, 3), np.uint8)
    soft = lawn | plant
    for lab in range(1, num):
        comp = labels == lab
        area = int(comp.sum())
        if area <= float(min_area_px):
            n_small += 1
            continue
        dil = cv2.dilate(comp.astype(np.uint8), kern) > 0
        ring = dil & ~comp
        if ring.any() and np.any(ring & hard):
            n_hard += 1
            continue
        # Prefer patches that actually touch lawn (true “island in green”).
        if ring.any() and not np.any(ring & lawn):
            # Surrounded only by plant (interior of a larger blob after split) — skip.
            n_hard += 1
            continue
        if ring.any() and not np.any(ring & soft):
            n_hard += 1
            continue
        dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
        diameters.append(float(2.0 * dt.max()))
    if not diameters:
        return dict(
            mean_diameter_px=0.0, n_isolated=0, n_rejected_small=n_small,
            n_rejected_touch_hard=n_hard, diameters=[],
        )
    return dict(
        mean_diameter_px=float(np.mean(diameters)),
        n_isolated=len(diameters),
        n_rejected_small=n_small,
        n_rejected_touch_hard=n_hard,
        diameters=diameters,
    )


def restore_r1_plant(
    image,
    green_mask,
    orange_mask,
    blocking_mask,
    *,
    circle_packing_enabled: bool = False,
    clear_r1_plants_first: bool = False,
    noise_area_max_px: float = 6.0,
    min_isolated_area_px: Optional[float] = None,
    crown_jitter: float = 0.25,
    rng_seed: Optional[int] = None,
    road_relax_px: float = 2.0,
    diagnostics: Optional[dict] = None,
    **_unused,
):
    """Keep R1 plant domain; optional circles sized from isolated plant patches.

    When packing is on, crown diameter = mean diameter of plant CCs that touch
    only lawn (not road/building/pavement/water). Area floor defaults to
    ``min_isolated_area_px`` or ``max(noise_area_max_px, 40)`` so crumbs do not
    dominate the mean. Per-centre radius jitters by ±``crown_jitter``.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert('RGB')
    else:
        image = image.copy()
    arr = np.asarray(image, dtype=np.uint8).copy()
    blocked = _as_bool(blocking_mask)
    if orange_mask is not None:
        blocked = blocked | _as_bool(orange_mask)
    work = _as_bool(green_mask) & ~blocked
    packing_on = bool(circle_packing_enabled)
    centres: List[Center] = []
    disks = np.zeros(work.shape, dtype=bool)
    crown_diameter = 0.0
    crown_radius = 0.0
    cleared_px = 0
    plant_noise_removed_px = 0
    iso_stats: Dict = {}
    rng = np.random.default_rng(None if rng_seed is None else int(rng_seed))

    # Denoise: drop plant CCs with area ≤ noise_area_max_px → lawn.
    # Road (and other blockers) are never recolored — noise → lawn only on soft plant.
    noise_thr = float(noise_area_max_px)
    if work.any() and noise_thr >= 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            work.astype(np.uint8), connectivity=8)
        drop = np.zeros_like(work)
        for lab in range(1, num):
            if int(stats[lab, cv2.CC_STAT_AREA]) <= noise_thr:
                drop |= labels == lab
        drop = drop & work & ~blocked
        if drop.any():
            plant_noise_removed_px = int(drop.sum())
            work = work & ~drop
            arr[drop] = LAWN_RGB

    if packing_on:
        lawn = np.all(arr == LAWN_RGB, axis=2)
        hard = blocked | np.all(arr == (255, 255, 0), axis=2) | np.all(arr == (0, 255, 255), axis=2)
        if orange_mask is not None:
            hard = hard | _as_bool(orange_mask)
        hard = hard | np.all(arr == (255, 0, 255), axis=2)
        if min_isolated_area_px is None:
            min_iso = float(max(float(noise_area_max_px), 40.0))
        else:
            min_iso = float(min_isolated_area_px)
        iso_stats = estimate_isolated_plant_mean_diameter(
            work, lawn, hard, min_area_px=min_iso)
        iso_stats['min_isolated_area_px'] = min_iso
        target = float(iso_stats.get('mean_diameter_px') or 0.0)
        if target < 2.0 and work.any():
            # Fallback: mean inscribed diameter of plant CCs above the same floor.
            num, labels = cv2.connectedComponents(work.astype(np.uint8), connectivity=8)
            ds = []
            for lab in range(1, num):
                comp = labels == lab
                if int(comp.sum()) <= min_iso:
                    continue
                dt = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
                ds.append(float(2.0 * dt.max()))
            target = float(np.mean(ds)) if ds else 0.0
            iso_stats['fallback_all_cc_mean'] = target
        crown_radius = float(max(1, int(np.floor(target / 2.0))))
        crown_diameter = float(2.0 * crown_radius)
        jitter = max(0.0, float(crown_jitter))
        # Seed centres with base radius; then apply per-centre jitter.
        base = select_centers_min_dist(
            work,
            d_min=float(crown_diameter),
            radius_min=float(crown_radius),
            radius_max=float(crown_radius),
        )
        for y, x, _r in base:
            if jitter > 0:
                scale = float(rng.uniform(1.0 - jitter, 1.0 + jitter))
            else:
                scale = 1.0
            r = float(max(1, int(np.floor(crown_radius * scale))))
            # Keep disk inside local plant EDT.
            # (select already required dt>=crown_radius at seed; jittered r may need clip)
            centres.append((y, x, r))
        soft = soft_clip_for_disks(
            work, blocked, road_relax_px=float(road_relax_px))
        disks = _stamp_relaxed(work.shape, centres, work, soft) & ~blocked
        do_clear = bool(clear_r1_plants_first) and len(centres) > 0
        if do_clear and work.any():
            plant_here = np.all(arr == PLANT_RGB, axis=2) & work
            cleared_px = int(plant_here.sum())
            arr[plant_here] = LAWN_RGB
            if disks.any():
                arr[disks] = PLANT_RGB
            keep = np.zeros_like(work)
        else:
            if work.any():
                arr[work] = PLANT_RGB
            if disks.any():
                arr[disks] = PLANT_RGB
            keep = work
        canopy = (keep | disks) & ~blocked
        note = (
            f'keep_r1_plant + circles; crown_diameter={crown_diameter:.2f} '
            f'from isolated_plant_mean (n={iso_stats.get("n_isolated", 0)}); '
            f'jitter=±{jitter:.2f}; road_relax={float(road_relax_px):.1f}; '
            f'n_centres={len(centres)}; clear_first={do_clear}'
        )
    else:
        if work.any():
            arr[work] = PLANT_RGB
        canopy = work
        note = 'no circle packing; R1 plant restored on non-blocked pixels only'

    if diagnostics is not None:
        diagnostics.update(
            vegetation_method='keep_r1_plant',
            circle_packing_enabled=packing_on,
            crown_source='isolated_plant_mean_diameter' if packing_on else None,
            isolated_plant_n=iso_stats.get('n_isolated') if packing_on else None,
            isolated_rejected_small=iso_stats.get('n_rejected_small') if packing_on else None,
            isolated_rejected_touch_hard=iso_stats.get('n_rejected_touch_hard') if packing_on else None,
            min_isolated_area_px=iso_stats.get('min_isolated_area_px') if packing_on else None,
            crown_diameter_px=float(crown_diameter) if packing_on else None,
            crown_radius_px=float(crown_radius) if packing_on else None,
            crown_jitter=float(crown_jitter) if packing_on else None,
            n_large_centres=len(centres),
            n_small_centres=0,
            branch_counts={},
            tree_groups=[],
            n_tree_groups=0,
            n_grouped_multi_centre=0,
            group_bridge_plant_px=0,
            group_lawn_island_fill_px=0,
            noise_absorbed_to_lawn_px=int(plant_noise_removed_px),
            cleared_plant_to_lawn_px=int(cleared_px),
            disks_written_px=int(disks.sum()),
            canopy_symbol_px=int(canopy.sum()),
            centres=[{'y': int(y), 'x': int(x), 'r': float(r)} for y, x, r in centres],
            note=note + f'; plant_noise_removed_px={plant_noise_removed_px}',
        )
    return Image.fromarray(arr)


def fill_green_areas_with_circles(
    image,
    green_mask,
    orange_mask,
    blocking_mask,
    large_size_threshold=40,
    *,
    min_center_dist_px: float = 16.0,
    large_radius_min_px: float = 8.0,
    small_min_center_dist_px: float = 8.0,
    small_radius_min_px: float = 2.0,
    use_orange_radius_cap: bool = False,
    clump_close_px: float = 0.0,
    clear_r1_plants_first: bool = False,
    large_component_area_px: Optional[float] = None,
    noise_area_max_px: float = 6.0,
    circle_packing_enabled: bool = True,
    thin_max_width_px: float = 2.0,
    noise_aspect_min: float = 3.0,
    noise_compact_max: float = 0.35,
    max_lawn_island_px: float = 22.0,
    island_ring_frac: float = 0.8,
    group_close_px: float = 4.0,
    patch_enhance_pct: float = 20.0,
    road_relax_px: float = 2.0,
    circle_smooth_window: int = 5,
    circle_smooth_passes: int = 1,
    circle_smooth_band_px: int = 2,
    circle_smooth_close_px: int = 0,
    circle_smooth_min_area_px: int = 20,
    circle_smooth_hole_max_px: int = 0,
    diagnostics: Optional[dict] = None,
):
    """Inscribed-circle vegetation.

    R = max(1, floor(D)) at local maxima of distance-to-boundary. A large
    component repeats that step on the uncovered remainder; a smaller
    component keeps one centre. Near a blocker the drawn radius is
    min(R+rho, max(1, q)) when q < R+rho, and R otherwise. Disks do not
    enter road, building or pavement. Plant pixels outside every disk
    become lawn.
    """
    area_thr = float(
        large_component_area_px if large_component_area_px is not None else large_size_threshold)
    packing_on = bool(circle_packing_enabled)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert('RGB')
    else:
        image = image.copy()
    cap = float(estimate_orange_width(orange_mask)) if use_orange_radius_cap else None
    blocked = _as_bool(blocking_mask)
    if orange_mask is not None:
        blocked = blocked | _as_bool(orange_mask)

    packed = pack_vegetation_circles(
        green_mask, blocked,
        min_center_dist_px=min_center_dist_px,
        large_radius_min_px=large_radius_min_px,
        small_min_center_dist_px=small_min_center_dist_px,
        small_radius_min_px=small_radius_min_px,
        radius_cap_px=cap,
        clump_close_px=clump_close_px,
        large_component_area_px=area_thr,
        noise_area_max_px=float(noise_area_max_px),
        thin_max_width_px=float(thin_max_width_px),
        noise_aspect_min=float(noise_aspect_min),
        noise_compact_max=float(noise_compact_max),
        road_relax_px=float(road_relax_px),
    )
    arr = np.asarray(image, dtype=np.uint8).copy()
    work = packed['work_mask']
    cleared_px = 0
    # The painted plant is the union of disks. Crown pixels outside every
    # disk become lawn. Disks do not enter road, building or pavement.

    # Noise absorb: legal recolour plant→lawn inside work, never into blockers.
    absorb = packed.get('absorb_to_lawn')
    absorbed_px = 0
    if absorb is not None and np.any(absorb):
        dest = absorb & work & ~blocked
        absorbed_px = int(dest.sum())
        arr[dest] = LAWN_RGB

    group_bridge_px = 0
    island_fill_px = 0
    disks_written_px = 0
    circle_smooth_px = 0
    packing_ops_skipped = []

    if packing_on:
        centres = list(packed['large_centres']) + list(packed['small_centres'])
        disks = np.zeros(work.shape, dtype=bool)
        h, w = work.shape
        for y, x, r in centres:
            rad = max(1.0, float(r))
            y0 = max(0, int(np.floor(y - rad)))
            y1 = min(h, int(np.ceil(y + rad)) + 1)
            x0 = max(0, int(np.floor(x - rad)))
            x1 = min(w, int(np.ceil(x + rad)) + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            disks[y0:y1, x0:x1] |= (
                (yy - y) * (yy - y) + (xx - x) * (xx - x) <= rad * rad)
        keep = work & ~blocked
        if absorb is not None:
            keep = keep & ~absorb
        lawn = np.all(arr == np.asarray(LAWN_RGB, dtype=np.uint8), axis=2)
        disks = disks & ~blocked & (lawn | keep)
        smoothed, smooth_info = smooth_tree_circle_edges(
            disks, blocked, lawn,
            window=int(circle_smooth_window),
            passes=int(circle_smooth_passes),
            band_px=int(circle_smooth_band_px),
            close_px=int(circle_smooth_close_px),
            min_area_px=int(circle_smooth_min_area_px),
            hole_max_px=int(circle_smooth_hole_max_px))
        circle_smooth_px = int(smooth_info.get('changed_px') or 0)
        hole_px = int(smooth_info.get('hole_filled_px') or 0)
        disks = smoothed
        # Reclaim original plant that sits in the healed group or its 1 px rim.
        near = cv2.dilate(
            disks.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        disks = disks | (keep & near & ~blocked)
        uncovered = keep & ~disks
        cleared_px = int(uncovered.sum())
        if uncovered.any():
            arr[uncovered] = LAWN_RGB
        disks_written_px = int(disks.sum())
        group_bridge_px = 0
        island_fill_px = hole_px
        enhance_px = int((disks & ~keep).sum())
        if disks.any():
            arr[disks] = PLANT_RGB
        canopy = disks.copy()
        contrast_scope = 'paper_circle_union_healed_cluster'
    else:
        # OFF: shared noise absorb already applied; keep remaining plant masks.
        # Skip disk stamp + centre-dependent bridging (no alternate centres).
        packing_ops_skipped = [
            'disk_raster_write',
            'centre_dependent_group_bridge',
            'centre_dependent_lawn_island_fill',
            'optional_clump_close_on_disks',
            'mid_band_patch_enhance',
        ]
        enhance_px = 0
        keep_plant = work & ~blocked
        if absorb is not None:
            keep_plant = keep_plant & ~absorb
        if keep_plant.any():
            arr[keep_plant] = PLANT_RGB
        canopy = keep_plant.copy()
        contrast_scope = (
            'circle_packing_stage_and_dependent_operations'
            ' (disk write + centre-dependent bridge/island skipped; '
            'shared noise absorb + plant-mask keep retained)'
        )

    if diagnostics is not None:
        spill = int(((packed['cover_large'] | packed['cover_small']) & ~work).sum())
        road_hit = int((canopy & _as_bool(orange_mask)).sum()) if orange_mask is not None else 0
        diagnostics.update(
            vegetation_method='three_branch_noise_small_cluster_edt',
            circle_packing_enabled=packing_on,
            packing_contrast_scope=contrast_scope,
            packing_ops_skipped=packing_ops_skipped,
            disks_written_px=int(disks_written_px),
            circle_edge_smooth_px=int(circle_smooth_px),
            min_center_dist_px=float(min_center_dist_px),
            large_radius_min_px=float(large_radius_min_px),
            small_min_center_dist_px=float(small_min_center_dist_px),
            small_radius_min_px=float(small_radius_min_px),
            clump_close_px=float(clump_close_px or 0.0),
            clear_r1_plants_first=bool(clear_r1_plants_first),
            cleared_plant_to_lawn_px=cleared_px,
            noise_absorbed_to_lawn_px=absorbed_px,
            group_bridge_plant_px=group_bridge_px,
            group_lawn_island_fill_px=island_fill_px,
            mid_band_enhance_px=int(enhance_px),
            large_component_area_px=area_thr,
            noise_area_max_px=float(noise_area_max_px),
            thin_max_width_px=float(thin_max_width_px),
            patch_enhance_pct=float(patch_enhance_pct),
            max_lawn_island_px=float(max_lawn_island_px),
            island_ring_frac=float(island_ring_frac),
            large_size_threshold_applied=area_thr,
            n_large_centres=len(packed['large_centres']),
            n_small_centres=len(packed['small_centres']),
            branch_counts=packed.get('branch_counts'),
            remain_after_large_px=packed['remain_after_large_px'],
            uncovered_after_both_px=packed['uncovered_after_both_px'],
            inscribed_spill_px=spill,
            plant_on_road_px=road_hit,
            orange_radius_cap=cap,
            scale_log=packed.get('scale_log'),
            tree_groups=packed.get('tree_groups') if packing_on else [],
            n_tree_groups=(
                len(packed.get('tree_groups') or []) if packing_on else 0),
            n_grouped_multi_centre=(
                sum(1 for e in (packed.get('scale_log') or [])
                    if e.get('mode') == 'cluster_circle_union'
                    and int(e.get('paper_n') or 0) > 1)
                if packing_on else 0),
            canopy_symbol_px=int(canopy.sum()),
            note=('disk union written as plant; uncovered plant pixels become lawn'
                  if packing_on else
                  'packing OFF: shared absorb/keep only; no disk/bridge write'),
        )
    return Image.fromarray(arr)


def render_vegetation_groups_preview(
    base_rgb: np.ndarray,
    tree_groups: Optional[Sequence[dict]] = None,
    *,
    plant_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Symbol preview only: group outlines + centre marks. Not a formal label / metric."""
    out = np.asarray(base_rgb, dtype=np.uint8).copy()
    if not tree_groups:
        return out
    h, w = out.shape[:2]
    for g in tree_groups:
        centres = g.get('centres') or []
        mode = g.get('mode') or ''
        # Outline colour: multi-centre groups cyan-ish, singletons magenta-ish (preview only).
        col = (0, 220, 255) if mode == 'grouped_multi_centre' else (255, 128, 0)
        for c in centres:
            cy, cx = int(c.get('y', 0)), int(c.get('x', 0))
            r = max(1, int(round(float(c.get('r', 2)))))
            if 0 <= cy < h and 0 <= cx < w:
                cv2.circle(out, (cx, cy), r, col, 1, lineType=cv2.LINE_8)
                cv2.circle(out, (cx, cy), 1, (255, 255, 255), -1, lineType=cv2.LINE_8)
        # If member CCs known but no centres, mark centroid of first member via plant_mask skip.
        if not centres and plant_mask is not None and g.get('n_members', 0) >= 2:
            pass
    return out
