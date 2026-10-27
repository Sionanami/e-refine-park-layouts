# -*- coding: utf-8 -*-
"""High-quality vector extraction v2 from palette layout B.

Pipeline: upsample → color masks → morphology → skeleton prune/link →
building contour orthogonalize → plant circularity filter → downsample coords.
Select params on train only; report val once.
"""
from __future__ import annotations
import os

import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects, skeletonize

FIXED = Path(os.environ.get("SD15_MANIFEST", "manifests"))
OUT = Path(os.environ.get("SD15_VECTOR_OUT", "vector_out"))

ROAD = np.array([241, 145, 73], np.int16)
BUILDING = np.array([255, 0, 255], np.int16)
PLANT = np.array([0, 152, 67], np.int16)
WATER = np.array([0, 255, 255], np.int16)
PAVEMENT = np.array([255, 255, 0], np.int16)


@dataclass
class Params:
    scale: int = 1
    road_tol: int = 10
    road_close: int = 1
    road_open: int = 0
    spur_len: int = 3
    link_dist: int = 8
    link_cos: float = 0.15
    bld_tol: int = 18
    bld_min_area: int = 30
    poly_tol: float = 1.2
    plant_tol: int = 14
    plant_min_area: int = 10
    plant_max_area: int = 120
    plant_circ: float = 0.55
    water_tol: int = 14
    water_min_area: int = 30


def mask_tol(img: np.ndarray, rgb: np.ndarray, tol: int) -> np.ndarray:
    x = img.astype(np.int16)
    return np.all(np.abs(x - rgb) <= tol, axis=2)


def road_mask(img: np.ndarray, tol: int) -> np.ndarray:
    """Palette orange plus anti-aliased browns common on 256 layouts."""
    m = mask_tol(img, ROAD, tol)
    r = img[..., 0].astype(np.int16)
    g = img[..., 1].astype(np.int16)
    b = img[..., 2].astype(np.int16)
    heur = (r > 195) & (g > 95) & (g < 195) & (b < 130) & (r > g + 20) & (g > b + 10)
    return m | heur


def building_mask(img: np.ndarray, tol: int) -> np.ndarray:
    """Magenta buildings + AA pinks."""
    m = mask_tol(img, BUILDING, tol)
    r = img[..., 0].astype(np.int16)
    g = img[..., 1].astype(np.int16)
    b = img[..., 2].astype(np.int16)
    heur = (r > 180) & (b > 180) & (g < 120) & (r > g + 40) & (b > g + 40)
    return m | heur


def plant_mask(img: np.ndarray, tol: int) -> np.ndarray:
    # exact palette only — broad green heuristics merge lawn blobs
    return mask_tol(img, PLANT, tol)


def water_mask(img: np.ndarray, tol: int) -> np.ndarray:
    return mask_tol(img, WATER, tol)


def neighbor_count(skel: np.ndarray) -> np.ndarray:
    ker = np.ones((3, 3), np.uint8)
    return ndimage.convolve(skel.astype(np.uint8), ker, mode="constant", cval=0) - skel.astype(np.uint8)


def prune_spurs(skel: np.ndarray, max_len: int) -> np.ndarray:
    """Remove only short endpoint→junction branches (length-based, not peel-all)."""
    if max_len <= 0 or not skel.any():
        return skel.astype(bool)
    out = skel.astype(bool).copy()
    h, w = out.shape

    def nbrs(y, x):
        res = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx]:
                    res.append((ny, nx))
        return res

    changed = True
    while changed:
        changed = False
        n = neighbor_count(out)
        eps = list(zip(*np.where(out & (n == 1))))
        for ey, ex in eps:
            if not out[ey, ex]:
                continue
            path = [(ey, ex)]
            py, px = ey, ex
            # step inward
            cands = nbrs(ey, ex)
            if not cands:
                continue
            cy, cx = cands[0]
            while True:
                path.append((cy, cx))
                deg = len(nbrs(cy, cx))
                if deg != 2:
                    break
                nxts = [p for p in nbrs(cy, cx) if p != (py, px)]
                if not nxts:
                    break
                py, px = cy, cx
                cy, cx = nxts[0]
                if len(path) > max_len + 2:
                    break
            # path is endpoint ... junction/end; spur length = pixels before last
            if len(path) <= max_len + 1:
                # delete all but keep junction pixel
                for y, x in path[:-1]:
                    out[y, x] = False
                changed = True
    if out.sum() == 0:
        return skel.astype(bool)
    return out


def endpoint_list(skel: np.ndarray):
    n = neighbor_count(skel)
    return list(zip(*np.where((skel) & (n == 1))))


def local_dir(skel: np.ndarray, y: int, x: int, steps: int = 4):
    h, w = skel.shape
    path = [(y, x)]
    prev = None
    cy, cx = y, x
    for _ in range(steps):
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and (ny, nx) != prev:
                    nbrs.append((ny, nx))
        if not nbrs:
            break
        prev = (cy, cx)
        # prefer continuing into degree-2
        cy, cx = nbrs[0]
        path.append((cy, cx))
    if len(path) < 2:
        return np.array([0.0, 0.0])
    vy = path[0][0] - path[-1][0]
    vx = path[0][1] - path[-1][1]
    nrm = math.hypot(vx, vy) + 1e-6
    return np.array([vx / nrm, vy / nrm])


def link_endpoints(skel: np.ndarray, max_dist: int, min_cos: float) -> np.ndarray:
    out = skel.astype(bool).copy()
    eps = endpoint_list(out)
    if len(eps) < 2:
        return out
    dirs = {p: local_dir(out, p[0], p[1]) for p in eps}
    used = set()
    pairs = []
    for i, a in enumerate(eps):
        for b in eps[i + 1 :]:
            dy = a[0] - b[0]
            dx = a[1] - b[1]
            dist = math.hypot(dx, dy)
            if dist < 1.5 or dist > max_dist:
                continue
            # direction from a to b should align with a's outward? use opposite of inward dir
            va = dirs[a]
            vb = dirs[b]
            ab = np.array([b[1] - a[1], b[0] - a[0]], dtype=float)
            ab /= np.linalg.norm(ab) + 1e-6
            ba = -ab
            # endpoint direction points into skeleton; want connection opposite to that
            ca = float(np.dot(-va, ab))
            cb = float(np.dot(-vb, ba))
            if ca >= min_cos and cb >= min_cos:
                pairs.append((dist, a, b))
    pairs.sort()
    for dist, a, b in pairs:
        if a in used or b in used:
            continue
        rr, cc = _line(a[0], a[1], b[0], b[1])
        # avoid painting through dense non-road? already on skeleton canvas
        for y, x in zip(rr, cc):
            out[y, x] = True
        used.add(a)
        used.add(b)
    return out


def _line(y0, x0, y1, x1):
    from skimage.draw import line

    return line(int(y0), int(x0), int(y1), int(x1))


def chains_from_skel(skel: np.ndarray):
    """Trace degree-2 paths between junctions/endpoints into polylines."""
    s = skel.astype(bool)
    n = neighbor_count(s)
    nodes = set(zip(*np.where(s & ((n != 2)))))
    visited_edge = set()
    polylines = []
    h, w = s.shape
    max_len = h + w + 8

    def nbrs(y, x):
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and s[ny, nx]:
                    out.append((ny, nx))
        return out

    def walk(sy, sx, ny, nx):
        e0 = tuple(sorted(((sy, sx), (ny, nx))))
        if e0 in visited_edge:
            return None
        path = [(sy, sx), (ny, nx)]
        visited_edge.add(e0)
        py, px = sy, sx
        cy, cx = ny, nx
        while len(path) < max_len:
            if (cy, cx) in nodes and (cy, cx) != (sy, sx):
                break
            if (cy, cx) == (sy, sx) and len(path) > 2:
                break
            nxt = None
            for c in nbrs(cy, cx):
                if c == (py, px):
                    continue
                ee = tuple(sorted(((cy, cx), c)))
                if ee in visited_edge:
                    continue
                nxt = c
                break
            if nxt is None:
                break
            visited_edge.add(tuple(sorted(((cy, cx), nxt))))
            py, px = cy, cx
            cy, cx = nxt
            path.append((cy, cx))
        return path if len(path) >= 2 else None

    starts = list(nodes) if nodes else list(zip(*np.where(s)))
    for sy, sx in starts:
        for ny, nx in nbrs(sy, sx):
            path = walk(sy, sx, ny, nx)
            if path is not None:
                polylines.append(path)
    # leftover loops (all degree 2)
    ys, xs = np.where(s)
    for y, x in zip(ys.tolist(), xs.tolist()):
        for ny, nx in nbrs(y, x):
            path = walk(y, x, ny, nx)
            if path is not None and len(path) >= 8:
                polylines.append(path)
    return polylines


def _poly_length(pts_yx) -> float:
    if len(pts_yx) < 2:
        return 0.0
    t = 0.0
    for i in range(1, len(pts_yx)):
        t += math.hypot(pts_yx[i][0] - pts_yx[i - 1][0], pts_yx[i][1] - pts_yx[i - 1][1])
    return t


def simplify_poly(pts, tol):
    if len(pts) < 3:
        return pts
    # drop consecutive duplicates
    cleaned = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - cleaned[-1][0]) + abs(p[1] - cleaned[-1][1]) > 1e-6:
            cleaned.append(p)
    if len(cleaned) < 3:
        return cleaned
    arr = np.array([[p[1], p[0]] for p in cleaned], dtype=float)  # x,y
    simp = approximate_polygon(arr, tolerance=max(tol, 0.5))
    out = [(float(y), float(x)) for x, y in simp]
    # collapse tiny self-loops / oscillating remnants
    if len(out) > 4:
        uniq = {(round(p[0], 1), round(p[1], 1)) for p in out}
        if len(uniq) <= 3:
            return [out[0], out[len(out) // 2], out[-1]]
    return out


def orthogonalize(pts_yx, angle_snap_deg=20.0):
    if len(pts_yx) < 3:
        return pts_yx
    out = [pts_yx[0]]
    for i in range(1, len(pts_yx)):
        y0, x0 = out[-1]
        y1, x1 = pts_yx[i]
        dy, dx = y1 - y0, x1 - x0
        ang = abs(math.degrees(math.atan2(dy, dx))) % 90
        if ang < angle_snap_deg or ang > 90 - angle_snap_deg:
            if abs(dx) >= abs(dy):
                out.append((y0, x1))
            else:
                out.append((y1, x0))
        else:
            out.append((y1, x1))
    # drop near-duplicates after snap
    cleaned = [out[0]]
    for p in out[1:]:
        if math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > 0.5:
            cleaned.append(p)
    return cleaned


def building_rect_from_mask(comp: np.ndarray):
    """Min-area rectangle; snap to axis-aligned when nearly orthogonal (palette B)."""
    ys, xs = np.where(comp)
    if len(xs) < 4:
        return None
    # default axis-aligned box (palette buildings are usually AA)
    y0, y1 = float(ys.min()) - 0.5, float(ys.max()) + 0.5
    x0, x1 = float(xs.min()) - 0.5, float(xs.max()) + 0.5
    aa = [(y0, x0), (y0, x1), (y1, x1), (y1, x0), (y0, x0)]
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    mean = coords.mean(axis=0)
    c = coords - mean
    cov = (c.T @ c) / max(len(c), 1)
    try:
        evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return aa
    # principal axis angle to nearest axis
    vx, vy = float(evecs[0, 1]), float(evecs[1, 1])
    ang = abs(math.degrees(math.atan2(vy, vx))) % 90
    tilt = min(ang, 90 - ang)
    if tilt < 12.0:
        return aa
    proj = c @ evecs
    mn = proj.min(axis=0)
    mx = proj.max(axis=0)
    corners_local = np.array(
        [
            [mn[0], mn[1]],
            [mx[0], mn[1]],
            [mx[0], mx[1]],
            [mn[0], mx[1]],
            [mn[0], mn[1]],
        ]
    )
    corners = corners_local @ evecs.T + mean
    return [(float(y), float(x)) for x, y in corners]


def circularity(mask_comp: np.ndarray) -> float:
    from skimage.measure import perimeter

    area = float(mask_comp.sum())
    if area <= 0:
        return 0.0
    peri = float(perimeter(mask_comp.astype(np.uint8), neighborhood=4))
    peri = max(peri, 1.0)
    return float(min(1.0, 4 * math.pi * area / (peri * peri)))


def plant_ok(mask_comp: np.ndarray, min_circ: float) -> tuple[bool, float]:
    from skimage.measure import regionprops

    circ = circularity(mask_comp)
    lab = mask_comp.astype(np.uint8)
    props = regionprops(lab)
    if not props:
        return False, circ
    p = props[0]
    # prefer compact near-circular blobs
    if circ < min_circ:
        return False, circ
    if p.eccentricity > 0.82:
        return False, circ
    if p.solidity < 0.55:
        return False, circ
    return True, circ


def extract_one(img0: np.ndarray, p: Params, full: bool = True) -> dict:
    scale = max(1, int(p.scale))
    if scale != 1:
        im = Image.fromarray(img0).resize((img0.shape[1] * scale, img0.shape[0] * scale), Image.NEAREST)
        img = np.array(im)
    else:
        img = img0

    road = road_mask(img, p.road_tol)
    if p.road_close > 0:
        road = binary_closing(road, footprint=disk(p.road_close))
    if p.road_open > 0:
        road = binary_opening(road, footprint=disk(p.road_open))
    road = remove_small_objects(road.astype(bool), min_size=8)

    skel = skeletonize(road).astype(bool) if road.any() else np.zeros_like(road, dtype=bool)
    if skel.any() and p.spur_len > 0:
        skel = prune_spurs(skel, p.spur_len)
    if skel.any() and p.link_dist > 0:
        skel = link_endpoints(skel, p.link_dist, p.link_cos)
        skel = skeletonize(skel).astype(bool)

    roads = []
    if full and skel.any():
        min_road_len = 4.0 * scale
        for ch in chains_from_skel(skel):
            if _poly_length(ch) < min_road_len:
                continue
            simp = simplify_poly(ch, tol=max(1.0, p.poly_tol * scale / 2))
            if len(simp) < 2:
                continue
            if _poly_length(simp) < min_road_len * 0.75:
                continue
            roads.append([[pt[1] / scale, pt[0] / scale] for pt in simp])

    bld = building_mask(img, p.bld_tol)
    bld = binary_closing(bld, footprint=disk(1))
    bld = remove_small_objects(bld, min_size=max(4, int(p.bld_min_area * scale * scale // 2)))
    buildings = []
    lab, n_b = ndimage.label(bld.astype(np.uint8))
    if full:
        for i in range(1, n_b + 1):
            comp = lab == i
            area = int(comp.sum())
            if area < p.bld_min_area * (scale**2) / 2:
                continue
            pts = building_rect_from_mask(comp)
            if pts is None or len(pts) < 4:
                pad = np.pad(comp.astype(float), 1, mode="constant")
                cons = find_contours(pad, 0.5)
                if not cons:
                    continue
                cons.sort(key=lambda c: -len(c))
                c = cons[0] - 1.0
                pts = [(float(y), float(x)) for y, x in c]
                pts = simplify_poly(pts, tol=p.poly_tol * scale)
                pts = orthogonalize(pts)
            if len(pts) < 3:
                continue
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            ys = [q[0] for q in pts]
            xs = [q[1] for q in pts]
            buildings.append(
                {
                    "polygon_xy": [[x / scale, y / scale] for y, x in pts],
                    "center_xy": [float(np.mean(xs) / scale), float(np.mean(ys) / scale)],
                    "area_px": area / (scale**2),
                }
            )
    else:
        buildings = [{"area_px": 1} for _ in range(n_b)]  # count only

    plant = plant_mask(img, p.plant_tol)
    plants = []
    lab, n_p = ndimage.label(plant.astype(np.uint8))
    min_a = p.plant_min_area * (scale**2) / 2
    max_a = p.plant_max_area * (scale**2)
    for i in range(1, n_p + 1):
        comp = lab == i
        area = int(comp.sum())
        if area < min_a or area > max_a:
            continue
        ok, circ = plant_ok(comp, p.plant_circ)
        if not ok:
            continue
        ys, xs = np.where(comp)
        plants.append(
            {
                "center_xy": [float(xs.mean() / scale), float(ys.mean() / scale)],
                "area_px": area / (scale**2),
                "circularity": circ,
            }
        )
    plants = _nms_points(plants, min_dist=3.0 * scale / 2)

    waters = []
    if full:
        water = water_mask(img, p.water_tol)
        water = binary_closing(water, footprint=disk(2))
        water = remove_small_objects(water, min_size=max(8, int(p.water_min_area * scale * scale / 2)))
        lab, n_w = ndimage.label(water.astype(np.uint8))
        for i in range(1, n_w + 1):
            comp = lab == i
            pad = np.pad(comp.astype(float), 1, mode="constant")
            cons = find_contours(pad, 0.5)
            if not cons:
                continue
            cons.sort(key=lambda c: -len(c))
            c = cons[0] - 1.0
            pts = simplify_poly([(float(y), float(x)) for y, x in c], tol=p.poly_tol * scale)
            if len(pts) < 3:
                continue
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            waters.append({"polygon_xy": [[x / scale, y / scale] for y, x in pts], "area_px": float(comp.sum() / scale**2)})

    if road.any() and skel.any():
        dist = ndimage.distance_transform_edt(~skel)
        half_w = ndimage.distance_transform_edt(road)
        rad = np.maximum(2.0 * scale, 0.85 * half_w + 1.0)
        cover = float(((dist <= rad) & road).sum() / road.sum())
    else:
        cover = 0.0
    ep = endpoint_list(skel)
    skel_px_orig = float(skel.sum()) / float(scale)
    if scale != 1:
        h0, w0 = img0.shape[:2]
        sh = np.zeros((h0, w0), dtype=bool)
        yy, xx = np.where(skel)
        sh[(yy // scale).clip(0, h0 - 1), (xx // scale).clip(0, w0 - 1)] = True
        skel_ds = sh
    else:
        skel_ds = skel

    n_bld = len(buildings) if full else n_b
    return {
        "roads": roads,
        "buildings": buildings if full else [],
        "plants": plants,
        "waters": waters,
        "metrics": {
            "skel_px": int(round(skel_px_orig)),
            "skel_endpoints": len(ep),
            "n_road_lines": len(roads) if full else -1,
            "n_buildings": n_bld,
            "n_plants": len(plants),
            "n_waters": len(waters),
            "road_cover": cover,
            "ep_per_100_skel": (100.0 * len(ep) / max(float(skel.sum()), 1.0)),
            "mean_plant_circ": float(np.mean([q["circularity"] for q in plants])) if plants else 0.0,
            "has_road": skel_px_orig >= 20,
            "has_building": n_bld >= 1,
        },
        "skel": skel_ds.astype(np.uint8),
    }


def _nms_points(plants, min_dist: float):
    plants = sorted(plants, key=lambda q: -q["circularity"])
    kept = []
    for p in plants:
        ok = True
        for k in kept:
            dx = p["center_xy"][0] - k["center_xy"][0]
            dy = p["center_xy"][1] - k["center_xy"][1]
            if math.hypot(dx, dy) < min_dist:
                ok = False
                break
        if ok:
            kept.append(p)
    return kept


def score_train(rows_metrics: list[dict]) -> float:
    """Higher better. Prefer coverage, fewer spurs, presence of elements."""
    cover = np.mean([m["road_cover"] for m in rows_metrics])
    ep100 = np.mean([m["ep_per_100_skel"] for m in rows_metrics])
    road_rate = np.mean([m["has_road"] for m in rows_metrics])
    bld_rate = np.mean([m["has_building"] for m in rows_metrics])
    plant_n = np.median([m["n_plants"] for m in rows_metrics])
    # plant count: prefer 15..80
    plant_pen = 0.0 if 15 <= plant_n <= 80 else abs(plant_n - 40) / 40.0
    # prefer high cover, element presence, and cleaner skeletons
    return float(3.0 * cover + 1.2 * road_rate + 1.2 * bld_rate - 0.05 * ep100 - 0.3 * plant_pen)


def load_split(name: str):
    return list(csv.DictReader((FIXED / f"{name}.csv").open(encoding="utf-8")))


def overlay(img: np.ndarray, feat: dict) -> Image.Image:
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for line in feat["roads"]:
        if len(line) >= 2:
            d.line([(p[0], p[1]) for p in line], fill=(255, 60, 0, 255), width=2)
    for b in feat["buildings"]:
        poly = [(p[0], p[1]) for p in b["polygon_xy"]]
        if len(poly) >= 3:
            d.line(poly, fill=(255, 0, 255, 255), width=2)
            cx, cy = b["center_xy"]
            d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 0, 255, 255))
    for p in feat["plants"]:
        cx, cy = p["center_xy"]
        d.ellipse([cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5], outline=(0, 255, 80, 255), width=1)
        d.point((cx, cy), fill=(0, 255, 80, 255))
    for w in feat["waters"]:
        poly = [(p[0], p[1]) for p in w["polygon_xy"]]
        if len(poly) >= 3:
            d.line(poly, fill=(0, 180, 255, 255), width=2)
    return Image.alpha_composite(im, ov).convert("RGB")


def eval_split(rows, params: Params, max_n: int = 0, full: bool = False):
    metrics = []
    feats = []
    use = rows if max_n <= 0 else rows[:max_n]
    for r in use:
        img = np.array(Image.open(r["B_path"]).convert("RGB"))
        feat = extract_one(img, params, full=full)
        m = dict(feat["metrics"])
        m["case_id"] = r["case_id"]
        metrics.append(m)
        if full:
            feats.append((r["case_id"], img, feat))
    return metrics, feats


def main():
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    train = load_split("train")
    val = load_split("val")
    export_only = "--export-only" in sys.argv

    if export_only and (OUT / "BEST_PARAMS.json").exists():
        summary0 = json.loads((OUT / "BEST_PARAMS.json").read_text(encoding="utf-8"))
        params = Params(**summary0["params"])
        print("export-only with frozen params", asdict(params), flush=True)
        print("exporting full vectors...", flush=True)
        val_mets, val_feats = eval_split(val, params, full=True)
        train_mets_full, train_feats = eval_split(train, params, full=True)
        vec_dir = OUT / "vectors"
        ov_dir = OUT / "overlays"
        vec_dir.mkdir(parents=True, exist_ok=True)
        ov_dir.mkdir(parents=True, exist_ok=True)

        def dump_feats(feats, split):
            for cid, img, feat in feats:
                payload = {
                    "case_id": cid,
                    "split": split,
                    "params": asdict(params),
                    "roads": feat["roads"],
                    "buildings": feat["buildings"],
                    "plants": [{k: v for k, v in p.items()} for p in feat["plants"]],
                    "waters": feat["waters"],
                    "metrics": feat["metrics"],
                }
                (vec_dir / f"{split}_{cid}.json").write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                overlay(img, feat).save(ov_dir / f"{split}_{cid}.png")

        dump_feats(val_feats, "val")
        dump_feats(train_feats, "train")
        thumbs = [overlay(img, feat) for cid, img, feat in val_feats]
        if thumbs:
            w = sum(t.width for t in thumbs[:5]) + 4 * 4
            h = max(t.height for t in thumbs[:5])
            sheet = Image.new("RGB", (w, h), (30, 30, 30))
            x = 0
            for t in thumbs[:5]:
                sheet.paste(t, (x, 0))
                x += t.width + 4
            sheet.save(OUT / "VAL_OVERVIEW.png")

        def agg(mets):
            return {
                "n": len(mets),
                "road_cover_mean": float(np.mean([m["road_cover"] for m in mets])),
                "ep100_mean": float(np.mean([m["ep_per_100_skel"] for m in mets])),
                "ep_median": float(np.median([m["skel_endpoints"] for m in mets])),
                "skel_median": float(np.median([m["skel_px"] for m in mets])),
                "road_rate": float(np.mean([m["has_road"] for m in mets])),
                "bld_rate": float(np.mean([m["has_building"] for m in mets])),
                "plant_median": float(np.median([m["n_plants"] for m in mets])),
                "building_median": float(np.median([m["n_buildings"] for m in mets])),
                "water_median": float(np.median([m["n_waters"] for m in mets])),
                "mean_plant_circ": float(np.mean([m["mean_plant_circ"] for m in mets])),
                "road_lines_median": float(np.median([m["n_road_lines"] for m in mets])),
            }

        summary = {
            "params": asdict(params),
            "train_score": summary0.get("train_score"),
            "gates": summary0.get("gates"),
            "gates_passed": summary0.get("gates_passed", True),
            "train": agg(train_mets_full),
            "val": agg(val_mets),
            "v1_reference": summary0.get("v1_reference"),
        }
        (OUT / "BEST_PARAMS.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("DONE export-only train", summary["train"], "val", summary["val"], flush=True)
        return

    # coarse grid on a train subset for speed, then refine on full train top candidates
    subset = train[::5]  # ~18 cases for coarse search
    grid = {
        "scale": [1],
        "road_tol": [10],
        "road_close": [0, 1, 2],
        "spur_len": [2, 3, 4],
        "link_dist": [4, 8],
        "link_cos": [0.15],
        "plant_circ": [0.5, 0.55, 0.65],
        "plant_min_area": [10],
        "bld_min_area": [25, 35],
        "bld_tol": [18],
    }
    keys = list(grid.keys())
    cands = []
    print(f"grid search on {len(subset)} train cases...", flush=True)
    for vals in itertools.product(*[grid[k] for k in keys]):
        kw = dict(zip(keys, vals))
        p = Params(**kw)
        mets, _ = eval_split(subset, p, full=False)
        sc = score_train(mets)
        st0 = {
            "cover": float(np.mean([m["road_cover"] for m in mets])),
            "ep100": float(np.mean([m["ep_per_100_skel"] for m in mets])),
            "road_rate": float(np.mean([m["has_road"] for m in mets])),
            "bld_rate": float(np.mean([m["has_building"] for m in mets])),
            "plant_med": float(np.median([m["n_plants"] for m in mets])),
        }
        cands.append((sc, p, st0))
        if len(cands) % 6 == 0:
            print(f"  searched {len(cands)} best_so_far={max(c[0] for c in cands):.3f}", flush=True)
    cands.sort(key=lambda x: -x[0])
    top = cands[:5]
    print("top5 subset:", flush=True)
    for sc, p, st in top:
        print(sc, asdict(p), st, flush=True)

    # re-score top on full train
    best = None
    for sc0, p, _ in top:
        mets, _ = eval_split(train, p, full=False)
        sc = score_train(mets)
        st = {
            "cover": float(np.mean([m["road_cover"] for m in mets])),
            "ep100": float(np.mean([m["ep_per_100_skel"] for m in mets])),
            "road_rate": float(np.mean([m["has_road"] for m in mets])),
            "bld_rate": float(np.mean([m["has_building"] for m in mets])),
            "plant_med": float(np.median([m["n_plants"] for m in mets])),
            "skel_med": float(np.median([m["skel_px"] for m in mets])),
            "ep_med": float(np.median([m["skel_endpoints"] for m in mets])),
        }
        print("full_train", sc, st, flush=True)
        if best is None or sc > best[0]:
            best = (sc, p, st, mets)

    sc, params, st, train_mets = best
    # quality gates for "high quality"
    gates = {
        "road_cover_mean_min": 0.78,
        "ep100_mean_max": 12.0,
        "road_rate_min": 0.90,
        "bld_rate_min": 0.80,
        "plant_med_min": 15,
        "plant_med_max": 100,
        "ep_median_max": 35,
    }
    def gates_ok(st_):
        return (
            st_["cover"] >= gates["road_cover_mean_min"]
            and st_["ep100"] <= gates["ep100_mean_max"]
            and st_["road_rate"] >= gates["road_rate_min"]
            and st_["bld_rate"] >= gates["bld_rate_min"]
            and gates["plant_med_min"] <= st_["plant_med"] <= gates["plant_med_max"]
            and st_.get("ep_med", 999) <= gates["ep_median_max"]
        )

    passed = gates_ok(st)

    # local refine around best if not passed
    if not passed:
        print("gates not met; local refine...", flush=True)
        refine_space = []
        base = asdict(params)
        for spur in {params.spur_len, params.spur_len + 1, max(1, params.spur_len - 1)}:
            for close in {params.road_close, min(3, params.road_close + 1), max(0, params.road_close - 1)}:
                for link in {params.link_dist, params.link_dist + 4, max(2, params.link_dist - 2)}:
                    for circ in {params.plant_circ, min(0.7, params.plant_circ + 0.05), max(0.4, params.plant_circ - 0.05)}:
                        for bmin in {params.bld_min_area, max(15, params.bld_min_area - 10)}:
                            kw = dict(base)
                            kw.update(
                                spur_len=spur,
                                road_close=close,
                                link_dist=link,
                                plant_circ=circ,
                                bld_min_area=bmin,
                            )
                            refine_space.append(Params(**kw))
        # unique
        uniq = {json.dumps(asdict(p), sort_keys=True): p for p in refine_space}
        for p in uniq.values():
            mets, _ = eval_split(train, p, full=False)
            sc2 = score_train(mets)
            st2 = {
                "cover": float(np.mean([m["road_cover"] for m in mets])),
                "ep100": float(np.mean([m["ep_per_100_skel"] for m in mets])),
                "road_rate": float(np.mean([m["has_road"] for m in mets])),
                "bld_rate": float(np.mean([m["has_building"] for m in mets])),
                "plant_med": float(np.median([m["n_plants"] for m in mets])),
                "skel_med": float(np.median([m["skel_px"] for m in mets])),
                "ep_med": float(np.median([m["skel_endpoints"] for m in mets])),
            }
            ok2 = gates_ok(st2)
            print("refine", sc2, ok2, st2, flush=True)
            if ok2 and (not passed or sc2 > sc):
                best = (sc2, p, st2, mets)
                sc, params, st, train_mets = best
                passed = True

    # final val + export (full vectorization)
    print("exporting full vectors...", flush=True)
    val_mets, val_feats = eval_split(val, params, full=True)
    train_mets_full, train_feats = eval_split(train, params, full=True)

    vec_dir = OUT / "vectors"
    ov_dir = OUT / "overlays"
    vec_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    def dump_feats(feats, split):
        for cid, img, feat in feats:
            payload = {
                "case_id": cid,
                "split": split,
                "params": asdict(params),
                "roads": feat["roads"],
                "buildings": feat["buildings"],
                "plants": [{k: v for k, v in p.items()} for p in feat["plants"]],
                "waters": feat["waters"],
                "metrics": feat["metrics"],
            }
            (vec_dir / f"{split}_{cid}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            overlay(img, feat).save(ov_dir / f"{split}_{cid}.png")

    dump_feats(val_feats, "val")
    dump_feats(train_feats, "train")

    def agg(mets):
        return {
            "n": len(mets),
            "road_cover_mean": float(np.mean([m["road_cover"] for m in mets])),
            "ep100_mean": float(np.mean([m["ep_per_100_skel"] for m in mets])),
            "ep_median": float(np.median([m["skel_endpoints"] for m in mets])),
            "skel_median": float(np.median([m["skel_px"] for m in mets])),
            "road_rate": float(np.mean([m["has_road"] for m in mets])),
            "bld_rate": float(np.mean([m["has_building"] for m in mets])),
            "plant_median": float(np.median([m["n_plants"] for m in mets])),
            "building_median": float(np.median([m["n_buildings"] for m in mets])),
            "water_median": float(np.median([m["n_waters"] for m in mets])),
            "mean_plant_circ": float(np.mean([m["mean_plant_circ"] for m in mets])),
        }

    summary = {
        "params": asdict(params),
        "train_score": sc,
        "gates": gates,
        "gates_passed": passed,
        "train": agg(train_mets_full),
        "val": agg(val_mets),
        "v1_reference": {
            "train_ep_median": 69,
            "train_skel_median": 441,
            "note": "v1 had no prune/link/orthogonal/circ filter",
        },
    }
    (OUT / "BEST_PARAMS.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # overview strip of val
    thumbs = []
    for cid, img, feat in val_feats:
        thumbs.append(overlay(img, feat))
    if thumbs:
        w = sum(t.width for t in thumbs[:5]) + 4 * 4
        h = max(t.height for t in thumbs[:5])
        sheet = Image.new("RGB", (w, h), (30, 30, 30))
        x = 0
        for t in thumbs[:5]:
            sheet.paste(t, (x, 0))
            x += t.width + 4
        sheet.save(OUT / "VAL_OVERVIEW.png")

    md = [
        "# Vector v2 extraction",
        "",
        f"- gates_passed: **{passed}**",
        f"- params: `{json.dumps(asdict(params))}`",
        "",
        "## Train vs Val",
        "",
        "| 指标 | train | val |",
        "|------|------:|----:|",
        f"| 道路覆盖 road_cover | {summary['train']['road_cover_mean']:.3f} | {summary['val']['road_cover_mean']:.3f} |",
        f"| 端点/100骨架 | {summary['train']['ep100_mean']:.2f} | {summary['val']['ep100_mean']:.2f} |",
        f"| 端点中位数 (v1=69) | {summary['train']['ep_median']:.0f} | {summary['val']['ep_median']:.0f} |",
        f"| 骨架像素中位数 | {summary['train']['skel_median']:.0f} | {summary['val']['skel_median']:.0f} |",
        f"| 有道路折线比例 | {summary['train']['road_rate']:.1%} | {summary['val']['road_rate']:.1%} |",
        f"| 有建筑比例 | {summary['train']['bld_rate']:.1%} | {summary['val']['bld_rate']:.1%} |",
        f"| 树点中位数 | {summary['train']['plant_median']:.0f} | {summary['val']['plant_median']:.0f} |",
        f"| 建筑数中位数 | {summary['train']['building_median']:.1f} | {summary['val']['building_median']:.1f} |",
        "",
        "矢量 JSON: `vectors/`；叠图: `overlays/`.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("DONE passed", passed, "train", summary["train"], "val", summary["val"], flush=True)


if __name__ == "__main__":
    main()
