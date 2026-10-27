#!/usr/bin/env python
"""Abstract road + pavement into one circulation network (manuscript-aligned).

Manuscript (E-Refine):
  - Composition: keep RAR (road) and PVR (pavement) as separate area ratios.
  - Structure: LCR / RER evaluate the *road network* / circulation continuity.
  - R2 text: connectivity may extend to adjacent roads, entrances, or pavement
    boundaries — circulation is one topological system.

This script:
  1) Extracts road (orange continuum) and pavement (yellow) masks separately.
  2) Builds circulation_mask = road ∪ pavement (site-clipped if A available).
  3) Skeleton → spur prune → constrained endpoint link → polylines.
  4) Reports LCR, RER on circulation skeleton; also RAR/PVR pixel fractions.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.morphology import binary_closing, disk, remove_small_objects, skeletonize

_HERE = Path(__file__).resolve().parent
ROOT = _HERE
OUT = _HERE / "outputs"
CODE_NIGHT = _HERE
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from expand_road_gamut import road_mask_expanded  # noqa: E402
from optimize_road_plant import (  # noqa: E402
    OptParams,
    count_links_proxy,
    load_split,
    plant_mask_opt,
)
from vector_v2 import (  # noqa: E402
    chains_from_skel,
    endpoint_list,
    link_endpoints,
    local_dir,
    prune_spurs,
    simplify_poly,
    _line,
    _poly_length,
)

SUM_PATH = ROOT / "08_road_plant_opt" / "road_gamut_expand" / "SUMMARY.json"
PAVEMENT_RGB = np.array([255, 255, 0], np.int16)

# Trees sitting on / breaking paths: bridge within this distance (px @256)
# Ablation (09/.../plant_bridge_ablation): peak score at 16; useful range 14–18.
DEFAULT_PLANT_BRIDGE_DIST = 16
DEFAULT_PLANT_GAP_FRAC = 0.35
DEFAULT_PLANT_ISLAND_MAX = 50  # only small mid-path tree blobs into network


def pavement_mask(img: np.ndarray, tol: int = 18) -> np.ndarray:
    """Yellow pavement (manuscript palette) + near-yellow AA."""
    x = img.astype(np.int16)
    m = np.max(np.abs(x - PAVEMENT_RGB), axis=2) <= tol
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    # bright yellow / cream path fills
    heur = (r > 200) & (g > 180) & (b < 120) & (r > b + 70) & (g > b + 60)
    # exclude orange-dominant already counted as road preference later via union
    return (m | heur) & ~((r > 200) & (g < 170) & (b < 120) & (r > g + 40))


def site_mask_from_A(a: np.ndarray, tol: int = 2) -> np.ndarray:
    d = np.max(np.abs(a.astype(np.int16) - np.array([0, 255, 0], np.int16)), axis=2)
    return d <= tol


def largest_component_ratio(mask: np.ndarray) -> float:
    """Manuscript LCR on binary mask."""
    m = mask.astype(bool)
    if not m.any():
        return float("nan")
    lab, n = ndimage.label(m)
    if n == 0:
        return float("nan")
    sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
    return float(np.max(sizes) / m.sum())


def road_endpoint_ratio(skel: np.ndarray) -> float:
    """Manuscript RER on skeleton."""
    s = skel.astype(bool)
    if not s.any():
        return float("nan")
    eps = endpoint_list(s)
    return float(len(eps) / max(s.sum(), 1))


def absorb_plant_islands(
    circ: np.ndarray,
    plant: np.ndarray,
    max_area: int = DEFAULT_PLANT_ISLAND_MAX,
) -> tuple[np.ndarray, int]:
    """Absorb only plants sitting *in the middle of* a path (strongly enclosed by circulation).

    Roadside trees (touching path on one side only) are NOT absorbed — those are
    handled by endpoint bridging if they create a gap. Only affects network mask.
    """
    out = circ.astype(bool).copy()
    if not plant.any() or not circ.any():
        return out, 0
    circ_d = ndimage.binary_dilation(out, structure=np.ones((3, 3)))
    lab, n = ndimage.label(plant.astype(bool))
    n_abs = 0
    for i in range(1, n + 1):
        comp = lab == i
        area = int(comp.sum())
        if area < 3 or area > max_area:
            continue
        if not (comp & circ_d).any():
            continue
        # 1-pixel ring around plant
        ring = ndimage.binary_dilation(comp, structure=np.ones((3, 3))) & ~comp
        ring_n = int(ring.sum())
        if ring_n < 4:
            continue
        touch = int((ring & out).sum())
        enclose = touch / ring_n
        # must be mostly surrounded by path (≥55% of ring is circulation)
        if enclose < 0.55:
            continue
        out |= comp
        n_abs += 1
    return out, n_abs


def bridge_endpoints_across_plants(
    skel: np.ndarray,
    circ: np.ndarray,
    plant: np.ndarray,
    max_dist: int = DEFAULT_PLANT_BRIDGE_DIST,
    min_plant_frac: float = DEFAULT_PLANT_GAP_FRAC,
    min_cos: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, int, list]:
    """Actively link circulation endpoints if the gap is largely plant-occupied.

    Returns (skel_out, circ_out, n_bridges, bridge_segments_yx).
    """
    skel_out = skel.astype(bool).copy()
    circ_out = circ.astype(bool).copy()
    plant_d = ndimage.binary_dilation(plant.astype(bool), structure=np.ones((3, 3)))
    eps = endpoint_list(skel_out)
    if len(eps) < 2:
        return skel_out, circ_out, 0, []

    dirs = {p: local_dir(skel_out, p[0], p[1]) for p in eps}
    pairs = []
    for i, a in enumerate(eps):
        for b in eps[i + 1 :]:
            dy = a[0] - b[0]
            dx = a[1] - b[1]
            dist = math.hypot(dx, dy)
            if dist < 1.5 or dist > max_dist:
                continue
            va, vb = dirs[a], dirs[b]
            ab = np.array([b[1] - a[1], b[0] - a[0]], dtype=float)
            ab /= np.linalg.norm(ab) + 1e-6
            ba = -ab
            ca = float(np.dot(-va, ab))
            cb = float(np.dot(-vb, ba))
            # plant bridges: relax direction a bit vs normal road links
            if ca < min_cos or cb < min_cos:
                continue
            rr, cc = _line(a[0], a[1], b[0], b[1])
            gap_plant = 0
            gap_total = 0
            for y, x in zip(rr, cc):
                if skel_out[y, x] or circ_out[y, x]:
                    continue
                gap_total += 1
                if plant_d[y, x]:
                    gap_plant += 1
            if gap_total < 1:
                continue
            frac = gap_plant / gap_total
            # require plant involvement in the gap (tree interrupting the path)
            if frac < min_plant_frac and gap_plant < 2:
                continue
            # prefer pairs where both ends near plant
            near_a = plant_d[
                max(0, a[0] - 2) : a[0] + 3, max(0, a[1] - 2) : a[1] + 3
            ].any()
            near_b = plant_d[
                max(0, b[0] - 2) : b[0] + 3, max(0, b[1] - 2) : b[1] + 3
            ].any()
            if not (near_a or near_b or frac >= min_plant_frac):
                continue
            pairs.append((dist, -frac, a, b, list(zip(rr.tolist(), cc.tolist()))))

    pairs.sort()
    used = set()
    bridges = []
    n_bridges = 0
    for dist, neg_frac, a, b, pts in pairs:
        if a in used or b in used:
            continue
        for y, x in pts:
            skel_out[y, x] = True
            circ_out[y, x] = True
        used.add(a)
        used.add(b)
        bridges.append(
            {
                "a": [int(a[1]), int(a[0])],
                "b": [int(b[1]), int(b[0])],
                "dist": float(dist),
                "plant_frac": float(-neg_frac),
            }
        )
        n_bridges += 1
    return skel_out, circ_out, n_bridges, bridges


def extract_circulation(
    img: np.ndarray,
    a_img: np.ndarray | None,
    params: OptParams,
    center_tol: int = 18,
    plant_bridge_dist: int = DEFAULT_PLANT_BRIDGE_DIST,
    plant_gap_frac: float = DEFAULT_PLANT_GAP_FRAC,
    enable_plant_bridge: bool = True,
):
    road = road_mask_expanded(img, center_tol=center_tol, include_pavement=False)
    pav = pavement_mask(img, tol=18)
    plant = plant_mask_opt(img, tol=8, use_frozen=True)
    # circulation = road ∪ pavement (manuscript network abstraction)
    circ = road | pav

    site = None
    if a_img is not None:
        site = site_mask_from_A(a_img)
        # soft clip: keep circulation mostly in/near site (dilate site slightly)
        site_d = binary_closing(site, footprint=disk(2)) | site
        # urban context roads outside site may exist; keep them if connected to site circ
        # for network metrics prefer site-relevant: union outside only if orange road
        circ_in = circ & site_d
        circ_out_road = road & ~site_d  # external approach roads
        circ = circ_in | circ_out_road

    circ = binary_closing(circ, footprint=disk(max(1, params.road_close)))
    circ = remove_small_objects(circ.astype(bool), min_size=8)

    n_plant_absorb = 0
    n_plant_bridges = 0
    plant_bridges = []
    if enable_plant_bridge and plant.any():
        circ, n_plant_absorb = absorb_plant_islands(circ, plant)

    skel = skeletonize(circ).astype(bool) if circ.any() else np.zeros_like(circ, bool)
    if skel.any() and params.spur_len > 0:
        skel = prune_spurs(skel, params.spur_len)
    skel_pre = skel.copy()
    n_links = 0
    if skel.any() and params.link_dist > 0:
        skel = link_endpoints(skel, params.link_dist, params.link_cos)
        n_links = count_links_proxy(skel_pre, skel)
        skel = skeletonize(skel).astype(bool)

    # Plant-aware bridging: connect across trees that interrupt road/pavement
    if enable_plant_bridge and plant.any() and skel.any():
        skel, circ, n_plant_bridges, plant_bridges = bridge_endpoints_across_plants(
            skel,
            circ,
            plant,
            max_dist=plant_bridge_dist,
            min_plant_frac=plant_gap_frac,
            min_cos=0.05,
        )
        skel = skeletonize(skel).astype(bool)
        if params.spur_len > 0:
            skel = prune_spurs(skel, max(1, params.spur_len // 2))

    polylines = []
    min_len = 4.0
    if skel.any():
        for ch in chains_from_skel(skel):
            if _poly_length(ch) < min_len:
                continue
            simp = simplify_poly(ch, tol=max(1.0, params.poly_tol))
            if len(simp) < 2 or _poly_length(simp) < min_len * 0.75:
                continue
            # tag segments by majority source under polyline samples
            orange = 0
            yellow = 0
            for y, x in simp[:: max(1, len(simp) // 8)]:
                yi, xi = int(round(y)), int(round(x))
                if 0 <= yi < road.shape[0] and 0 <= xi < road.shape[1]:
                    if road[yi, xi]:
                        orange += 1
                    if pav[yi, xi]:
                        yellow += 1
            src = "road" if orange >= yellow else ("pavement" if yellow > orange else "mixed")
            polylines.append({"xy": [[float(p[1]), float(p[0])] for p in simp], "source_hint": src})

    h, w = img.shape[:2]
    canvas = float(h * w)
    site_px = float(site.sum()) if site is not None else canvas
    site_px = max(site_px, 1.0)
    metrics = {
        "road_px": int(road.sum()),
        "pavement_px": int(pav.sum()),
        "plant_px": int(plant.sum()),
        "circulation_px": int(circ.sum()),
        "skel_px": int(skel.sum()),
        "skel_endpoints": len(endpoint_list(skel)),
        "n_polylines": len(polylines),
        "n_link_components": n_links,
        "n_plant_absorb": n_plant_absorb,
        "n_plant_bridges": n_plant_bridges,
        # manuscript composition (fractions of canvas; also site-relative if A given)
        "RAR_canvas": float(road.sum() / canvas),
        "PVR_canvas": float(pav.sum() / canvas),
        "RAR_site": float((road & site).sum() / site_px) if site is not None else float("nan"),
        "PVR_site": float((pav & site).sum() / site_px) if site is not None else float("nan"),
        # manuscript structure on circulation network
        "LCR_circulation": largest_component_ratio(circ),
        "LCR_skel": largest_component_ratio(skel),
        "RER": road_endpoint_ratio(skel),
        "has_network": int(skel.sum() >= 20),
    }
    return {
        "road": road.astype(np.uint8),
        "pavement": pav.astype(np.uint8),
        "plant": plant.astype(np.uint8),
        "circulation": circ.astype(np.uint8),
        "skel": skel.astype(np.uint8),
        "polylines": polylines,
        "plant_bridges": plant_bridges,
        "metrics": metrics,
        "site": site.astype(np.uint8) if site is not None else None,
    }


def overlay(img, feat, *, legend: bool = True) -> Image.Image:
    """Balanced semantic overlay: soft fills + strong circulation lines + optional legend."""
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    rgba = np.zeros((*img.shape[:2], 4), dtype=np.uint8)
    # low-alpha fills so no single class dominates the view
    if "plant" in feat and feat["plant"] is not None:
        rgba[feat["plant"] > 0] = (0, 160, 70, 36)
    rgba[feat["pavement"] > 0] = (255, 230, 40, 55)
    rgba[feat["road"] > 0] = (241, 145, 73, 70)
    ov = Image.alpha_composite(ov, Image.fromarray(rgba, "RGBA"))
    d = ImageDraw.Draw(ov)
    # circulation polylines thicker / higher contrast (structure first)
    for pl in feat["polylines"]:
        pts = [(p[0], p[1]) for p in pl["xy"]]
        if len(pts) >= 2:
            color = (255, 48, 0, 255) if pl["source_hint"] == "road" else (
                (230, 170, 0, 255) if pl["source_hint"] == "pavement" else (255, 110, 0, 255)
            )
            d.line(pts, fill=color, width=2)
    for br in feat.get("plant_bridges") or []:
        d.line([tuple(br["a"]), tuple(br["b"])], fill=(0, 220, 255, 255), width=2)
    composed = Image.alpha_composite(im, ov).convert("RGB")
    if legend:
        m = feat.get("metrics") or {}
        road_px = int(m.get("road_px") or 0)
        pav_px = int(m.get("pavement_px") or 0)
        plant_px = int(m.get("plant_px") or 0)
        circ_px = int(m.get("circulation_px") or max(road_px + pav_px, 1))
        canvas = max(img.shape[0] * img.shape[1], 1)
        txt = (
            f"RAR {road_px / canvas:.3f}  PVR {pav_px / canvas:.3f}  "
            f"plant {plant_px / canvas:.3f}  circ {circ_px / canvas:.3f}"
        )
        # warn semantic imbalance
        if plant_px > 6 * max(circ_px, 1):
            txt += "  !plant>>circ"
        if road_px + pav_px < 0.005 * canvas:
            txt += "  !circ_thin"
        draw = ImageDraw.Draw(composed)
        draw.rectangle([0, 0, composed.size[0], 16], fill=(20, 20, 24))
        draw.text((3, 2), txt, fill=(240, 240, 240))
    return composed


def summarize(mets: list[dict]) -> dict:
    def med(k):
        vals = [m[k] for m in mets if m[k] == m[k]]  # drop nan
        return float(np.median(vals)) if vals else float("nan")

    def mean(k):
        vals = [m[k] for m in mets if m[k] == m[k]]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n": len(mets),
        "network_rate": mean("has_network"),
        "skel_median": med("skel_px"),
        "LCR_circ_median": med("LCR_circulation"),
        "LCR_skel_median": med("LCR_skel"),
        "RER_median": med("RER"),
        "RAR_site_median": med("RAR_site"),
        "PVR_site_median": med("PVR_site"),
        "polylines_median": med("n_polylines"),
        "plant_bridges_mean": mean("n_plant_bridges") if mets and "n_plant_bridges" in mets[0] else 0.0,
        "plant_absorb_mean": mean("n_plant_absorb") if mets and "n_plant_absorb" in mets[0] else 0.0,
        "ep_median": med("skel_endpoints"),
        "low_skel_lt50": int(sum(1 for m in mets if m["skel_px"] < 50)),
    }


def main():
    out_root = ROOT / "09_circulation_network" / "plant_bridge"
    out_root.mkdir(parents=True, exist_ok=True)
    for sub in ("layers", "overlays", "vectors", "panels"):
        (out_root / sub).mkdir(exist_ok=True)

    if SUM_PATH.exists():
        sumj = json.loads(SUM_PATH.read_text(encoding="utf-8"))
        params = OptParams(**{k: v for k, v in sumj["params"].items() if k in asdict(OptParams())})
        center_tol = int(sumj["config"].get("center_tol", 18))
    else:
        params = OptParams(road_close=1, spur_len=4, link_dist=6, link_cos=0.25)
        center_tol = 18

    plant_bridge_dist = DEFAULT_PLANT_BRIDGE_DIST
    plant_gap_frac = DEFAULT_PLANT_GAP_FRAC

    meta = {
        "policy": {
            "composition": "RAR=road orange only; PVR=pavement yellow only (manuscript)",
            "network": "circulation = road ∪ pavement; plant islands absorbed; endpoints bridged across plant gaps",
            "plant_bridge": {
                "max_dist_px": plant_bridge_dist,
                "min_plant_frac_in_gap": plant_gap_frac,
                "island_max_area": DEFAULT_PLANT_ISLAND_MAX,
                "note": "Bridges only affect network mask/centerline, not RAR/PVR semantic area",
            },
            "manuscript_refs": [
                "RAR and PVR as separate area ratios",
                "LCR/RER for road-network continuity",
                "R2 connectivity may reach pavement boundaries; trees often interrupt paths",
            ],
            "center_tol": center_tol,
            "params": asdict(params),
        },
        "splits": {},
    }

    for split in ("train", "val"):
        rows = load_split(split)
        mets = []
        panel_rows = []
        for i, r in enumerate(rows):
            cid = r["case_id"]
            img = np.array(Image.open(r["B_path"]).convert("RGB"))
            a_img = np.array(Image.open(r["A_path"]).convert("RGB")) if r.get("A_path") else None
            feat = extract_circulation(
                img,
                a_img,
                params,
                center_tol=center_tol,
                plant_bridge_dist=plant_bridge_dist,
                plant_gap_frac=plant_gap_frac,
                enable_plant_bridge=True,
            )
            m = dict(feat["metrics"])
            m["case_id"] = cid
            mets.append(m)

            Image.fromarray(feat["road"] * 255).save(out_root / "layers" / f"{split}_{cid}_road.png")
            Image.fromarray(feat["pavement"] * 255).save(out_root / "layers" / f"{split}_{cid}_pavement.png")
            Image.fromarray(feat["plant"] * 255).save(out_root / "layers" / f"{split}_{cid}_plant.png")
            Image.fromarray(feat["circulation"] * 255).save(out_root / "layers" / f"{split}_{cid}_circulation.png")
            Image.fromarray(feat["skel"] * 255).save(out_root / "layers" / f"{split}_{cid}_centerline.png")
            ov = overlay(img, feat)
            ov.save(out_root / "overlays" / f"{split}_{cid}.png")
            payload = {
                "case_id": cid,
                "split": split,
                "polylines": feat["polylines"],
                "plant_bridges": feat["plant_bridges"],
                "metrics": feat["metrics"],
            }
            (out_root / "vectors" / f"{split}_{cid}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            # highlight cases with plant bridges in panels
            if split == "val" or m["n_plant_bridges"] > 0 or m["n_plant_absorb"] > 0 or (split == "train" and i < 6):
                road_rgb = np.zeros_like(img)
                road_rgb[feat["road"] > 0] = (241, 145, 73)
                pav_rgb = np.zeros_like(img)
                pav_rgb[feat["pavement"] > 0] = (255, 255, 0)
                plant_rgb = np.zeros_like(img)
                plant_rgb[feat["plant"] > 0] = (0, 160, 60)
                circ_rgb = np.array(ov)
                tile = np.concatenate([img, road_rgb, pav_rgb, plant_rgb, circ_rgb], axis=1)
                im = Image.fromarray(tile)
                ImageDraw.Draw(im).text(
                    (5, 5),
                    f"{cid} B|road|pav|plant|net  bridges={m['n_plant_bridges']} absorb={m['n_plant_absorb']} LCR={m['LCR_circulation']:.2f} RER={m['RER']:.3f}",
                    fill=(255, 0, 0),
                )
                panel_rows.append(np.array(im))

            if (i + 1) % 20 == 0:
                print(split, i + 1, flush=True)

        with open(out_root / f"{split}_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(mets[0].keys()))
            w.writeheader()
            w.writerows(mets)
        meta["splits"][split] = summarize(mets)
        print(split, meta["splits"][split], flush=True)
        if panel_rows:
            # cap panel height
            Image.fromarray(np.concatenate(panel_rows[:16], 0)).save(out_root / "panels" / f"{split}_overview.png")

    (out_root / "SUMMARY.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tr, va = meta["splits"]["train"], meta["splits"]["val"]
    report = f"""# Circulation Network + Plant Bridging

Road ∪ pavement as one network; **trees interrupting paths are bridged within {plant_bridge_dist}px**.

## Rules

1. Semantic layers unchanged: road / pavement / plant kept separate (RAR/PVR still valid).
2. Network mask: start from road ∪ pavement.
3. **Absorb**: small plant islands (area ≤ {DEFAULT_PLANT_ISLAND_MAX}) tightly surrounded by path → fill into network only.
4. **Bridge**: skeleton endpoints within {plant_bridge_dist}px whose gap is mostly plant → draw a link (cyan in overlays).

## Results

| split | network_rate | skel_med | LCR_circ | RER | ep_med | plant_bridges_mean | absorb_mean | low_skel |
|-------|-------------:|---------:|--------:|----:|-------:|-------------------:|------------:|---------:|
| train | {tr['network_rate']:.3f} | {tr['skel_median']:.0f} | {tr['LCR_circ_median']:.3f} | {tr['RER_median']:.3f} | {tr['ep_median']:.0f} | {tr['plant_bridges_mean']:.2f} | {tr['plant_absorb_mean']:.2f} | {tr['low_skel_lt50']} |
| val | {va['network_rate']:.3f} | {va['skel_median']:.0f} | {va['LCR_circ_median']:.3f} | {va['RER_median']:.3f} | {va['ep_median']:.0f} | {va['plant_bridges_mean']:.2f} | {va['plant_absorb_mean']:.2f} | {va['low_skel_lt50']} |

## Outputs

`{out_root}`

- cyan lines in overlays = plant bridges
- `vectors/*.json` includes `plant_bridges` list
"""
    (out_root / "REPORT.md").write_text(report, encoding="utf-8")
    # also mirror short note to parent
    (ROOT / "09_circulation_network" / "PLANT_BRIDGE_NOTE.md").write_text(
        report + "\n\nFull artifacts under `plant_bridge/`.\n", encoding="utf-8"
    )
    print(report, flush=True)
    print("DONE", out_root, flush=True)


if __name__ == "__main__":
    main()
