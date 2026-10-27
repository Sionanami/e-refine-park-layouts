#!/usr/bin/env python
"""Expand road color gamut and re-extract (train report / val once).

Tiers:
  A) FROZEN road centers with larger L∞ tol
  B) broader orange–brown / khaki heuristic (AA strokes)
  C) optional yellow pavement as path-as-road (thin structures preferred)
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects, skeletonize

_HERE = Path(__file__).resolve().parent
ROOT = _HERE
OUT = _HERE / "outputs"
CODE = _HERE
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from optimize_road_plant import (  # noqa: E402
    OptParams,
    PLANT_CENTERS,
    ROAD_CENTERS,
    extract_road_plant,
    load_split,
    mask_centers,
    overlay,
    plant_mask_opt,
    summarize,
)
from vector_v2 import (  # noqa: E402
    chains_from_skel,
    endpoint_list,
    link_endpoints,
    prune_spurs,
    simplify_poly,
    _nms_points,
    _poly_length,
)
from optimize_road_plant import plant_ok_strict, count_links_proxy  # noqa: E402

BEST = json.loads((_HERE / "road_plant_best_params.json").read_text(encoding="utf-8"))


def road_mask_expanded(
    img: np.ndarray,
    center_tol: int = 14,
    include_pavement: bool = True,
    pavement_as_thin_only: bool = True,
) -> np.ndarray:
    """Expanded road / path mask."""
    r = img[..., 0].astype(np.int16)
    g = img[..., 1].astype(np.int16)
    b = img[..., 2].astype(np.int16)

    # A) frozen centers with expanded tol (override tiny tol=1)
    centers = [(rgb, max(tol, center_tol)) for rgb, tol in ROAD_CENTERS]
    # also explicit extra orange seeds seen in unmatched census
    extra = [
        ((241, 145, 73), center_tol),
        ((230, 139, 70), center_tol),
        ((226, 136, 69), center_tol),
        ((255, 176, 79), center_tol),
        ((255, 186, 69), max(10, center_tol - 2)),
        ((226, 159, 67), max(10, center_tol - 2)),
        ((198, 164, 88), 12),  # khaki path
        ((168, 150, 46), 12),
        ((187, 147, 104), 12),
        ((200, 177, 74), 12),
    ]
    centers = centers + extra
    m = mask_centers(img, centers)

    # B) broad orange-brown continuum
    orange = (
        (r > 170)
        & (r < 256)
        & (g > 70)
        & (g < 210)
        & (b < 140)
        & (r > g + 15)
        & (g > b)
        & ((r - b) > 40)
    )
    # khaki / olive path strokes (not pure yellow pavement)
    khaki = (
        (r > 150)
        & (r < 230)
        & (g > 120)
        & (g < 200)
        & (b < 130)
        & (np.abs(r - g) < 55)
        & (r > b + 40)
        & (g > b + 30)
    )
    m = m | orange | khaki

    # C) yellow pavement as path (optional)
    if include_pavement:
        pav = (r > 200) & (g > 180) & (b < 100) & (r > b + 80) & (g > b + 80)
        if pavement_as_thin_only and pav.any():
            # keep elongated / thin components; drop huge solid plazas
            lab, n = ndimage.label(pav.astype(np.uint8))
            keep = np.zeros_like(pav)
            for i in range(1, n + 1):
                comp = lab == i
                area = int(comp.sum())
                if area < 12:
                    continue
                if area > 6000:
                    # likely plaza — skip unless ring-like (high perimeter)
                    from skimage.measure import perimeter

                    peri = float(perimeter(comp.astype(np.uint8), neighborhood=4))
                    circ = 4 * np.pi * area / max(peri * peri, 1.0)
                    if circ > 0.35:
                        continue  # blob-like
                # elongation via bbox aspect
                ys, xs = np.where(comp)
                h = ys.max() - ys.min() + 1
                w = xs.max() - xs.min() + 1
                aspect = max(h, w) / max(min(h, w), 1)
                if aspect >= 1.8 or area <= 800:
                    keep |= comp
            pav = keep
        m = m | pav

    # exclude pure magenta buildings / cyan water / plant green
    building = (r > 180) & (b > 180) & (g < 120)
    water = (b > 180) & (g > 180) & (r < 80)
    plant = (g > 120) & (g > r + 30) & (g > b + 20) & (r < 100)
    white = (r > 245) & (g > 245) & (b > 245)
    m = m & ~building & ~water & ~plant & ~white
    return m


def extract_with_road_mask(img0: np.ndarray, p: OptParams, road: np.ndarray) -> dict:
    """Same as extract_road_plant but inject external road mask."""
    scale = max(1, int(p.scale))
    if scale != 1:
        img = np.array(Image.fromarray(img0).resize((img0.shape[1] * scale, img0.shape[0] * scale), Image.NEAREST))
        road_u = np.array(Image.fromarray((road.astype(np.uint8) * 255)).resize((img.shape[1], img.shape[0]), Image.NEAREST)) > 127
    else:
        img = img0
        road_u = road

    if p.road_close > 0:
        road_u = binary_closing(road_u, footprint=disk(p.road_close))
    if p.road_open > 0:
        road_u = binary_opening(road_u, footprint=disk(p.road_open))
    road_u = remove_small_objects(road_u.astype(bool), min_size=max(6, 4 * scale))

    skel0 = skeletonize(road_u).astype(bool) if road_u.any() else np.zeros_like(road_u, bool)
    skel = skel0.copy()
    if skel.any() and p.spur_len > 0:
        skel = prune_spurs(skel, p.spur_len)
    skel_pre = skel.copy()
    n_links = 0
    if skel.any() and p.link_dist > 0:
        skel = link_endpoints(skel, p.link_dist, p.link_cos)
        n_links = count_links_proxy(skel_pre, skel)
        skel = skeletonize(skel).astype(bool)

    roads = []
    min_road_len = 4.0 * scale
    if skel.any():
        for ch in chains_from_skel(skel):
            if _poly_length(ch) < min_road_len:
                continue
            simp = simplify_poly(ch, tol=max(1.0, p.poly_tol * scale / 2))
            if len(simp) < 2 or _poly_length(simp) < min_road_len * 0.75:
                continue
            roads.append([[pt[1] / scale, pt[0] / scale] for pt in simp])

    # plants unchanged
    plant = plant_mask_opt(img, p.plant_tol, p.use_frozen_palette)
    plants, shrubs = [], []
    lab, n_p = ndimage.label(plant.astype(np.uint8))
    min_a = p.plant_min_area * (scale**2) / 2
    max_a = p.plant_max_area * (scale**2)
    for i in range(1, n_p + 1):
        comp = lab == i
        area = int(comp.sum())
        if area < min_a:
            continue
        if area > max_a:
            if area <= max_a * 4:
                ys, xs = np.where(comp)
                shrubs.append({"center_xy": [float(xs.mean() / scale), float(ys.mean() / scale)], "area_px": area / (scale**2)})
            continue
        ok, circ = plant_ok_strict(comp, p.plant_circ, p.plant_ecc_max, p.plant_solid_min)
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
    plants = _nms_points(plants, min_dist=float(p.plant_nms))

    if road_u.any() and skel.any():
        dist = ndimage.distance_transform_edt(~skel)
        half_w = ndimage.distance_transform_edt(road_u)
        rad = np.maximum(2.0 * scale, 0.85 * half_w + 1.0)
        cover = float(((dist <= rad) & road_u).sum() / road_u.sum())
    else:
        cover = 0.0
    ep = endpoint_list(skel)
    skel_px = float(skel.sum()) / float(scale)

    if scale != 1:
        h0, w0 = img0.shape[:2]
        skel_ds = np.zeros((h0, w0), bool)
        road_ds = np.zeros((h0, w0), bool)
        yy, xx = np.where(skel)
        skel_ds[(yy // scale).clip(0, h0 - 1), (xx // scale).clip(0, w0 - 1)] = True
        yy, xx = np.where(road_u)
        road_ds[(yy // scale).clip(0, h0 - 1), (xx // scale).clip(0, w0 - 1)] = True
    else:
        skel_ds, road_ds = skel, road_u

    return {
        "roads": roads,
        "plants": plants,
        "shrubs": shrubs,
        "skel": skel_ds.astype(np.uint8),
        "road_mask": road_ds.astype(np.uint8),
        "metrics": {
            "skel_px": int(round(skel_px)),
            "skel_endpoints": len(ep),
            "n_road_lines": len(roads),
            "n_plants": len(plants),
            "n_shrubs": len(shrubs),
            "road_cover": cover,
            "ep_per_100_skel": 100.0 * len(ep) / max(float(skel.sum()), 1.0),
            "n_link_components": n_links,
            "mean_plant_circ": float(np.mean([q["circularity"] for q in plants])) if plants else 0.0,
            "has_road": skel_px >= 20,
            "road_mask_px": int(road_ds.sum()),
        },
    }


def eval_mode(rows, params: OptParams, center_tol: int, include_pavement: bool):
    mets, feats = [], []
    for r in rows:
        img = np.array(Image.open(r["B_path"]).convert("RGB"))
        road = road_mask_expanded(img, center_tol=center_tol, include_pavement=include_pavement)
        feat = extract_with_road_mask(img, params, road)
        m = dict(feat["metrics"])
        m["case_id"] = r["case_id"]
        mets.append(m)
        feats.append((r["case_id"], img, feat))
    return mets, feats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(exist_ok=True)
    (OUT / "layers").mkdir(exist_ok=True)

    params = OptParams(**{k: v for k, v in BEST["params"].items() if k in asdict(OptParams())})
    # slightly more closing once gamut is wider (bridges AA gaps)
    params = replace(params, road_close=max(params.road_close, 1))

    train = load_split("train")
    val = load_split("val")

    modes = [
        ("narrow_prev", None),  # special: previous extract_road_plant
        ("expand_tol14", dict(center_tol=14, include_pavement=False)),
        ("expand_tol18", dict(center_tol=18, include_pavement=False)),
        ("expand_tol14_pav", dict(center_tol=14, include_pavement=True)),
        ("expand_tol18_pav", dict(center_tol=18, include_pavement=True)),
    ]

    from optimize_road_plant import extract_road_plant  # noqa

    results = {}
    for name, cfg in modes:
        print("mode", name, flush=True)
        if cfg is None:
            mets_tr, feats_tr = [], []
            for r in train:
                img = np.array(Image.open(r["B_path"]).convert("RGB"))
                feat = extract_road_plant(img, params)
                m = dict(feat["metrics"])
                m["case_id"] = r["case_id"]
                mets_tr.append(m)
            mets_va, feats_va = [], []
            for r in val:
                img = np.array(Image.open(r["B_path"]).convert("RGB"))
                feat = extract_road_plant(img, params)
                m = dict(feat["metrics"])
                m["case_id"] = r["case_id"]
                mets_va.append(m)
                feats_va.append((r["case_id"], img, feat))
        else:
            mets_tr, _ = eval_mode(train, params, **cfg)
            mets_va, feats_va = eval_mode(val, params, **cfg)
        results[name] = {"train": summarize(mets_tr), "val": summarize(mets_va)}
        print(" ", name, results[name]["train"], flush=True)

    # pick best: maximize has_road rate then cover, keep ep_median from exploding
    def pick_key(s):
        t = s["train"]
        return (t["road_rate"], t["skel_median"], -t["ep_median"], t["road_cover_mean"])

    # prefer a mode with pavement if it lifts road_rate without ep_median > 55
    candidates = []
    for name, s in results.items():
        if name == "narrow_prev":
            continue
        t = s["train"]
        if t["ep_median"] <= 55 and t["skel_median"] >= 300:
            candidates.append((pick_key(s), name))
    candidates.sort(reverse=True)
    best_name = candidates[0][1] if candidates else "expand_tol14_pav"
    print("selected", best_name, flush=True)

    cfg = dict(modes[[m[0] for m in modes].index(best_name)][1])
    # re-export best
    _, train_f = eval_mode(train, params, **cfg)
    mets_va, val_f = eval_mode(val, params, **cfg)

    # also save prev vs best montage for previously low cases
    low_ids = [
        r["case_id"]
        for r in csv.DictReader(open(ROOT / "08_road_plant_opt" / "train_metrics.csv", encoding="utf-8"))
        if int(r["skel_px"]) < 50
    ]
    id2 = {r["case_id"]: r for r in train}
    rows_img = []
    for cid in low_ids[:12]:
        if cid not in id2:
            continue
        img = np.array(Image.open(id2[cid]["B_path"]).convert("RGB"))
        prev = extract_road_plant(img, params)
        road = road_mask_expanded(img, **cfg)
        cur = extract_with_road_mask(img, params, road)
        tile = np.concatenate([img, np.array(overlay(img, prev)), np.array(overlay(img, cur))], axis=1)
        im = Image.fromarray(tile)
        ImageDraw.Draw(im).text(
            (5, 5),
            f"{cid} raw|narrow|expanded  mask {prev['metrics']['road_mask_px']}->{cur['metrics']['road_mask_px']} skel {prev['metrics']['skel_px']}->{cur['metrics']['skel_px']}",
            fill=(255, 0, 0),
        )
        rows_img.append(np.array(im))
        # layers for low cases
        Image.fromarray((cur["road_mask"] * 255).astype(np.uint8)).save(OUT / "layers" / f"train_{cid}_road_mask.png")
        Image.fromarray((cur["skel"] * 255).astype(np.uint8)).save(OUT / "layers" / f"train_{cid}_road_centerline.png")
        overlay(img, cur).save(OUT / "overlays" / f"train_{cid}.png")

    if rows_img:
        Image.fromarray(np.concatenate(rows_img, 0)).save(OUT / "LOWCASE_NARROW_VS_EXPANDED.png")

    for cid, img, feat in val_f:
        Image.fromarray((feat["road_mask"] * 255).astype(np.uint8)).save(OUT / "layers" / f"val_{cid}_road_mask.png")
        Image.fromarray((feat["skel"] * 255).astype(np.uint8)).save(OUT / "layers" / f"val_{cid}_road_centerline.png")
        overlay(img, feat).save(OUT / "overlays" / f"val_{cid}.png")

    # full train metrics for selected
    mets_tr = [dict(f[2]["metrics"], case_id=f[0]) for f in train_f]
    with open(OUT / "train_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mets_tr[0].keys()))
        w.writeheader()
        w.writerows(mets_tr)
    with open(OUT / "val_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mets_va[0].keys()))
        w.writeheader()
        w.writerows(mets_va)

    summary = {
        "selected_mode": best_name,
        "config": cfg,
        "params": asdict(params),
        "comparison": results,
        "selected_train": summarize(mets_tr),
        "selected_val": summarize(mets_va),
        "n_low_before": len(low_ids),
        "n_low_after": sum(1 for m in mets_tr if m["skel_px"] < 50),
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    prev_tr = results["narrow_prev"]["train"]
    cur_tr = summary["selected_train"]
    report = f"""# Road Gamut Expansion

## Selected mode: `{best_name}`
Config: `{json.dumps(cfg)}`

## Train comparison

| mode | road_rate | skel_med | ep_med | cover | mask note |
|------|----------:|---------:|-------:|------:|-----------|
"""
    for name, s in results.items():
        t = s["train"]
        report += f"| {name} | {t['road_rate']:.3f} | {t['skel_median']:.0f} | {t['ep_median']:.0f} | {t['road_cover_mean']:.3f} | |\n"

    report += f"""
## Lift on previous low-skel cases
- low skel (<50) before: {summary['n_low_before']}
- low skel (<50) after selected: {summary['n_low_after']}

## Interpretation
- Expanded orange/khaki continuum catches AA strokes the frozen tol=1–2 missed.
- `*_pav` modes also treat elongated yellow pavement as path (plazas filtered by area/circularity).
- Check `LOWCASE_NARROW_VS_EXPANDED.png` for residual misses / over-include.

## Outputs
- overlays/, layers/, SUMMARY.json, train/val_metrics.csv
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
