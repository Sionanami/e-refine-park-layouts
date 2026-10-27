#!/usr/bin/env python
"""Optimize road + tree extraction (AutoResearch VECTOR_OPT), train-freeze / val-once.

Baseline = vector_v2 BEST_PARAMS.
Optimizations:
  - FROZEN_PALETTE multi-center masks (road + plant supplementary colors)
  - road: small close → skeleton → spur prune → constrained endpoint link
  - plant: plant colors only (no lawn green), area+circularity+eccentricity, stronger NMS
  - optional 2x NEAREST upsample for thin roads
"""
from __future__ import annotations

import csv
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass, replace
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

from vector_v2 import (  # noqa: E402
    Params,
    chains_from_skel,
    circularity,
    endpoint_list,
    link_endpoints,
    load_split,
    plant_ok,
    prune_spurs,
    simplify_poly,
    _nms_points,
    _poly_length,
)

PALETTE_PATH = _HERE / "frozen_palette.json"
BASELINE_JSON = _HERE / "vector_best_params.json"


@dataclass
class OptParams(Params):
    use_frozen_palette: bool = True
    plant_nms: float = 4.0
    plant_ecc_max: float = 0.78
    plant_solid_min: float = 0.60
    # false-link proxy: links longer than this fraction of image diagonal counted separately
    link_report_max: int = 12


def load_frozen_centers():
    pal = json.loads(PALETTE_PATH.read_text(encoding="utf-8"))
    roads, plants = [], []
    for c in pal.get("all_centers", []):
        rgb = tuple(c["rgb"])
        tol = int(c.get("tol", 2))
        if c["class"] == "road":
            roads.append((rgb, tol))
        elif c["class"] == "plant":
            plants.append((rgb, tol))
    return roads, plants


ROAD_CENTERS, PLANT_CENTERS = load_frozen_centers()


def mask_centers(img: np.ndarray, centers) -> np.ndarray:
    x = img.astype(np.int16)
    m = np.zeros(img.shape[:2], dtype=bool)
    for rgb, tol in centers:
        d = np.max(np.abs(x - np.array(rgb, dtype=np.int16)), axis=2)
        m |= d <= tol
    return m


def road_mask_opt(img: np.ndarray, tol: int, use_frozen: bool) -> np.ndarray:
    if use_frozen and ROAD_CENTERS:
        m = mask_centers(img, ROAD_CENTERS)
    else:
        m = np.all(np.abs(img.astype(np.int16) - np.array([241, 145, 73], np.int16)) <= tol, axis=2)
    # light AA brown heuristic (kept narrow)
    r, g, b = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16), img[..., 2].astype(np.int16)
    heur = (r > 200) & (g > 100) & (g < 180) & (b < 120) & (r > g + 25) & (g > b + 15)
    return m | heur


def plant_mask_opt(img: np.ndarray, tol: int, use_frozen: bool) -> np.ndarray:
    """Plant only — never merge lawn green (0,255,0)."""
    if use_frozen and PLANT_CENTERS:
        return mask_centers(img, PLANT_CENTERS)
    return np.all(np.abs(img.astype(np.int16) - np.array([0, 152, 67], np.int16)) <= tol, axis=2)


def plant_ok_strict(comp: np.ndarray, min_circ: float, ecc_max: float, solid_min: float):
    from skimage.measure import regionprops

    circ = circularity(comp)
    props = regionprops(comp.astype(np.uint8))
    if not props:
        return False, circ
    p = props[0]
    if circ < min_circ:
        return False, circ
    if p.eccentricity > ecc_max:
        return False, circ
    if p.solidity < solid_min:
        return False, circ
    return True, circ


def count_links_proxy(skel_before: np.ndarray, skel_after: np.ndarray) -> int:
    """Pixels added by linking (rough proxy for number/length of bridges)."""
    added = skel_after & ~skel_before
    if not added.any():
        return 0
    _, n = ndimage.label(added.astype(np.uint8))
    return int(n)


def extract_road_plant(img0: np.ndarray, p: OptParams) -> dict:
    scale = max(1, int(p.scale))
    if scale != 1:
        img = np.array(Image.fromarray(img0).resize((img0.shape[1] * scale, img0.shape[0] * scale), Image.NEAREST))
    else:
        img = img0

    # ---- road ----
    road = road_mask_opt(img, p.road_tol, p.use_frozen_palette)
    if p.road_close > 0:
        road = binary_closing(road, footprint=disk(p.road_close))
    if p.road_open > 0:
        road = binary_opening(road, footprint=disk(p.road_open))
    road = remove_small_objects(road.astype(bool), min_size=max(6, 4 * scale))

    skel0 = skeletonize(road).astype(bool) if road.any() else np.zeros_like(road, bool)
    skel = skel0.copy()
    if skel.any() and p.spur_len > 0:
        skel = prune_spurs(skel, p.spur_len)
    skel_pre_link = skel.copy()
    n_links = 0
    if skel.any() and p.link_dist > 0:
        skel = link_endpoints(skel, p.link_dist, p.link_cos)
        n_links = count_links_proxy(skel_pre_link, skel)
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

    # ---- plants / trees ----
    plant = plant_mask_opt(img, p.plant_tol, p.use_frozen_palette)
    plants, shrubs = [], []
    lab, n_p = ndimage.label(plant.astype(np.uint8))
    min_a = p.plant_min_area * (scale**2) / 2
    max_a = p.plant_max_area * (scale**2)
    shrub_max = max_a * 4  # larger blobs → shrub/face, not tree points
    for i in range(1, n_p + 1):
        comp = lab == i
        area = int(comp.sum())
        if area < min_a:
            continue
        if area > max_a:
            if area <= shrub_max:
                ys, xs = np.where(comp)
                shrubs.append(
                    {
                        "center_xy": [float(xs.mean() / scale), float(ys.mean() / scale)],
                        "area_px": area / (scale**2),
                        "kind": "shrub_patch",
                    }
                )
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
                "kind": "tree_point",
            }
        )
    plants = _nms_points(plants, min_dist=float(p.plant_nms) * scale / max(scale, 1))

    # coverage of road mask by thickened skeleton
    if road.any() and skel.any():
        dist = ndimage.distance_transform_edt(~skel)
        half_w = ndimage.distance_transform_edt(road)
        rad = np.maximum(2.0 * scale, 0.85 * half_w + 1.0)
        cover = float(((dist <= rad) & road).sum() / road.sum())
    else:
        cover = 0.0

    ep = endpoint_list(skel)
    skel_px = float(skel.sum()) / float(scale)

    # downsample skel for export
    if scale != 1:
        h0, w0 = img0.shape[:2]
        sh = np.zeros((h0, w0), dtype=bool)
        yy, xx = np.where(skel)
        sh[(yy // scale).clip(0, h0 - 1), (xx // scale).clip(0, w0 - 1)] = True
        skel_ds = sh
        road_ds = np.zeros((h0, w0), dtype=bool)
        yy, xx = np.where(road)
        road_ds[(yy // scale).clip(0, h0 - 1), (xx // scale).clip(0, w0 - 1)] = True
    else:
        skel_ds, road_ds = skel, road

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


def score_road_plant(mets: list[dict]) -> float:
    """Train score: high cover, fewer endpoints/spur, plant count in band, penalize many links."""
    cover = np.mean([m["road_cover"] for m in mets])
    ep100 = np.mean([m["ep_per_100_skel"] for m in mets])
    road_rate = np.mean([m["has_road"] for m in mets])
    plant_med = float(np.median([m["n_plants"] for m in mets]))
    links = np.mean([m["n_link_components"] for m in mets])
    plant_pen = 0.0 if 12 <= plant_med <= 60 else abs(plant_med - 30) / 40.0
    # prefer cover & presence; discourage messy skeletons and aggressive linking
    return float(3.2 * cover + 1.3 * road_rate - 0.06 * ep100 - 0.04 * links - 0.35 * plant_pen)


def summarize(mets: list[dict]) -> dict:
    def med(k):
        return float(np.median([m[k] for m in mets]))

    def mean(k):
        return float(np.mean([m[k] for m in mets]))

    return {
        "n": len(mets),
        "road_cover_mean": mean("road_cover"),
        "ep100_mean": mean("ep_per_100_skel"),
        "ep_median": med("skel_endpoints"),
        "skel_median": med("skel_px"),
        "road_rate": mean("has_road"),
        "road_lines_median": med("n_road_lines"),
        "plant_median": med("n_plants"),
        "shrub_median": med("n_shrubs"),
        "mean_plant_circ": mean("mean_plant_circ"),
        "link_components_mean": mean("n_link_components"),
        "score": score_road_plant(mets),
    }


def eval_split(rows, params: OptParams):
    mets, feats = [], []
    for r in rows:
        img = np.array(Image.open(r["B_path"]).convert("RGB"))
        feat = extract_road_plant(img, params)
        m = dict(feat["metrics"])
        m["case_id"] = r["case_id"]
        mets.append(m)
        feats.append((r["case_id"], img, feat))
    return mets, feats


def overlay(img, feat) -> Image.Image:
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    # road mask faint
    rm = feat["road_mask"]
    if rm.any():
        rgba = np.zeros((*rm.shape, 4), dtype=np.uint8)
        rgba[rm > 0] = (241, 145, 73, 60)
        ov = Image.alpha_composite(ov, Image.fromarray(rgba, "RGBA"))
        d = ImageDraw.Draw(ov)
    for line in feat["roads"]:
        if len(line) >= 2:
            d.line([(p[0], p[1]) for p in line], fill=(255, 40, 0, 255), width=2)
    for p in feat["plants"]:
        cx, cy = p["center_xy"]
        d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], outline=(0, 220, 60, 255), width=1)
        d.point((cx, cy), fill=(0, 255, 80, 255))
    for p in feat.get("shrubs", []):
        cx, cy = p["center_xy"]
        d.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], outline=(0, 140, 40, 180), width=1)
    return Image.alpha_composite(im, ov).convert("RGB")


def to_opt_params(d: dict) -> OptParams:
    base = asdict(OptParams())
    base.update({k: v for k, v in d.items() if k in base})
    return OptParams(**base)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(exist_ok=True)
    (OUT / "layers").mkdir(exist_ok=True)
    (OUT / "vectors").mkdir(exist_ok=True)

    train = load_split("train")
    val = load_split("val")
    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))["params"]
    base_p = to_opt_params({**baseline, "use_frozen_palette": False, "plant_nms": 3.0})

    print("baseline eval...", flush=True)
    base_train_m, _ = eval_split(train, base_p)
    base_val_m, base_val_f = eval_split(val, base_p)
    base_sum = {"train": summarize(base_train_m), "val": summarize(base_val_m), "params": asdict(base_p)}
    (OUT / "baseline_summary.json").write_text(json.dumps(base_sum, indent=2), encoding="utf-8")
    print("baseline", base_sum["train"], flush=True)

    # small grid — train only (VECTOR_OPT priorities)
    grid = []
    for scale in (1, 2):
        for road_close in (0, 1):
            for spur_len in (2, 4):
                for link_dist, link_cos in ((6, 0.25), (8, 0.20), (10, 0.15)):
                    for plant_circ in (0.55, 0.62, 0.68):
                        for plant_nms in (3.5, 5.0):
                            for plant_max in (80, 100):
                                grid.append(
                                    replace(
                                        base_p,
                                        scale=scale,
                                        road_close=road_close,
                                        spur_len=spur_len,
                                        link_dist=link_dist,
                                        link_cos=link_cos,
                                        plant_circ=plant_circ,
                                        plant_nms=plant_nms,
                                        plant_max_area=plant_max,
                                        plant_min_area=8,
                                        use_frozen_palette=True,
                                        road_tol=10,
                                        plant_tol=8,
                                    )
                                )
    # dedupe by asdict json
    uniq, seen = [], set()
    for p in grid:
        key = json.dumps(asdict(p), sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    # subsample if huge: keep diversity
    if len(uniq) > 48:
        rng = np.random.RandomState(20261014)
        idx = rng.choice(len(uniq), size=48, replace=False)
        uniq = [uniq[i] for i in sorted(idx)]
    print(f"grid size={len(uniq)}", flush=True)

    best_p, best_score, best_train_sum = None, -1e9, None
    search_rows = []
    for i, p in enumerate(uniq):
        mets, _ = eval_split(train, p)
        s = summarize(mets)
        # hard gates (AutoResearch-style)
        gates_ok = (
            s["road_cover_mean"] >= 0.78
            and s["road_rate"] >= 0.85
            and s["ep_median"] <= 45
            and s["skel_median"] >= 250  # don't collapse length vs v1~441 / v2~395
            and 10 <= s["plant_median"] <= 70
        )
        score = s["score"] if gates_ok else s["score"] - 5.0
        search_rows.append({**{f"p_{k}": v for k, v in asdict(p).items()}, **s, "gates_ok": int(gates_ok), "score_gated": score})
        if score > best_score:
            best_score, best_p, best_train_sum = score, p, s
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{len(uniq)}] best_score={best_score:.3f} gates={best_train_sum}", flush=True)

    with open(OUT / "search_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(search_rows[0].keys()))
        w.writeheader()
        w.writerows(search_rows)

    print("best params", asdict(best_p), "train", best_train_sum, flush=True)

    # val once
    opt_train_m, opt_train_f = eval_split(train, best_p)
    opt_val_m, opt_val_f = eval_split(val, best_p)
    opt_sum = {
        "params": asdict(best_p),
        "train": summarize(opt_train_m),
        "val": summarize(opt_val_m),
        "baseline_train": base_sum["train"],
        "baseline_val": base_sum["val"],
        "delta_train": {
            "ep_median": summarize(opt_train_m)["ep_median"] - base_sum["train"]["ep_median"],
            "skel_median": summarize(opt_train_m)["skel_median"] - base_sum["train"]["skel_median"],
            "plant_median": summarize(opt_train_m)["plant_median"] - base_sum["train"]["plant_median"],
            "road_cover_mean": summarize(opt_train_m)["road_cover_mean"] - base_sum["train"]["road_cover_mean"],
            "link_components_mean": summarize(opt_train_m)["link_components_mean"] - base_sum["train"].get("link_components_mean", 0),
        },
        "n_frozen_road_centers": len(ROAD_CENTERS),
        "n_frozen_plant_centers": len(PLANT_CENTERS),
    }
    (OUT / "BEST_PARAMS.json").write_text(json.dumps(opt_sum, indent=2), encoding="utf-8")

    # export layers + overlays for val + sample train
    def dump(feats, split: str):
        for cid, img, feat in feats:
            payload = {
                "case_id": cid,
                "split": split,
                "roads": feat["roads"],
                "plants": feat["plants"],
                "shrubs": feat["shrubs"],
                "metrics": feat["metrics"],
            }
            (OUT / "vectors" / f"{split}_{cid}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            # layers
            Image.fromarray((feat["road_mask"] * 255).astype(np.uint8)).save(OUT / "layers" / f"{split}_{cid}_road_mask.png")
            Image.fromarray((feat["skel"] * 255).astype(np.uint8)).save(OUT / "layers" / f"{split}_{cid}_road_centerline.png")
            # plant points raster
            pt = np.zeros(img.shape[:2], dtype=np.uint8)
            for q in feat["plants"]:
                x, y = int(round(q["center_xy"][0])), int(round(q["center_xy"][1]))
                if 0 <= y < pt.shape[0] and 0 <= x < pt.shape[1]:
                    pt[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = 255
            Image.fromarray(pt).save(OUT / "layers" / f"{split}_{cid}_plant_points.png")
            overlay(img, feat).save(OUT / "overlays" / f"{split}_{cid}.png")

    dump(opt_val_f, "val")
    dump(opt_train_f[:12], "train")

    # paired baseline vs opt montage (val)
    rows_img = []
    for (cid, img, feat_b), (_, _, feat_o) in zip(base_val_f, opt_val_f):
        left = overlay(img, feat_b)
        right = overlay(img, feat_o)
        tile = np.concatenate([img, np.array(left), np.array(right)], axis=1)
        im = Image.fromarray(tile)
        ImageDraw.Draw(im).text((5, 5), f"{cid}  raw | baseline | optimized", fill=(255, 0, 0))
        rows_img.append(np.array(im))
    Image.fromarray(np.concatenate(rows_img, axis=0)).save(OUT / "VAL_BASELINE_VS_OPT.png")

    # per-case csv
    with open(OUT / "val_metrics.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["case_id"] + [k for k in opt_val_m[0] if k != "case_id"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(opt_val_m)
    with open(OUT / "train_metrics.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["case_id"] + [k for k in opt_train_m[0] if k != "case_id"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(opt_train_m)

    bt, bv = base_sum["train"], base_sum["val"]
    ot, ov = opt_sum["train"], opt_sum["val"]
    report = f"""# Road + Plant Optimization Report

Aligned with AutoResearch `AUTORESEARCH_VECTOR_OPT.md`.

## Baseline (vector_v2 BEST, no frozen multi-center)

| metric | train | val |
|--------|------:|----:|
| road_cover | {bt['road_cover_mean']:.3f} | {bv['road_cover_mean']:.3f} |
| ep_median | {bt['ep_median']:.1f} | {bv['ep_median']:.1f} |
| skel_median | {bt['skel_median']:.1f} | {bv['skel_median']:.1f} |
| plant_median | {bt['plant_median']:.1f} | {bv['plant_median']:.1f} |
| mean_plant_circ | {bt['mean_plant_circ']:.3f} | {bv['mean_plant_circ']:.3f} |

## Optimized (frozen palette + close/prune/link + stricter trees)

Params: `{json.dumps(asdict(best_p))}`

| metric | train | val |
|--------|------:|----:|
| road_cover | {ot['road_cover_mean']:.3f} | {ov['road_cover_mean']:.3f} |
| ep_median | {ot['ep_median']:.1f} | {ov['ep_median']:.1f} |
| skel_median | {ot['skel_median']:.1f} | {ov['skel_median']:.1f} |
| plant_median | {ot['plant_median']:.1f} | {ov['plant_median']:.1f} |
| shrub_median | {ot['shrub_median']:.1f} | {ov['shrub_median']:.1f} |
| mean_plant_circ | {ot['mean_plant_circ']:.3f} | {ov['mean_plant_circ']:.3f} |
| link_components_mean | {ot['link_components_mean']:.2f} | {ov['link_components_mean']:.2f} |

## Deltas (train)

{json.dumps(opt_sum['delta_train'], indent=2)}

## Reading

- Lower `ep_median` with skel length not collapsed → cleaner roads (good).
- Lower `plant_median` toward 15–40 with high circularity → fewer lawn fragments as fake trees (good).
- Always check `n_link_components` — links can create false connections; see overlays.
- Empty orange-road cases remain; pavement-as-path not auto-merged.

## Outputs

- `BEST_PARAMS.json`, `search_log.csv`
- `layers/*_road_mask.png`, `*_road_centerline.png`, `*_plant_points.png`
- `overlays/`, `VAL_BASELINE_VS_OPT.png`
- `vectors/*.json`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("DONE", OUT, flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (OUT / "FAILURE.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
