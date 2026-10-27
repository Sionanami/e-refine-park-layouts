#!/usr/bin/env python
"""Strict semantic color-block paint in fixed layer order.

Order (bottom → top, later overwrites earlier):
  1. 草地 green   (0, 255, 0)
  2. 水体 water   (0, 255, 255)
  3. 道路 road    (241, 145, 73)
  4. 铺装 pavement (255, 255, 0)  # 铺装在道路之上
  5. 建筑 building (255, 0, 255)
  6. 树木 plant   (0, 152, 67)    # 树木在铺装/建筑之上（最顶除紫路）

Policy:
  - Exact palette RGB only (no AA blends).
  - Outside site: keep A.
  - Element classes (water/pavement/building/plant) prefer **B** when available —
    raw composeY usually lacks them; Y alone cannot invent buildings/trees/water.
  - Road may come from B and/or Y (union of centerlines).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.morphology import skeletonize

_HERE = Path(__file__).resolve().parent
ROOT = _HERE
NIGHT = _HERE
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from site_compose import compose_Y  # noqa: E402

PALETTE = {
    "green": (0, 255, 0),
    "water": (0, 255, 255),
    "pavement": (255, 255, 0),
    "road": (241, 145, 73),
    "building": (255, 0, 255),
    "plant": (0, 152, 67),
    "urban_road": (137, 86, 161),
}

LAYER_ORDER = (
    ("green", "草地"),
    ("water", "水体"),
    ("road", "道路"),
    ("pavement", "铺装"),  # above road
    ("building", "建筑"),
    ("plant", "树木"),
)

# Element classes that must stay visible — take from B when possible
ELEMENT_KEYS = ("water", "pavement", "building", "plant")


def _tol(img: np.ndarray, rgb, tol: int) -> np.ndarray:
    x = img.astype(np.int16)
    c = np.array(rgb, dtype=np.int16)
    return np.max(np.abs(x - c), axis=2) <= tol


def site_mask(a: np.ndarray) -> np.ndarray:
    return _tol(a, PALETTE["green"], 30)


def _clean_cc(
    mask: np.ndarray,
    min_area: int = 8,
    max_area: Optional[int] = None,
    max_box: Optional[int] = None,
) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return m
    lab, n = ndimage.label(m.astype(np.uint8))
    keep = np.zeros_like(m)
    for i in range(1, n + 1):
        comp = lab == i
        area = int(comp.sum())
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if max_box is not None:
            ys, xs = np.where(comp)
            if (ys.max() - ys.min() + 1) > max_box or (xs.max() - xs.min() + 1) > max_box:
                continue
        keep |= comp
    return keep


def extract_layer_masks(
    src: np.ndarray,
    *,
    road_as_centerline: bool = True,
    road_width: int = 2,
    plant_as_disks: bool = True,
    loose: bool = False,
    drop_over_area: bool = True,
) -> Dict[str, np.ndarray]:
    """Detect class masks from a source RGB (B or raw Y)."""
    r = src[..., 0].astype(np.int16)
    g = src[..., 1].astype(np.int16)
    b = src[..., 2].astype(np.int16)
    wt, bt, pt = (36, 40, 28) if loose else (24, 28, 18)

    green = _tol(src, PALETTE["green"], 45) | (
        (g > 190) & (g > r + 45) & (g > b + 45) & (r < 140) & (b < 140)
    )

    water = _tol(src, PALETTE["water"], wt) | (
        (b > 160) & (g > 160) & (r < 140) & (b > r + 40) & (g > r + 40)
    )
    if loose:
        # cyan / teal blobs common in raw Y
        water = water | ((b > 140) & (g > 130) & (r < 160) & ((b + g) > 2 * r + 40))
    water = _clean_cc(water, min_area=8 if loose else 12)

    pavement = _tol(src, PALETTE["pavement"], 32) | (
        (r > 190) & (g > 170) & (b < 140) & (r > b + 55) & (g > b + 45)
    )
    pavement = _clean_cc(pavement, min_area=8)

    road = _tol(src, PALETTE["road"], 24) | (
        (r > 175) & (g > 85) & (g < 200) & (b < 140) & (r > g + 12) & (g > b + 5)
    )
    road = _clean_cc(road, min_area=4)
    if road_as_centerline and road.any():
        sk = skeletonize(road.astype(bool))
        if road_width > 1:
            sk = ndimage.binary_dilation(sk, structure=np.ones((road_width, road_width)))
        road = sk

    building = _tol(src, PALETTE["building"], bt) | (
        (r > 160) & (b > 160) & (g < 140) & (r > g + 40) & (b > g + 40)
    )
    if loose:
        building = building | ((r > 150) & (b > 140) & (g < 160) & (r + b > 2 * g + 30))
    # Cap building blobs early — gen often over-paints magenta
    building = _clean_cc(
        building,
        min_area=12 if loose else 16,
        max_area=(4000 if loose else 6000) if drop_over_area else None,
    )

    plant = _tol(src, PALETTE["plant"], pt) | (
        (g > 90) & (g < 210) & (r < 110) & (b < 130) & (g > r + 25) & (g > b + 10)
    )
    if loose:
        # stricter: avoid muddy green/yellow mislabel as trees
        plant = plant | ((g > 100) & (g < 200) & (r < 100) & (b < 120) & (g > r + 30) & (g > b + 15))
        plant = plant & ~((g > 240) & (r < 40) & (b < 40))
        plant = plant & ~((r > 160) & (g > 140))  # not pavement/road oranges
    plant = _clean_cc(plant, min_area=6, max_area=(220 if loose else 160) if drop_over_area else None)

    if plant_as_disks and plant.any():
        lab, n = ndimage.label(plant.astype(np.uint8))
        disks = np.zeros_like(plant)
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            area = int(ys.size)
            if area < 4 or area > 800:
                continue
            # preserve diameter from blob area: r ≈ sqrt(A/π); do not force tiny 3px trees
            rad = float(max(2.0, min(12.0, np.sqrt(area / np.pi))))
            bh, bw = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
            if bh > 48 or bw > 48:
                # elongated / canopy cluster: keep original filled mask
                disks[lab == i] = True
                continue
            cy, cx = float(ys.mean()), float(xs.mean())
            yy, xx = np.ogrid[: src.shape[0], : src.shape[1]]
            disks |= (yy - cy) ** 2 + (xx - cx) ** 2 <= rad**2
        if disks.any():
            plant = disks

    return {
        "green": green.astype(bool),
        "water": water.astype(bool),
        "pavement": pavement.astype(bool),
        "road": road.astype(bool),
        "building": building.astype(bool),
        "plant": plant.astype(bool),
    }


def paint_strict_layers(
    masks: Dict[str, np.ndarray],
    shape: Tuple[int, int],
    *,
    site: Optional[np.ndarray] = None,
) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if site is None:
        site = np.ones((h, w), dtype=bool)
    out[site] = PALETTE["green"]
    for name, _zh in LAYER_ORDER[1:]:
        m = masks.get(name)
        if m is None:
            continue
        m = m.astype(bool) & site
        if m.any():
            out[m] = PALETTE[name]
    return out


def merge_masks(
    primary: Dict[str, np.ndarray],
    secondary: Dict[str, np.ndarray],
    *,
    prefer_primary_elements: bool = True,
) -> Dict[str, np.ndarray]:
    """Merge B (primary) + Y (secondary). Elements from B; road = B∪Y."""
    out = {k: primary[k].copy() for k in primary}
    if prefer_primary_elements:
        for k in ELEMENT_KEYS:
            # if B empty for a class, fall back to Y
            if not out[k].any() and secondary.get(k) is not None:
                out[k] = secondary[k]
        out["road"] = primary["road"] | secondary.get("road", np.zeros_like(primary["road"]))
    else:
        for k in out:
            out[k] = primary[k] | secondary.get(k, np.zeros_like(primary[k]))
    out["green"] = primary.get("green", out.get("green"))
    return out


def strict_semantic_from_rgb(
    src_rgb: np.ndarray,
    a_rgb: Optional[np.ndarray] = None,
    *,
    element_rgb: Optional[np.ndarray] = None,
    road_as_centerline: bool = True,
    road_width: int = 2,
    plant_as_disks: bool = True,
    compose_with_a: bool = True,
    loose_y: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Paint strict layers.

    - element_rgb: usually B — supplies water/pavement/building/plant
    - src_rgb: structure/road source (Y or B)
    """
    site = site_mask(a_rgb) if a_rgb is not None else np.ones(src_rgb.shape[:2], dtype=bool)

    def _resize(im, shape):
        if im.shape[:2] == shape:
            return im
        return np.array(Image.fromarray(im).resize((shape[1], shape[0]), Image.NEAREST))

    src_rgb = np.asarray(src_rgb)
    if element_rgb is not None:
        element_rgb = _resize(np.asarray(element_rgb), src_rgb.shape[:2])
    if a_rgb is not None:
        a_rgb = _resize(np.asarray(a_rgb), src_rgb.shape[:2])
        site = site_mask(a_rgb)

    m_struct = extract_layer_masks(
        src_rgb,
        road_as_centerline=road_as_centerline,
        road_width=road_width,
        plant_as_disks=plant_as_disks,
        loose=loose_y,
    )
    if element_rgb is not None:
        m_elem = extract_layer_masks(
            element_rgb,
            road_as_centerline=road_as_centerline,
            road_width=road_width,
            plant_as_disks=plant_as_disks,
            loose=False,
        )
        masks = merge_masks(m_elem, m_struct, prefer_primary_elements=True)
    else:
        masks = m_struct

    masks["green"] = site
    # keep elements inside site only
    for k in masks:
        masks[k] = masks[k] & site

    painted = paint_strict_layers(masks, src_rgb.shape[:2], site=site)
    if compose_with_a and a_rgb is not None:
        y, _ = compose_Y(painted, a_rgb)
        return y, masks
    return painted, masks


def strict_semantic_pil(
    src: Image.Image,
    a: Optional[Image.Image] = None,
    *,
    element: Optional[Image.Image] = None,
    **kwargs,
) -> Image.Image:
    src_rgb = np.array(src.convert("RGB"))
    a_rgb = np.array(a.convert("RGB")) if a is not None else None
    el_rgb = np.array(element.convert("RGB")) if element is not None else None
    y, _ = strict_semantic_from_rgb(src_rgb, a_rgb, element_rgb=el_rgb, **kwargs)
    return Image.fromarray(y)


def layer_legend_strip(height: int = 24) -> Image.Image:
    cell = 90
    im = Image.new("RGB", (cell * len(LAYER_ORDER), height), (30, 30, 32))
    d = ImageDraw.Draw(im)
    for i, (name, zh) in enumerate(LAYER_ORDER):
        x0 = i * cell
        d.rectangle([x0, 0, x0 + 18, height - 1], fill=PALETTE[name])
        d.text((x0 + 22, 5), f"{i + 1}.{zh}", fill=(240, 240, 240))
    return im


def _frac(arr: np.ndarray, rgb) -> float:
    return float(np.all(arr == np.array(rgb, dtype=np.uint8), axis=2).mean())


def main_export_val():
    """Val10: B supplies water/building/plant/pavement; road from B∪Y; strict order paint."""
    from extract_circulation_network import OptParams, extract_circulation, overlay

    panels = NIGHT / "10_night_site_road" / "stage5_raw_opt" / "panels_v2300_best"
    out = ROOT / "10_fused_gen_quality" / "strict_semantic_v2"
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("strict_Y", "strict_B", "overlays_Y", "overlays_B", "masks", "panels"):
        (out / sub).mkdir(exist_ok=True)

    val = NIGHT / "data_manifests_fixed_v1" / "val.csv"
    rows = list(csv.DictReader(val.open(encoding="utf-8")))
    legend = layer_legend_strip()
    legend.save(out / "LEGEND.png")
    stats = []
    params = OptParams(road_close=1, spur_len=4, link_dist=6, link_cos=0.25)

    for r in rows:
        cid = str(r.get("base_case_id") or r.get("case_id"))
        y_path = panels / cid / "composeY.png"
        if not y_path.exists():
            print("skip", cid)
            continue
        a = Image.open(r["A_path"]).convert("RGB")
        b = Image.open(r["B_path"]).convert("RGB")
        y_raw = Image.open(y_path).convert("RGB")

        # Full elements from B; roads from B∪Y
        y_strict = strict_semantic_pil(
            y_raw,
            a,
            element=b,
            road_as_centerline=True,
            road_width=2,
            plant_as_disks=True,
            loose_y=True,
        )
        b_strict = strict_semantic_pil(
            b,
            a,
            element=b,
            road_as_centerline=True,
            road_width=2,
            plant_as_disks=True,
        )
        y_strict.save(out / "strict_Y" / f"{cid}.png")
        b_strict.save(out / "strict_B" / f"{cid}.png")
        y_strict.resize(y_raw.size, Image.NEAREST).save(panels / cid / "composeY_strict.png")
        y_strict.resize(y_raw.size, Image.NEAREST).save(panels / cid / "composeY_snapped.png")

        def to256(im: Image.Image) -> np.ndarray:
            return np.array(im.resize((256, 256), Image.NEAREST))

        a256, b256, y256 = to256(a), to256(b_strict), to256(y_strict)
        feat_b = extract_circulation(b256, a256, params, 18, 16, 0.35, True)
        feat_y = extract_circulation(y256, a256, params, 18, 16, 0.35, True)
        overlay(b256, feat_b).save(out / "overlays_B" / f"{cid}.png")
        overlay(y256, feat_y).save(out / "overlays_Y" / f"{cid}.png")

        _, masks = strict_semantic_from_rgb(
            to256(y_raw), a256, element_rgb=to256(b), compose_with_a=False, loose_y=True
        )
        tiles = []
        for name, _zh in LAYER_ORDER:
            tile = np.zeros((256, 256, 3), dtype=np.uint8)
            tile[:] = (40, 40, 42)
            tile[masks[name]] = PALETTE[name]
            tiles.append(tile)
        Image.fromarray(np.concatenate(tiles, 1)).save(out / "masks" / f"{cid}_layers.png")

        fr = {name: _frac(y256, PALETTE[name]) for name, _ in LAYER_ORDER}
        stats.append({"case_id": cid, "fractions": fr})
        print(
            cid,
            f"water={fr['water']:.3f} bldg={fr['building']:.3f} plant={fr['plant']:.3f} "
            f"road={fr['road']:.3f} pav={fr['pavement']:.3f}",
            flush=True,
        )

        strip = Image.new("RGB", (256 * 5 + 16, 256 + 28 + legend.height), (18, 18, 20))
        strip.paste(legend.resize((strip.width, legend.height)), (0, 0))
        imgs = [
            Image.fromarray(a256),
            Image.fromarray(b256),
            to256(y_raw).__array__() if False else Image.fromarray(to256(y_raw)),
            Image.fromarray(y256),
            overlay(y256, feat_y),
        ]
        # fix: to256 returns ndarray
        imgs = [
            Image.fromarray(a256),
            Image.fromarray(b256),
            Image.fromarray(to256(y_raw)),
            Image.fromarray(y256),
            overlay(y256, feat_y),
        ]
        labs = ["A", "B严格", "Y原图", "Y严格(B要素)", "overlay"]
        d = ImageDraw.Draw(strip)
        for i, (im, lab) in enumerate(zip(imgs, labs)):
            x = i * (256 + 4)
            d.text((x + 2, legend.height + 2), lab, fill=(200, 220, 255))
            strip.paste(im, (x, legend.height + 18))
        strip.save(out / "panels" / f"{cid}_strip.png")

    med = {
        k: float(np.median([s["fractions"][k] for s in stats]))
        for k, _ in LAYER_ORDER
    }
    meta = {
        "layer_order": [{"name": n, "zh": z, "rgb": list(PALETTE[n])} for n, z in LAYER_ORDER],
        "policy": (
            "Paint order grass→water→road→pavement→building→plant (pavement above road). "
            "Water/building/plant/pavement from B; road = B∪Y centerline; outside site = A."
        ),
        "median_fractions": med,
        "cases": stats,
    }
    (out / "STRICT_SEMANTIC.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("MEDIAN", med, flush=True)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main_export_val()
