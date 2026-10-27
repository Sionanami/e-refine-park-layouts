# -*- coding: utf-8 -*-
"""Training targets for park layout.

Modes:
  - site_compose: Y = M*B + (1-M)*A  (legacy RGB B inside site)
  - centerline_compose: clean palette layout from B masks + thin road centerline
    (Sat2Graph-style: supervise structure, not AA blobs)
  - fused_centerline_compose: road∪pavement circulation + plant-bridge (overnight fuse)
  - proportion_compose: B-faithful shapes/fracs; thin road-only skel; z=road→pav→bldg→plant
  - skeleton_first_compose: thin circulation only (+ compact buildings/water); kills blob fills
"""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.morphology import skeletonize

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from site_compose import compose_Y  # noqa: E402

PALETTE = {
    "building": (255, 0, 255),
    "pavement": (255, 255, 0),
    "green": (0, 255, 0),
    "road": (241, 145, 73),
    "water": (0, 255, 255),
    "plant": (0, 152, 67),
    "urban_road": (137, 86, 161),
    "redline": (220, 20, 60),
}

# Editable park classes only (never urban_road / redline / outside)
SEMANTIC_KEYS = ("building", "water", "plant", "road", "pavement")

CENTERLINE_ROOT = Path(os.environ.get("SD15_CENTERLINE", str(_HERE / "centerline_targets")))


def compose_train_target(a_img: Image.Image, b_img: Image.Image) -> Image.Image:
    a = a_img.convert("RGB")
    b = b_img.convert("RGB")
    if b.size != a.size:
        b = b.resize(a.size, Image.NEAREST)
    y, _meta = compose_Y(np.array(b), np.array(a))
    return Image.fromarray(y)


def _mask_tol(img: np.ndarray, rgb, tol: int) -> np.ndarray:
    x = img.astype(np.int16)
    c = np.array(rgb, dtype=np.int16)
    return np.all(np.abs(x - c) <= tol, axis=2)


def _road_mask(img: np.ndarray, tol: int = 10) -> np.ndarray:
    m = _mask_tol(img, PALETTE["road"], tol)
    r, g, b = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16), img[..., 2].astype(np.int16)
    heur = (r > 195) & (g > 95) & (g < 195) & (b < 130) & (r > g + 20) & (g > b + 10)
    return m | heur


def _site_mask(a: np.ndarray) -> np.ndarray:
    return _mask_tol(a, PALETTE["green"], 30)


def build_clean_element_canvas(b_rgb: np.ndarray, *, road_width: int = 2) -> np.ndarray:
    """Discrete palette layout: grass→water→road→pavement→building→tree (pav above road)."""
    try:
        from strict_semantic_layers import strict_semantic_from_rgb

        painted, _ = strict_semantic_from_rgb(
            b_rgb,
            None,
            road_as_centerline=True,
            road_width=road_width,
            plant_as_disks=True,
            compose_with_a=False,
        )
        return painted
    except Exception:
        pass
    h, w = b_rgb.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # 1 grass
    out[:] = PALETTE["green"]
    # 2 water → 3 road → 4 pavement (covers road) → 5 building → 6 plant
    for name, tol in (("water", 14), ("building", 18)):
        m = _mask_tol(b_rgb, PALETTE[name], tol)
        lab, n = ndimage.label(m.astype(np.uint8))
        keep = np.zeros_like(m)
        min_a = 8 if name != "building" else 20
        for i in range(1, n + 1):
            comp = lab == i
            if int(comp.sum()) >= min_a:
                keep |= comp
        out[keep] = PALETTE[name]
    rm = _road_mask(b_rgb, 10)
    if rm.any():
        sk = skeletonize(rm)
        if road_width > 1:
            sk = ndimage.binary_dilation(sk, structure=np.ones((road_width, road_width)))
        out[sk] = PALETTE["road"]
    pav = _mask_tol(b_rgb, PALETTE["pavement"], 14)
    lab, n = ndimage.label(pav.astype(np.uint8))
    keep = np.zeros_like(pav)
    for i in range(1, n + 1):
        comp = lab == i
        if int(comp.sum()) >= 8:
            keep |= comp
    out[keep] = PALETTE["pavement"]
    plant = _mask_tol(b_rgb, PALETTE["plant"], 14)
    lab, n = ndimage.label(plant.astype(np.uint8))
    im = Image.fromarray(out)
    draw = ImageDraw.Draw(im)
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        area = int(ys.size)
        if area < 8 or area > 120:
            continue
        if (ys.max() - ys.min() + 1) > 24 or (xs.max() - xs.min() + 1) > 24:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=PALETTE["plant"])
    return np.array(im)


def compose_centerline_train_target(
    a_img: Image.Image,
    b_img: Image.Image,
    *,
    case_id: str | None = None,
    road_width: int = 2,
) -> Image.Image:
    """Clean element map inside site; outside site keep A (constraints)."""
    a = a_img.convert("RGB")
    b = b_img.convert("RGB")
    if b.size != a.size:
        b = b.resize(a.size, Image.NEAREST)
    a_np = np.array(a)
    b_np = np.array(b)

    # prefer pre-exported centerline if present
    clean = build_clean_element_canvas(b_np, road_width=road_width)
    if case_id:
        # try train then val
        for split in ("train", "val"):
            p = CENTERLINE_ROOT / split / "centerline" / f"{case_id}_bin.png"
            if p.exists():
                cl = np.array(Image.open(p).convert("L")) > 127
                if cl.shape != clean.shape[:2]:
                    cl = np.array(Image.fromarray(cl.astype(np.uint8) * 255).resize((clean.shape[1], clean.shape[0]), Image.NEAREST)) > 127
                if road_width > 1:
                    cl = ndimage.binary_dilation(cl, structure=np.ones((road_width, road_width)))
                # redraw roads from exported centerline (authoritative)
                site = _site_mask(a_np)
                # clear previous road inside site then paint
                road_px = np.all(clean == PALETTE["road"], axis=2)
                clean[road_px & site] = PALETTE["green"]
                clean[cl & site] = PALETTE["road"]
                # pavement above road, then building/plant on top
                pav = _mask_tol(b_np, PALETTE["pavement"], 18) & site
                lab, n = ndimage.label(pav.astype(np.uint8))
                for i in range(1, n + 1):
                    comp = lab == i
                    if int(comp.sum()) >= 24:
                        clean[comp] = PALETTE["pavement"]
                clean = _repaint_building_plant(clean, b_np, site)
                break

    y, _ = compose_Y(clean, a_np)
    return Image.fromarray(y)


def _clean_class_mask(b_np: np.ndarray, name: str, *, tol: int = 14, min_area: int = 8) -> np.ndarray:
    m = _mask_tol(b_np, PALETTE[name], tol)
    lab, n = ndimage.label(m.astype(np.uint8))
    keep = np.zeros_like(m)
    for i in range(1, n + 1):
        comp = lab == i
        if int(comp.sum()) >= min_area:
            keep |= comp
    return keep


def _road_only_skel(b_np: np.ndarray) -> np.ndarray:
    """Thin skeleton from park roads only — no pavement union, no plant bridge (avoids dense nets)."""
    try:
        from expand_road_gamut import road_mask_expanded

        road = road_mask_expanded(b_np, center_tol=16, include_pavement=False)
    except Exception:
        road = _road_mask(b_np, 12)
    if not road.any():
        return np.zeros(b_np.shape[:2], dtype=bool)
    # light close then skeleton; drop tiny spurs via size filter on components before skel
    road = ndimage.binary_closing(road, structure=np.ones((2, 2)))
    return skeletonize(road.astype(bool))


def load_class_area_budget(cls: str = "building") -> dict:
    """Per-class train frac policy: hard ≤ p90; target in [p25,p75] median band."""
    defaults = {
        "building": {"max_hard": 0.079, "target_max": 0.037, "target_min": 0.0, "p50": 0.015},
        "plant": {"max_hard": 0.361, "target_max": 0.296, "target_min": 0.143, "p50": 0.229},
    }
    out = dict(defaults.get(cls, {"max_hard": 0.25, "target_max": 0.20, "target_min": 0.0, "p50": 0.10}))
    priors = Path(__file__).resolve().parent / "COMPOSITION_PRIORS.json"
    if priors.exists():
        try:
            blob = json.loads(priors.read_text(encoding="utf-8"))
            b = (blob.get("classes") or {}).get(cls) or {}
            out["max_hard"] = float(b.get("max_hard", b.get("p90", out["max_hard"])))
            out["target_max"] = float(b.get("target_max", b.get("p75", out["target_max"])))
            out["target_min"] = float(b.get("target_min", b.get("p25", out["target_min"])))
            out["p50"] = float(b.get("p50", out["p50"]))
        except Exception:
            pass
    return out


def load_building_area_budget() -> dict:
    return load_class_area_budget("building")


def clamp_building_mask(bld: np.ndarray, site: np.ndarray, max_frac: float) -> np.ndarray:
    """Keep largest components until area/site ≤ max_frac (works for any class mask)."""
    from scipy import ndimage as _ndi

    b = bld.astype(bool) & site.astype(bool)
    site_n = max(int(site.sum()), 1)
    budget = int(max_frac * site_n)
    if int(b.sum()) <= budget:
        return b
    lab, n = _ndi.label(b.astype(np.uint8))
    if n == 0:
        return b
    sizes = [(int(_ndi.sum(b, lab, index=i)), i) for i in range(1, n + 1)]
    sizes.sort(reverse=True)
    keep = np.zeros_like(b)
    used = 0
    for area, i in sizes:
        if used + area > budget and used > 0:
            break
        if used + area > budget and used == 0:
            comp = lab == i
            while int(comp.sum()) > budget and comp.any():
                comp = _ndi.binary_erosion(comp)
            keep |= comp
            used = int(keep.sum())
            break
        keep |= lab == i
        used += area
        if used >= budget:
            break
    return keep


def _repaint_building_plant(clean: np.ndarray, b_np: np.ndarray, site: np.ndarray) -> np.ndarray:
    """Z-order top: building then plants (diameter from B area: r≈√(A/π)).

    Buildings ≤ train p75; plants clamped into train median band [p25,p75]
    (oversized canopies cut; do not invent trees if B is sparse).
    Never paint plant/building over road∪pavement circulation.
    """
    out = clean.copy()
    circ = (
        np.all(out == PALETTE["road"], axis=2)
        | np.all(out == PALETTE["pavement"], axis=2)
    ) & site
    bld = _clean_class_mask(b_np, "building", tol=18, min_area=12) & site & ~circ
    bud_b = load_class_area_budget("building")
    bld = clamp_building_mask(bld, site, float(bud_b["target_max"]))
    out[bld] = PALETTE["building"]
    plant = _mask_tol(b_np, PALETTE["plant"], 14) & site & ~circ & ~bld
    lab, n = ndimage.label(plant.astype(np.uint8))
    im = Image.fromarray(out)
    draw = ImageDraw.Draw(im)
    elongated = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        area = int(ys.size)
        if area < 6 or area > 800:
            continue
        bh, bw = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
        if bh > 48 or bw > 48:
            elongated.append(i)
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        rad = float(max(2.0, min(12.0, np.sqrt(area / np.pi))))
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=PALETTE["plant"])
    out = np.array(im)
    for i in elongated:
        out[lab == i] = PALETTE["plant"]
    # re-assert circulation / building on top of disks that spilled
    out[circ] = clean[circ]
    out[bld] = PALETTE["building"]
    # clamp plant area to median-band upper (p75)
    plant_px = np.all(out == PALETTE["plant"], axis=2) & site
    bud_p = load_class_area_budget("plant")
    plant_px = clamp_building_mask(plant_px, site, float(bud_p["target_max"]))
    # clear oversized plant then repaint kept
    was_plant = np.all(out == PALETTE["plant"], axis=2) & site
    out[was_plant] = PALETTE["green"]
    out[plant_px] = PALETTE["plant"]
    out[circ] = clean[circ]
    out[bld] = PALETTE["building"]
    return out


def apply_hard_constraint_to_target(
    target: Image.Image,
    a_img: Image.Image,
    *,
    building_max_frac: float | None = None,
) -> Image.Image:
    """Bake geometry-preserving hard constraint into the training/supervision RGB target.

    Default building cap = train p90 (COMPOSITION_PRIORS max_hard), not the 25% outlier max.
    """
    from hard_constraint_compose import hard_constraint_compose

    if building_max_frac is None:
        building_max_frac = float(load_building_area_budget()["max_hard"])
    a = a_img.convert("RGB")
    t = target.convert("RGB")
    if t.size != a.size:
        t = t.resize(a.size, Image.NEAREST)
    y, _ = hard_constraint_compose(
        np.array(t),
        np.array(a),
        building_max_frac=float(building_max_frac),
        preserve_geometry=True,
        road_as_centerline=False,
        plant_as_disks=False,
        road_width=2,
    )
    return Image.fromarray(y)


def _circulation_skel_from_b(b_np: np.ndarray, a_np: np.ndarray | None) -> np.ndarray:
    """Fused circulation skeleton: expand road ∪ pavement + plant bridge (scaled by res)."""
    h = b_np.shape[0]
    # ablation @256 → dist=16; scale linearly
    bridge_dist = max(8, int(round(16 * (h / 256.0))))
    try:
        from extract_circulation_network import OptParams, extract_circulation

        feat = extract_circulation(
            b_np,
            a_np,
            OptParams(road_close=1, spur_len=4, link_dist=6, link_cos=0.25),
            center_tol=18,
            plant_bridge_dist=bridge_dist,
            plant_gap_frac=0.35,
            enable_plant_bridge=True,
        )
        return feat["skel"].astype(bool)
    except Exception:
        # fallback: expand road ∪ yellow pavement → skeleton
        try:
            from expand_road_gamut import road_mask_expanded

            road = road_mask_expanded(b_np, center_tol=18, include_pavement=False)
        except Exception:
            road = _road_mask(b_np, 12)
        pav = _mask_tol(b_np, PALETTE["pavement"], 18)
        circ = road | pav
        if not circ.any():
            return np.zeros(b_np.shape[:2], dtype=bool)
        return skeletonize(circ.astype(bool))


def compose_fused_centerline_train_target(
    a_img: Image.Image,
    b_img: Image.Image,
    *,
    case_id: str | None = None,
    road_width: int = 2,
) -> Image.Image:
    """Like centerline_compose, but roads = fused circulation centerline (road∪pav + plant bridge)."""
    a = a_img.convert("RGB")
    b = b_img.convert("RGB")
    if b.size != a.size:
        b = b.resize(a.size, Image.NEAREST)
    a_np = np.array(a)
    b_np = np.array(b)

    clean = build_clean_element_canvas(b_np, road_width=road_width)
    site = _site_mask(a_np)
    sk = _circulation_skel_from_b(b_np, a_np)
    if sk.any():
        if road_width > 1:
            sk = ndimage.binary_dilation(sk, structure=np.ones((road_width, road_width)))
        # clear old orange roads inside site; paint fused circulation as road orange
        road_px = np.all(clean == PALETTE["road"], axis=2)
        clean[road_px & site] = PALETTE["green"]
        clean[sk & site] = PALETTE["road"]
        # pavement above road (not on skel), then building/plant on top of pavement
        pav = _mask_tol(b_np, PALETTE["pavement"], 18) & site & ~sk
        lab, n = ndimage.label(pav.astype(np.uint8))
        for i in range(1, n + 1):
            comp = lab == i
            if int(comp.sum()) >= 40:
                clean[comp] = PALETTE["pavement"]
        clean = _repaint_building_plant(clean, b_np, site)

    y, _ = compose_Y(clean, a_np)
    return Image.fromarray(y)


def compose_proportion_train_target(
    a_img: Image.Image,
    b_img: Image.Image,
    *,
    case_id: str | None = None,
    road_width: int = 2,
    apply_hard: bool = True,
) -> Image.Image:
    """B-faithful shapes/proportions + hard constraint baked into supervision.

    Z-order: grass → water → road → pavement → building → plant.
    Primary structure: **B road∪pavement circulation network** (topology teacher).
    Pavement: B yellow shapes, area-capped to ≈B×1.05 and ≤15% site.
    Building ≤p75; plant ∈[p25,p75] median band; neither covers circulation.
    """
    del case_id
    a = a_img.convert("RGB")
    b = b_img.convert("RGB")
    if b.size != a.size:
        b = b.resize(a.size, Image.NEAREST)
    a_np = np.array(a)
    b_np = np.array(b)
    site = _site_mask(a_np)
    site_n = max(int(site.sum()), 1)

    clean = build_clean_element_canvas(b_np, road_width=max(2, road_width))
    # replace roads with cleaned B road mask (keeps stroke width)
    road_px = np.all(clean == PALETTE["road"], axis=2)
    clean[road_px & site] = PALETTE["green"]
    try:
        from expand_road_gamut import road_mask_expanded

        road_m = road_mask_expanded(b_np, center_tol=16, include_pavement=False) & site
    except Exception:
        road_m = _road_mask(b_np, 12) & site
    road_m = ndimage.binary_closing(road_m, structure=np.ones((2, 2)))
    lab, n = ndimage.label(road_m.astype(np.uint8))
    keep_r = np.zeros_like(road_m)
    for i in range(1, n + 1):
        comp = lab == i
        if int(comp.sum()) >= 8:
            keep_r |= comp
    clean[keep_r] = PALETTE["road"]

    # pavement from B, area budget ≈ B; do not paint over roads/buildings/plants
    pav_px = np.all(clean == PALETTE["pavement"], axis=2)
    clean[pav_px & site] = PALETTE["green"]
    bld_m = _clean_class_mask(b_np, "building", tol=18, min_area=12) & site
    plant_m = _mask_tol(b_np, PALETTE["plant"], 14) & site
    pav = _mask_tol(b_np, PALETTE["pavement"], 16) & site & ~keep_r & ~bld_m & ~plant_m
    b_pav_frac = float((_mask_tol(b_np, PALETTE["pavement"], 16) & site).sum() / site_n)
    budget = int(min(0.15, b_pav_frac * 1.05) * site_n)
    lab, n = ndimage.label(pav.astype(np.uint8))
    comps = []
    for i in range(1, n + 1):
        comp = lab == i
        area = int(comp.sum())
        if area >= 30:
            comps.append((area, comp))
    comps.sort(key=lambda t: t[0], reverse=True)
    used = 0
    for area, comp in comps:
        if used >= budget:
            break
        if used + area > budget and used > 0:
            continue
        clean[comp] = PALETTE["pavement"]
        used += area

    water = _clean_class_mask(b_np, "water", tol=14, min_area=8) & site
    green_px = np.all(clean == PALETTE["green"], axis=2)
    clean[water & green_px] = PALETTE["water"]

    clean = _repaint_building_plant(clean, b_np, site)
    y, _ = compose_Y(clean, a_np)
    out = Image.fromarray(y)
    if apply_hard:
        # Supervise inside median band (≤p75); sample legality still uses p90 elsewhere
        out = apply_hard_constraint_to_target(
            out, a, building_max_frac=float(load_building_area_budget()["target_max"])
        )
    return out


def _masks_from_A(a_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (site_interior, urban_road, redline) bool masks from original A."""
    try:
        from mask_from_A import _green_fill, _largest_component, _purple_road, _red_border_strict
        from PIL import ImageFilter

        green = _green_fill(a_np)
        g_closed = np.array(Image.fromarray((green.astype(np.uint8) * 255)).filter(ImageFilter.MaxFilter(3))) > 0
        g_closed = np.array(Image.fromarray((g_closed.astype(np.uint8) * 255)).filter(ImageFilter.MinFilter(3))) > 0
        main = _largest_component(g_closed)
        main = main | (
            green & (np.array(Image.fromarray((main.astype(np.uint8) * 255)).filter(ImageFilter.MaxFilter(3))) > 0)
        )
        urban = _purple_road(a_np)
        red = _red_border_strict(a_np)
        site = main & (~urban)
        return site, urban, red
    except Exception:
        r, g, b = a_np[..., 0].astype(np.int16), a_np[..., 1].astype(np.int16), a_np[..., 2].astype(np.int16)
        site = (g > 180) & (g > r + 40) & (g > b + 40)
        rf, gf, bf = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
        urban = (rf > 80) & (bf > 80) & (gf < rf * 0.85) & (rf + bf > 200) & ~((rf > 220) & (gf > 220) & (bf > 220))
        red = (r > 180) & (g < 120) & (b < 120) & (r > g + 60) & (r > b + 60)
        lab, n = ndimage.label(site.astype(np.uint8))
        if n > 0:
            sizes = ndimage.sum(site, lab, index=np.arange(1, n + 1))
            site = lab == (int(np.argmax(sizes)) + 1)
        site = site & (~urban)
        return site, urban, red


def forbidden_mask_from_A(a_np: np.ndarray, *, urban_dilate: int = 1) -> np.ndarray:
    """禁生区：场外 + 城市道路 + 红线（可轻微膨胀城路，避免建筑色渗入路边）。"""
    site, urban, red = _masks_from_A(a_np)
    if urban_dilate > 0 and urban.any():
        urban = ndimage.binary_dilation(urban, structure=np.ones((urban_dilate * 2 + 1, urban_dilate * 2 + 1)))
    return (~site) | urban | red


def paste_forbidden_from_A(
    gen: np.ndarray | Image.Image,
    a: np.ndarray | Image.Image,
    *,
    urban_dilate: int = 1,
) -> np.ndarray:
    """把禁生区像素强制贴回原 A（城路/红线/场外保持不变）。"""
    if isinstance(gen, Image.Image):
        g = np.array(gen.convert("RGB"))
    else:
        g = np.asarray(gen).copy()
    if isinstance(a, Image.Image):
        a_np = np.array(a.convert("RGB"))
    else:
        a_np = np.asarray(a)
    if g.shape[:2] != a_np.shape[:2]:
        g = np.array(Image.fromarray(g).resize((a_np.shape[1], a_np.shape[0]), Image.NEAREST))
    forbid = forbidden_mask_from_A(a_np, urban_dilate=urban_dilate)
    out = g.copy()
    out[forbid] = a_np[forbid]
    return out


def sanitize_condition_A(a_img: Image.Image) -> Image.Image:
    """Adapter 条件：白底 + 场地绿 + 紫城路 + 红线（精确调色板，无底部图例条）。

    城路与红线是输入端硬限制，原 A 几何保持不变；仅场地内部可生成五类语义。
    """
    a = np.array(a_img.convert("RGB"))
    out = np.full_like(a, 255)
    site, urban, red = _masks_from_A(a)
    out[site] = PALETTE["green"]
    out[urban] = PALETTE["urban_road"]
    out[red] = PALETTE["redline"]
    # 城路/红线优先于场地绿（交叠时保持限制色）
    out[urban] = PALETTE["urban_road"]
    out[red] = PALETTE["redline"]
    return Image.fromarray(out)


def compose_skeleton_first_train_target(
    a_img: Image.Image,
    b_img: Image.Image,
    *,
    case_id: str | None = None,
    road_width: int = 2,
) -> Image.Image:
    """Thin fused circulation + compact buildings/water; no pavement/plant blob fills."""
    del case_id  # reserved for future cached skels
    a = a_img.convert("RGB")
    b = b_img.convert("RGB")
    if b.size != a.size:
        b = b.resize(a.size, Image.NEAREST)
    a_np = np.array(a)
    b_np = np.array(b)
    h, w = b_np.shape[:2]
    clean = np.zeros((h, w, 3), dtype=np.uint8)
    clean[:] = PALETTE["green"]
    site = _site_mask(a_np)

    for name, tol, min_a, max_box in (
        ("building", 18, 20, 80),
        ("water", 14, 12, 120),
    ):
        m = _mask_tol(b_np, PALETTE[name], tol) & site
        lab, n = ndimage.label(m.astype(np.uint8))
        for i in range(1, n + 1):
            comp = lab == i
            area = int(comp.sum())
            if area < min_a:
                continue
            ys, xs = np.where(comp)
            if (ys.max() - ys.min() + 1) > max_box or (xs.max() - xs.min() + 1) > max_box:
                continue
            clean[comp] = PALETTE[name]

    sk = _circulation_skel_from_b(b_np, a_np)
    if sk.any():
        if road_width > 1:
            sk = ndimage.binary_dilation(sk, structure=np.ones((road_width, road_width)))
        clean[sk & site] = PALETTE["road"]

    y, _ = compose_Y(clean, a_np)
    return Image.fromarray(y)


def palette_snap_raw(
    img: Image.Image | np.ndarray,
    tol: int = 40,
    a_img: Image.Image | np.ndarray | None = None,
    *,
    balance: bool = True,
) -> Image.Image:
    """Snap/paint to frozen palette.

    Default: strict layer order 草地→水体→道路→铺装→建筑→树木 (exact RGB; pav above road).
    balance=False: legacy nearest-center snap.
    """
    del tol  # kept for API compat
    if isinstance(img, Image.Image):
        src = img.convert("RGB")
    else:
        src = Image.fromarray(np.asarray(img)).convert("RGB")
    a_pil = None
    if a_img is not None:
        a_pil = a_img.convert("RGB") if isinstance(a_img, Image.Image) else Image.fromarray(np.asarray(a_img))

    if balance:
        try:
            from strict_semantic_layers import strict_semantic_pil

            # element=None here; callers with B should use strict_semantic_pil(..., element=b)
            return strict_semantic_pil(
                src, a_pil, road_as_centerline=True, road_width=2, plant_as_disks=True, loose_y=True
            )
        except Exception:
            pass

    rgb = np.array(src)
    h, w = rgb.shape[:2]
    a_np = np.array(a_pil) if a_pil is not None else None
    if a_np is not None and a_np.shape[:2] != (h, w):
        a_np = np.array(Image.fromarray(a_np).resize((w, h), Image.NEAREST))
    site = _site_mask(a_np) if a_np is not None else np.ones((h, w), dtype=bool)
    # ordered paint (pavement after road → covers road where they overlap)
    order = ("green", "water", "road", "pavement", "building", "plant")
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[site] = PALETTE["green"]
    x = rgb.astype(np.int16)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    masks = {
        "green": site,
        "water": site & ((b > 180) & (g > 180) & (r < 120)),
        "pavement": site & ((r > 200) & (g > 180) & (b < 120)),
        "road": site & ((r > 190) & (g > 90) & (g < 190) & (b < 130) & (r > g + 20)),
        "building": site & ((r > 180) & (b > 180) & (g < 120)),
        "plant": site & ((g > 100) & (g < 200) & (r < 90) & (b < 110) & (g > r + 35)),
    }
    for name in order[1:]:
        out[masks[name]] = PALETTE[name]
    if a_np is not None:
        y, _ = compose_Y(out, a_np)
        return Image.fromarray(y)
    return Image.fromarray(out)
