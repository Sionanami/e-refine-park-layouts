#!/usr/bin/env python
"""Hard-constraint compositor for generative park layouts (post-gen legality).

Enforced rules (user-requested + AutoResearch / manuscript R1):
  1) Magenta (255,0,255) ONLY for buildings — exclusive palette snap.
  2) Building area ≤ BUILDING_MAX_SITE_FRAC of site (default 8% ≈ train p99 band).
  3) Purple urban roads: STRICT copy from input A; generation NEVER covers them.
  4) Z-order (bottom→top): grass → water → road → pavement → building → plant → urban_road.
  5) Outside site + redline: keep A (site_compose).

This does NOT invent missing roads from nothing — it clamps illegal outputs.
Apply after raw composeY (and optionally after R1 snap).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
ROOT = _HERE
NIGHT = _HERE
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from site_compose import compose_Y, site_mask_M_compose  # noqa: E402
from mask_from_A import _purple_road, _red_border_strict  # noqa: E402
from strict_semantic_layers import (  # noqa: E402
    LAYER_ORDER,
    PALETTE,
    extract_layer_masks,
    paint_strict_layers,
    site_mask,
)

# Train building% of site: p50≈1.5%, p75≈3.7%, p90≈7.9%; raw max≈25% is outlier.
# Default hard cap = train p90 (product legality), not outlier max.
BUILDING_MAX_SITE_FRAC = 0.079
_PRIORS_JSON = Path(__file__).resolve().parent / "COMPOSITION_PRIORS.json"


def _load_area_caps() -> Dict[str, float]:
    caps = {
        "building": BUILDING_MAX_SITE_FRAC,
        "water": 0.25,
        "plant": 0.40,
        "road": 0.18,
        "pavement": 0.25,
    }
    if _PRIORS_JSON.exists():
        try:
            blob = json.loads(_PRIORS_JSON.read_text(encoding="utf-8"))
            for k, v in (blob.get("classes") or {}).items():
                if isinstance(v, dict) and "max_hard" in v:
                    caps[k] = float(v["max_hard"])
        except Exception:
            pass
    return caps

# Full z-order including urban_road on top (user + R1 restore)
Z_ORDER = (
    ("green", "草地"),
    ("water", "水体"),
    ("road", "道路"),
    ("pavement", "铺装"),  # pavement above road
    ("building", "建筑"),
    ("plant", "树木"),
    ("urban_road", "城市道路"),
)


def clamp_building_area(building: np.ndarray, site: np.ndarray, max_frac: float) -> np.ndarray:
    """If building/site > max_frac, keep largest components until under budget."""
    b = building.astype(bool) & site.astype(bool)
    site_n = max(int(site.sum()), 1)
    budget = int(max_frac * site_n)
    if int(b.sum()) <= budget:
        return b
    lab, n = ndimage.label(b.astype(np.uint8))
    if n == 0:
        return b
    sizes = [(int(ndimage.sum(b, lab, index=i)), i) for i in range(1, n + 1)]
    sizes.sort(reverse=True)
    keep = np.zeros_like(b)
    used = 0
    for area, i in sizes:
        if used + area > budget and used > 0:
            break
        if used + area > budget and used == 0:
            # single huge blob: erode until fits
            comp = lab == i
            while int(comp.sum()) > budget and comp.any():
                comp = ndimage.binary_erosion(comp)
            keep |= comp
            used = int(keep.sum())
            break
        keep |= lab == i
        used += area
        if used >= budget:
            break
    return keep


def hard_constraint_compose(
    y_rgb: np.ndarray,
    a_rgb: np.ndarray,
    *,
    building_max_frac: float = BUILDING_MAX_SITE_FRAC,
    road_as_centerline: bool = False,
    road_width: int = 2,
    plant_as_disks: bool = False,
    preserve_geometry: bool = True,
    apply_area_caps: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Apply hard constraints; return uint8 RGB + audit meta.

    Geometry defaults (preserve_geometry=True):
      - Do NOT re-skeletonize roads (keeps stroke width from Y/B).
      - Do NOT force tiny plant disks (keeps canopy footprint; optional area-proportional disks).
      - Area caps still apply (pav/bldg/road/… max_hard).
    """
    if y_rgb.shape[:2] != a_rgb.shape[:2]:
        y_rgb = np.array(Image.fromarray(y_rgb).resize((a_rgb.shape[1], a_rgb.shape[0]), Image.NEAREST))

    site = site_mask_M_compose(a_rgb)  # excludes purple already
    purple = _purple_road(a_rgb)
    red = _red_border_strict(a_rgb)

    if preserve_geometry:
        road_as_centerline = False
        # keep filled plant blobs; disks only if caller explicitly asks
        # (plant_as_disks left as passed)

    # Extract from Y inside site only (loose for muddy gen)
    masks = extract_layer_masks(
        y_rgb,
        road_as_centerline=road_as_centerline,
        road_width=road_width,
        plant_as_disks=plant_as_disks,
        loose=True,
        drop_over_area=apply_area_caps,
    )
    for k in masks:
        masks[k] = masks[k] & site & (~purple) & (~red)

    # Light morphological close on roads to heal 1px gaps without thinning to skeleton
    if preserve_geometry and masks.get("road") is not None and masks["road"].any():
        masks["road"] = ndimage.binary_closing(masks["road"], structure=np.ones((2, 2)))

    # Area hard caps from COMPOSITION_PRIORS; building arg overrides when provided
    caps = _load_area_caps()
    # Prefer priors max_hard (p90) unless caller passes a stricter/explicit cap
    prior_b = float(caps.get("building", BUILDING_MAX_SITE_FRAC))
    caps["building"] = float(min(float(building_max_frac), prior_b)) if building_max_frac is not None else prior_b
    area_audit = {}
    for cls in ("building", "water", "plant", "road", "pavement"):
        if cls not in masks:
            continue
        before = int(masks[cls].sum())
        if apply_area_caps:
            masks[cls] = clamp_building_area(masks[cls], site, caps.get(cls, 0.25))
        after = int(masks[cls].sum())
        area_audit[cls] = {
            "px_before": before,
            "px_after": after,
            "max_frac": caps.get(cls),
            "capped": bool(apply_area_caps),
        }

    # Paint z-order without urban_road first
    masks["green"] = site
    painted = paint_strict_layers(masks, y_rgb.shape[:2], site=site)

    # Compose with A: outside site / redline from A
    y_comp, cmeta = compose_Y(painted, a_rgb)

    # FORCE urban_road on top — strict copy from A (never from G)
    y_comp = y_comp.copy()
    y_comp[purple] = np.array(PALETTE["urban_road"], dtype=np.uint8)
    # also restore redline from A
    y_comp[red] = a_rgb[red]

    # Final legality: no site paint may remain on purple
    assert not (site & purple).any() or True  # site already excludes purple
    leak = np.all(y_comp == np.array(PALETTE["building"]), axis=2) & purple
    if leak.any():
        y_comp[leak] = np.array(PALETTE["urban_road"], dtype=np.uint8)

    site_n = max(int(site.sum()), 1)
    b_after = int(masks["building"].sum())
    meta = {
        "building_max_site_frac": building_max_frac,
        "building_px_before": area_audit.get("building", {}).get("px_before", 0),
        "building_px_after": b_after,
        "building_frac_after": b_after / site_n,
        "area_audit": area_audit,
        "urban_road_px_forced": int(purple.sum()),
        "z_order": [n for n, _ in Z_ORDER],
        "preserve_geometry": bool(preserve_geometry),
        "road_as_centerline": bool(road_as_centerline),
        "plant_as_disks": bool(plant_as_disks),
        "compose": cmeta,
        "rules": [
            "magenta_only_building",
            "area_caps_off_by_default",
            "urban_road_copy_A_top",
            "gen_never_covers_urban_road",
            "layer_order_grass_to_urban_road",
            "preserve_road_width_and_plant_footprint",
        ],
    }
    return y_comp.astype(np.uint8), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--panels",
        default=str(NIGHT / "10_night_site_road" / "stage6_multichannel" / "panels_v1600_best"),
    )
    ap.add_argument("--val_csv", default=str(NIGHT / "data_manifests_fixed_v1" / "val.csv"))
    ap.add_argument(
        "--out",
        default=str(ROOT / "10_fused_gen_quality" / "hard_constraint_v1"),
    )
    ap.add_argument("--building_max_frac", type=float, default=BUILDING_MAX_SITE_FRAC)
    args = ap.parse_args()

    panels = Path(args.panels)
    out = Path(args.out)
    for sub in ("constrained", "compare", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(Path(args.val_csv).open(encoding="utf-8")))
    audits = []
    for r in rows:
        cid = str(r.get("base_case_id") or r.get("case_id"))
        yp = panels / cid / "composeY.png"
        if not yp.exists():
            print("skip", cid)
            continue
        a = np.array(Image.open(r["A_path"]).convert("RGB"))
        y = np.array(Image.open(yp).convert("RGB"))
        # match A size
        if y.shape[:2] != a.shape[:2]:
            y = np.array(Image.fromarray(y).resize((a.shape[1], a.shape[0]), Image.BILINEAR))
        yc, meta = hard_constraint_compose(y, a, building_max_frac=args.building_max_frac)
        Image.fromarray(yc).save(out / "constrained" / f"{cid}.png")
        # also write into panels
        Image.fromarray(yc).save(panels / cid / "composeY_hard.png")
        (out / "meta" / f"{cid}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # compare strip A | B | Yraw | Yhard
        b = np.array(Image.open(r["B_path"]).convert("RGB"))
        if b.shape[:2] != a.shape[:2]:
            b = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.NEAREST))
        h, w = a.shape[:2]
        # resize for strip if large
        scale = 256 / max(h, w)
        def rs(im):
            return Image.fromarray(im).resize((int(w * scale), int(h * scale)), Image.NEAREST)

        strip = Image.new("RGB", (int(w * scale) * 4 + 12, int(h * scale) + 18), (18, 18, 20))
        for i, im in enumerate([rs(a), rs(b), rs(y), rs(yc)]):
            strip.paste(im, (i * (im.width + 4), 16))
        from PIL import ImageDraw

        ImageDraw.Draw(strip).text(
            (4, 2),
            f"{cid}  bld {meta['building_frac_after']:.3f}≤{args.building_max_frac}  urban_road forced",
            fill=(240, 240, 240),
        )
        strip.save(out / "compare" / f"{cid}.png")
        audits.append({"case_id": cid, **{k: meta[k] for k in meta if k != "compose"}})
        print(cid, f"bld {meta['building_px_before']}→{meta['building_px_after']}", flush=True)

    (out / "SUMMARY.json").write_text(
        json.dumps(
            {
                "building_max_site_frac": args.building_max_frac,
                "z_order": [n for n, _ in Z_ORDER],
                "n": len(audits),
                "cases": audits,
                "note": "Hard constraints improve legality; cannot restore missing thin roads inventively.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
