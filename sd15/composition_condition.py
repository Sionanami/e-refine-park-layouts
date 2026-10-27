# -*- coding: utf-8 -*-
"""Composition-aware Adapter condition (3ch, 6GB-compatible).

DEPRECATED bars: bottom area-budget legend strips are DISABLED — they leak non-semantic
colors into Adapter condition and confuse RAW generation. ``build_composition_condition_A``
now equals sanitize (site green + urban purple + redline), same as ``sanitize_condition_A``.

Area priors remain in COMPOSITION_PRIORS.json for hard_constraint / audits only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from site_target import PALETTE, sanitize_condition_A, _mask_tol, _site_mask

CLASS_KEYS = ("building", "water", "plant", "road", "pavement")
DEFAULT_PRIORS_PATH = Path(__file__).resolve().parent.parent / "manifests" / "COMPOSITION_PRIORS.json"


def class_fracs_in_site(b_rgb: np.ndarray, a_rgb: np.ndarray) -> Dict[str, float]:
    """Pixel fraction of each semantic class inside site (A green)."""
    site = _site_mask(a_rgb)
    site_n = max(int(site.sum()), 1)
    out = {}
    for k in CLASS_KEYS:
        m = _mask_tol(b_rgb, PALETTE[k], 14 if k != "road" else 18) & site
        if k == "road":
            # include heuristic orange
            r, g, b = b_rgb[..., 0].astype(np.int16), b_rgb[..., 1].astype(np.int16), b_rgb[..., 2].astype(np.int16)
            heur = (r > 195) & (g > 95) & (g < 195) & (b < 130) & (r > g + 20) & site
            m = m | heur
        out[k] = float(m.sum() / site_n)
    return out


def compute_priors_from_train_csv(train_csv: Path, out_json: Path) -> dict:
    rows = list(csv.DictReader(train_csv.open(encoding="utf-8")))
    buckets = {k: [] for k in CLASS_KEYS}
    for r in rows:
        a = np.array(Image.open(r["A_path"]).convert("RGB"))
        b = np.array(Image.open(r["B_path"]).convert("RGB"))
        if b.shape[:2] != a.shape[:2]:
            b = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.NEAREST))
        fr = class_fracs_in_site(b, a)
        for k in CLASS_KEYS:
            buckets[k].append(fr[k])
    priors = {"n_train": len(rows), "classes": {}}
    for k in CLASS_KEYS:
        arr = np.asarray(buckets[k], dtype=np.float64)
        priors["classes"][k] = {
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "mean": float(arr.mean()),
            "max_hard": float(min(0.25, np.percentile(arr, 95) * 1.15)),
        }
    # building hard = train p90 (not outlier max); target_max = p75 median-band
    b = priors["classes"]["building"]
    b["max_hard"] = float(b.get("p90", 0.079))
    b["target_max"] = float(b.get("p75", b["max_hard"] * 0.5))
    b["max"] = float(b.get("max", b["max_hard"]))
    priors["road_width_px"] = 2
    priors["note"] = (
        "p50 used as infer condition bars; p10-p90 as soft bands; "
        "max_hard for post hard_constraint. Train condition may use case B fracs."
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(priors, indent=2, ensure_ascii=False), encoding="utf-8")
    return priors


def load_priors(path: Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_PRIORS_PATH
    if not p.exists():
        return {
            "classes": {
                "building": {"p10": 0.005, "p50": 0.017, "p90": 0.089, "max_hard": 0.10},
                "water": {"p10": 0.0, "p50": 0.01, "p90": 0.08, "max_hard": 0.15},
                "plant": {"p10": 0.01, "p50": 0.05, "p90": 0.15, "max_hard": 0.20},
                "road": {"p10": 0.01, "p50": 0.04, "p90": 0.12, "max_hard": 0.18},
                "pavement": {"p10": 0.02, "p50": 0.08, "p90": 0.20, "max_hard": 0.25},
            },
            "road_width_px": 2,
            "fallback": True,
        }
    return json.loads(p.read_text(encoding="utf-8"))


def default_infer_fracs(priors: dict) -> Dict[str, float]:
    return {k: float(priors["classes"][k]["p50"]) for k in CLASS_KEYS}


def _draw_frac_legend(out: np.ndarray, fracs: Dict[str, float], *, bar_h: int = 40) -> np.ndarray:
    """DISABLED: previously drew bottom budget bars that poisoned RAW semantics."""
    del fracs, bar_h
    return out


def build_composition_condition_A(
    a_img: Image.Image,
    *,
    fracs: Optional[Dict[str, float]] = None,
    priors: Optional[dict] = None,
    b_img: Optional[Image.Image] = None,
    bar_h: int = 40,
) -> Image.Image:
    """Same as sanitize_condition_A (no legend bars). fracs/priors ignored for pixels."""
    del fracs, priors, b_img, bar_h
    return sanitize_condition_A(a_img)


def build_condition_A(
    a_img: Image.Image,
    *,
    mode: str = "composition",
    b_img: Optional[Image.Image] = None,
    priors: Optional[dict] = None,
) -> Image.Image:
    """mode: raw | sanitize | composition."""
    if mode == "raw":
        return a_img.convert("RGB")
    if mode == "sanitize":
        return sanitize_condition_A(a_img)
    return build_composition_condition_A(a_img, b_img=b_img, priors=priors)
