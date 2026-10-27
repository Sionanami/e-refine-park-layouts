# -*- coding: utf-8 -*-
"""Run weighted R1, then R2 and R2'."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent
_LEGACY = _ROOT / "original_reference"


def _boot() -> None:
    sys.path = [str(_LEGACY), str(_ROOT)] + [
        p for p in sys.path
        if Path(p).resolve() not in {_LEGACY.resolve(), _ROOT.resolve()}
    ]
    for name in list(sys.modules):
        if name == "config" or name == "util" or name.startswith("util."):
            del sys.modules[name]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--parameters", type=Path, default=_ROOT / "parameters.json")
    ap.add_argument("--r2-out", type=Path, required=True)
    ap.add_argument("--r2p-out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--building-policy", default="local_repair")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with args.samples.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("no samples")

    for d in (args.r2_out, args.r2p_out):
        d.mkdir(parents=True, exist_ok=True)
        existing = list(d.glob("*.png"))
        if existing and not args.force:
            raise FileExistsError(f"refusing non-empty output: {d}")

    _boot()
    from core import generate as rev
    from core.params.parameter_config import (
        load_parameters, resolve_building, resolve_r1_weights, resolve_road_revision,
        resolve_surface, resolve_vegetation, road_kwargs,
    )
    from core.study_common import check_raw, json_write, load_constraints, rgb

    cfg, _ = load_parameters(args.parameters)
    api = rev.original_api()
    r1_weights, r1_source = resolve_r1_weights(cfg)
    veg_settings, veg_source = resolve_vegetation(cfg)
    road_settings, road_source = resolve_road_revision(cfg)
    surf_settings, surf_source = resolve_surface(cfg)
    bld_settings, bld_source = resolve_building(cfg)
    effective_kwargs, _ = road_kwargs(cfg, api["road"])
    api.update(
        r1_weights=r1_weights,
        vegetation_settings=veg_settings,
        road_settings=road_settings,
        surface_settings=surf_settings,
        building_settings=bld_settings,
        r2_settings=cfg["r2"],
        road_kwargs=effective_kwargs,
        road_diagnostics=[],
    )

    ok, failed = [], []
    for i, row in enumerate(rows, 1):
        vid = row["view_id"]
        print(f"[{i}/{len(rows)}] RUN {vid}", flush=True)
        try:
            raw_path = Path(row["raw_path"])
            if not raw_path.is_file():
                raise FileNotFoundError(vid)
            row = dict(row)
            row["raw_path"] = str(raw_path)
            cons = load_constraints(row)
            raw = rgb(raw_path)
            check_raw(raw, cons)
            random.seed(args.seed)
            np.random.seed(args.seed)
            try:
                import cv2
                cv2.setRNGSeed(args.seed)
            except ImportError:
                pass
            r1 = rev.r1_image(raw, cons, api)
            for key in ("vegetation_diagnostics", "surface_diagnostics", "building_diagnostics"):
                api[key] = {}
            api["road_diagnostics"] = []
            api["approved_building_mask"] = None
            r2, _ = rev.r2_image(r1, raw, cons, api, building_policy=args.building_policy)
            random.seed(args.seed)
            np.random.seed(args.seed)
            try:
                import cv2
                cv2.setRNGSeed(args.seed)
            except ImportError:
                pass
            for key in ("vegetation_diagnostics", "surface_diagnostics", "building_diagnostics"):
                api[key] = {}
            api["road_diagnostics"] = []
            api["approved_building_mask"] = None
            r2p, _ = rev.r2_image(raw, raw, cons, api, building_policy=args.building_policy)
            Image.fromarray(r2).save(args.r2_out / f"{vid}.png")
            Image.fromarray(r2p).save(args.r2p_out / f"{vid}.png")
            ok.append(vid)
        except Exception as exc:
            failed.append({"view_id": vid, "error": repr(exc)})
            print(f"FAIL {vid}: {exc!r}", flush=True)

    json_write(args.r2_out / "manifest.json", {
        "parameters": args.parameters.name,
        "r1_source": r1_source,
        "road_source": road_source,
        "surface_source": surf_source,
        "vegetation_source": veg_source,
        "building_source": bld_source,
        "n_ok": len(ok),
        "n_fail": len(failed),
        "failed": failed,
    })
    print(json.dumps({"n_ok": len(ok), "n_fail": len(failed)}))


if __name__ == "__main__":
    main()
