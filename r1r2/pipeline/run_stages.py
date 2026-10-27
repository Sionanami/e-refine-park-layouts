"""Generate N/R1/R2 without opening annotations. All outputs in a new directory."""
from __future__ import annotations
import argparse
import importlib
import random
import sys
from pathlib import Path
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.paths import CODE_ROOT
from core.study_common import (read_manifest, rgb, load_constraints, check_raw, nearest_rgb,
                          exact_labels, csv_write, json_write, sha, source_hashes)
from core.params.parameter_config import (load_parameters, resolve_r1_weights, resolve_vegetation,
                              resolve_road_revision, road_kwargs)
from core import generate as revision_generate


def _revision_code_hashes():
    acc = {}
    for folder in ('core', 'pipeline', 'evaluation', 'tools'):
        root = CODE_ROOT / folder
        acc.update({f'{folder}/{rel}': digest for rel, digest in source_hashes(root).items()})
    return acc


def run(manifest, legacy_root, out, building_policy='preserve', skip=(), seed=42, parameters=None):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite results: {out}')
    rows = read_manifest(manifest)
    cfg, parameter_record = load_parameters(parameters)
    rev = revision_generate
    if legacy_root is None:
        legacy_root = rev.bundled_legacy_root()
    legacy_root = Path(legacy_root).resolve()
    if not (legacy_root/'config.py').is_file() or not (legacy_root/'util').is_dir():
        raise FileNotFoundError(
            'Need a project root with config.py and util/. '
            f'Default bundle is {rev.bundled_legacy_root()}')
    # Legacy util/config first; code_1024 stays on path so `core` still imports.
    sys.path = [str(legacy_root), str(CODE_ROOT)] + [
        p for p in sys.path if Path(p).resolve() not in {legacy_root, CODE_ROOT}]
    for name in list(sys.modules):
        if name == 'config' or name == 'util' or name.startswith('util.'):
            del sys.modules[name]
    api = rev.original_api()
    for module_name in ('config','util.color_extraction','util.mask_processing','util.shape_drawing'):
        module = importlib.import_module(module_name)
        if not Path(module.__file__).resolve().is_relative_to(legacy_root):
            raise ValueError(f'Wrong original dependency imported: {module_name}, {module.__file__}')
    r1_image = rev.r1_image
    r2_image = rev.r2_image
    r1_weights, r1_source = resolve_r1_weights(cfg)
    veg_settings, veg_source = resolve_vegetation(cfg)
    road_settings, road_source = resolve_road_revision(cfg)
    effective_kwargs, binding_record = road_kwargs(cfg, api['road'])
    if 'road' in skip and binding_record['applied_native_controls']:
        raise ValueError('Native road controls requested while road module is skipped')
    # allow_cross_vegetation is passed into repair_roads_cross_vegetation via
    # road_settings / refine overlay — do not silently swap to legacy util.
    # Legacy remains available as api['road_legacy'] for explicit ablations only.
    api.update(r1_weights=r1_weights, vegetation_settings=veg_settings,
               road_settings=road_settings, r2_settings=cfg['r2'],
               road_kwargs=effective_kwargs, road_diagnostics=[])
    # Validate every RAW/mask before beginning output.
    assets = []
    for row in rows:
        raw, cons = rgb(row['raw_path']), load_constraints(row)
        check_raw(raw, cons)
        assets.append((row, raw, cons))
    out.mkdir(parents=True)
    original_hashes = source_hashes(legacy_root/'util')
    code_hashes = _revision_code_hashes()
    log, checks, r1_diagnostics = [], [], []
    hashes = []
    for row, raw, cons in assets:
        # Seed every view independently: removing a different view does not shift RNG.
        random.seed(seed)
        np.random.seed(seed)
        try:
            import cv2
            cv2.setRNGSeed(seed)
        except ImportError:
            pass  # Original utilities will fail explicitly if they require it.
        n = nearest_rgb(raw, cons['editable'])
        mapping_info = dict(view_id=row['view_id'])
        r1 = r1_image(raw, cons, api, diagnostics=mapping_info)
        r1_diagnostics.append(mapping_info)
        r2, edits = r2_image(r1, raw, cons, api, building_policy=building_policy, skip=skip)
        for stage, image in [('N', n), ('R1', r1), ('R2', r2)]:
            dest = out/stage/f"{row['view_id']}.png"
            dest.parent.mkdir(exist_ok=True)
            Image.fromarray(image).save(dest)
            labels = exact_labels(image)
            labels[~cons['editable']] = 0
            np.save(dest.with_suffix('.npy'), labels)
            invalid = int(((labels == 0) & cons['editable']).sum())
            # R2 output normalization is an explicit logged generation step;
            # never change or repair predictions inside the evaluator.
            if invalid:
                raise ValueError(f'Encoding failed for {stage}/{row["view_id"]}')
            checks.append(dict(view_id=row['view_id'], stage=stage, invalid_editable_px=invalid))
            hashes.append(dict(view_id=row['view_id'], stage=stage, path=str(dest), sha256=sha(dest)))
        log.extend(dict(view_id=row['view_id'], **e) for e in edits)
        for key in ('raw_path','condition_path','site_path','protect_path','urban_path'):
            hashes.append(dict(view_id=row['view_id'], stage=key, path=row[key], sha256=sha(row[key])))
    csv_write(out/'edit_log.csv', log)
    csv_write(out/'r1_mapping_log.csv',r1_diagnostics)
    csv_write(out/'encoding_checks.csv', checks)
    csv_write(out/'asset_hashes.csv', hashes)
    json_write(out/'run_manifest.json', dict(
        method='original_source_minimal_revision_v5_fused_original_reference', complete=True,
        legacy_root=str(legacy_root),
        annotations_read=False, semantic_patch=False, development_set=False,
        manifest_path=str(Path(manifest).resolve()), manifest_sha256=sha(manifest),
        n_views=len(rows), n_parks=len({r['park_id'] for r in rows}),
        pipeline='RAW->R1 weighted snap; R2: pavement/water -> road -> keep_r1_plant (no circles); single R2 PNG',
        building_policy=building_policy, skip_modules=list(skip), seed=seed,
        original_TOLERANCE=api['tolerance'], original_config_sha256=sha(legacy_root/'config.py'),
        r1_weights=r1_weights, r1_weight_source=r1_source,
        absolute_distance_tau=None,
        vegetation_settings=veg_settings, vegetation_source=veg_source,
        road_settings=road_settings, road_source=road_source,
        road_parameter_binding=binding_record,
        **parameter_record,
        original_util_sha256=original_hashes, revision_code_sha256=code_hashes,
        fixed_reference='supplied already-composed RAW, verified against source_A',
        r1_building_dilation=False, conditional_orthogonalization=False,
        R2_output_normalization='explicit final nearest-palette encoding, before building footprint preservation',
        R2_invalid_px_total=sum(c['invalid_editable_px'] for c in checks if c['stage']=='R2')))
    print(f"Generated {len(rows)} views. No annotations read. Freeze outputs before evaluate_stages.py.")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--legacy-root', type=Path, default=None,
                   help='Project root with config.py + util/. Default: bundled original_reference/')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--building-policy', choices=['preserve','legacy_bbox'], default='preserve')
    p.add_argument('--skip', action='append', choices=['road','vegetation','pavement','water'], default=[])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--parameters', type=Path, help='Explicit parameters.json; default is core/params/parameters.json')
    a = p.parse_args()
    run(a.manifest, a.legacy_root, a.out, a.building_policy, a.skip, a.seed, a.parameters)

if __name__ == '__main__':
    main()
