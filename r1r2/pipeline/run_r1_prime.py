"""Generate R1' only: paper absolute-distance τ gate (internal ablation).

Does not modify frozen weighted R1 / R2. Writes a fresh output directory with
R1_prime/ images, labels, mapping log, and a small run manifest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.study_common import (
    DEFAULT_ABSOLUTE_DISTANCE_TAU,
    check_raw,
    csv_write,
    exact_labels,
    json_write,
    load_constraints,
    read_manifest,
    rgb,
    sha,
    source_hashes,
)
from core.r1.tau_gate import r1_prime_image


def run(manifest, out, tau=DEFAULT_ABSOLUTE_DISTANCE_TAU, r1_dir=None):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite results: {out}')
    rows = read_manifest(manifest)
    r1_dir = Path(r1_dir).resolve() if r1_dir else None
    if r1_dir is not None and not r1_dir.is_dir():
        raise FileNotFoundError(f'--r1-dir not found: {r1_dir}')

    assets = []
    for row in rows:
        raw, cons = rgb(row['raw_path']), load_constraints(row)
        check_raw(raw, cons)
        r1_rgb = None
        if r1_dir is not None:
            r1_path = r1_dir / f"{row['view_id']}.png"
            if not r1_path.is_file():
                raise FileNotFoundError(f'Missing frozen R1 image: {r1_path}')
            r1_rgb = rgb(r1_path)
            if r1_rgb.shape != raw.shape:
                raise ValueError(f'R1 shape mismatch for {row["view_id"]}')
        assets.append((row, raw, cons, r1_rgb))

    out.mkdir(parents=True)
    stage_dir = out / 'R1_prime'
    stage_dir.mkdir()
    log, checks, hashes = [], [], []
    for row, raw, cons, r1_rgb in assets:
        info = dict(view_id=row['view_id'])
        image = r1_prime_image(raw, cons, tau=tau, r1_rgb=r1_rgb, diagnostics=info)
        dest = stage_dir / f"{row['view_id']}.png"
        Image.fromarray(image).save(dest)
        labels = exact_labels(image)
        labels[~cons['editable']] = 0
        np.save(dest.with_suffix('.npy'), labels)
        invalid = int(((labels == 0) & cons['editable']).sum())
        if invalid:
            raise ValueError(f'Encoding failed for R1_prime/{row["view_id"]}')
        checks.append(dict(
            view_id=row['view_id'], stage='R1_prime', invalid_editable_px=invalid))
        hashes.append(dict(
            view_id=row['view_id'], stage='R1_prime', path=str(dest), sha256=sha(dest)))
        for key in ('raw_path', 'condition_path', 'site_path', 'protect_path', 'urban_path'):
            hashes.append(dict(
                view_id=row['view_id'], stage=key, path=row[key], sha256=sha(row[key])))
        log.append(info)

    csv_write(out / 'r1_prime_mapping_log.csv', log)
    csv_write(out / 'encoding_checks.csv', checks)
    csv_write(out / 'asset_hashes.csv', hashes)
    json_write(out / 'run_manifest_r1_prime.json', dict(
        method='r1_prime_paper_tau_gate_internal_ablation',
        complete=True,
        annotations_read=False,
        semantic_patch=False,
        development_set=False,
        manifest_path=str(Path(manifest).resolve()),
        manifest_sha256=sha(manifest),
        n_views=len(rows),
        n_parks=len({r['park_id'] for r in rows}),
        pipeline='RAW->R1_prime tau_gate (nearest fallback); no R2',
        absolute_distance_tau=float(tau),
        assignment='tau_gate',
        r1_dir=str(r1_dir) if r1_dir else None,
        frozen_R1_unchanged=True,
        note=(
            'With nearest fallback, R1_prime labels match unweighted N; '
            'use mapping log tau_hit_px / fallback_to_nearest_px and '
            'changed_vs_R1_px for internal comparison to frozen weighted R1.'
        ),
        revision_r1_tau_gate_sha256=source_hashes(_ROOT / 'core' / 'r1').get('tau_gate.py'),
    ))
    print(f"Generated R1_prime for {len(rows)} views -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--tau', type=float, default=DEFAULT_ABSOLUTE_DISTANCE_TAU,
                   help=f'Absolute RGB-L2 τ (default {DEFAULT_ABSOLUTE_DISTANCE_TAU})')
    p.add_argument('--r1-dir', type=Path, default=None,
                   help='Optional frozen R1/ folder for changed_vs_R1_px in the log')
    a = p.parse_args()
    run(a.manifest, a.out, tau=a.tau, r1_dir=a.r1_dir)


if __name__ == '__main__':
    main()
