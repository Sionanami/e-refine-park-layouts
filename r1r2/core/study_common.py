"""Shared RGB/label/mask rules. No annotations are imported by generation."""
from __future__ import annotations
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image

# Preserve IDs in the supplied palette_human.csv. 0 = invalid/unassigned.
PALETTE = {
    1: (255, 0, 255), 2: (0, 255, 0), 3: (255, 255, 0),
    4: (0, 255, 255), 5: (241, 145, 73), 6: (0, 152, 67),
}
NAMES = {1: 'building', 2: 'green', 3: 'pavement', 4: 'water', 5: 'road', 6: 'plant'}
# Frozen metric-repair INTERNAL order (tie-break = lower index wins).
INTERNAL_ORDER = ('building', 'pavement', 'green', 'road', 'water', 'plant')
NAME_TO_ID = {name: idx for idx, name in NAMES.items()}
DEFAULT_PLANT_DIST_WEIGHT = 0.55
DEFAULT_GREEN_DIST_WEIGHT = 0.72
DEFAULT_PAV_OVER_ROAD_MARGIN = 28.0
# Paper-style absolute RGB-L2 τ for internal R1' ablation (config.TOLERANCE).
DEFAULT_ABSOLUTE_DISTANCE_TAU = 30.0
VIEW_RE = re.compile(r'^(?P<park>.+)_r(?P<rotation>000|090|180|270)(?:_(?:fake_B|raw|output_raw))?$')

def identify(stem):
    m = VIEW_RE.fullmatch(stem)
    if not m:
        raise ValueError(f'Unrecognized RAW view name: {stem}; expected park_r000[optional _raw/_fake_B]')
    return m['park'], int(m['rotation']), f"{m['park']}_r{m['rotation']}"

def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)

def mask(path):
    arr = np.asarray(Image.open(path).convert('L'))
    if not set(np.unique(arr)).issubset({0, 1, 255}):
        raise ValueError(f'Expected binary mask (0/1 or 0/255), no resampling: {path}')
    return arr != 0

def exact_labels(image):
    out = np.zeros(image.shape[:2], dtype=np.uint8)
    for k, color in PALETTE.items():
        out[np.all(image == color, axis=2)] = k
    return out

def _palette_l2_nearest(image):
    """Unweighted RGB-L2 distances to PALETTE (id order 1..6); returns idx0, min_d, labels."""
    cols = np.asarray(list(PALETTE.values()), dtype=np.float32)
    dist = np.linalg.norm(
        image.astype(np.float32)[:, :, None, :] - cols[None, None, :, :], axis=3)
    nearest_idx = np.argmin(dist, axis=2)
    min_d = np.take_along_axis(dist, nearest_idx[:, :, None], axis=2)[:, :, 0]
    labels = (nearest_idx.astype(np.uint8) + 1)
    return nearest_idx, min_d, labels

def nearest_labels(image):
    # Unweighted N baseline. float32 prevents uint8 overflow; PALETTE order resolves ties.
    return _palette_l2_nearest(image)[2]

def _validate_tau(tau):
    tau = float(tau)
    if not np.isfinite(tau) or tau < 0:
        raise ValueError(f'tau must be a finite nonnegative number, got {tau!r}')
    return tau

def tau_gate_labels(image, tau=DEFAULT_ABSOLUTE_DISTANCE_TAU, fallback='nearest'):
    """Paper-style absolute-distance τ gate (internal R1' ablation).

    Assign argmin_k ||P-c_k||_2 when min_d < τ. Miss handling:
    - ``nearest``: unweighted nearest (labels == N; default EC=1 audit)
    - ``lawn``: class 2 green open space (τ changes visible class areas)
    - ``unassigned``: label 0 on misses (EC may be <1; visual/noise audit)
    """
    tau = _validate_tau(tau)
    _, min_d, nearest = _palette_l2_nearest(image)
    hit = min_d < tau
    if fallback == 'nearest':
        return nearest
    if fallback == 'lawn':
        out = nearest.copy()
        out[~hit] = 2
        return out
    if fallback == 'unassigned':
        out = np.zeros(nearest.shape, dtype=np.uint8)
        out[hit] = nearest[hit]
        return out
    raise ValueError(f'Unknown tau_gate fallback: {fallback!r}')

def tau_gate_hit_mask(image, tau=DEFAULT_ABSOLUTE_DISTANCE_TAU):
    """True where min RGB-L2 distance to any palette colour is < τ."""
    tau = _validate_tau(tau)
    _, min_d, _ = _palette_l2_nearest(image)
    return min_d < tau

def weighted_nearest_labels(image, plant_dist_weight=DEFAULT_PLANT_DIST_WEIGHT,
                            green_dist_weight=DEFAULT_GREEN_DIST_WEIGHT,
                            pav_over_road_margin=DEFAULT_PAV_OVER_ROAD_MARGIN):
    """Weighted RGB L2 argmin used by R1 (no absolute distance-tau gate).

    Matches frozen r1r2_metric_repair._snap_internal:
    plant/green distances are scaled down; road is suppressed when pavement is
    within pav_over_road_margin of the road distance.
    """
    ids = np.asarray([NAME_TO_ID[name] for name in INTERNAL_ORDER], dtype=np.uint8)
    cols = np.stack([np.asarray(PALETTE[int(i)], np.float32) for i in ids], axis=0)
    dist = np.linalg.norm(image.astype(np.float32)[:, :, None, :] - cols[None, None, :, :], axis=3)
    dist = dist.copy()
    dist[:, :, INTERNAL_ORDER.index('plant')] *= float(plant_dist_weight)
    dist[:, :, INTERNAL_ORDER.index('green')] *= float(green_dist_weight)
    i_road = INTERNAL_ORDER.index('road')
    i_pav = INTERNAL_ORDER.index('pavement')
    closer_pav = dist[:, :, i_pav] <= dist[:, :, i_road] + float(pav_over_road_margin)
    dist[:, :, i_road] = np.where(closer_pav, np.inf, dist[:, :, i_road])
    return ids[dist.argmin(axis=2)]

def label_rgb(labels, base, editable):
    out = np.array(base, copy=True)
    for k, color in PALETTE.items():
        out[(labels == k) & editable] = color
    return out

def nearest_rgb(image, editable):
    return label_rgb(nearest_labels(image), image, editable)

def weighted_nearest_rgb(image, editable, **weights):
    return label_rgb(weighted_nearest_labels(image, **weights), image, editable)

def tau_gate_rgb(image, editable, tau=DEFAULT_ABSOLUTE_DISTANCE_TAU, fallback='nearest'):
    """Snap editable pixels by τ gate; ``keep_raw`` leaves misses as original RGB."""
    if fallback == 'keep_raw':
        tau = _validate_tau(tau)
        _, min_d, nearest = _palette_l2_nearest(image)
        hit = (min_d < tau) & editable
        out = np.array(image, copy=True)
        for k, color in PALETTE.items():
            out[hit & (nearest == k)] = color
        return out
    return label_rgb(tau_gate_labels(image, tau=tau, fallback=fallback), image, editable)

def load_constraints(row):
    a = rgb(row['condition_path'])
    site, protect, urban = (mask(row[k]) for k in ('site_path', 'protect_path', 'urban_path'))
    if any(m.shape != a.shape[:2] for m in (site, protect, urban)):
        raise ValueError(f"Constraint shape mismatch: {row['view_id']}")
    if not site.any():
        raise ValueError(f"Empty site: {row['view_id']}")
    if np.any(urban & site):
        raise ValueError('External urban road overlaps site; correct the manifest/masks')
    # S can contain preserved objects; the editable domain excludes protection.
    return dict(a=a, site=site, protect=protect, urban=urban, editable=site & ~protect & ~urban)

def restore_limits(image, raw, cons):
    if image.shape != raw.shape or raw.shape != cons['a'].shape:
        raise ValueError('Image/A/mask sizes differ; never resize silently')
    out = np.array(image, copy=True)
    fixed = ~cons['editable']
    out[fixed] = raw[fixed]  # The supplied RAW has already been composed upstream.
    return out

def check_raw(raw, cons):
    if raw.shape != cons['a'].shape:
        raise ValueError('RAW and frozen masks/A have different sizes')
    # Verify rather than silently recompose or claim an R1 improvement.
    fixed = ~cons['editable']
    if np.any(raw[fixed] != cons['a'][fixed]):
        raise ValueError('RAW differs from source_A on fixed pixels. Supply the actual composition reference/masks.')

def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty manifest')
    required = ('view_id','park_id','rotation','raw_path','condition_path','site_path','protect_path','urban_path')
    seen = set()
    for r in rows:
        if any(not r.get(k) for k in required):
            raise ValueError(f'Missing manifest value: {r}')
        park, angle, view = identify(r['view_id'])
        if park != r['park_id'] or angle != int(r['rotation']) or view in seen:
            raise ValueError('Inconsistent or duplicate park/view identity')
        seen.add(view)
        for k in required[3:]:
            p = Path(r[k]).expanduser()
            if not p.is_absolute():
                p = path.parent / p
            if not p.is_file():
                raise FileNotFoundError(p)
            r[k] = str(p.resolve())
    parks = {r['park_id'] for r in rows}
    if {r['park_id'] for r in rows if int(r['rotation']) == 0} != parks:
        raise ValueError('Every park must have exactly one r000 view')
    return rows

def csv_write(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts}
