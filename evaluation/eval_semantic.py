# -*- coding: utf-8 -*-
"""Human-reference semantic accuracy (SA) on aligned annotation points."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.study_common import NAMES

# human label ids match study_common: 1..6
CLASS_NAMES = {1: 'building', 2: 'green', 3: 'pavement', 4: 'water', 5: 'road', 6: 'plant'}


def load_annotation_points(audit_csv: Path, sa_dir: Path) -> Tuple[List[dict], dict]:
    """Load valid points from annotation_audit.csv; cross-check SA_anotation_set."""
    rows = []
    with Path(audit_csv).open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    meta = dict(
        n_rows=len(rows),
        n_valid=sum(1 for r in rows if r.get('status') == 'valid'),
        n_excluded=sum(1 for r in rows if r.get('status') != 'valid'),
        exclude_reasons={},
        parks=sorted({r['case_id'] for r in rows}),
        source_audit=str(Path(audit_csv).resolve()),
        sa_dir=str(Path(sa_dir).resolve()),
    )
    for r in rows:
        if r.get('status') != 'valid':
            reason = r.get('reason') or r.get('status') or 'unknown'
            meta['exclude_reasons'][reason] = meta['exclude_reasons'].get(reason, 0) + 1
    # Verify SA folder parks exist
    sa_parks = sorted(p.name for p in Path(sa_dir).iterdir() if p.is_dir())
    meta['sa_folder_parks'] = sa_parks
    meta['parks_missing_in_sa'] = sorted(set(meta['parks']) - set(sa_parks))
    meta['sa_extra_parks'] = sorted(set(sa_parks) - set(meta['parks']))
    return rows, meta


def predict_at_points(
    Q: np.ndarray,
    domain: np.ndarray,
    protect: np.ndarray,
    points: List[dict],
    park_id: str,
    stage: str,
    view_id: str,
    rotation: int,
) -> List[dict]:
    """Evaluate points for one park/view. Points are in r000 image coordinates.

    For rotated views, annotations apply only when rotation==0 (r000). Callers
    should skip non-r000 for SA primary stats.
    """
    h, w = Q.shape
    out = []
    for r in points:
        if r['case_id'] != park_id:
            continue
        status = r.get('status', '')
        x = int(float(r['x']))
        y = int(float(r['y']))
        human = int(float(r['human'])) if r.get('human') not in ('', None) else None
        rec = dict(
            view_id=view_id, park_id=park_id, rotation=rotation, stage=stage,
            point_id=r.get('point_id'), x=x, y=y, human=human,
            human_name=r.get('class_name') or CLASS_NAMES.get(human, ''),
            annot_status=status, reason=r.get('reason', ''),
            pred=None, pred_name='', correct=None,
            in_bounds=None, in_domain=None, is_fixed=None, is_editable=None,
            decidable=False,
        )
        if not (0 <= x < w and 0 <= y < h):
            rec['in_bounds'] = False
            rec['reason'] = rec['reason'] or 'OOB'
            out.append(rec)
            continue
        rec['in_bounds'] = True
        rec['in_domain'] = bool(domain[y, x])
        rec['is_fixed'] = bool(protect[y, x]) or (not domain[y, x])
        # Fixed = non-editable: protect OR outside site OR urban (domain already excludes)
        rec['is_editable'] = bool(domain[y, x])
        if status != 'valid' or human is None or human not in CLASS_NAMES:
            rec['decidable'] = False
            out.append(rec)
            continue
        # Main SA: decidable on-site points (inside original site semantics).
        # Use domain∪protect for "in park body" — protect is still site-interior.
        # Author: use original decidable on-site points; also report editable-only.
        on_site = bool(domain[y, x] or protect[y, x])
        if not on_site:
            rec['decidable'] = False
            rec['reason'] = rec['reason'] or 'outside_site'
            out.append(rec)
            continue
        pred = int(Q[y, x])
        rec['pred'] = pred
        rec['pred_name'] = CLASS_NAMES.get(pred, '')
        rec['correct'] = bool(pred == human)
        rec['decidable'] = True
        out.append(rec)
    return out


def summarize_sa(point_rows: List[dict]) -> Dict[str, Any]:
    decidable = [r for r in point_rows if r.get('decidable')]
    correct = sum(1 for r in decidable if r.get('correct'))
    n = len(decidable)
    # confusion + recall
    labels = list(range(1, 7))
    conf = np.zeros((6, 6), dtype=int)
    for r in decidable:
        h, p = int(r['human']), int(r['pred'])
        if 1 <= h <= 6 and 1 <= p <= 6:
            conf[h - 1, p - 1] += 1
    recalls = {}
    supports = {}
    for i, lab in enumerate(labels):
        support = int(conf[i, :].sum())
        supports[CLASS_NAMES[lab]] = support
        recalls[CLASS_NAMES[lab]] = (
            float(conf[i, i] / support) if support > 0 else None)
    valid_recalls = [v for v in recalls.values() if v is not None]
    macro = float(np.mean(valid_recalls)) if valid_recalls else None
    editable = [r for r in decidable if r.get('is_editable')]
    e_correct = sum(1 for r in editable if r.get('correct'))
    return dict(
        n_correct=correct,
        n_decidable=n,
        SA=float(correct / n) if n else None,
        macro_recall=macro,
        recalls=recalls,
        supports=supports,
        confusion=conf.tolist(),
        n_editable_decidable=len(editable),
        n_editable_correct=e_correct,
        SA_editable=float(e_correct / len(editable)) if editable else None,
        n_fixed_decidable=sum(1 for r in decidable if r.get('is_fixed')),
    )
