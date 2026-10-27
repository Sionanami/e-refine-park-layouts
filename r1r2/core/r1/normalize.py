"""R1: weighted RGB L2 argmin. No distance threshold."""
from __future__ import annotations

import numpy as np

from ..study_common import (
    nearest_rgb, weighted_nearest_rgb, weighted_nearest_labels,
    nearest_labels, restore_limits,
    DEFAULT_PLANT_DIST_WEIGHT, DEFAULT_GREEN_DIST_WEIGHT,
    DEFAULT_PAV_OVER_ROAD_MARGIN,
)


def r1_image(raw, cons, api, *, diagnostics=None):
    """Assign every editable pixel by weighted distance to the class colours.

    There is no acceptance threshold. Pixels outside the site, on the
    protected mask, or on the external road are copied from the input.
    """
    weights = api.get('r1_weights') or dict(
        plant_dist_weight=DEFAULT_PLANT_DIST_WEIGHT,
        green_dist_weight=DEFAULT_GREEN_DIST_WEIGHT,
        pav_over_road_margin=DEFAULT_PAV_OVER_ROAD_MARGIN,
    )
    unweighted = nearest_rgb(raw, cons['editable'])
    out = weighted_nearest_rgb(raw, cons['editable'], **weights)
    if diagnostics is not None:
        d = cons['editable']
        wlab = weighted_nearest_labels(raw, **weights)
        nlab = nearest_labels(raw)
        diagnostics.update(
            assignment='weighted_argmin',
            absolute_distance_tau=None,
            plant_dist_weight=float(weights['plant_dist_weight']),
            green_dist_weight=float(weights['green_dist_weight']),
            pav_over_road_margin=float(weights['pav_over_road_margin']),
            changed_vs_unweighted_nearest_px=int(((wlab != nlab) & d).sum()),
            changed_vs_unweighted_rgb_px=int((np.any(out != unweighted, axis=2) & d).sum()),
        )
    return restore_limits(out, raw, cons)
