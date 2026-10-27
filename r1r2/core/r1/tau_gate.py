"""R1' paper-style absolute-distance τ gate (internal ablation only).

Frozen R1 remains weighted_argmin with absolute_distance_tau=null.
"""
from __future__ import annotations

import numpy as np

from ..study_common import (
    DEFAULT_ABSOLUTE_DISTANCE_TAU,
    exact_labels,
    nearest_labels,
    nearest_rgb,
    restore_limits,
    tau_gate_hit_mask,
    tau_gate_rgb,
    weighted_nearest_labels,
)


def r1_prime_image(raw, cons, tau=DEFAULT_ABSOLUTE_DISTANCE_TAU, *,
                   fallback='nearest', r1_rgb=None, diagnostics=None,
                   require_full_coverage=None):
    """Encode RAW with paper τ gate.

    ``fallback``: ``nearest`` | ``lawn`` | ``unassigned`` | ``keep_raw``.
    ``require_full_coverage`` defaults True for nearest/lawn, False otherwise.
    """
    tau = float(tau)
    dmask = cons['editable']
    hit = tau_gate_hit_mask(raw, tau=tau)
    out = tau_gate_rgb(raw, dmask, tau=tau, fallback=fallback)
    out = restore_limits(out, raw, cons)
    n_rgb = nearest_rgb(raw, dmask)
    nlab = nearest_labels(raw)
    plab = exact_labels(out)
    if require_full_coverage is None:
        require_full_coverage = fallback in ('nearest', 'lawn')
    if require_full_coverage:
        invalid = int(((plab == 0) & dmask).sum())
        if invalid:
            raise ValueError(f'R1_prime encoding left {invalid} unassigned editable px')
    if diagnostics is not None:
        wlab = weighted_nearest_labels(raw)
        diagnostics.update(
            assignment='tau_gate',
            absolute_distance_tau=tau,
            fallback=fallback,
            tau_hit_px=int((hit & dmask).sum()),
            fallback_px=int((~hit & dmask).sum()),
            fallback_to_nearest_px=int((~hit & dmask).sum()) if fallback == 'nearest' else 0,
            changed_vs_N_px=int((np.any(out != n_rgb, axis=2) & dmask).sum()),
            changed_vs_N_labels_px=int(((plab != nlab) & dmask).sum()),
            changed_vs_weighted_R1_labels_px=int(((plab != wlab) & dmask).sum()),
            unassigned_editable_px=int(((plab == 0) & dmask).sum()),
        )
        if r1_rgb is not None:
            r1lab = exact_labels(np.asarray(r1_rgb))
            diagnostics['changed_vs_R1_px'] = int(((plab != r1lab) & dmask).sum())
        else:
            diagnostics['changed_vs_R1_px'] = None
    return out
