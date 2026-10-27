"""Facade: legacy util binding plus R1/R2 entry points.

R1 lives in core/r1, R2 in core/r2, parameters in core/params.
"""
from __future__ import annotations
from pathlib import Path

from .paths import LEGACY_ROOT as BUNDLE_ROOT
from .r1.normalize import r1_image
from .r1.tau_gate import r1_prime_image
from .r2.refine import r2_image

__all__ = ['bundled_legacy_root', 'original_api', 'r1_image', 'r1_prime_image', 'r2_image']


def bundled_legacy_root() -> Path:
    """Self-contained original util + config shipped under code_1024."""
    if not (BUNDLE_ROOT / 'config.py').is_file() or not (BUNDLE_ROOT / 'util').is_dir():
        raise FileNotFoundError(f'Bundled original_reference incomplete: {BUNDLE_ROOT}')
    return BUNDLE_ROOT


def original_api():
    """Load ORIGINAL util from the project root currently on sys.path."""
    from config import TOLERANCE
    from util.color_extraction import extract_color_regions
    from util.mask_processing import (
        refine_green_mask,
        repair_and_paste_orange,
        mask_to_bounding_boxes,
    )
    from util.shape_drawing import paste_color_with_smooth_edges, draw_geometric_blocks
    from .r2.vegetation import fill_green_areas_with_circles
    from .r2.road_connect import repair_roads_cross_vegetation

    def _optional(module_name, names):
        out = {}
        try:
            mod = __import__(module_name, fromlist=list(names))
        except ImportError:
            return {n: None for n in names}
        for n in names:
            out[n] = getattr(mod, n, None)
        return out

    opt_color = _optional('util.color_extraction', ('extract_orange_region',))
    opt_mask = _optional('util.mask_processing', (
        'keep_original_if_orange', 'connect_road_skeleton', 'close_and_fill_mask',
        'normalize_line_width', 'orange_curve_from_image', 'orange_curve_connect_nearest'))
    opt_draw = _optional('util.shape_drawing', (
        'smooth_mask_and_draw', 'smooth_region_edges', 'draw_orange_smooth_curve',
        'draw_shape_preserve_edges'))
    return dict(
        tolerance=TOLERANCE,
        extract=extract_color_regions,
        extract_orange=opt_color['extract_orange_region'],
        refine=refine_green_mask,
        road=repair_roads_cross_vegetation,
        road_legacy=repair_and_paste_orange,
        boxes=mask_to_bounding_boxes,
        keep_if_orange=opt_mask['keep_original_if_orange'],
        connect_skel=opt_mask['connect_road_skeleton'],
        close_fill=opt_mask['close_and_fill_mask'],
        normalize_width=opt_mask['normalize_line_width'],
        orange_curve=opt_mask['orange_curve_from_image'],
        orange_connect=opt_mask['orange_curve_connect_nearest'],
        blocks=draw_geometric_blocks,
        smooth=paste_color_with_smooth_edges,
        smooth_mask=opt_draw['smooth_mask_and_draw'],
        smooth_edges=opt_draw['smooth_region_edges'],
        orange_smooth_curve=opt_draw['draw_orange_smooth_curve'],
        draw_preserve=opt_draw['draw_shape_preserve_edges'],
        trees=fill_green_areas_with_circles,
    )


if __name__ == '__main__':
    import sys
    _root = str(Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from pipeline.run_stages import main
    main()
