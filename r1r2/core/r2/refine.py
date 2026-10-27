"""R2 staged refinement with explicit coverage write-authority.

Coverage (see coverage.py):
  fixed/A > building > road > pavement > water; plant/lawn soft.
  Water may be covered by other elements; water/plant/lawn/pavement must not cover roads;
  water must not cover pavement; roads may overlay pavement and cover water.

Stage map (selected pc2 order):
  1–2  pavement/water internal green fill, then denoise (rem→lawn)
  3    planar boundary smooth (road protected; shrink→lawn)
  4    building contour (ortho/hybrid) → interior fill → top mask
  5    vegetation inscribed circles before road; disks stay off road, building, pavement
  6    road copy + connect (priority second only to building)
  7    palette + building protect (top) + road protect + restore A markers

Structured thresholds / shared-vs-diff notes: ``core/r2_stages/README.md``.
"""
from __future__ import annotations

import inspect
import cv2
import numpy as np
from PIL import Image

from ..study_common import nearest_rgb, restore_limits, exact_labels
from .coverage import (
    class_masks,
    protected_from_pavement,
    protected_from_water,
    road_hard_blockers,
    vegetation_blockers,
    COVERAGE_RULES_TEXT,
)
from .surface import regularize_surface_rgb
from .vegetation import restore_r1_plant
# Structured planar stages (pavement → water → building); see core/r2_stages/README.md
from ..r2_stages.stage_building import run_building_stage
from ..r2_stages.stage_pavement import run_pavement_stage
from ..r2_stages.stage_water import run_water_stage
from ..r2_stages.thresholds import (
    BuildingThresholds,
    PavementThresholds,
    SharedSurfaceMorph,
    WaterThresholds,
)

BLACK = (0, 0, 0)
PURPLE_URBAN = (137, 86, 161)
RED_MARKER = (255, 0, 0)


def restore_condition_markers(image: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Force black / urban purple / red from condition A on top."""
    out = np.array(image, copy=True)
    out[np.all(a == BLACK, axis=2)] = BLACK
    out[np.all(a == PURPLE_URBAN, axis=2)] = PURPLE_URBAN
    out[np.all(a == RED_MARKER, axis=2)] = RED_MARKER
    return out


def _refresh_barriers(arr, fake, extract, tol, *, cyan, magenta, green, yellow, lawn_rgb, fixed):
    """Current canvas ∪ R1 extracts for structural classes."""
    mask_cyan = np.all(arr == cyan, axis=2) | np.asarray(
        extract(fake, cyan, tolerance=tol), dtype=bool)
    mask_building = np.all(arr == magenta, axis=2) | np.asarray(
        extract(fake, magenta, tolerance=tol), dtype=bool)
    mask_pavement = np.all(arr == yellow, axis=2) | np.asarray(
        extract(fake, yellow, tolerance=tol), dtype=bool)
    mask_lawn = np.all(arr == lawn_rgb, axis=2)
    mask_plant = np.all(arr == green, axis=2)
    mask_road = np.all(arr == (241, 145, 73), axis=2)
    return dict(
        cyan=mask_cyan,
        building=mask_building,
        pavement=mask_pavement,
        lawn=mask_lawn,
        plant=mask_plant,
        road=mask_road,
        fixed=fixed,
    )


def r2_image(r1, raw, cons, api, *, building_policy='local_repair', skip=()):
    """Staged R2 with coverage write-authority.

    building_policy:
      local_repair — fill enclosed micro-holes, edge despeckle, drop abnormal crumbs;
                     final protect uses *approved* building mask (default this round).
      preserve — historical: force R1 building footprint at end.
      legacy_bbox — old MBB path (not used for quality claims).
    """
    if building_policy not in ('preserve', 'legacy_bbox', 'local_repair'):
        raise ValueError('Unknown building policy')
    fake = Image.fromarray(r1)
    result = fake.copy()
    extract = api['extract']
    tol = api['tolerance']
    road_settings = api.get('r2_settings', {'road_copy_tolerance': 40})
    native_road_kwargs = api.get('road_kwargs', {'tolerance': 30, 'max_width': 2})
    cyan, green, yellow = (0, 255, 255), (0, 152, 67), (255, 255, 0)
    orange, magenta = (241, 145, 73), (255, 0, 255)
    lawn_rgb = (0, 255, 0)
    fixed = ~cons['editable']
    a = cons['a']
    log = []
    api['vegetation_diagnostics'] = {}
    api['surface_diagnostics'] = {}
    api['building_diagnostics'] = {}
    api['approved_building_mask'] = None
    api['coverage_rules'] = COVERAGE_RULES_TEXT
    surf_cfg = api.get('surface_settings') or {}
    bld_cfg = api.get('building_settings') or {}
    approved_building = np.all(r1 == magenta, axis=2) & cons['editable']
    # Road lock: R1 roads protected through fill/denoise/building/veg.
    # Road topology steps may revise the mask; lock is refreshed afterward.
    api['road_lock_mask'] = np.all(r1 == orange, axis=2) & cons['editable']
    _road_ops_skip_lock = {
        'road_r1_copy', 'road_connect_cross_vegetation', 'road_protect_approved_mask',
        'building_local_repair', 'building_protect_approved_mask',
        'building_preserve_R1_footprint', 'building_legacy_bbox',
    }

    def step(name, func):
        nonlocal result
        before = np.asarray(result).copy()
        proposed = np.asarray(func(result).convert('RGB'), dtype=np.uint8)
        after = restore_limits(proposed, raw, cons)
        # Restore locked roads unless road topology or building (building outranks road).
        lock = api.get('road_lock_mask')
        road_restored = 0
        if lock is not None and name not in _road_ops_skip_lock:
            lock = np.asarray(lock, dtype=bool) & cons['editable']
            bld = api.get('approved_building_mask')
            if bld is not None:
                lock = lock & ~np.asarray(bld, dtype=bool)
            # Do not paint road over current building pixels.
            bld_now = np.all(after == magenta, axis=2)
            miss = lock & ~np.all(after == orange, axis=2) & ~bld_now
            road_restored = int(miss.sum())
            after[miss] = orange
        log.append(dict(
            operation=name,
            proposed_changed_px=int(np.any(proposed != before, axis=2).sum()),
            fixed_proposal_changed_px=int((np.any(proposed != before, axis=2) & fixed).sum()),
            actual_changed_px=int(np.any(after != before, axis=2).sum()),
            road_lock_restored_px=road_restored,
        ))
        result = Image.fromarray(after)

    # --- Stages 1–2: planar surface via r2_stages (pavement → water) ---
    _morph = SharedSurfaceMorph(
        edge_kernel_px=int(surf_cfg.get('edge_kernel_px', 5)),
        edge_iterations=int(surf_cfg.get('edge_iterations', 2)),
        boundary_band_px=int(surf_cfg.get('boundary_band_px', 3)),
        micro_hole_area_px=int(surf_cfg.get('pavement_hole_area_px', 16)),
        enable_thin_linear_pores=bool(surf_cfg.get('enable_thin_linear_pores', False)),
        thin_pore_max_width_px=float(surf_cfg.get('thin_pore_max_width_px', 3.0)),
        thin_pore_min_aspect=float(surf_cfg.get('thin_pore_min_aspect', 2.5)),
        enable_enclosed_area_holes=bool(
            surf_cfg.get('enable_enclosed_area_holes', False)),
        enclosed_hole_max_area_px2=int(surf_cfg.get('enclosed_hole_max_area_px2', 500)),
        enclosed_hole_max_area_ratio=float(
            surf_cfg.get('enclosed_hole_max_area_ratio', 0.20)),
    )

    _fill_on = bool(surf_cfg.get('enable_internal_green_fill', False))
    _pav_cfg = PavementThresholds(
        morph=SharedSurfaceMorph(
            **{**_morph.__dict__,
               'micro_hole_area_px': int(surf_cfg.get('pavement_hole_area_px', 16))}),
        crumb_area_px=int(surf_cfg.get('pavement_crumb_area_px', 10)),
        expand_band_px=int(surf_cfg.get('pavement_expand_band_px', 1)),
        thin_pore_max_area_px=int(
            surf_cfg.get('pavement_thin_pore_max_area_px',
                         surf_cfg.get('thin_pore_max_area_px', 160))),
    )
    _wat_cfg = WaterThresholds(
        morph=SharedSurfaceMorph(
            **{**_morph.__dict__,
               'micro_hole_area_px': int(surf_cfg.get('water_hole_area_px', 16))}),
        crumb_area_px=int(surf_cfg.get('water_crumb_area_px', 16)),
        thin_pore_max_area_px=int(
            surf_cfg.get('water_thin_pore_max_area_px',
                         surf_cfg.get('thin_pore_max_area_px', 128))),
    )
    _edge = bool(surf_cfg.get('enable_edge_despeckle', True))
    _holes = bool(surf_cfg.get('enable_micro_holes', True))
    _crumbs = bool(surf_cfg.get('enable_noise_crumbs', True))
    _pav_protect_water = bool(surf_cfg.get('pavement_protect_water', False))

    if _fill_on:
        # Phase A: 铺装/水体内部绿地填充（先于去噪）
        if 'pavement' not in skip:
            def pavement_green_fill(o):
                out, slog = run_pavement_stage(
                    np.asarray(o, dtype=np.uint8),
                    fixed=fixed, smooth_fn=api['smooth'], smooth_source=fake,
                    tolerance=tol, cfg=_pav_cfg,
                    enable_edge=False, enable_holes=False, enable_crumbs=False,
                    enable_internal_green_fill=True,
                    internal_green_max_px=int(
                        surf_cfg.get('pavement_internal_green_max_px', 0)),
                    do_smooth=True, do_regularize=False,
                    protect_water_from_pavement=_pav_protect_water,
                )
                api['surface_diagnostics']['pavement_fill'] = slog
                return Image.fromarray(out)

            step('pavement_internal_green_fill', pavement_green_fill)

        if 'water' not in skip:
            def water_green_fill(o):
                out, slog = run_water_stage(
                    np.asarray(o, dtype=np.uint8),
                    fixed=fixed, smooth_fn=api['smooth'], smooth_source=fake,
                    tolerance=tol, cfg=_wat_cfg,
                    enable_edge=False, enable_holes=False, enable_crumbs=False,
                    enable_internal_green_fill=True,
                    internal_green_max_px=int(
                        surf_cfg.get('water_internal_green_max_px', 0)),
                    do_smooth=True, do_regularize=False,
                )
                api['surface_diagnostics']['water_fill'] = slog
                return Image.fromarray(out)

            step('water_internal_green_fill', water_green_fill)

        # Phase B: 面状去噪（碎屑/微孔/边缘）——填充完成之后
        if 'pavement' not in skip:
            def pavement_denoise(o):
                out, slog = run_pavement_stage(
                    np.asarray(o, dtype=np.uint8),
                    fixed=fixed, smooth_fn=api['smooth'], smooth_source=fake,
                    tolerance=tol, cfg=_pav_cfg,
                    enable_edge=_edge, enable_holes=_holes, enable_crumbs=_crumbs,
                    enable_internal_green_fill=False,
                    do_smooth=False, do_regularize=True,
                    protect_water_from_pavement=_pav_protect_water,
                )
                api['surface_diagnostics']['pavement'] = slog
                return Image.fromarray(out)

            step('pavement_denoise', pavement_denoise)

        if 'water' not in skip:
            def water_denoise(o):
                out, slog = run_water_stage(
                    np.asarray(o, dtype=np.uint8),
                    fixed=fixed, smooth_fn=api['smooth'], smooth_source=fake,
                    tolerance=tol, cfg=_wat_cfg,
                    enable_edge=_edge, enable_holes=_holes, enable_crumbs=_crumbs,
                    enable_internal_green_fill=False,
                    do_smooth=False, do_regularize=True,
                )
                api['surface_diagnostics']['water'] = slog
                return Image.fromarray(out)

            step('water_denoise', water_denoise)
    else:
        if 'pavement' not in skip:
            def smooth_pavement(o):
                out, slog = run_pavement_stage(
                    np.asarray(o, dtype=np.uint8),
                    fixed=fixed, smooth_fn=api['smooth'], smooth_source=fake,
                    tolerance=tol, cfg=_pav_cfg,
                    enable_edge=_edge, enable_holes=_holes, enable_crumbs=_crumbs,
                    enable_internal_green_fill=False,
                    protect_water_from_pavement=_pav_protect_water,
                )
                api['surface_diagnostics']['pavement'] = slog
                return Image.fromarray(out)

            step('pavement_original_smoothing', smooth_pavement)

        if 'water' not in skip:
            def smooth_water(o):
                out, slog = run_water_stage(
                    np.asarray(o, dtype=np.uint8),
                    fixed=fixed, smooth_fn=api['smooth'], smooth_source=fake,
                    tolerance=tol, cfg=_wat_cfg,
                    enable_edge=_edge, enable_holes=_holes, enable_crumbs=_crumbs,
                    enable_internal_green_fill=False,
                )
                api['surface_diagnostics']['water'] = slog
                return Image.fromarray(out)

            step('water_original_smoothing', smooth_water)

    # --- Planar boundary smooth (after fill+denoise; road protected; rem→lawn) ---
    if bool(surf_cfg.get('enable_planar_boundary_smooth', False)):
        def planar_boundary_smooth(o):
            arr = np.asarray(o, dtype=np.uint8).copy()
            m = class_masks(arr, fixed)
            k = int(surf_cfg.get('planar_smooth_kernel_px', 7))
            it = int(surf_cfg.get('planar_smooth_iterations', 2))
            band = int(surf_cfg.get('planar_smooth_band_px', 4))
            slog = {}
            arr, plog = regularize_surface_rgb(
                arr, m['pavement'], yellow,
                protected_from_pavement(m, protect_water=_pav_protect_water),
                enable_edge=True, enable_holes=False, enable_crumbs=False,
                edge_kernel_px=k, edge_iterations=it, boundary_band_px=band,
            )
            slog['pavement'] = plog
            m2 = class_masks(arr, fixed)
            arr, wlog = regularize_surface_rgb(
                arr, m2['water'], cyan, protected_from_water(m2),
                enable_edge=True, enable_holes=False, enable_crumbs=False,
                edge_kernel_px=k, edge_iterations=it, boundary_band_px=band,
            )
            slog['water'] = wlog
            api['surface_diagnostics']['planar_boundary_smooth'] = slog
            return Image.fromarray(arr)

        step('planar_boundary_smooth', planar_boundary_smooth)

    # --- Building contour → interior fill → top-priority approved mask ---
    if building_policy == 'local_repair' and 'building' not in skip:
        def repair_buildings(o):
            nonlocal approved_building
            bcfg = BuildingThresholds(
                max_hole_area_px=int(bld_cfg.get('max_hole_area_px', 400)),
                max_crumb_area_px=int(
                    bld_cfg.get('max_crumb_area_px',
                                surf_cfg.get('building_crumb_area_px', 16))),
                edge_kernel_px=int(bld_cfg.get(
                    'edge_kernel_px', surf_cfg.get('edge_kernel_px', 5))),
                boundary_band_px=int(bld_cfg.get(
                    'boundary_band_px', surf_cfg.get('boundary_band_px', 3))),
                enable_conditional_ortho=bool(
                    bld_cfg.get('enable_conditional_ortho', True)),
                enable_hybrid_curve_smooth=bool(
                    bld_cfg.get('enable_hybrid_curve_smooth', True)),
                angle_tol_deg=float(bld_cfg.get('ANGLE_TOL_DEG',
                                                bld_cfg.get('angle_tol_deg', 12.0))),
                near_rect_ratio=float(bld_cfg.get('NEAR_RECT_RATIO',
                                                  bld_cfg.get('near_rect_ratio', 0.85))),
                near_rect_max_area_px=float(
                    bld_cfg.get('NEAR_RECT_MAX_AREA_PX',
                                bld_cfg.get('near_rect_max_area_px', 500))),
                area_change_tol=float(bld_cfg.get('AREA_CHANGE_TOL',
                                                  bld_cfg.get('area_change_tol', 0.12))),
                min_building_area=int(bld_cfg.get('MIN_BUILDING_AREA',
                                                  bld_cfg.get('min_building_area', 40))),
                min_edge_len=float(bld_cfg.get('MIN_EDGE_LEN',
                                               bld_cfg.get('min_edge_len', 6.0))),
                max_aspect_ratio=float(bld_cfg.get('MAX_ASPECT_RATIO',
                                                   bld_cfg.get('max_aspect_ratio', 8.0))),
                dp_epsilon_frac=float(bld_cfg.get('DP_EPSILON_FRAC',
                                                  bld_cfg.get('dp_epsilon_frac', 0.03))),
                multi_axis_spread_deg=float(
                    bld_cfg.get('MULTI_AXIS_ANGLE_SPREAD_DEG',
                                bld_cfg.get('multi_axis_spread_deg', 25.0))),
                curve_edge_frac=float(bld_cfg.get('CURVE_EDGE_FRAC',
                                                  bld_cfg.get('curve_edge_frac', 0.35))),
                force_ortho_solidity=float(
                    bld_cfg.get('FORCE_ORTHO_SOLIDITY',
                                bld_cfg.get('force_ortho_solidity', 0.85))),
                concave_solidity_min=float(
                    bld_cfg.get('CONCAVE_SOLIDITY_MIN',
                                bld_cfg.get('concave_solidity_min', 0.92))),
                concave_defect_depth_frac=float(
                    bld_cfg.get('CONCAVE_DEFECT_DEPTH_FRAC',
                                bld_cfg.get('concave_defect_depth_frac', 0.08))),
                enable_rotate_ortho=bool(bld_cfg.get('enable_rotate_ortho', True)),
                stage2_angle_deg=float(
                    bld_cfg.get('STAGE2_ANGLE_DEG',
                                bld_cfg.get('stage2_angle_deg', 30.0))),
                stage2_max_edge_px=float(
                    bld_cfg.get('STAGE2_MAX_EDGE_PX',
                                bld_cfg.get('stage2_max_edge_px', 10.0))),
                complex_isoperimetric_q_max=float(
                    bld_cfg.get('complex_isoperimetric_q_max', 0.35)),
                complex_solidity_max=float(
                    bld_cfg.get('complex_solidity_max', 0.78)),
                complex_preclose_px=int(bld_cfg.get('complex_preclose_px', 5)),
                straight_dev_px=float(bld_cfg.get('straight_dev_px', 2.0)),
                corner_angle_deg=float(bld_cfg.get('corner_angle_deg', 28.0)),
                curve_smooth_iters=int(bld_cfg.get('curve_smooth_iters', 4)),
                hybrid_band_px=int(bld_cfg.get('hybrid_band_px', 8)),
                hybrid_pre_dp_eps_px=float(bld_cfg.get('hybrid_pre_dp_eps_px', 1.5)),
                hybrid_curve_dp_eps_px=float(
                    bld_cfg.get('hybrid_curve_dp_eps_px', 1.5)),
                hybrid_curve_ma_win=int(bld_cfg.get('hybrid_curve_ma_win', 7)),
            )
            out, new_b, blog = run_building_stage(
                np.asarray(o, dtype=np.uint8),
                fixed=fixed, r1_rgb=r1, cfg=bcfg,
                ortho_debug_dir=bld_cfg.get('ortho_debug_dir'),
                ortho_debug_tag=str(bld_cfg.get('ortho_debug_tag', 'building_ortho')),
            )
            approved_building = new_b
            api['approved_building_mask'] = new_b
            api['building_diagnostics'] = blog
            api['surface_diagnostics']['building'] = blog
            # Building outranks road: drop building footprint from road lock.
            lock = api.get('road_lock_mask')
            if lock is not None and new_b is not None:
                api['road_lock_mask'] = np.asarray(lock, dtype=bool) & ~new_b
            return Image.fromarray(out)

        step('building_local_repair', repair_buildings)
    elif building_policy == 'preserve':
        api['surface_diagnostics']['building'] = dict(
            enabled=False, reason='building_policy_preserve')
        approved_building = np.all(r1 == magenta, axis=2) & cons['editable']
        api['approved_building_mask'] = approved_building
    else:
        api['surface_diagnostics']['building'] = dict(
            enabled=False, reason='legacy_bbox_no_local_repair')

    mask_plant_r1 = np.asarray(extract(fake, green, tolerance=tol), dtype=bool)
    mask_veg_r1 = mask_plant_r1

    barriers = _refresh_barriers(
        np.asarray(result), fake, extract, tol,
        cyan=cyan, magenta=magenta, green=green, yellow=yellow,
        lawn_rgb=lawn_rgb, fixed=fixed)
    mask_vegetation = mask_plant_r1 | barriers['lawn'] | barriers['plant']

    # --- Vegetation circle packing (before road; building/road-lock protected) ---
    if 'vegetation' not in skip:
        def paint_vegetation(o):
            arr = np.asarray(o)
            m = class_masks(arr, fixed)
            b = _refresh_barriers(
                arr, fake, extract, tol,
                cyan=cyan, magenta=magenta, green=green, yellow=yellow,
                lawn_rgb=lawn_rgb, fixed=fixed)
            m['building'] = m['building'] | b['building']
            if approved_building is not None:
                m['building'] = m['building'] | approved_building
            m['pavement'] = m['pavement'] | b['pavement']
            # Pre-road: protect R1 / lock roads so packing cannot paint over them.
            lock = api.get('road_lock_mask')
            if lock is not None:
                m['road'] = m['road'] | np.asarray(lock, dtype=bool)
            m['road'] = m['road'] | b['road']
            veg = dict(api.get('vegetation_settings') or {})
            blocking = vegetation_blockers(
                m, protect_water=bool(veg.get('protect_water_from_vegetation', False)))
            veg_work = mask_veg_r1 & ~blocking.astype(bool)
            vdiag = {}
            method = str(veg.get('method') or 'three_branch_noise_small_cluster_edt')
            if method in ('keep_r1_plant', 'keep_r1_plant_no_circles'):
                packing = bool(veg.get('circle_packing_enabled', False))
                if method == 'keep_r1_plant_no_circles':
                    packing = False
                out = restore_r1_plant(
                    o, veg_work.astype(np.uint8), m['road'].astype(np.uint8),
                    blocking,
                    circle_packing_enabled=packing,
                    clear_r1_plants_first=bool(veg.get('clear_r1_plants_first', False)),
                    noise_area_max_px=float(veg.get('noise_area_max_px', 6.0)),
                    min_isolated_area_px=float(
                        veg.get('min_isolated_area_px',
                                veg.get('large_component_area_px', 40.0))),
                    crown_jitter=float(veg.get('crown_jitter', 0.25)),
                    rng_seed=veg.get('rng_seed'),
                    road_relax_px=float(veg.get('road_relax_px', 2.0)),
                    diagnostics=vdiag)
            elif method in ('three_branch_noise_small_cluster_edt',
                            'min_dist_two_pass_edt'):
                skip_keys = {
                    'enable_lawn_clump', 'method',
                    'noise_area_max_px', 'circle_packing_enabled',
                    'large_component_area_px',
                    'crown_jitter', 'rng_seed', 'min_isolated_area_px',
                    'protect_water_from_vegetation',
                }
                out = api['trees'](
                    o, veg_work.astype(np.uint8), m['road'].astype(np.uint8),
                    blocking,
                    large_size_threshold=int(veg.get('large_component_area_px', 40)),
                    noise_area_max_px=float(veg.get('noise_area_max_px', 6.0)),
                    circle_packing_enabled=bool(
                        veg.get('circle_packing_enabled', True)),
                    diagnostics=vdiag,
                    **{k: v for k, v in veg.items() if k not in skip_keys})
            else:
                raise ValueError(f'Unknown vegetation method: {method}')
            out_a = np.asarray(out, dtype=np.uint8).copy()
            # Restore building top + road lock after packing.
            if approved_building is not None:
                out_a[np.asarray(approved_building, dtype=bool) & ~fixed] = magenta
            if lock is not None:
                hit = np.asarray(lock, dtype=bool) & ~fixed
                if approved_building is not None:
                    hit = hit & ~np.asarray(approved_building, dtype=bool)
                restored = int(np.count_nonzero(
                    hit & ~np.all(out_a == orange, axis=2)))
                out_a[hit] = orange
                vdiag['road_restored_after_veg_px'] = restored
            api['vegetation_diagnostics'] = vdiag
            return Image.fromarray(out_a)

        step('vegetation_circle_packing', paint_vegetation)

    # --- Road last (priority second only to building) ---
    if 'road' not in skip:
        road_copy = np.asarray(
            extract(fake, orange, tolerance=road_settings['road_copy_tolerance']),
            dtype=bool) & ~fixed

        def copy_road(o):
            arr = np.asarray(o).copy()
            paste = road_copy & ~np.all(arr == magenta, axis=2)
            if approved_building is not None:
                paste = paste & ~approved_building
            arr[paste] = r1[paste]
            return Image.fromarray(arr)

        step('road_r1_copy', copy_road)
        barriers = _refresh_barriers(
            np.asarray(result), fake, extract, tol,
            cyan=cyan, magenta=magenta, green=green, yellow=yellow,
            lawn_rgb=lawn_rgb, fixed=fixed)
        mask_vegetation = mask_plant_r1 | barriers['lawn'] | barriers['plant']

        road_rev = api.get('road_settings') or {}
        road_kwargs = dict(native_road_kwargs)
        for key in ('morph_close_px', 'connect_max_gap_px', 'max_links_per_endpoint',
                    'allow_cross_vegetation', 'direction_tolerance_deg',
                    'branch_trace_px', 'enable_t_junction',
                    'reject_third_road_contact', 'enable_building_proxy',
                    'pavement_contact_radius_px',
                    'prune_min_area_px', 'prune_max_span_px', 'prune_spur_max_px',
                    'enable_width_regularization',
                    'width_half_min_px', 'width_half_max_px',
                    'width_mode',
                    'access_stub_area_px', 'large_cc_ee_cap_px', 'medium_gap_px',
                    'pinch_gap_px', 'large_cc_area_px'):
            if key in road_rev:
                road_kwargs[key] = road_rev[key]
        m_now = class_masks(np.asarray(result), fixed)
        if approved_building is not None:
            m_now['building'] = m_now['building'] | approved_building
        hard_blocks = road_hard_blockers(m_now)
        road_diag = []
        api['road_diagnostics'] = road_diag
        use_raw_guide = bool(road_rev.get('use_raw_guide', True))
        extra = dict(
            vegetation_masks=[mask_vegetation],
            guide_rgb=(raw if use_raw_guide else None),
            diagnostics=road_diag,
            pavement_mask=m_now['pavement'],
            building_mask=m_now['building'],
            water_mask=m_now['water'],
        )
        road_fn = api['road']
        try:
            sig = inspect.signature(road_fn)
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not accepts_var_kw:
                allowed = set(sig.parameters)
                road_kwargs = {k: v for k, v in road_kwargs.items() if k in allowed}
                extra = {k: v for k, v in extra.items() if k in allowed}
        except (TypeError, ValueError):
            pass
        step('road_connect_cross_vegetation', lambda o: road_fn(
            o, fake, orange_rgb=orange,
            blocking_masks=hard_blocks,
            **extra,
            **road_kwargs))
        approved = np.all(np.asarray(result) == orange, axis=2) & cons['editable']
        if approved_building is not None:
            approved = approved & ~approved_building
        api['approved_road_mask'] = approved
        api['road_lock_mask'] = np.asarray(approved, dtype=bool)

    # --- Stage 8 ---
    if building_policy == 'legacy_bbox':
        road40 = np.asarray(
            extract(fake, orange, tolerance=road_settings['road_copy_tolerance']), dtype=bool)
        bmask = barriers['building'] & ~road40 & ~fixed
        boxes = api['boxes']((bmask * 255).astype(np.uint8))
        step('building_legacy_bbox', lambda o: api['blocks'](o, boxes, magenta))

    step('output_palette_normalization',
         lambda o: Image.fromarray(nearest_rgb(np.asarray(o), cons['editable'])))

    if building_policy == 'preserve':
        original_b = np.all(r1 == magenta, axis=2) & cons['editable']

        def keep_buildings(o):
            arr = np.asarray(o).copy()
            current_b = np.all(arr == magenta, axis=2) & cons['editable']
            arr[current_b & ~original_b] = r1[current_b & ~original_b]
            arr[original_b] = magenta
            return Image.fromarray(arr)

        step('building_preserve_R1_footprint', keep_buildings)
    elif building_policy == 'local_repair' and approved_building is not None:
        ref_b = approved_building & cons['editable']

        def keep_approved_buildings(o):
            arr = np.asarray(o).copy()
            # Force approved footprint; do not resurrect R1 holes/crumbs.
            cur = np.all(arr == magenta, axis=2) & cons['editable']
            extra = cur & ~ref_b
            if extra.any():
                arr[extra] = lawn_rgb
            arr[ref_b] = magenta
            return Image.fromarray(arr)

        step('building_protect_approved_mask', keep_approved_buildings)

    # Final road protect (never overwrite building — building is top).
    approved_road = api.get('approved_road_mask')
    if approved_road is not None and 'road' not in skip:
        ref_r = np.asarray(approved_road, dtype=bool) & cons['editable']
        if approved_building is not None:
            ref_r = ref_r & ~approved_building

        def keep_approved_roads(o):
            arr = np.asarray(o, dtype=np.uint8).copy()
            # Re-assert building first.
            if approved_building is not None:
                arr[np.asarray(approved_building, dtype=bool) & ~fixed] = magenta
            miss = ref_r & ~np.all(arr == orange, axis=2)
            arr[ref_r] = orange
            api.setdefault('vegetation_diagnostics', {})
            prev = int(api['vegetation_diagnostics'].get('road_restored_after_veg_px', 0) or 0)
            api['vegetation_diagnostics']['road_restored_final_px'] = prev + int(miss.sum())
            return Image.fromarray(arr)

        step('road_protect_approved_mask', keep_approved_roads)

    def restore_markers(o):
        return Image.fromarray(restore_condition_markers(np.asarray(o), a))

    step('restore_A_black_purple_red', restore_markers)

    final = restore_limits(np.asarray(result), raw, cons)
    final = restore_condition_markers(final, a)
    if building_policy == 'preserve':
        original_b = (exact_labels(r1) == 1) & cons['editable']
        final_b = (exact_labels(final) == 1) & cons['editable']
        if not np.array_equal(original_b, final_b):
            raise RuntimeError('Building-preserve invariant violated')
    elif building_policy == 'local_repair' and approved_building is not None:
        ref_b = approved_building & cons['editable']
        final_b = (exact_labels(final) == 1) & cons['editable']
        if not np.array_equal(ref_b, final_b):
            # Soft enforce: rewrite once more
            final = final.copy()
            final[ref_b] = magenta
            leak = final_b & ~ref_b
            final[leak] = lawn_rgb
            final = restore_condition_markers(final, a)
    return final, log
