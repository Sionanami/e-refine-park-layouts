"""R1 weighted snap parameters and checked binding to ORIGINAL road parameters.

Paper τ (“assign if distance < τ”) is not used. R1 assigns every editable pixel
by weighted RGB L2 argmin (plant 0.55, green 0.72, pav-over-road margin 28).
The supplied code.zip had no absolute-distance tau; connection-cost weights for
R2 are still unverified in the original util API.
"""
from __future__ import annotations
import inspect
import json
import math
from numbers import Real
from pathlib import Path
from ..study_common import (sha, DEFAULT_PLANT_DIST_WEIGHT, DEFAULT_GREEN_DIST_WEIGHT,
                          DEFAULT_PAV_OVER_ROAD_MARGIN)
from ..paths import PARAMETERS_PATH as DEFAULT_PARAMETERS
CONTROLS = ('candidate_search_radius_px','direction_tolerance_deg',
            'connection_cost_weights','local_width_threshold_px')
R1_KEYS = {'distance_metric','assignment','plant_dist_weight','green_dist_weight',
           'pav_over_road_margin','absolute_distance_tau'}
VEG_KEYS = {'method','circle_packing_enabled','min_center_dist_px','large_radius_min_px',
            'small_min_center_dist_px','small_radius_min_px','use_orange_radius_cap',
            'clump_close_px','clear_r1_plants_first','large_component_area_px',
            'noise_area_max_px','enable_lawn_clump','crown_jitter','rng_seed',
            'min_isolated_area_px','road_relax_px',
            'circle_smooth_window','circle_smooth_passes','circle_smooth_band_px',
            'circle_smooth_close_px','circle_smooth_min_area_px',
            'circle_smooth_hole_max_px','protect_water_from_vegetation'}
ROAD_REV_KEYS = {'allow_cross_vegetation','morph_close_px','connect_max_gap_px',
                 'max_links_per_endpoint','direction_tolerance_deg','branch_trace_px',
                 'enable_t_junction','use_raw_guide','reject_third_road_contact',
                 'enable_building_proxy','pavement_contact_radius_px'}
ROAD_REV_OPTIONAL_KEYS = {
    'enable_width_regularization', 'width_half_min_px', 'width_half_max_px',
    'width_mode', 'prune_min_area_px', 'prune_max_span_px', 'prune_spur_max_px',
    'access_stub_area_px', 'large_cc_ee_cap_px', 'medium_gap_px',
    'pinch_gap_px', 'large_cc_area_px',
}
SURFACE_KEYS = {'enable_edge_despeckle','enable_micro_holes','enable_noise_crumbs',
                'edge_kernel_px','edge_iterations','boundary_band_px'}
# Optional surface crumb/hole areas (forwarded to refine; not required for schema=3).
SURFACE_OPTIONAL_KEYS = {
    'pavement_crumb_area_px', 'water_crumb_area_px',
    'pavement_hole_area_px', 'water_hole_area_px',
    'building_crumb_area_px',
    'enable_internal_green_fill',
    'pavement_internal_green_max_px', 'water_internal_green_max_px',
    'enable_planar_boundary_smooth',
    'planar_smooth_kernel_px', 'planar_smooth_iterations', 'planar_smooth_band_px',
    'pavement_protect_water',
}
BUILDING_OPTIONAL_KEYS = {
    'enable_conditional_ortho', 'enable_hybrid_curve_smooth',
    'max_hole_area_px', 'max_crumb_area_px', 'edge_kernel_px', 'boundary_band_px',
    'straight_dev_px', 'corner_angle_deg', 'curve_smooth_iters', 'hybrid_band_px',
    'curve_edge_frac', 'force_ortho_solidity',
    'complex_isoperimetric_q_max', 'complex_solidity_max', 'complex_preclose_px',
}
VEG_METHODS = {
    'keep_r1_plant',  # no three-branch path
    'keep_r1_plant_no_circles',
    'min_dist_two_pass_edt',  # legacy
    'three_branch_noise_small_cluster_edt',  # production packing path (+ circle_packing_enabled)
}

def finite_real(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)

def load_parameters(path=None):
    path = Path(path or DEFAULT_PARAMETERS).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    allowed_top = {'schema_version', 'r1', 'r2', 'meta'}
    if set(cfg) - allowed_top or not {'schema_version', 'r1', 'r2'} <= set(cfg):
        raise ValueError('Unsupported or misspelled parameter configuration (expect schema_version=3)')
    if cfg['schema_version'] != 3:
        raise ValueError('Unsupported or misspelled parameter configuration (expect schema_version=3)')
    r1, r2 = cfg['r1'], cfg['r2']
    if set(r1) != R1_KEYS:
        raise ValueError('R1 requires weighted_argmin fields (and absolute_distance_tau=null)')
    if r1['distance_metric'] != 'rgb_l2' or r1['assignment'] != 'weighted_argmin':
        raise ValueError('R1 must use rgb_l2 weighted_argmin (no priority/tau gate)')
    if r1['absolute_distance_tau'] is not None:
        raise ValueError('Paper absolute-distance tau gate is retired; set absolute_distance_tau to null')
    for key, lo, hi in (('plant_dist_weight', 0, 1), ('green_dist_weight', 0, 1),
                        ('pav_over_road_margin', 0, math.sqrt(3)*255)):
        if not finite_real(r1[key]) or not lo <= r1[key] <= hi:
            raise ValueError(f'Invalid R1 {key}')
    allowed_r2 = {'road_color_tolerance','road_copy_tolerance','legacy_max_width_px',
                  'road_revision','native_controls','vegetation','surface','building'}
    if set(r2) - allowed_r2:
        raise ValueError('Unsupported/misspelled R2 parameter')
    if not allowed_r2 - {'surface', 'building'} <= set(r2):
        raise ValueError('Missing required R2 parameter block')
    for key in ('road_color_tolerance','road_copy_tolerance','legacy_max_width_px'):
        if not finite_real(r2[key]) or r2[key] < 0:
            raise ValueError(f'Invalid R2 {key}')
    if not isinstance(r2['legacy_max_width_px'], int) or r2['legacy_max_width_px'] < 1:
        raise ValueError('Original road max_width must be a positive integer')
    road_rev = r2['road_revision']
    if (not ROAD_REV_KEYS.issubset(set(road_rev))
            or set(road_rev) - ROAD_REV_KEYS - ROAD_REV_OPTIONAL_KEYS):
        raise ValueError('R2 road_revision keys mismatch')
    if not isinstance(road_rev['allow_cross_vegetation'], bool):
        raise ValueError('road_revision.allow_cross_vegetation must be boolean')
    if not isinstance(road_rev['morph_close_px'], int) or road_rev['morph_close_px'] < 0:
        raise ValueError('road_revision.morph_close_px must be a nonnegative int')
    if not finite_real(road_rev['connect_max_gap_px']) or road_rev['connect_max_gap_px'] < 0:
        raise ValueError('Invalid road_revision.connect_max_gap_px')
    if not isinstance(road_rev['max_links_per_endpoint'], int) or road_rev['max_links_per_endpoint'] < 1:
        raise ValueError('road_revision.max_links_per_endpoint must be a positive int')
    if not finite_real(road_rev['direction_tolerance_deg']) or not 0 <= road_rev['direction_tolerance_deg'] <= 180:
        raise ValueError('road_revision.direction_tolerance_deg must be in [0,180]')
    if not isinstance(road_rev['branch_trace_px'], int) or road_rev['branch_trace_px'] < 2:
        raise ValueError('road_revision.branch_trace_px must be an int >= 2')
    if not isinstance(road_rev['enable_t_junction'], bool):
        raise ValueError('road_revision.enable_t_junction must be boolean')
    if not isinstance(road_rev['use_raw_guide'], bool):
        raise ValueError('road_revision.use_raw_guide must be boolean')
    if not isinstance(road_rev['reject_third_road_contact'], bool):
        raise ValueError('road_revision.reject_third_road_contact must be boolean')
    if not isinstance(road_rev['enable_building_proxy'], bool):
        raise ValueError('road_revision.enable_building_proxy must be boolean')
    if not isinstance(road_rev['pavement_contact_radius_px'], int) or road_rev['pavement_contact_radius_px'] < 0:
        raise ValueError('road_revision.pavement_contact_radius_px must be a nonnegative int')
    if 'enable_width_regularization' in road_rev and not isinstance(
            road_rev['enable_width_regularization'], bool):
        raise ValueError('road_revision.enable_width_regularization must be boolean')
    for wk in ('width_half_min_px', 'width_half_max_px'):
        if wk in road_rev and (not finite_real(road_rev[wk]) or road_rev[wk] < 0):
            raise ValueError(f'road_revision.{wk} must be a nonnegative real')
    veg = r2['vegetation']
    required_veg = VEG_KEYS - {
        'crown_jitter', 'rng_seed', 'min_isolated_area_px', 'road_relax_px',
        'circle_smooth_window', 'circle_smooth_passes', 'circle_smooth_band_px',
        'circle_smooth_close_px', 'circle_smooth_min_area_px',
        'circle_smooth_hole_max_px', 'protect_water_from_vegetation'}
    if not required_veg.issubset(set(veg)) or set(veg) - VEG_KEYS or veg['method'] not in VEG_METHODS:
        raise ValueError('R2 vegetation requires known method and Min-dist/three-branch fields')
    if 'crown_jitter' in veg and (not finite_real(veg['crown_jitter']) or veg['crown_jitter'] < 0):
        raise ValueError('vegetation.crown_jitter must be a nonnegative real')
    if 'min_isolated_area_px' in veg and (
            not finite_real(veg['min_isolated_area_px']) or veg['min_isolated_area_px'] < 0):
        raise ValueError('vegetation.min_isolated_area_px must be a nonnegative real')
    if 'road_relax_px' in veg and (
            not finite_real(veg['road_relax_px']) or veg['road_relax_px'] < 0):
        raise ValueError('vegetation.road_relax_px must be a nonnegative real')
    for sk in ('circle_smooth_window', 'circle_smooth_passes', 'circle_smooth_band_px',
               'circle_smooth_close_px', 'circle_smooth_min_area_px',
               'circle_smooth_hole_max_px'):
        if sk in veg and (not isinstance(veg[sk], int) or veg[sk] < 0):
            raise ValueError(f'vegetation.{sk} must be a nonnegative int')
    if not isinstance(veg['use_orange_radius_cap'], bool):
        raise ValueError('vegetation.use_orange_radius_cap must be boolean')
    if not isinstance(veg['clear_r1_plants_first'], bool):
        raise ValueError('vegetation.clear_r1_plants_first must be boolean')
    if not isinstance(veg['enable_lawn_clump'], bool):
        raise ValueError('vegetation.enable_lawn_clump must be boolean')
    if not isinstance(veg['circle_packing_enabled'], bool):
        raise ValueError('vegetation.circle_packing_enabled must be boolean')
    for key in ('min_center_dist_px','large_radius_min_px','small_min_center_dist_px',
                'small_radius_min_px','clump_close_px','large_component_area_px',
                'noise_area_max_px'):
        if not finite_real(veg[key]) or veg[key] < 0:
            raise ValueError(f'Invalid vegetation {key}')
    if veg['small_radius_min_px'] > veg['large_radius_min_px']:
        raise ValueError('small_radius_min_px must be <= large_radius_min_px')
    if 'surface' in r2:
        surf = r2['surface']
        if not SURFACE_KEYS.issubset(set(surf)) or set(surf) - SURFACE_KEYS - SURFACE_OPTIONAL_KEYS:
            raise ValueError('R2 surface keys mismatch')
        for bk in ('enable_edge_despeckle', 'enable_micro_holes', 'enable_noise_crumbs'):
            if not isinstance(surf[bk], bool):
                raise ValueError(f'surface.{bk} must be boolean')
        for ik in ('edge_kernel_px', 'edge_iterations', 'boundary_band_px'):
            if not isinstance(surf[ik], int) or surf[ik] < 0:
                raise ValueError(f'surface.{ik} must be a nonnegative int')
        for bk in ('enable_internal_green_fill', 'enable_planar_boundary_smooth'):
            if bk in surf and not isinstance(surf[bk], bool):
                raise ValueError(f'surface.{bk} must be boolean')
        for ok in SURFACE_OPTIONAL_KEYS - {
                'enable_internal_green_fill', 'enable_planar_boundary_smooth'}:
            if ok in surf and (not isinstance(surf[ok], int) or surf[ok] < 0):
                raise ValueError(f'surface.{ok} must be a nonnegative int')
    if 'building' in r2:
        bld = r2['building']
        if set(bld) - BUILDING_OPTIONAL_KEYS:
            raise ValueError('R2 building keys mismatch')
        for bk in ('enable_conditional_ortho', 'enable_hybrid_curve_smooth'):
            if bk in bld and not isinstance(bld[bk], bool):
                raise ValueError(f'building.{bk} must be boolean')
        for ik in ('max_hole_area_px', 'max_crumb_area_px', 'edge_kernel_px',
                   'boundary_band_px', 'curve_smooth_iters', 'hybrid_band_px'):
            if ik in bld and (not isinstance(bld[ik], int) or bld[ik] < 0):
                raise ValueError(f'building.{ik} must be a nonnegative int')
        for fk in ('straight_dev_px', 'corner_angle_deg', 'curve_edge_frac',
                   'force_ortho_solidity'):
            if fk in bld and (not finite_real(bld[fk]) or bld[fk] < 0):
                raise ValueError(f'building.{fk} must be a nonnegative real')
    if set(r2['native_controls']) != set(CONTROLS):
        raise ValueError('Missing/unknown native road control')
    for key, entry in r2['native_controls'].items():
        if set(entry) != {'value','argument'}:
            raise ValueError(f'Invalid control structure: {key}')
        value, argument = entry['value'], entry['argument']
        if value is None:
            if argument is not None:
                raise ValueError(f'Unassigned value with a binding: {key}')
            continue
        if key == 'connection_cost_weights':
            if not isinstance(value,dict) or not value or any(not finite_real(v) or v < 0 for v in value.values()) or sum(value.values()) <= 0:
                raise ValueError('Connection weights require named nonnegative values with positive sum')
            if not isinstance(argument, (str,dict)):
                raise ValueError('Bind weights to one dict argument or explicit component arguments')
            if isinstance(argument,dict) and set(argument) != set(value):
                raise ValueError('Every requested weight needs an explicit native argument')
        else:
            if not finite_real(value) or value < 0:
                raise ValueError(f'Invalid control value: {key}')
            if key == 'direction_tolerance_deg' and value > 180:
                raise ValueError('Direction tolerance must be in [0,180] degrees')
            if not isinstance(argument,str):
                raise ValueError(f'An explicit native argument is required: {key}')
        args = argument.values() if isinstance(argument,dict) else [argument]
        if any(not isinstance(a,str) or not a.isidentifier() for a in args):
            raise ValueError(f'Invalid native argument name: {key}')
    return cfg, dict(parameters_path=str(path), parameters_sha256=sha(path), parameter_config=cfg)

def resolve_r1_weights(cfg):
    r1 = cfg['r1']
    weights = dict(
        plant_dist_weight=float(r1['plant_dist_weight']),
        green_dist_weight=float(r1['green_dist_weight']),
        pav_over_road_margin=float(r1['pav_over_road_margin']),
    )
    source = (
        f"weighted_argmin plant={weights['plant_dist_weight']}, "
        f"green={weights['green_dist_weight']}, "
        f"pav_over_road_margin={weights['pav_over_road_margin']}; "
        f"paper absolute-distance tau not applied"
    )
    defaults = (weights['plant_dist_weight'] == DEFAULT_PLANT_DIST_WEIGHT and
                weights['green_dist_weight'] == DEFAULT_GREEN_DIST_WEIGHT and
                weights['pav_over_road_margin'] == DEFAULT_PAV_OVER_ROAD_MARGIN)
    if defaults:
        source += '; matches frozen r1r2_metric_repair._snap_internal defaults'
    return weights, source

def resolve_vegetation(cfg):
    veg = cfg['r2']['vegetation']
    method = str(veg.get('method') or 'three_branch_noise_small_cluster_edt')
    settings = dict(
        method=method,
        circle_packing_enabled=bool(veg.get('circle_packing_enabled', True)),
        min_center_dist_px=float(veg['min_center_dist_px']),
        large_radius_min_px=float(veg['large_radius_min_px']),
        small_min_center_dist_px=float(veg['small_min_center_dist_px']),
        small_radius_min_px=float(veg['small_radius_min_px']),
        use_orange_radius_cap=bool(veg['use_orange_radius_cap']),
        clump_close_px=float(veg['clump_close_px']),
        clear_r1_plants_first=bool(veg['clear_r1_plants_first']),
        large_component_area_px=float(veg['large_component_area_px']),
        noise_area_max_px=float(veg.get('noise_area_max_px', 6.0)),
        enable_lawn_clump=bool(veg['enable_lawn_clump']),
        crown_jitter=float(veg.get('crown_jitter', 0.25)),
        rng_seed=veg.get('rng_seed'),
        min_isolated_area_px=float(
            veg['min_isolated_area_px']) if 'min_isolated_area_px' in veg
            else float(veg.get('large_component_area_px', 40.0)),
        road_relax_px=float(veg.get('road_relax_px', 2.0)),
        circle_smooth_window=int(veg.get('circle_smooth_window', 5)),
        circle_smooth_passes=int(veg.get('circle_smooth_passes', 1)),
        circle_smooth_band_px=int(veg.get('circle_smooth_band_px', 2)),
        circle_smooth_close_px=int(veg.get('circle_smooth_close_px', 0)),
        circle_smooth_min_area_px=int(veg.get('circle_smooth_min_area_px', 20)),
        circle_smooth_hole_max_px=int(veg.get('circle_smooth_hole_max_px', 0)),
        protect_water_from_vegetation=bool(
            veg.get('protect_water_from_vegetation', False)),
    )
    if method == 'keep_r1_plant_no_circles':
        source = f"{method}: restore R1 plant; circle packing forced off"
    elif method == 'keep_r1_plant':
        if settings['circle_packing_enabled']:
            source = (
                f"{method}: restore R1 plant + circles; "
                f"crown_diameter = mean isolated plant patch (lawn-only); "
                f"jitter=±{settings['crown_jitter']}; "
                f"noise_area_px2={settings['noise_area_max_px']}; "
                f"clear_first={settings['clear_r1_plants_first']}"
            )
        else:
            source = f"{method}: restore R1 plant; circle packing off"
    else:
        source = (
            f"{method} circle_packing_enabled={settings['circle_packing_enabled']}, "
            f"small_r_min={settings['small_radius_min_px']}, "
            f"clump_close={settings['clump_close_px']} (0=off), "
            f"clear_first={settings['clear_r1_plants_first']}, "
            f"area_split_px2={settings['large_component_area_px']}, "
            f"noise_area_px2={settings['noise_area_max_px']}, "
            f"road_relax={settings['road_relax_px']}, "
            f"lawn_clump={settings['enable_lawn_clump']}"
        )
    return settings, source


def resolve_road_revision(cfg):
    rev = cfg['r2']['road_revision']
    settings = dict(
        morph_close_px=int(rev['morph_close_px']),
        connect_max_gap_px=float(rev['connect_max_gap_px']),
        max_links_per_endpoint=int(rev['max_links_per_endpoint']),
        allow_cross_vegetation=bool(rev['allow_cross_vegetation']),
        direction_tolerance_deg=float(rev['direction_tolerance_deg']),
        branch_trace_px=int(rev['branch_trace_px']),
        enable_t_junction=bool(rev['enable_t_junction']),
        use_raw_guide=bool(rev['use_raw_guide']),
        reject_third_road_contact=bool(rev['reject_third_road_contact']),
        enable_building_proxy=bool(rev.get('enable_building_proxy', True)),
        pavement_contact_radius_px=int(rev.get('pavement_contact_radius_px', 2)),
        enable_width_regularization=bool(rev.get('enable_width_regularization', True)),
        width_half_min_px=float(rev.get('width_half_min_px', 1.0)),
        width_half_max_px=float(rev.get('width_half_max_px', 4.0)),
        width_mode=str(rev.get('width_mode', 'median')),
        prune_min_area_px=int(rev.get('prune_min_area_px', 14)),
        prune_max_span_px=int(rev.get('prune_max_span_px', 10)),
        prune_spur_max_px=int(rev.get('prune_spur_max_px', 0)),
        access_stub_area_px=int(rev.get('access_stub_area_px', 40)),
        large_cc_ee_cap_px=float(rev.get('large_cc_ee_cap_px', 12.0)),
        medium_gap_px=float(rev.get('medium_gap_px', 16.0)),
        pinch_gap_px=float(rev.get('pinch_gap_px', 10.0)),
        large_cc_area_px=int(rev.get('large_cc_area_px', 160)),
    )
    source = (
        f"cross_veg={settings['allow_cross_vegetation']}, "
        f"morph_close={settings['morph_close_px']}, "
        f"max_gap={settings['connect_max_gap_px']}, "
        f"links/end={settings['max_links_per_endpoint']}, "
        f"dir_tol_deg={settings['direction_tolerance_deg']} (engineering default), "
        f"t_junction={settings['enable_t_junction']}, "
        f"raw_guide={settings['use_raw_guide']}, "
        f"reject_third={settings['reject_third_road_contact']}, "
        f"building_proxy={settings['enable_building_proxy']}, "
        f"pav_contact_r={settings['pavement_contact_radius_px']}, "
        f"spur={settings['prune_spur_max_px']}, "
        f"stub_area={settings['access_stub_area_px']}, "
        f"large_ee={settings['large_cc_ee_cap_px']}, "
        f"med_gap={settings['medium_gap_px']}, "
        f"width_reg={settings['enable_width_regularization']} "
        f"mode={settings['width_mode']} "
        f"half=[{settings['width_half_min_px']},{settings['width_half_max_px']}]"
    )
    return settings, source


def resolve_surface(cfg):
    surf = cfg['r2'].get('surface') or dict(
        enable_edge_despeckle=True, enable_micro_holes=True, enable_noise_crumbs=True,
        edge_kernel_px=3, edge_iterations=1, boundary_band_px=2)
    settings = dict(
        enable_edge_despeckle=bool(surf['enable_edge_despeckle']),
        enable_micro_holes=bool(surf['enable_micro_holes']),
        enable_noise_crumbs=bool(surf['enable_noise_crumbs']),
        edge_kernel_px=int(surf['edge_kernel_px']),
        edge_iterations=int(surf['edge_iterations']),
        boundary_band_px=int(surf['boundary_band_px']),
        pavement_crumb_area_px=int(surf.get('pavement_crumb_area_px', 10)),
        water_crumb_area_px=int(surf.get('water_crumb_area_px', 16)),
        pavement_hole_area_px=int(surf.get('pavement_hole_area_px', 16)),
        water_hole_area_px=int(surf.get('water_hole_area_px', 16)),
        building_crumb_area_px=int(surf.get('building_crumb_area_px', 16)),
        enable_internal_green_fill=bool(surf.get('enable_internal_green_fill', False)),
        pavement_internal_green_max_px=int(
            surf.get('pavement_internal_green_max_px', 0)),
        water_internal_green_max_px=int(
            surf.get('water_internal_green_max_px', 0)),
        enable_planar_boundary_smooth=bool(
            surf.get('enable_planar_boundary_smooth', False)),
        planar_smooth_kernel_px=int(surf.get('planar_smooth_kernel_px', 7)),
        planar_smooth_iterations=int(surf.get('planar_smooth_iterations', 2)),
        planar_smooth_band_px=int(surf.get('planar_smooth_band_px', 4)),
        pavement_protect_water=bool(surf.get('pavement_protect_water', False)),
    )
    source = (
        f"edge_despeckle={settings['enable_edge_despeckle']} "
        f"k={settings['edge_kernel_px']} iters={settings['edge_iterations']} "
        f"band={settings['boundary_band_px']} "
        f"crumbs={settings['enable_noise_crumbs']} "
        f"pav_crumb={settings['pavement_crumb_area_px']} "
        f"water_crumb={settings['water_crumb_area_px']} "
        f"bld_crumb={settings['building_crumb_area_px']} "
        f"green_fill={settings['enable_internal_green_fill']} "
        f"pav_green≤{settings['pavement_internal_green_max_px']} "
        f"water_green≤{settings['water_internal_green_max_px']} "
        f"planar_smooth={settings['enable_planar_boundary_smooth']} "
        f"pav_protect_water={settings['pavement_protect_water']}"
    )
    return settings, source


def resolve_building(cfg):
    bld = cfg['r2'].get('building') or {}
    settings = dict(
        enable_conditional_ortho=bool(bld.get('enable_conditional_ortho', True)),
        enable_hybrid_curve_smooth=bool(bld.get('enable_hybrid_curve_smooth', True)),
        max_hole_area_px=int(bld.get('max_hole_area_px', 400)),
        max_crumb_area_px=int(bld.get('max_crumb_area_px', 48)),
        edge_kernel_px=int(bld.get('edge_kernel_px', 5)),
        boundary_band_px=int(bld.get('boundary_band_px', 3)),
        straight_dev_px=float(bld.get('straight_dev_px', 2.0)),
        corner_angle_deg=float(bld.get('corner_angle_deg', 28.0)),
        curve_smooth_iters=int(bld.get('curve_smooth_iters', 4)),
        hybrid_band_px=int(bld.get('hybrid_band_px', 8)),
        curve_edge_frac=float(bld.get('curve_edge_frac', 0.35)),
        force_ortho_solidity=float(bld.get('force_ortho_solidity', 0.85)),
        complex_isoperimetric_q_max=float(bld.get('complex_isoperimetric_q_max', 0.35)),
        complex_solidity_max=float(bld.get('complex_solidity_max', 0.78)),
        complex_preclose_px=int(bld.get('complex_preclose_px', 5)),
    )
    source = (
        f"ortho={settings['enable_conditional_ortho']} "
        f"hybrid={settings['enable_hybrid_curve_smooth']} "
        f"straight_dev={settings['straight_dev_px']} "
        f"curve_iters={settings['curve_smooth_iters']} "
        f"band={settings['hybrid_band_px']} "
        f"complex_Q<{settings['complex_isoperimetric_q_max']}|"
        f"sol<{settings['complex_solidity_max']} "
        f"preclose={settings['complex_preclose_px']}"
    )
    return settings, source

def road_kwargs(cfg, road_function):
    """Reject an unknown/unused binding before generating any image.

    **kwargs alone is not proof that a parameter is implemented. A requested
    control must match a named keyword-capable parameter in the original API.
    """
    r2 = cfg['r2']
    kwargs = dict(tolerance=r2['road_color_tolerance'], max_width=r2['legacy_max_width_px'])
    signature = inspect.signature(road_function)
    applied, pending = {}, []
    reserved = {'orange_rgb','blocking_masks','tolerance','max_width',
                'vegetation_masks','guide_rgb','diagnostics','allow_cross_vegetation',
                'morph_close_px','connect_max_gap_px','max_links_per_endpoint'}
    for key, entry in r2['native_controls'].items():
        value, argument = entry['value'], entry['argument']
        if value is None:
            pending.append(key)
            continue
        pairs = {argument: value} if isinstance(argument,str) else {argument[k]:v for k,v in value.items()}
        for name, v in pairs.items():
            if name in reserved or name in kwargs:
                raise ValueError(f'Duplicate/reserved road binding: {name}')
            p = signature.parameters.get(name)
            if p is None or p.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD,inspect.Parameter.KEYWORD_ONLY):
                raise ValueError(f'Requested {key} cannot bind to explicit original parameter {name}; supply the original road implementation')
            kwargs[name] = v
        applied[key] = dict(value=value, argument=argument)
    return kwargs, dict(original_road_signature=str(signature), applied_native_controls=applied,
                        unconfigured_native_controls=pending, effective_road_kwargs=kwargs,
                        note='Unconfigured controls retain original implementation; no micro-tuning claimed')
