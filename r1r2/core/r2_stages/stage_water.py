# -*- coding: utf-8 -*-
"""阶段② 水体（water）

顺序位置：紧接铺装之后、建筑之前。
支持两段式：smooth+内部绿地填充 → 稍后 regularize 去噪。
与铺装不同：保护集多 pavement；无 expand_band；碎屑/细缝阈值略不同；最软结构类。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from ..r2.coverage import (
    apply_color_write_clipped,
    class_masks,
    protected_from_water,
)
from ..r2.surface import fill_internal_green_holes, regularize_surface_rgb
from .thresholds import SharedSurfaceMorph, WaterThresholds

CYAN = (0, 255, 255)


def run_water_stage(
    rgb: np.ndarray,
    *,
    fixed: np.ndarray,
    smooth_fn,
    smooth_source,
    tolerance: float,
    cfg: Optional[WaterThresholds] = None,
    enable_edge: bool = True,
    enable_holes: bool = True,
    enable_crumbs: bool = True,
    enable_internal_green_fill: bool = False,
    internal_green_max_px: int = 0,
    do_smooth: bool = True,
    do_regularize: bool = True,
    enable_thin_pores_experimental: bool = False,
    enable_enclosed_holes_experimental: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """执行水体阶段，返回 (rgb_out, diagnostics)。"""
    cfg = cfg or WaterThresholds()
    morph: SharedSurfaceMorph = cfg.morph
    before = np.asarray(rgb, dtype=np.uint8)
    slog: Dict[str, Any] = {'stage': 'water'}
    work = before

    if do_smooth:
        proposed = np.asarray(
            smooth_fn(
                Image.fromarray(before), smooth_source,
                color_rgb=CYAN, tolerance=tolerance,
            ).convert('RGB'),
            dtype=np.uint8,
        )
        m = class_masks(before, fixed)
        plant_lawn = m['plant'] | m['lawn']
        prot_smooth = protected_from_water(m) | plant_lawn
        work = np.asarray(
            apply_color_write_clipped(before, proposed, CYAN, prot_smooth),
            dtype=np.uint8,
        )
        wanted = np.all(proposed == CYAN, axis=2) & plant_lawn & ~m['water']
        slog['smooth_plant_lawn_blocked_px'] = int(wanted.sum())

    ginfo = None
    if enable_internal_green_fill and int(internal_green_max_px) > 0:
        m_fill = class_masks(work, fixed)
        hard = (
            m_fill['fixed'] | m_fill['building'] | m_fill['road'] | m_fill['pavement']
        )
        green = m_fill['lawn'] | m_fill['plant']
        work, ginfo = fill_internal_green_holes(
            work, m_fill['water'], green, CYAN,
            max_area_px=int(internal_green_max_px),
            hard_protected=hard,
        )
        slog['internal_green_fill'] = ginfo

    if do_regularize:
        m2 = class_masks(work, fixed)
        prot = protected_from_water(m2) | m2['plant'] | m2['lawn']
        use_thin = bool(
            enable_thin_pores_experimental or morph.enable_thin_linear_pores)
        use_enc = bool(
            enable_enclosed_holes_experimental or morph.enable_enclosed_area_holes)

        work, rlog = regularize_surface_rgb(
            work, m2['water'], CYAN, prot,
            enable_edge=bool(enable_edge),
            enable_holes=bool(enable_holes),
            enable_crumbs=bool(enable_crumbs),
            enable_thin_pores=use_thin,
            enable_enclosed_area_holes=use_enc,
            edge_kernel_px=int(morph.edge_kernel_px),
            edge_iterations=int(morph.edge_iterations),
            boundary_band_px=int(morph.boundary_band_px),
            max_hole_area_px=int(morph.micro_hole_area_px),
            max_crumb_area_px=int(cfg.crumb_area_px),
            thin_pore_max_width_px=float(morph.thin_pore_max_width_px),
            thin_pore_max_area_px=int(cfg.thin_pore_max_area_px),
            thin_pore_min_aspect=float(morph.thin_pore_min_aspect),
            enclosed_hole_max_area_px2=int(morph.enclosed_hole_max_area_px2),
            enclosed_hole_max_area_ratio=float(morph.enclosed_hole_max_area_ratio),
        )
        slog.update(rlog)
        slog['experimental_thin_pores'] = use_thin
        slog['experimental_enclosed_holes'] = use_enc

    slog['stage'] = 'water'
    slog['do_smooth'] = bool(do_smooth)
    slog['do_regularize'] = bool(do_regularize)
    slog['did_green_fill'] = ginfo is not None
    return work, slog
