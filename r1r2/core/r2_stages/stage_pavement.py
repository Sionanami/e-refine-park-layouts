# -*- coding: utf-8 -*-
"""阶段① 铺装（pavement）

顺序位置：表面最先。之后才是水体、建筑。
支持两段式调用（refine 在开启内部绿地填充时使用）：
  1) smooth + 内部绿地填充（不去噪）
  2) regularize 去噪（碎屑/微孔/边缘；不再填充）
与水体：默认可盖水体；开启 protect_water_from_pavement 后不可盖水。
1px lawn 扩展带；碎屑阈值更严；植物硬保护；道路始终硬保护。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from ..r2.coverage import (
    apply_color_write_clipped,
    class_masks,
    protected_from_pavement,
)
from ..r2.surface import fill_internal_green_holes, regularize_surface_rgb
from .thresholds import PavementThresholds, SharedSurfaceMorph

YELLOW = (255, 255, 0)


def run_pavement_stage(
    rgb: np.ndarray,
    *,
    fixed: np.ndarray,
    smooth_fn,
    smooth_source,
    tolerance: float,
    cfg: Optional[PavementThresholds] = None,
    enable_edge: bool = True,
    enable_holes: bool = True,
    enable_crumbs: bool = True,
    enable_internal_green_fill: bool = False,
    internal_green_max_px: int = 0,
    do_smooth: bool = True,
    do_regularize: bool = True,
    protect_water_from_pavement: bool = False,
    # --- 冗余实验开关（默认关；打开仅用于对照）---
    enable_thin_pores_experimental: bool = False,
    enable_enclosed_holes_experimental: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """执行铺装阶段，返回 (rgb_out, diagnostics)。"""
    cfg = cfg or PavementThresholds()
    morph: SharedSurfaceMorph = cfg.morph
    before = np.asarray(rgb, dtype=np.uint8)
    slog: Dict[str, Any] = {
        'stage': 'pavement',
        'protect_water_from_pavement': bool(protect_water_from_pavement),
    }
    work = before
    _pw = bool(protect_water_from_pavement)

    if do_smooth:
        proposed = np.asarray(
            smooth_fn(
                Image.fromarray(before), smooth_source,
                color_rgb=YELLOW, tolerance=tolerance,
            ).convert('RGB'),
            dtype=np.uint8,
        )
        m = class_masks(before, fixed)
        plant_lawn = m['plant'] | m['lawn']
        prot_smooth = protected_from_pavement(m, protect_water=_pw) | plant_lawn
        work = np.asarray(
            apply_color_write_clipped(before, proposed, YELLOW, prot_smooth),
            dtype=np.uint8,
        )
        wanted = np.all(proposed == YELLOW, axis=2) & plant_lawn & ~m['pavement']
        slog['smooth_plant_lawn_blocked_px'] = int(wanted.sum())
    else:
        proposed = None

    ginfo = None
    if enable_internal_green_fill and int(internal_green_max_px) > 0:
        m_fill = class_masks(work, fixed)
        hard = m_fill['fixed'] | m_fill['building'] | m_fill['road']
        if _pw:
            hard = hard | m_fill['water']
        green = m_fill['lawn'] | m_fill['plant']
        work, ginfo = fill_internal_green_holes(
            work, m_fill['pavement'], green, YELLOW,
            max_area_px=int(internal_green_max_px),
            hard_protected=hard,
        )
        slog['internal_green_fill'] = ginfo

    if do_regularize:
        m2 = class_masks(work, fixed)
        plant0, lawn0, pav0 = m2['plant'], m2['lawn'], m2['pavement']
        prot = protected_from_pavement(m2, protect_water=_pw) | plant0 | lawn0
        band = int(cfg.expand_band_px)
        if band > 0:
            import cv2
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
            allow = (cv2.dilate(pav0.astype(np.uint8), k) > 0) & ~pav0
            if _pw:
                allow = allow & ~m2['water']
            prot = protected_from_pavement(m2, protect_water=_pw) | plant0 | (lawn0 & ~allow)

        use_thin = bool(
            enable_thin_pores_experimental or morph.enable_thin_linear_pores)
        use_enc = bool(
            enable_enclosed_holes_experimental or morph.enable_enclosed_area_holes)

        work, rlog = regularize_surface_rgb(
            work, m2['pavement'], YELLOW, prot,
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
        slog['expand_band_px'] = band
        slog['experimental_thin_pores'] = use_thin
        slog['experimental_enclosed_holes'] = use_enc

    slog['stage'] = 'pavement'
    slog['do_smooth'] = bool(do_smooth)
    slog['do_regularize'] = bool(do_regularize)
    slog['did_green_fill'] = ginfo is not None
    return work, slog
