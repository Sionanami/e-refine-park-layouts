# -*- coding: utf-8 -*-
"""阶段③ 建筑（building）

顺序位置：铺装、水体之后。产出 approved footprint，供道路/植被硬保护。
与表面相同思路：闭孔 → 去碎屑 → 边缘去毛刺。
与表面不同：无 B-spline；孔洞阈值更大 + 环向占比；可选条件正交 + 仅豁免分量的曲线 hybrid。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..r2.building_repair import repair_building_mask
from ..r2.coverage import class_masks
from .thresholds import BuildingThresholds

MAGENTA = (255, 0, 255)
LAWN = (0, 255, 0)


def run_building_stage(
    rgb: np.ndarray,
    *,
    fixed: np.ndarray,
    r1_rgb: Optional[np.ndarray] = None,
    cfg: Optional[BuildingThresholds] = None,
    # 生产建议：ortho+hybrid 开（L3_debug / 175843 逻辑）
    enable_ortho: Optional[bool] = None,
    enable_hybrid: Optional[bool] = None,
    ortho_debug_dir: Optional[str] = None,
    ortho_debug_tag: str = 'building_ortho',
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """执行建筑阶段。

    Returns
    -------
    rgb_out : 重着色后的 RGB
    approved_mask : 批准建筑掩膜（后续 protect 用）
    diagnostics : 日志
    """
    cfg = cfg or BuildingThresholds()
    before = np.asarray(rgb, dtype=np.uint8)
    m = class_masks(before, fixed)
    seed = m['building'].copy()
    if r1_rgb is not None:
        r1b = np.all(np.asarray(r1_rgb) == MAGENTA, axis=2) & ~fixed
        seed |= r1b

    use_ortho = cfg.enable_conditional_ortho if enable_ortho is None else bool(enable_ortho)
    use_hybrid = (
        cfg.enable_hybrid_curve_smooth if enable_hybrid is None else bool(enable_hybrid))

    # ------------------------------------------------------------------
    # 已验证后回退、不再默认启用的建筑实验（保留说明，勿在此重新打开）：
    # 1) concave_prefer_ortho：凹陷一律 ORTHO → 000041 全员拉直过猛
    # 2) AREA_CHANGE_TOL=0.15 + poly|erode(core)：救 FALLBACK，视觉不如 175843
    # 3) adaptive DP 顶点回退：同上批次
    # 当前 ortho = 175843：sol>0.85 强制 ORTHO；低 sol 凹形可 CURVE_EXEMPT
    #    + 旋转正交 + 两阶段吸附；FALLBACK=原始轮廓（禁止 MABR）
    # ------------------------------------------------------------------

    new_b, blog = repair_building_mask(
        seed, fixed,
        enable_holes=True,
        enable_crumbs=True,
        enable_edge=True,
        enable_conditional_ortho=use_ortho,
        enable_hybrid_curve_smooth=use_hybrid,
        max_hole_area_px=int(cfg.max_hole_area_px),
        max_crumb_area_px=int(cfg.max_crumb_area_px),
        edge_kernel_px=int(cfg.edge_kernel_px),
        boundary_band_px=int(cfg.boundary_band_px),
        angle_tol_deg=float(cfg.angle_tol_deg),
        near_rect_ratio=float(cfg.near_rect_ratio),
        area_change_tol=float(cfg.area_change_tol),
        min_building_area=int(cfg.min_building_area),
        min_edge_len=float(cfg.min_edge_len),
        max_aspect_ratio=float(cfg.max_aspect_ratio),
        dp_epsilon_frac=float(cfg.dp_epsilon_frac),
        multi_axis_spread_deg=float(cfg.multi_axis_spread_deg),
        curve_edge_frac=float(cfg.curve_edge_frac),
        near_rect_max_area_px=float(cfg.near_rect_max_area_px),
        concave_solidity_min=float(cfg.concave_solidity_min),
        concave_defect_depth_frac=float(cfg.concave_defect_depth_frac),
        force_ortho_solidity=float(cfg.force_ortho_solidity),
        enable_rotate_ortho=bool(cfg.enable_rotate_ortho),
        stage2_angle_deg=float(cfg.stage2_angle_deg),
        stage2_max_edge_px=float(cfg.stage2_max_edge_px),
        complex_isoperimetric_q_max=float(cfg.complex_isoperimetric_q_max),
        complex_solidity_max=float(cfg.complex_solidity_max),
        complex_preclose_px=int(cfg.complex_preclose_px),
        straight_dev_px=float(cfg.straight_dev_px),
        corner_angle_deg=float(cfg.corner_angle_deg),
        curve_smooth_iters=int(cfg.curve_smooth_iters),
        hybrid_band_px=int(cfg.hybrid_band_px),
        hybrid_pre_dp_eps_px=float(cfg.hybrid_pre_dp_eps_px),
        hybrid_curve_dp_eps_px=float(cfg.hybrid_curve_dp_eps_px),
        hybrid_curve_ma_win=int(cfg.hybrid_curve_ma_win),
        enable_ortho_debug_viz=bool(ortho_debug_dir),
        ortho_debug_dir=ortho_debug_dir,
        ortho_debug_tag=ortho_debug_tag,
        fallback_mode='original',
    )
    new_b = new_b & ~fixed

    # Contour-interior fill after smooth (recover holes; top-priority footprint).
    from ..r2.building_repair import fill_enclosed_building_holes
    filled_b, hole_info = fill_enclosed_building_holes(
        new_b, fixed, max_hole_area_px=int(cfg.max_hole_area_px))
    filled_b = filled_b & ~fixed

    out = before.copy()
    add = filled_b & ~m['building']
    rem = m['building'] & ~filled_b & ~fixed
    out[add] = np.asarray(MAGENTA, dtype=np.uint8)
    # Shrink releases → lawn only (never promote to road via neighbour majority).
    if rem.any():
        out[rem] = np.asarray(LAWN, dtype=np.uint8)
    # Building sits on top: paint full approved footprint last.
    out[filled_b] = np.asarray(MAGENTA, dtype=np.uint8)

    blog = dict(blog)
    blog['stage'] = 'building'
    blog['enable_conditional_ortho'] = use_ortho
    blog['enable_hybrid_curve_smooth'] = use_hybrid
    blog['post_contour_hole_fill'] = hole_info
    blog['shrink_to_lawn_px'] = int(rem.sum())
    blog['expand_building_px'] = int(add.sum())
    return out, filled_b, blog