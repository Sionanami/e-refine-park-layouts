# -*- coding: utf-8 -*-
"""三阶段统一阈值（L3_debug 已验证配置快照）。

修改阈值请只改本文件；stage_* 模块从这里读取。
标注「实验/默认关闭」的项在样本上已证明增益有限或为负，流水线默认不启用。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


# ---------------------------------------------------------------------------
# 共用：表面类（铺装 / 水体）正则化共用的形态学参数
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SharedSurfaceMorph:
    """铺装与水体在 regularize_class_mask 中的相同部分。"""
    edge_kernel_px: int = 5
    edge_iterations: int = 2
    boundary_band_px: int = 3
    # 微孔：绝对面积上限（px²）
    micro_hole_area_px: int = 16
    # --- 以下默认关闭：样本上闭孔几乎全 skip_protected / filled=0 ---
    enable_thin_linear_pores: bool = False  # L3_debug 曾开；生产 L3 关
    thin_pore_max_width_px: float = 3.0
    thin_pore_min_aspect: float = 2.5
    enable_enclosed_area_holes: bool = False  # 实验：AND(500px², 20%)；样例有效填充≈0
    enclosed_hole_max_area_px2: int = 500
    enclosed_hole_max_area_ratio: float = 0.20


@dataclass(frozen=True)
class PavementThresholds:
    """铺装专有。"""
    morph: SharedSurfaceMorph = field(default_factory=SharedSurfaceMorph)
    crumb_area_px: int = 10
    expand_band_px: int = 1  # 仅铺装：边缘可略吃 lawn（植物硬保护）
    thin_pore_max_area_px: int = 160  # 若开启细缝填孔
    # 平滑写保护 = fixed|building|road|plant|lawn（与水体对齐植物硬保护）


@dataclass(frozen=True)
class WaterThresholds:
    """水体专有。"""
    morph: SharedSurfaceMorph = field(default_factory=SharedSurfaceMorph)
    crumb_area_px: int = 16  # 比铺装略松
    thin_pore_max_area_px: int = 128
    # 平滑写保护 = fixed|building|road|pavement|plant|lawn
    # 无 expand_band；不可盖铺装


@dataclass(frozen=True)
class BuildingThresholds:
    """建筑专有（local_repair + 条件正交 175843 逻辑）。"""
    # --- 与表面类似的局部修补 ---
    max_hole_area_px: int = 400
    hole_ring_frac_min: float = 0.85  # 建筑孔洞：环向建筑占比
    max_crumb_area_px: int = 16
    edge_kernel_px: int = 5
    boundary_band_px: int = 3
    # ortho 开启时边缘改为轻量：kernel≤3, iters=1, band≤2

    # --- 条件正交（默认开于 L3_debug；生产 L3 关）---
    enable_conditional_ortho: bool = True
    enable_hybrid_curve_smooth: bool = True  # 仅 CURVE_EXEMPT
    angle_tol_deg: float = 12.0
    near_rect_ratio: float = 0.85
    near_rect_max_area_px: float = 500  # 大于此禁止 MABR NEAR_RECT
    area_change_tol: float = 0.12
    min_building_area: int = 40
    min_edge_len: float = 6.0
    max_aspect_ratio: float = 8.0
    dp_epsilon_frac: float = 0.03
    multi_axis_spread_deg: float = 25.0
    curve_edge_frac: float = 0.35
    force_ortho_solidity: float = 0.85  # sol>此值禁止 CURVE_EXEMPT
    concave_solidity_min: float = 0.92
    concave_defect_depth_frac: float = 0.08
    enable_rotate_ortho: bool = True
    stage2_angle_deg: float = 30.0
    stage2_max_edge_px: float = 10.0
    # 破碎/扭曲轮廓：Q=4πA/P² 低于此 或 solidity 低于此 → CURVE_EXEMPT+hybrid
    # （复杂度↑对应 Q↓、solidity↓；勿用「周长/面积低」）
    complex_isoperimetric_q_max: float = 0.35
    complex_solidity_max: float = 0.78
    complex_preclose_px: int = 5  # 分支前椭圆闭运算，弥合碎斑；<3 关闭

    # --- 曲线豁免 hybrid（000125 强化后）---
    straight_dev_px: float = 2.0
    corner_angle_deg: float = 28.0
    curve_smooth_iters: int = 4
    hybrid_band_px: int = 8
    hybrid_pre_dp_eps_px: float = 1.5
    hybrid_curve_dp_eps_px: float = 1.5
    hybrid_curve_ma_win: int = 7

    # --- 已回退 / 不要再开（保留注释备忘）---
    # concave_prefer_ortho: 凹陷一律 ORTHO → 000041 过猛，已回退
    # area_change_tol=0.15 + poly|core: 为救 FALLBACK，视觉不如 175843
    # adaptive DP backoff: 同上批次，已回退


@dataclass(frozen=True)
class StageOrder:
    """三阶段相对顺序（嵌入完整 R2 时的位置）。"""
    # 完整 R2：… → ①铺装 → ②水体 → ③建筑 → 道路 → 植被 → palette/protect
    pavement: int = 1
    water: int = 2
    building: int = 3
    note: str = (
        "铺装先于水体：铺装可盖水体、水体不可盖铺装。"
        "建筑在表面之后：用修补后 footprint 作为后续道路/植被硬保护。"
    )


def l3_debug_bundle() -> Dict[str, Any]:
    """当前整理包默认 = L3_debug 有效配置，但表面闭孔/细缝默认关闭。"""
    shared = SharedSurfaceMorph(
        enable_thin_linear_pores=False,
        enable_enclosed_area_holes=False,
    )
    return dict(
        order=asdict(StageOrder()),
        pavement=asdict(PavementThresholds(morph=shared)),
        water=asdict(WaterThresholds(morph=shared)),
        building=asdict(BuildingThresholds()),
    )


def l3_debug_aggressive_surface() -> Dict[str, Any]:
    """若需复现曾开启的细缝+闭孔实验（样例净收益≈0）。"""
    shared = SharedSurfaceMorph(
        enable_thin_linear_pores=True,
        enable_enclosed_area_holes=True,
    )
    return dict(
        order=asdict(StageOrder()),
        pavement=asdict(PavementThresholds(morph=shared)),
        water=asdict(WaterThresholds(morph=shared)),
        building=asdict(BuildingThresholds()),
    )
