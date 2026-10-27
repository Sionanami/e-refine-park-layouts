# -*- coding: utf-8 -*-
"""三阶段流水线入口：铺装 → 水体 → 建筑。

不替代完整 ``r2_image``（仍含道路/植被/palette/A 恢复）。
本模块只整理并暴露「表面三类」的局部正则顺序与阈值。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .stage_building import run_building_stage
from .stage_pavement import run_pavement_stage
from .stage_water import run_water_stage
from .thresholds import (
    BuildingThresholds,
    PavementThresholds,
    WaterThresholds,
    l3_debug_bundle,
)


def run_pavement_water_building(
    rgb: np.ndarray,
    *,
    fixed: np.ndarray,
    smooth_fn,
    smooth_source,
    tolerance: float = 30.0,
    r1_rgb: Optional[np.ndarray] = None,
    pavement_cfg: Optional[PavementThresholds] = None,
    water_cfg: Optional[WaterThresholds] = None,
    building_cfg: Optional[BuildingThresholds] = None,
    # 表面实验开关（默认关）
    enable_thin_pores_experimental: bool = False,
    enable_enclosed_holes_experimental: bool = False,
    ortho_debug_dir: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """按固定顺序跑完三阶段。

    Order
    -----
    1. pavement  — 可覆盖水体；植物硬保护
    2. water     — 不可覆盖铺装；植物硬保护（防绿岛被淹）
    3. building  — 局部修补 +（可选）条件正交 / 曲线 hybrid
    """
    log: Dict[str, Any] = dict(
        order=['pavement', 'water', 'building'],
        thresholds_snapshot=l3_debug_bundle(),
    )
    cur = np.asarray(rgb, dtype=np.uint8)

    cur, plog = run_pavement_stage(
        cur, fixed=fixed, smooth_fn=smooth_fn, smooth_source=smooth_source,
        tolerance=tolerance, cfg=pavement_cfg,
        enable_thin_pores_experimental=enable_thin_pores_experimental,
        enable_enclosed_holes_experimental=enable_enclosed_holes_experimental,
    )
    log['pavement'] = plog

    cur, wlog = run_water_stage(
        cur, fixed=fixed, smooth_fn=smooth_fn, smooth_source=smooth_source,
        tolerance=tolerance, cfg=water_cfg,
        enable_thin_pores_experimental=enable_thin_pores_experimental,
        enable_enclosed_holes_experimental=enable_enclosed_holes_experimental,
    )
    log['water'] = wlog

    cur, approved, blog = run_building_stage(
        cur, fixed=fixed, r1_rgb=r1_rgb, cfg=building_cfg,
        ortho_debug_dir=ortho_debug_dir,
    )
    log['building'] = blog
    log['approved_building_mask'] = approved
    return cur, log
