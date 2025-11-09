"""逆流吸收塔快速计算工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AbsorptionStage:
    stage_index: int
    x_in: float  # 液相进入该级（来自上一级）
    x_out: float  # 液相离开该级（流向下一级）
    y_out: float  # 气相离开该级（流向上一级）
    y_in: float  # 气相进入该级（来自下一级）


@dataclass
class AbsorptionResult:
    gas_flow: float
    liquid_flow: float
    y_top: float
    y_feed_required: float
    x_top: float
    x_bottom: float
    absorption_factor: float
    stage_count: int
    effective_stages: float
    meets_spec: bool
    stages: List[AbsorptionStage] = field(default_factory=list)


def minimum_solvent_flow(
    gas_flow: float,
    *,
    y_feed: float,
    y_target: float,
    x_solvent_in: float,
    equilibrium_slope: float,
) -> float:
    """最小溶剂量：顶部达到平衡极限时的 L。"""

    if not 0 <= x_solvent_in < 1:
        raise ValueError("溶剂入口组成需位于 [0, 1)")
    if not 0 < y_target < y_feed < 1:
        raise ValueError("需满足 y_target < y_feed 且二者位于 (0, 1)")
    if equilibrium_slope <= 0:
        raise ValueError("平衡线斜率 m 必须 > 0")

    x_star = y_feed / equilibrium_slope
    denom = x_star - x_solvent_in
    if denom <= 0:
        raise ValueError("溶剂过浓，无法达到吸收目的")
    return gas_flow * (y_feed - y_target) / denom


def simulate_absorption(
    *,
    gas_flow: float,
    liquid_flow: float,
    y_feed: float,
    y_target: float,
    x_solvent_in: float,
    equilibrium_slope: float,
    max_stages: int = 40,
    tol: float = 1e-6,
) -> AbsorptionResult:
    """基于等摩流 + 线性平衡的逆流吸收逐板计算。"""

    if gas_flow <= 0 or liquid_flow <= 0:
        raise ValueError("气体与溶剂流量必须为正")
    if not 0 <= x_solvent_in < 1:
        raise ValueError("溶剂入口组成需位于 [0, 1)")
    if not 0 < y_target < y_feed < 1:
        raise ValueError("需满足 y_target < y_feed < 1")
    if equilibrium_slope <= 0:
        raise ValueError("平衡线斜率 m 必须 > 0")

    l_over_v = liquid_flow / gas_flow
    absorption_factor = liquid_flow / (equilibrium_slope * gas_flow)

    l_min = minimum_solvent_flow(
        gas_flow,
        y_feed=y_feed,
        y_target=y_target,
        x_solvent_in=x_solvent_in,
        equilibrium_slope=equilibrium_slope,
    )
    if liquid_flow <= l_min * (1 - 1e-6):
        raise ValueError(
            f"给定溶剂流量 {liquid_flow:.3f} kmol/h 低于最小需求 {l_min:.3f} kmol/h，"
            "请提升溶剂量或放宽产品指标"
        )

    stages: List[AbsorptionStage] = []
    y_leaving = y_target  # 顶部气相
    x_entering = x_solvent_in

    while y_leaving < y_feed - tol and len(stages) < max_stages:
        stage_idx = len(stages) + 1
        x_leaving = y_leaving / equilibrium_slope
        y_entering = y_leaving + l_over_v * (x_leaving - x_entering)
        y_entering = max(0.0, min(1.0, y_entering))

        stages.append(
            AbsorptionStage(
                stage_index=stage_idx,
                x_in=x_entering,
                x_out=x_leaving,
                y_out=y_leaving,
                y_in=y_entering,
            )
        )

        x_entering = x_leaving
        y_leaving = y_entering

    meets_spec = y_leaving >= y_feed - tol
    stage_count = len(stages)
    effective_stages = float(stage_count)

    if meets_spec and stages:
        last = stages[-1]
        delta = last.y_in - last.y_out
        if abs(delta) > tol:
            fraction = (y_feed - last.y_out) / delta
            fraction = max(0.0, min(1.0, fraction))
            effective_stages = max(0.0, stage_count - 1 + fraction)

    if not meets_spec:
        raise RuntimeError(
            "在指定最大级数内未达到进料组成，请增加级数或调整溶剂条件"
        )

    x_bottom = stages[-1].x_out if stages else x_solvent_in

    return AbsorptionResult(
        gas_flow=gas_flow,
        liquid_flow=liquid_flow,
        y_top=y_target,
        y_feed_required=y_leaving,
        x_top=x_solvent_in,
        x_bottom=x_bottom,
        absorption_factor=absorption_factor,
        stage_count=stage_count,
        effective_stages=effective_stages,
        meets_spec=meets_spec,
        stages=stages,
    )
