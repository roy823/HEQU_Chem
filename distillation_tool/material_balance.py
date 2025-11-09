"""Material and extraction balance utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BinaryBalanceResult:
    feed_flow: float
    feed_x: float
    distillate_flow: float
    distillate_x: float
    bottoms_flow: float
    bottoms_x: float
    reflux_ratio: float
    recovery: Optional[float] = None

    def summary_table(self) -> Dict[str, Dict[str, float]]:
        """Return a dict that mirrors the tabular presentation used in the UI."""

        return {
            "进料 F": {"总流量": self.feed_flow, "轻键摩尔分率": self.feed_x},
            "塔顶 D": {"总流量": self.distillate_flow, "轻键摩尔分率": self.distillate_x},
            "塔釜 W": {"总流量": self.bottoms_flow, "轻键摩尔分率": self.bottoms_x},
        }


@dataclass
class ExtractionStageDetail:
    stage_index: int
    raffinate_x: float
    extract_y: float


@dataclass
class ExtractionBalanceResult:
    feed_flow: float
    feed_solute_frac: float
    solvent_flow: float
    distribution_coefficient: float
    recovery: float
    extract_flow: float
    extract_solute_frac: float
    raffinate_flow: float
    raffinate_solute_frac: float
    mode: str = "single"
    stage_count: Optional[int] = None
    stage_details: List[ExtractionStageDetail] = field(default_factory=list)
    solvent_solute_frac: float = 0.0

    def streams(self) -> Dict[str, Dict[str, float]]:
        return {
            "进料 F": {"总流量": self.feed_flow, "溶质摩尔分率": self.feed_solute_frac},
            "萃取剂 S": {"总流量": self.solvent_flow, "溶质摩尔分率": self.solvent_solute_frac},
            "萃取相 E": {"总流量": self.extract_flow, "溶质摩尔分率": self.extract_solute_frac},
            "萃余相 R": {"总流量": self.raffinate_flow, "溶质摩尔分率": self.raffinate_solute_frac},
        }


def solve_binary_balance(
    feed_flow: float,
    feed_x: float,
    *,
    distillate_x: float,
    bottoms_x: Optional[float] = None,
    reflux_ratio: float,
    recovery: Optional[float] = None,
) -> BinaryBalanceResult:
    """Perform overall material balance for a binary distillation column."""

    if feed_flow <= 0:
        raise ValueError("进料流量必须大于 0")
    if not 0 <= feed_x <= 1:
        raise ValueError("feed_x 必须位于 [0, 1]")
    if not 0 < distillate_x <= 1:
        raise ValueError("x_D 必须位于 (0, 1]")

    feed_light = feed_flow * feed_x

    if recovery is not None:
        if not 0 < recovery <= 1:
            raise ValueError("回收率必须位于 (0, 1]")
        distillate_flow = recovery * feed_light / distillate_x
    else:
        if bottoms_x is None:
            raise ValueError("未指定回收率时必须提供塔釜组成 bottoms_x")
        if not 0 <= bottoms_x < distillate_x:
            raise ValueError("需满足 0 <= x_W < x_D")
        distillate_flow = feed_flow * (feed_x - bottoms_x) / (distillate_x - bottoms_x)

    bottoms_flow = feed_flow - distillate_flow
    if bottoms_flow <= 0:
        raise ValueError("塔釜流量为非正值，请检查输入参数")

    bottoms_light = feed_light - distillate_flow * distillate_x
    calc_bottoms_x = bottoms_light / bottoms_flow

    if bottoms_x is not None and abs(calc_bottoms_x - bottoms_x) > 1e-4 and recovery is None:
        raise ValueError(
            "给定的塔釜组成与整体物料衡算不一致，"
            f"计算得到 {calc_bottoms_x:.4f}, 输入 {bottoms_x:.4f}"
        )

    return BinaryBalanceResult(
        feed_flow=feed_flow,
        feed_x=feed_x,
        distillate_flow=distillate_flow,
        distillate_x=distillate_x,
        bottoms_flow=bottoms_flow,
        bottoms_x=calc_bottoms_x,
        reflux_ratio=reflux_ratio,
        recovery=recovery,
    )


def solve_single_stage_extraction(
    feed_flow: float,
    feed_solute_frac: float,
    *,
    distribution_coefficient: float,
    recovery: float,
    solvent_solute_frac: float = 0.0,
) -> ExtractionBalanceResult:
    """Single-stage extraction based on a linear distribution relation Y = K X."""

    if not 0 <= feed_solute_frac <= 1:
        raise ValueError("溶质摩尔分率需位于 [0, 1]")
    if distribution_coefficient <= 0:
        raise ValueError("分配系数 K 必须大于 0")
    if not 0 < recovery < 1:
        raise ValueError("回收率需位于 (0, 1)")
    if not 0 <= solvent_solute_frac <= 1:
        raise ValueError("溶剂中溶质分率需位于 [0, 1]")

    solvent_flow = recovery * feed_flow / (distribution_coefficient * (1 - recovery))

    solute_total = feed_flow * feed_solute_frac
    raffinate_solute_frac = solute_total / (feed_flow + solvent_flow * distribution_coefficient)
    extract_solute_frac = distribution_coefficient * raffinate_solute_frac

    stage_details = [
        ExtractionStageDetail(stage_index=1, raffinate_x=raffinate_solute_frac, extract_y=extract_solute_frac)
    ]

    return ExtractionBalanceResult(
        feed_flow=feed_flow,
        feed_solute_frac=feed_solute_frac,
        solvent_flow=solvent_flow,
        solvent_solute_frac=solvent_solute_frac,
        distribution_coefficient=distribution_coefficient,
        recovery=recovery,
        extract_flow=solvent_flow,
        extract_solute_frac=extract_solute_frac,
        raffinate_flow=feed_flow,
        raffinate_solute_frac=raffinate_solute_frac,
        mode="single",
        stage_count=1,
        stage_details=stage_details,
    )


def solve_countercurrent_extraction(
    feed_flow: float,
    feed_solute_frac: float,
    *,
    solvent_flow: float,
    distribution_coefficient: float,
    stage_count: Optional[int] = None,
    target_recovery: Optional[float] = None,
    solvent_solute_frac: float = 0.0,
    max_stages: int = 12,
) -> ExtractionBalanceResult:
    """Countercurrent extraction with linear equilibrium relation Y = K X."""

    if feed_flow <= 0 or solvent_flow <= 0:
        raise ValueError("进料与溶剂流量必须大于 0")
    if not 0 <= feed_solute_frac <= 1:
        raise ValueError("进料溶质分率需位于 [0, 1]")
    if not 0 <= solvent_solute_frac <= 1:
        raise ValueError("溶剂中溶质分率需位于 [0, 1]")
    if distribution_coefficient <= 0:
        raise ValueError("分配系数需大于 0")
    if stage_count is None and target_recovery is None:
        raise ValueError("需提供 stage_count 或 target_recovery")
    if stage_count is not None and stage_count <= 0:
        raise ValueError("stage_count 必须为正整数")
    if target_recovery is not None and not 0 < target_recovery < 1:
        raise ValueError("target_recovery 需位于 (0, 1)")
    if max_stages <= 0:
        raise ValueError("max_stages 必须为正整数")

    def _solve_for_stages(n_stages: int) -> ExtractionBalanceResult:
        xs = _countercurrent_profile(
            n_stages,
            raffinate_flow=feed_flow,
            solvent_flow=solvent_flow,
            distribution_coefficient=distribution_coefficient,
            feed_solute_frac=feed_solute_frac,
            solvent_solute_frac=solvent_solute_frac,
        )
        stage_details = [
            ExtractionStageDetail(stage_index=idx + 1, raffinate_x=x, extract_y=distribution_coefficient * x)
            for idx, x in enumerate(xs)
        ]
        raffinate_solute_frac = xs[0]
        extract_solute_frac = distribution_coefficient * xs[-1]
        recovery_val = 0.0 if feed_solute_frac == 0 else 1 - raffinate_solute_frac / feed_solute_frac
        return ExtractionBalanceResult(
            feed_flow=feed_flow,
            feed_solute_frac=feed_solute_frac,
            solvent_flow=solvent_flow,
            solvent_solute_frac=solvent_solute_frac,
            distribution_coefficient=distribution_coefficient,
            recovery=recovery_val,
            extract_flow=solvent_flow,
            extract_solute_frac=extract_solute_frac,
            raffinate_flow=feed_flow,
            raffinate_solute_frac=raffinate_solute_frac,
            mode="countercurrent",
            stage_count=n_stages,
            stage_details=stage_details,
        )

    if stage_count is None:
        for n in range(1, max_stages + 1):
            result = _solve_for_stages(n)
            if result.recovery >= (target_recovery or 0):
                return result
        raise RuntimeError(
            f"在 {max_stages} 级内未达到指定的回收率 {target_recovery:.3f}"
            if target_recovery is not None
            else "未找到满足条件的级数"
        )

    return _solve_for_stages(stage_count)


def _countercurrent_profile(
    stage_count: int,
    *,
    raffinate_flow: float,
    solvent_flow: float,
    distribution_coefficient: float,
    feed_solute_frac: float,
    solvent_solute_frac: float,
) -> List[float]:
    """Solve for the raffinate composition leaving each stage (x_1 ... x_N)."""

    L = raffinate_flow
    V = solvent_flow
    K = distribution_coefficient

    main = L + V * K
    upper = -L
    lower = -V * K

    mat = np.zeros((stage_count, stage_count))
    rhs = np.zeros(stage_count)

    for i in range(stage_count):
        mat[i, i] = main
        if i < stage_count - 1:
            mat[i, i + 1] = upper
        if i > 0:
            mat[i, i - 1] = lower

    rhs[0] = V * solvent_solute_frac
    rhs[-1] = L * feed_solute_frac

    xs = np.linalg.solve(mat, rhs)
    xs_list = xs.tolist()

    if any(x < -1e-8 for x in xs_list):
        raise RuntimeError("求解得到的萃余组成出现负值，请检查输入参数")

    return xs_list
