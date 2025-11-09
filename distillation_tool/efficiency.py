"""塔效率与实际塔板数计算。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class EfficiencyResult:
    theoretical_stages: float
    overall_efficiency: float
    actual_trays: float


def murphree_y(y_star: float, y_next: float, efficiency: float) -> float:
    """根据气相默弗里效率计算出口汽相组成。"""

    if not 0 <= efficiency <= 1:
        raise ValueError("效率需在 [0, 1]")
    return y_next + efficiency * (y_star - y_next)


def overall_stage_count(n_theoretical: float, efficiency: float) -> EfficiencyResult:
    if not 0 < efficiency <= 1:
        raise ValueError("整体效率需在 (0, 1]")
    actual = n_theoretical / efficiency
    return EfficiencyResult(
        theoretical_stages=n_theoretical,
        overall_efficiency=efficiency,
        actual_trays=actual,
    )


def tray_count_from_local_efficiencies(equilibrium_y: Sequence[float], actual_y: Sequence[float]) -> float:
    """根据逐层默弗里效率回算理论级数。

    equilibrium_y: 每级与液相平衡的汽相比值
    actual_y: 实际气相出口组成
    """

    if len(equilibrium_y) != len(actual_y):
        raise ValueError("平衡值与实际值长度需一致")
    efficiencies: List[float] = []
    for y_star, y_act_prev, y_act in zip(equilibrium_y[:-1], actual_y[:-1], actual_y[1:]):
        denom = y_star - y_act_prev
        if math.isclose(denom, 0.0, rel_tol=1e-9, abs_tol=1e-12):
            continue
        efficiencies.append((y_act - y_act_prev) / denom)
    if not efficiencies:
        raise ValueError("无法计算效率，请检查输入")
    avg_eff = sum(efficiencies) / len(efficiencies)
    return len(actual_y) * avg_eff

