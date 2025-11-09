"""闪蒸与单级平衡计算模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class BinaryFlashResult:
    feed_flow: float
    feed_fraction: float
    vapor_flow: float
    vapor_fraction: float
    liquid_flow: float
    liquid_fraction: float
    vapor_ratio: float
    equilibrium_pair: Tuple[float, float]


def binary_flash_lever_rule(
    feed_flow: float,
    feed_fraction: float,
    *,
    x_eq: float,
    y_eq: float,
) -> BinaryFlashResult:
    """基于杠杆法则求解单级等温闪蒸。

    适用于给定平衡线上的 (x_eq, y_eq) 点，例如根据 T-x-y 表查得。
    """

    if not 0.0 <= x_eq < y_eq <= 1.0:
        raise ValueError("需满足 0 <= x_eq < y_eq <= 1")
    if not 0.0 <= feed_fraction <= 1.0:
        raise ValueError("进料摩尔分率需在 [0, 1]")

    if not (x_eq <= feed_fraction <= y_eq):
        raise ValueError("进料组成需位于平衡线两端之间，即 x_eq <= z <= y_eq")

    beta = (feed_fraction - x_eq) / (y_eq - x_eq)
    vapor_flow = feed_flow * beta
    liquid_flow = feed_flow - vapor_flow

    return BinaryFlashResult(
        feed_flow=feed_flow,
        feed_fraction=feed_fraction,
        vapor_flow=vapor_flow,
        vapor_fraction=y_eq,
        liquid_flow=liquid_flow,
        liquid_fraction=x_eq,
        vapor_ratio=beta,
        equilibrium_pair=(x_eq, y_eq),
    )


@dataclass
class Stream:
    phase: str  # "V" | "L"
    flow: float
    fraction: float


def combine_streams(streams: Iterable[Stream]) -> Tuple[float, float]:
    total_flow = 0.0
    total_component = 0.0
    for stream in streams:
        if stream.flow < 0:
            raise ValueError("流量需为非负数")
        if not 0.0 <= stream.fraction <= 1.0:
            raise ValueError("摩尔分率需在 [0, 1]")
        total_flow += stream.flow
        total_component += stream.flow * stream.fraction
    if total_flow <= 0:
        raise ValueError("总流量需大于 0")
    z = total_component / total_flow
    return total_flow, z


def flash_with_mixed_feeds(streams: Iterable[Stream], *, x_eq: float, y_eq: float) -> BinaryFlashResult:
    total_flow, z = combine_streams(streams)
    return binary_flash_lever_rule(total_flow, z, x_eq=x_eq, y_eq=y_eq)


def rachford_rice(z: Sequence[float], k: Sequence[float], *, tol: float = 1e-8, max_iter: int = 200) -> float:
    """求解 Rachford-Rice 方程，返回汽化分率 β。"""

    z_arr = np.asarray(z, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    if np.any(z_arr < 0) or not np.isclose(z_arr.sum(), 1.0, atol=1e-6):
        raise ValueError("组成需非负并归一化")
    if np.any(k_arr <= 0):
        raise ValueError("平衡常数需大于 0")

    def f(beta: float) -> float:
        return float(np.sum(z_arr * (k_arr - 1) / (1 + beta * (k_arr - 1))))

    beta_low, beta_high = 0.0, 1.0
    f_low, f_high = f(beta_low), f(beta_high)

    if f_low * f_high > 0:
        raise ValueError("在 [0, 1] 范围内未找到满足质量分率的解，请检查 K 值")

    for _ in range(max_iter):
        beta_mid = 0.5 * (beta_low + beta_high)
        f_mid = f(beta_mid)
        if abs(f_mid) < tol:
            return beta_mid
        if f_low * f_mid <= 0:
            beta_high, f_high = beta_mid, f_mid
        else:
            beta_low, f_low = beta_mid, f_mid

    raise RuntimeError("Rachford-Rice 求解未收敛")


