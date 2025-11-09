"""理论级与操作线计算。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .material_balance import BinaryBalanceResult
from .vle import VLEModel


@dataclass
class OperatingLines:
    rectifying: Callable[[float], float]
    stripping: Callable[[float], float]
    q_line: Optional[Callable[[float], float]]
    q_vertical_x: Optional[float]
    feed_intersection: Tuple[float, float]
    rect_params: Tuple[float, float]
    strip_params: Tuple[float, float]


@dataclass
class StageProfile:
    stage_count: int
    feed_stage: Optional[int]
    x_sequence: List[float]
    y_sequence: List[float]
    steps: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    murphree_details: Optional[List["MurphreeDetail"]] = None


@dataclass
class MurphreeDetail:
    stage_index: int
    x_operating: float
    y_in: float
    y_equilibrium: float
    y_out: float


def fenske_minimum_stages(xd: float, xw: float, alpha: float) -> float:
    if not 0 < xd < 1 or not 0 < xw < 1:
        raise ValueError("组成需在 (0, 1) 内")
    if alpha <= 1:
        raise ValueError("α 必须大于 1 才能实现分离")
    return math.log((xd / (1 - xd)) * ((1 - xw) / xw), alpha)


def build_operating_lines(
    result: BinaryBalanceResult,
    *,
    q: float,
) -> OperatingLines:
    D = result.distillate_flow
    W = result.bottoms_flow
    F = result.feed_flow
    x_d = result.distillate_x
    x_w = result.bottoms_x

    R = result.reflux_ratio
    L = R * D
    V = (R + 1) * D
    Lp = L + q * F
    Vp = V - (1 - q) * F

    a_rect = R / (R + 1)
    b_rect = x_d / (R + 1)

    def rect(x: float) -> float:
        return a_rect * x + b_rect

    a_strip = Lp / Vp
    b_strip = -W * x_w / Vp

    def strip(x: float) -> float:
        return a_strip * x + b_strip

    if math.isclose(q, 1.0, rel_tol=1e-6):
        q_line = None
        q_vertical_x = result.feed_x
    else:
        slope = q / (q - 1)
        intercept = -result.feed_x / (q - 1)

        def q_line(x: float) -> float:
            return slope * x + intercept

        q_vertical_x = None

    # Feed 进料位置为精馏线与提馏线交点
    if abs(a_rect - a_strip) < 1e-9:
        x_inter = result.feed_x
    else:
        x_inter = (b_strip - b_rect) / (a_rect - a_strip)
    x_inter = max(0.0, min(1.0, x_inter))
    y_inter = rect(x_inter)

    return OperatingLines(
        rectifying=rect,
        stripping=strip,
        q_line=q_line,
        q_vertical_x=q_vertical_x,
        feed_intersection=(x_inter, y_inter),
        rect_params=(a_rect, b_rect),
        strip_params=(a_strip, b_strip),
    )


def minimum_reflux_ratio(
    result: BinaryBalanceResult,
    model: VLEModel,
    *,
    q: float,
) -> float:
    x_f = result.feed_x

    if math.isclose(q, 1.0, rel_tol=1e-6):
        x_q = x_f
        y_q = model.y(x_q)
    else:
        slope = q / (q - 1)
        intercept = -x_f / (q - 1)

        def func(x: float) -> float:
            return slope * x + intercept - model.y(x)

        grid = np.linspace(0, 1, 1001)
        values = func(grid)
        sign_change = np.where(np.sign(values[:-1]) * np.sign(values[1:]) <= 0)[0]
        if len(sign_change) == 0:
            raise ValueError("q 线与平衡线无交点，无法计算 Rmin")
        idx = sign_change[0]
        x_lower, x_upper = grid[idx], grid[idx + 1]
        for _ in range(50):
            x_mid = 0.5 * (x_lower + x_upper)
            if func(x_lower) * func(x_mid) <= 0:
                x_upper = x_mid
            else:
                x_lower = x_mid
        x_q = 0.5 * (x_lower + x_upper)
        y_q = slope * x_q + intercept

    x_d = result.distillate_x
    if math.isclose(y_q, x_q, rel_tol=1e-6):
        raise ValueError("进料点与平衡线重合，Rmin → ∞")
    return (x_d - y_q) / (y_q - x_q)


def gilliland_stages(n_min: float, r: float, r_min: float) -> float:
    if r <= r_min:
        raise ValueError("回流比必须大于最小回流比")

    y = (r - r_min) / (r + 1.0)

    def correlation(x: float) -> float:
        return 1 - math.exp((1 + 54.4 * x) / (11 + 117.2 * x) * (x - 1))

    def func(x: float) -> float:
        return correlation(x) - y

    lower, upper = 1e-6, 0.999
    f_lower = func(lower)
    f_upper = func(upper)
    if f_lower * f_upper > 0:
        grid = np.linspace(1e-6, 0.999, 500)
        values = [func(val) for val in grid]
        bracket_found = False
        for idx in range(len(grid) - 1):
            if values[idx] * values[idx + 1] <= 0:
                lower = grid[idx]
                upper = grid[idx + 1]
                f_lower = values[idx]
                f_upper = values[idx + 1]
                bracket_found = True
                break
        if not bracket_found:
            raise RuntimeError("Gilliland 相关式未找到可行根，请检查回流比/组分参数")

    for _ in range(60):
        mid = 0.5 * (lower + upper)
        f_mid = func(mid)
        if f_lower * f_mid <= 0:
            upper = mid
            f_upper = f_mid
        else:
            lower = mid
            f_lower = f_mid
    x = 0.5 * (lower + upper)
    return (x + n_min) / (1 - x)


def mccabe_thiele(
    model: VLEModel,
    result: BinaryBalanceResult,
    *,
    q: float,
    max_stages: int = 100,
    murphree_type: Optional[str] = None,
    murphree_efficiency: Optional[float] = None,
) -> StageProfile:
    lines = build_operating_lines(result, q=q)
    x_feed, _ = lines.feed_intersection
    tol = 1e-6

    use_murphree = murphree_type is not None and murphree_efficiency is not None
    if use_murphree:
        if not 0.0 < murphree_efficiency <= 1.0:
            raise ValueError("默弗里效率需在 (0, 1]")
        if murphree_type != "gas":
            raise NotImplementedError("目前仅支持气相默弗里效率的图解校正")

    if not use_murphree:
        x_curr = result.distillate_x
        y_curr = x_curr

        x_seq = [x_curr]
        y_seq = [y_curr]
        steps: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        stage_count = 0
        feed_stage: Optional[int] = None

        for _ in range(max_stages):
            x_eq = model.x(y_curr)
            steps.append(((x_curr, y_curr), (x_eq, y_curr)))

            if x_eq <= result.bottoms_x + tol:
                stage_count += 1
                steps.append(((x_eq, y_curr), (x_eq, result.bottoms_x)))
                x_seq.append(result.bottoms_x)
                y_seq.append(result.bottoms_x)
                break

            if feed_stage is None:
                cross_val = (x_curr - x_feed) * (x_eq - x_feed)
                if cross_val <= tol:
                    feed_stage = stage_count + 1

            if x_eq >= x_feed:
                y_next = lines.rectifying(x_eq)
            else:
                y_next = lines.stripping(x_eq)

            y_next = max(0.0, min(1.0, y_next))
            steps.append(((x_eq, y_curr), (x_eq, y_next)))

            stage_count += 1
            x_curr, y_curr = x_eq, y_next
            x_seq.append(x_curr)
            y_seq.append(y_curr)

            if x_curr <= result.bottoms_x + tol:
                break
        else:
            raise RuntimeError("逐板计算达到最大迭代次数尚未收敛")

        if feed_stage is None and stage_count > 0:
            feed_stage = stage_count

        return StageProfile(
            stage_count=stage_count,
            feed_stage=feed_stage,
            x_sequence=x_seq,
            y_sequence=y_seq,
            steps=steps,
            murphree_details=None,
        )

    # Murphree 校正路径（自塔底向上逐板绘制）
    eff = murphree_efficiency or 1.0
    rect_a, rect_b = lines.rect_params
    strip_a, strip_b = lines.strip_params

    def invert_line(y_val: float, a_val: float, b_val: float) -> float:
        if abs(a_val) < 1e-9:
            raise ValueError("操作线斜率接近 0，无法进行默弗里计算")
        return (y_val - b_val) / a_val

    x_points: List[float] = [result.bottoms_x]
    y_points: List[float] = [result.bottoms_x]
    y_current = model.y(result.bottoms_x)
    y_current = max(0.0, min(1.0, y_current))
    x_points.append(result.bottoms_x)
    y_points.append(y_current)

    stage_count = 0
    feed_stage: Optional[int] = None
    details: List[MurphreeDetail] = []

    for _ in range(max_stages):
        if y_current >= result.distillate_x - tol:
            break
        stage_count += 1
        if y_current <= lines.feed_intersection[1] + tol:
            section = "stripping"
            x_oper = invert_line(y_current, strip_a, strip_b)
        else:
            section = "rectifying"
            if feed_stage is None:
                feed_stage = stage_count
            x_oper = invert_line(y_current, rect_a, rect_b)
        x_oper = max(0.0, min(1.0, x_oper))

        x_points.append(x_oper)
        y_points.append(y_current)

        y_eq = model.y(x_oper)
        y_out = y_current + eff * (y_eq - y_current)
        y_out = max(0.0, min(1.0, y_out))

        x_points.append(x_oper)
        y_points.append(y_out)

        details.append(
            MurphreeDetail(
                stage_index=stage_count,
                x_operating=x_oper,
                y_in=y_current,
                y_equilibrium=y_eq,
                y_out=y_out,
            )
        )

        y_current = y_out

    if y_current < result.distillate_x - tol:
        raise RuntimeError("默弗里计算在给定级数内未达到塔顶组成")

    x_points.append(result.distillate_x)
    y_points.append(y_current)
    x_points.append(result.distillate_x)
    y_points.append(result.distillate_x)

    steps = [
        ((x_points[idx], y_points[idx]), (x_points[idx + 1], y_points[idx + 1]))
        for idx in range(len(x_points) - 1)
    ]

    if feed_stage is None and stage_count > 0:
        feed_stage = stage_count

    return StageProfile(
        stage_count=stage_count,
        feed_stage=feed_stage,
        x_sequence=x_points,
        y_sequence=y_points,
        steps=steps,
        murphree_details=details,
    )
