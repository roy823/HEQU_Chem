"""可视化输出模块。"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .material_balance import BinaryBalanceResult, ExtractionBalanceResult
from .stages import OperatingLines, StageProfile
from .vle import VLEModel, enrich_curve

if TYPE_CHECKING:  # pragma: no cover
    from .absorption import AbsorptionResult


def _ensure_chinese_font() -> None:
    if getattr(_ensure_chinese_font, "_configured", False):
        return
    preferred_fonts = ["SimHei", "Microsoft YaHei", "Microsoft JhengHei", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            _ensure_chinese_font._configured = True  # type: ignore[attr-defined]
            return
    _ensure_chinese_font._configured = True  # type: ignore[attr-defined]


_ensure_chinese_font()


def plot_mccabe_thiele(
    model: VLEModel,
    profile: StageProfile,
    lines: OperatingLines,
    *,
    show_q: bool = True,
    title: str = "McCabe-Thiele 图",
    actual_profile: Optional[StageProfile] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    xs, ys = enrich_curve(model)
    ax.plot(xs, ys, label="平衡线", color="#1f77b4")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="45°线")

    x_vals = np.linspace(0, 1, 200)
    ax.plot(x_vals, [lines.rectifying(x) for x in x_vals], label="精馏段操作线", color="#d62728")
    ax.plot(x_vals, [lines.stripping(x) for x in x_vals], label="提馏段操作线", color="#2ca02c")

    if show_q:
        if lines.q_line is not None:
            ax.plot(
                x_vals,
                [lines.q_line(x) for x in x_vals],
                label="q 线",
                color="#9467bd",
                linestyle=":",
            )
        elif lines.q_vertical_x is not None:
            ax.axvline(lines.q_vertical_x, linestyle=":", color="#9467bd", label="q 线 (垂直)")

    for idx, ((x1, y1), (x2, y2)) in enumerate(profile.steps):
        label = "理论级阶梯" if idx == 0 else None
        ax.plot([x1, x2], [y1, y2], color="#ff7f0e", linewidth=1.2, label=label)

    if actual_profile is not None:
        if actual_profile.murphree_details:
            ax.plot(
                actual_profile.x_sequence,
                actual_profile.y_sequence,
                color="#8c564b",
                linewidth=1.6,
                linestyle="--",
                label="默弗里效率路径",
            )
            for idx, detail in enumerate(actual_profile.murphree_details):
                label_star = "理论上升 (y*)" if idx == 0 else None
                label_actual = "实际上升 (η_M)" if idx == 0 else None
                ax.plot(
                    [detail.x_operating, detail.x_operating],
                    [detail.y_in, detail.y_equilibrium],
                    color="#d62728",
                    linestyle=":",
                    linewidth=1.2,
                    label=label_star,
                )
                ax.plot(
                    [detail.x_operating, detail.x_operating],
                    [detail.y_in, detail.y_out],
                    color="#d62728",
                    linewidth=2.0,
                    label=label_actual,
                )
        else:
            for idx, ((x1, y1), (x2, y2)) in enumerate(actual_profile.steps):
                label = "效率修正路径" if idx == 0 else None
                ax.plot([x1, x2], [y1, y2], color="#8c564b", linewidth=1.4, linestyle="--", label=label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("液相摩尔分率 x")
    ax.set_ylabel("汽相摩尔分率 y")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return fig


def plot_flow_table(
    distillation: BinaryBalanceResult,
    extraction: Optional[ExtractionBalanceResult] = None,
    *,
    title: str = "物流表",
) -> plt.Figure:
    from matplotlib.table import Table

    fig, ax = plt.subplots(figsize=(6, 2.5 if extraction is None else 4))
    ax.axis("off")

    table = Table(ax, bbox=[0, 0, 1, 1])
    headers = ["物流", "总流量 (kmol/h)", "轻/溶质摩尔分率"]
    cell_props = dict(facecolor="#e7f0fd", edgecolor="black")
    for col, header in enumerate(headers):
        table.add_cell(0, col, width=1 / len(headers), height=0.2, text=header, loc="center", **cell_props)

    rows = []
    for name, data in distillation.summary_table().items():
        rows.append((name, data["总流量"], data["轻键摩尔分率"]))

    if extraction is not None:
        for name, data in extraction.streams().items():
            rows.append((f"[萃取] {name}", data["总流量"], data["溶质摩尔分率"]))

    for idx, row in enumerate(rows, start=1):
        for col, value in enumerate(row):
            text = f"{value:.4f}" if isinstance(value, float) else value
            table.add_cell(idx, col, width=1 / len(headers), height=0.18, text=text, loc="center")

    ax.add_table(table)
    ax.set_title(title)
    return fig


def plot_process_flow(distillation: BinaryBalanceResult, extraction: Optional[ExtractionBalanceResult] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")

    if extraction is None:
        elements = {
            "进料": (0.1, 0.5),
            "精馏塔": (0.45, 0.5),
            "塔顶": (0.8, 0.75),
            "塔釜": (0.8, 0.25),
        }

        for label, (x, y) in elements.items():
            ax.add_patch(plt.Rectangle((x - 0.05, y - 0.05), 0.1, 0.1, fill=False))
            ax.text(x, y, label, ha="center", va="center")

        ax.annotate(
            f"F={distillation.feed_flow:.2f}\nx={distillation.feed_x:.3f}",
            xy=elements["进料"],
            xytext=(0.25, 0.5),
            arrowprops=dict(arrowstyle="->"),
            ha="left",
        )

        ax.annotate(
            f"D={distillation.distillate_flow:.2f}\nx={distillation.distillate_x:.3f}",
            xy=elements["塔顶"],
            xytext=(0.6, 0.8),
            arrowprops=dict(arrowstyle="->"),
            ha="left",
        )

        ax.annotate(
            f"W={distillation.bottoms_flow:.2f}\nx={distillation.bottoms_x:.3f}",
            xy=elements["塔釜"],
            xytext=(0.6, 0.2),
            arrowprops=dict(arrowstyle="->"),
            ha="left",
        )
    else:
        elements = {
            "进料": (0.1, 0.65),
            "萃取塔": (0.35, 0.65),
            "精馏塔": (0.65, 0.65),
            "塔顶": (0.9, 0.85),
            "塔釜": (0.9, 0.45),
            "溶剂循环": (0.35, 0.35),
        }

        for label, (x, y) in elements.items():
            ax.add_patch(plt.Rectangle((x - 0.05, y - 0.05), 0.1, 0.1, fill=False))
            ax.text(x, y, label, ha="center", va="center")

        ax.annotate(
            f"F={extraction.feed_flow:.2f}\nz={extraction.feed_solute_frac:.3f}",
            xy=elements["进料"],
            xytext=(0.2, 0.75),
            arrowprops=dict(arrowstyle="->"),
        )

        ax.annotate(
            f"S={extraction.solvent_flow:.2f}",
            xy=elements["萃取塔"],
            xytext=(0.35, 0.82),
            arrowprops=dict(arrowstyle="->"),
        )

        ax.annotate(
            f"E={extraction.extract_flow:.2f}\ny={extraction.extract_solute_frac:.3f}",
            xy=elements["精馏塔"],
            xytext=(0.55, 0.78),
            arrowprops=dict(arrowstyle="->"),
        )

        ax.annotate(
            f"R={extraction.raffinate_flow:.2f}\nx={extraction.raffinate_solute_frac:.3f}",
            xy=elements["萃取塔"],
            xytext=(0.35, 0.5),
            arrowprops=dict(arrowstyle="->"),
        )

        ax.annotate(
            f"D={distillation.distillate_flow:.2f}\nx={distillation.distillate_x:.3f}",
            xy=elements["塔顶"],
            xytext=(0.75, 0.9),
            arrowprops=dict(arrowstyle="->"),
        )

        ax.annotate(
            f"W={distillation.bottoms_flow:.2f}\nx={distillation.bottoms_x:.3f}",
            xy=elements["塔釜"],
            xytext=(0.75, 0.55),
            arrowprops=dict(arrowstyle="->"),
        )

    ax.set_title("流程示意")
    return fig


def plot_efficiency_curve(
    gas_velocity: Sequence[float],
    efficiencies: Sequence[float],
    *,
    title: str = "塔效率曲线",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(gas_velocity, efficiencies, marker="o")
    ax.set_xlabel("空塔气速 (m/s)")
    ax.set_ylabel("全塔板效率")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def plot_absorption_yx(
    result: "AbsorptionResult",
    equilibrium_slope: float,
    *,
    title: str = "吸收 Y-X 图",
) -> plt.Figure:
    """基于 AbsorptionResult 绘制 Y-X 阶梯图。"""

    fig, ax = plt.subplots(figsize=(6, 5))
    x_vals = np.linspace(0, 1, 200)
    ax.plot(x_vals, equilibrium_slope * x_vals, color="#1f77b4", label="平衡线 (y = m x)")

    slope = result.liquid_flow / result.gas_flow
    intercept = result.y_top - slope * result.x_top
    ax.plot(
        x_vals,
        slope * x_vals + intercept,
        color="#2ca02c",
        label="操作线",
    )

    if result.stages:
        points = []
        first = result.stages[0]
        points.append((first.x_in, first.y_out))
        for stage in result.stages:
            points.append((stage.x_out, stage.y_out))
            points.append((stage.x_out, stage.y_in))
        xs, ys = zip(*points)
        ax.plot(xs, ys, color="#ff7f0e", linewidth=1.4, label="理论级阶梯")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("液相摩尔分率 x")
    ax.set_ylabel("气相摩尔分率 y")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return fig

