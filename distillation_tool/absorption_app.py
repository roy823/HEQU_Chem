"""命令行版吸收塔设计计算。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from . import absorption, visualization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="逆流吸收塔快速计算")
    parser.add_argument("--gas-flow", type=float, required=True, help="气相总流量 V (kmol/h)")
    parser.add_argument("--gas-y-in", type=float, required=True, help="进塔气相摩尔分率 y_F")
    parser.add_argument("--target-y-out", type=float, required=True, help="塔顶气相指标 y_D")
    parser.add_argument("--equilibrium-m", type=float, required=True, help="平衡线斜率 m (y = m x)")
    parser.add_argument("--solvent-x-in", type=float, default=0.0, help="溶剂入口摩尔分率 x_L,in")
    parser.add_argument("--solvent-flow", type=float, help="溶剂流量 L (kmol/h)，缺省按安全系数放大 Lmin")
    parser.add_argument("--safety-factor", type=float, default=1.2, help="若未指明 L，则 L = SF * Lmin")
    parser.add_argument("--max-stages", type=int, default=20, help="逐板最大迭代级数")
    parser.add_argument("--plot-path", type=Path, help="输出吸收 Y-X 图的路径 (PNG)")
    return parser


def run(argv: Optional[list[str]] = None) -> absorption.AbsorptionResult:
    parser = build_parser()
    args = parser.parse_args(argv)

    l_min = absorption.minimum_solvent_flow(
        args.gas_flow,
        y_feed=args.gas_y_in,
        y_target=args.target_y_out,
        x_solvent_in=args.solvent_x_in,
        equilibrium_slope=args.equilibrium_m,
    )
    liquid_flow = args.solvent_flow or (args.safety_factor * l_min)

    result = absorption.simulate_absorption(
        gas_flow=args.gas_flow,
        liquid_flow=liquid_flow,
        y_feed=args.gas_y_in,
        y_target=args.target_y_out,
        x_solvent_in=args.solvent_x_in,
        equilibrium_slope=args.equilibrium_m,
        max_stages=args.max_stages,
    )

    print("======== 物料与操作参数 ========")
    print(f"V = {args.gas_flow:.3f} kmol/h")
    print(f"L_min = {l_min:.3f} kmol/h")
    print(f"L = {liquid_flow:.3f} kmol/h (L/V = {liquid_flow/args.gas_flow:.3f})")
    print(f"吸收因子 A = L/(mV) = {result.absorption_factor:.3f}")
    print(f"x_L,in = {args.solvent_x_in:.4f}")
    print(f"目标 y_D = {args.target_y_out:.4f}")

    print("\n======== 级数计算 ========")
    print(f"需要的整数级数: {result.stage_count}")
    print(f"插值级数: {result.effective_stages:.2f}")
    print(f"塔底溶剂组成 x_B = {result.x_bottom:.4f}")
    print(f"回推进料 y_F,calc = {result.y_feed_required:.4f} (目标 {args.gas_y_in:.4f})")

    print("\nStage | x_in    -> x_out   | y_out   -> y_in")
    for stage in result.stages:
        print(
            f"{stage.stage_index:>5} | "
            f"{stage.x_in:>7.4f} -> {stage.x_out:>7.4f} | "
            f"{stage.y_out:>7.4f} -> {stage.y_in:>7.4f}"
        )

    if args.plot_path:
        fig = visualization.plot_absorption_yx(result, args.equilibrium_m)
        args.plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plt_path = args.plot_path.resolve()
        print(f"\n吸收 Y-X 图已输出至: {plt_path}")

    return result


if __name__ == "__main__":  # pragma: no cover
    run()
