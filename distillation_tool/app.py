"""命令行主程序，支持包内或直接脚本执行。"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt

try:  # 优先使用包内相对导入
    from . import cases, diameter, efficiency, material_balance, stages, vle, visualization
except ImportError:  # 当直接执行 app.py 时使用绝对导入
    import sys

    PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from distillation_tool import (  # type: ignore  # pylint: disable=import-error
        cases,
        diameter,
        efficiency,
        material_balance,
        stages,
        vle,
        visualization,
    )


@dataclass
class CustomExtractionConfig:
    mode: str
    distribution_coefficient: float
    target_recovery: Optional[float]
    solvent_flow: Optional[float]
    solvent_to_feed: Optional[float]
    stage_count: Optional[int]
    solvent_solute_frac: float
    max_stages: int
    solute_is_light_key: bool


@dataclass
class CustomInput:
    feed_flow: float
    feed_x: float
    distillate_x: float
    bottoms_x: float
    reflux_ratio: float
    q: float
    recovery: Optional[float]
    vle_mode: str
    alpha: Optional[float]
    vle_file: Optional[Path]
    pressure: Optional[float]
    extraction: Optional[CustomExtractionConfig] = None
    component_labels: cases.ComponentLabels = field(default_factory=cases.ComponentLabels)
    distillate_flow: Optional[float] = None
    bottoms_flow: Optional[float] = None
    vle_preset: Optional[str] = None


@dataclass
class WorkflowResult:
    material_balance: material_balance.BinaryBalanceResult
    extraction_balance: Optional[material_balance.ExtractionBalanceResult]
    vle_model: vle.VLEModel
    operating_lines: stages.OperatingLines
    stage_profile: stages.StageProfile
    murphree_profile: Optional[stages.StageProfile]
    murphree_warning: Optional[str]
    n_min: float
    r_min: float
    n_theoretical: float
    efficiency_result: Optional[efficiency.EfficiencyResult]
    diameter_estimate: Optional[diameter.DiameterEstimate]
    murphree_type: Optional[str]
    murphree_value: Optional[float]
    murphree_actual_stages: Optional[float]
    gilliland_warning: Optional[str]
    q_value: float
    component_labels: cases.ComponentLabels


def _component_table_entries(result: WorkflowResult) -> list[dict[str, float]]:
    entries: list[dict[str, float]] = []
    mb = result.material_balance

    def add_stream(name: str, total: float, solute: float, solvent: float, diluent: float) -> None:
        if total <= 0 and solute <= 0 and solvent <= 0 and diluent <= 0:
            return
        entries.append(
            {
                "stream": name,
                "total": total,
                "solute": solute,
                "solvent": solvent,
                "diluent": diluent,
            }
        )

    def split_binary(total: float, solute_frac: float) -> tuple[float, float]:
        solute = total * solute_frac
        other = total - solute
        return solute, other

    solute_feed, diluent_feed = split_binary(mb.feed_flow, mb.feed_x)
    add_stream("原料液", mb.feed_flow, solute_feed, 0.0, diluent_feed)

    extraction = result.extraction_balance
    if extraction is not None:
        solute_solvent = extraction.solvent_flow * extraction.solvent_solute_frac
        solvent_pure = extraction.solvent_flow - solute_solvent
        add_stream("萃取剂 S", extraction.solvent_flow, solute_solvent, solvent_pure, 0.0)

        solute_extract, solvent_extract = split_binary(extraction.extract_flow, extraction.extract_solute_frac)
        add_stream("萃取相 E", extraction.extract_flow, solute_extract, solvent_extract, 0.0)

        solute_raff, diluent_raff = split_binary(extraction.raffinate_flow, extraction.raffinate_solute_frac)
        add_stream("萃余相 R", extraction.raffinate_flow, solute_raff, 0.0, diluent_raff)

        add_stream("精馏进料", extraction.extract_flow, solute_extract, solvent_extract, 0.0)
    else:
        add_stream("精馏进料", mb.feed_flow, solute_feed, 0.0, diluent_feed)

    solute_distillate, solvent_distillate = split_binary(mb.distillate_flow, mb.distillate_x)
    add_stream("塔顶 D", mb.distillate_flow, solute_distillate, solvent_distillate, 0.0)

    solute_bottoms, solvent_bottoms = split_binary(mb.bottoms_flow, mb.bottoms_x)
    add_stream("塔釜 W", mb.bottoms_flow, solute_bottoms, solvent_bottoms, 0.0)

    return entries


def _component_table_lines(result: WorkflowResult) -> list[str]:
    entries = _component_table_entries(result)
    if not entries:
        return []

    labels = result.component_labels
    header = f"{'物流':<12}{'总流量':>12}{labels.solute:>12}{labels.solvent:>12}{labels.diluent:>12}"
    separator = "-" * len(header)
    lines = [header, separator]
    for entry in entries:
        lines.append(
            f"{entry['stream']:<12}"
            f"{entry['total']:>12.2f}"
            f"{entry['solute']:>12.2f}"
            f"{entry['solvent']:>12.2f}"
            f"{entry['diluent']:>12.2f}"
        )
    return lines


def _energy_insights(result: WorkflowResult, vapor_flow: float) -> list[str]:
    insights: list[str] = []
    mb = result.material_balance
    ratio = None
    if result.r_min > 0:
        ratio = mb.reflux_ratio / result.r_min
        if ratio > 1.5:
            insights.append(
                f"回流比 / 最小回流比 = {ratio:.2f}，偏高，意味着冷凝/再沸器负荷较大，可考虑优化回流或热集成。"
            )

    insights.append(f"塔内蒸汽流量 V ≈ {vapor_flow:.2f} kmol/h，可据此估算蒸汽与冷却水需求。")

    extraction = result.extraction_balance
    if extraction is not None:
        if mb.feed_flow > 0:
            s_over_f = extraction.solvent_flow / mb.feed_flow
            if s_over_f > 1.0:
                insights.append(
                    f"萃取剂/原料比 S/F = {s_over_f:.2f}，溶剂循环量大，回收段蒸汽与冷却负荷显著。"
                )
        if extraction.distribution_coefficient < 2.0:
            insights.append(
                f"分配系数 K = {extraction.distribution_coefficient:.2f} 较低，可考虑开发选择性更高、较低再生温度的萃取剂。"
            )
        insights.append("萃取剂优选方向：高分配系数、高选择性、低沸点且与水互不溶，便于再生。")

    return insights


def estimate_alpha(model: vle.VLEModel, x_high: float, x_low: float) -> float:
    x_mid = max(1e-6, min(1 - 1e-6, 0.5 * (x_high + x_low)))
    y_mid = model.y(x_mid)
    return (y_mid / (1 - y_mid)) / (x_mid / (1 - x_mid))


def solve_workflow(
    *,
    feed_flow: float,
    feed_x: float,
    distillate_x: float,
    bottoms_x: float,
    reflux_ratio: float,
    q: float,
    recovery: Optional[float],
    vle_model: vle.VLEModel,
    overall_efficiency: Optional[float] = None,
    vapor_velocity: Optional[float] = None,
    vapor_temperature: Optional[float] = None,
    vapor_pressure: Optional[float] = None,
    extraction_result: Optional[material_balance.ExtractionBalanceResult] = None,
    murphree_type: Optional[str] = None,
    murphree_efficiency: Optional[float] = None,
    component_labels: Optional[cases.ComponentLabels] = None,
) -> WorkflowResult:
    balance = material_balance.solve_binary_balance(
        feed_flow,
        feed_x,
        distillate_x=distillate_x,
        bottoms_x=bottoms_x,
        reflux_ratio=reflux_ratio,
        recovery=recovery,
    )

    alpha_avg = estimate_alpha(vle_model, distillate_x, bottoms_x)
    n_min = stages.fenske_minimum_stages(distillate_x, bottoms_x, alpha_avg)
    lines = stages.build_operating_lines(balance, q=q)
    r_min = stages.minimum_reflux_ratio(balance, vle_model, q=q)
    gilliland_warning = None
    try:
        n_theoretical = stages.gilliland_stages(n_min, reflux_ratio, r_min)
    except RuntimeError as exc:
        n_theoretical = math.nan
        gilliland_warning = str(exc)
    profile = stages.mccabe_thiele(vle_model, balance, q=q)
    if math.isnan(n_theoretical):
        n_theoretical = float(profile.stage_count)

    eff_result = None
    if overall_efficiency is not None:
        eff_result = efficiency.overall_stage_count(n_theoretical, overall_efficiency)

    murphree_profile: Optional[stages.StageProfile] = None
    murphree_warning: Optional[str] = None
    murphree_actual = None
    if murphree_efficiency is not None and murphree_efficiency > 0:
        try:
            murphree_profile = stages.mccabe_thiele(
                vle_model,
                balance,
                q=q,
                murphree_type=murphree_type,
                murphree_efficiency=murphree_efficiency,
            )
            murphree_actual = murphree_profile.stage_count
        except (NotImplementedError, ValueError, RuntimeError) as exc:
            murphree_profile = None
            murphree_actual = profile.stage_count / murphree_efficiency
            murphree_warning = str(exc)

    diameter_est = None
    if vapor_velocity is not None and vapor_temperature is not None and vapor_pressure is not None:
        vapor_flow = (reflux_ratio + 1) * balance.distillate_flow
        diameter_est = diameter.ideal_gas_diameter(
            vapor_flow,
            temperature=vapor_temperature,
            pressure=vapor_pressure,
            superficial_velocity=vapor_velocity,
        )

    return WorkflowResult(
        material_balance=balance,
        extraction_balance=extraction_result,
        vle_model=vle_model,
        operating_lines=lines,
        stage_profile=profile,
        murphree_profile=murphree_profile,
        murphree_warning=murphree_warning,
        n_min=n_min,
        r_min=r_min,
        n_theoretical=n_theoretical,
        efficiency_result=eff_result,
        diameter_estimate=diameter_est,
        murphree_type=murphree_type,
        murphree_value=murphree_efficiency,
        murphree_actual_stages=murphree_actual,
        gilliland_warning=gilliland_warning,
        q_value=q,
        component_labels=component_labels or cases.ComponentLabels(),
    )


def run_case(
    case_id: str,
    *,
    overall_efficiency: Optional[float],
    vapor_velocity: Optional[float],
    vapor_temperature: Optional[float],
    vapor_pressure: Optional[float],
    murphree_type: Optional[str],
    murphree_efficiency: Optional[float],
) -> WorkflowResult:
    case = cases.get_case(case_id)
    if case.vle.preset is not None:
        vle_model = vle.build_preset_model(case.vle.preset)
    else:
        vle_model = vle.build_model(
            case.vle.mode,
            alpha=case.vle.alpha,
            file=case.vle.file,
            description=case.vle.description,
        )

    extraction_result = None
    feed_flow = case.feed.total_flow
    feed_x = case.feed.light_key_molfrac
    if case.extraction is not None:
        spec = case.extraction
        if spec.mode == "countercurrent":
            solvent_flow = spec.solvent_flow
            if solvent_flow is None and spec.solvent_to_feed_guess is not None:
                solvent_flow = spec.solvent_to_feed_guess * case.feed.total_flow
            if solvent_flow is None:
                solvent_flow = case.feed.total_flow
            extraction_result = material_balance.solve_countercurrent_extraction(
                feed_flow,
                feed_x,
                solvent_flow=solvent_flow,
                distribution_coefficient=spec.distribution_coefficient,
                stage_count=spec.stage_target,
                target_recovery=None if spec.stage_target is not None else spec.recovery,
                solvent_solute_frac=spec.solvent_solute_frac,
                max_stages=spec.max_stages,
            )
        else:
            extraction_result = material_balance.solve_single_stage_extraction(
                feed_flow,
                feed_x,
                distribution_coefficient=spec.distribution_coefficient,
                recovery=spec.recovery,
                solvent_solute_frac=spec.solvent_solute_frac,
            )
        feed_flow = extraction_result.extract_flow
        light_key_frac = extraction_result.extract_solute_frac if spec.solute_is_light_key else 1.0 - extraction_result.extract_solute_frac
        feed_x = max(0.0, min(1.0, light_key_frac))

    return solve_workflow(
        feed_flow=feed_flow,
        feed_x=feed_x,
        distillate_x=case.separation.light_key_distillate,
        bottoms_x=case.separation.light_key_bottoms,
        reflux_ratio=case.separation.reflux_ratio,
        q=case.feed.q,
        recovery=case.separation.recovery,
        vle_model=vle_model,
        overall_efficiency=overall_efficiency,
        vapor_velocity=vapor_velocity,
        vapor_temperature=vapor_temperature,
        vapor_pressure=vapor_pressure,
        extraction_result=extraction_result,
        murphree_type=murphree_type,
        murphree_efficiency=murphree_efficiency,
        component_labels=case.components,
    )


def run_custom(
    params: CustomInput,
    overall_efficiency: Optional[float],
    vapor_velocity: Optional[float],
    vapor_temperature: Optional[float],
    vapor_pressure: Optional[float],
    murphree_type: Optional[str],
    murphree_efficiency: Optional[float],
) -> WorkflowResult:
    if params.vle_preset is not None:
        vle_model = vle.build_preset_model(params.vle_preset)
    else:
        vle_model = vle.build_model(
            params.vle_mode,
            alpha=params.alpha,
            file=params.vle_file,
            description="自定义 VLE",
        )

    feed_flow = params.feed_flow
    feed_x = params.feed_x
    extraction_result: Optional[material_balance.ExtractionBalanceResult] = None
    if params.extraction is not None:
        config = params.extraction
        if config.mode == "single":
            if config.target_recovery is None:
                raise ValueError("单级萃取模式需提供 --extract-recovery")
            extraction_result = material_balance.solve_single_stage_extraction(
                feed_flow,
                feed_x,
                distribution_coefficient=config.distribution_coefficient,
                recovery=config.target_recovery,
                solvent_solute_frac=config.solvent_solute_frac,
            )
        else:
            solvent_flow = config.solvent_flow
            if solvent_flow is None and config.solvent_to_feed is not None:
                solvent_flow = config.solvent_to_feed * feed_flow
            if solvent_flow is None:
                raise ValueError("多级萃取模式需提供萃取剂流量或 --extract-solvent-ratio")
            extraction_result = material_balance.solve_countercurrent_extraction(
                feed_flow,
                feed_x,
                solvent_flow=solvent_flow,
                distribution_coefficient=config.distribution_coefficient,
                stage_count=config.stage_count,
                target_recovery=None if config.stage_count is not None else config.target_recovery,
                solvent_solute_frac=config.solvent_solute_frac,
                max_stages=config.max_stages,
            )
        feed_flow = extraction_result.extract_flow
        light_key_frac = (
            extraction_result.extract_solute_frac if config.solute_is_light_key else 1.0 - extraction_result.extract_solute_frac
        )
        feed_x = max(0.0, min(1.0, light_key_frac))

    return solve_workflow(
        feed_flow=feed_flow,
        feed_x=feed_x,
        distillate_x=params.distillate_x,
        bottoms_x=params.bottoms_x,
        reflux_ratio=params.reflux_ratio,
        q=params.q,
        recovery=params.recovery,
        vle_model=vle_model,
        overall_efficiency=overall_efficiency,
        vapor_velocity=vapor_velocity,
        vapor_temperature=vapor_temperature,
        vapor_pressure=vapor_pressure,
        extraction_result=extraction_result,
        murphree_type=murphree_type,
        murphree_efficiency=murphree_efficiency,
        component_labels=params.component_labels,
    )


def export_figures(result: WorkflowResult, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = {}

    fig_mt = visualization.plot_mccabe_thiele(
        result.vle_model,
        result.stage_profile,
        result.operating_lines,
        actual_profile=result.murphree_profile,
    )
    path_mt = out_dir / "mccabe_thiele.png"
    fig_mt.savefig(path_mt, dpi=300, bbox_inches="tight")
    plt.close(fig_mt)
    figs["mccabe_thiele"] = path_mt

    fig_table = visualization.plot_flow_table(result.material_balance, result.extraction_balance)
    path_table = out_dir / "stream_table.png"
    fig_table.savefig(path_table, dpi=300, bbox_inches="tight")
    plt.close(fig_table)
    figs["stream_table"] = path_table

    fig_flow = visualization.plot_process_flow(result.material_balance, result.extraction_balance)
    path_flow = out_dir / "process_flow.png"
    fig_flow.savefig(path_flow, dpi=300, bbox_inches="tight")
    plt.close(fig_flow)
    figs["process_flow"] = path_flow

    return figs


def workflow_to_text(result: WorkflowResult) -> str:
    lines: list[str] = []
    mb = result.material_balance
    lines.append("======== 物料衡算 ========")
    lines.append(f"F = {mb.feed_flow:.2f} kmol/h, x_F = {mb.feed_x:.4f}")
    lines.append(f"D = {mb.distillate_flow:.2f} kmol/h, x_D = {mb.distillate_x:.4f}")
    lines.append(f"W = {mb.bottoms_flow:.2f} kmol/h, x_W = {mb.bottoms_x:.4f}")
    if mb.recovery is not None:
        lines.append(f"回收率 = {mb.recovery * 100:.2f}%")

    if result.extraction_balance is not None:
        ex = result.extraction_balance
        lines.append("======== 萃取衡算 ========")
        lines.append(f"S = {ex.solvent_flow:.2f} kmol/h (K = {ex.distribution_coefficient:.2f})")
        lines.append(f"E = {ex.extract_flow:.2f} kmol/h, y = {ex.extract_solute_frac:.4f}")
        lines.append(f"R = {ex.raffinate_flow:.2f} kmol/h, x = {ex.raffinate_solute_frac:.4f}")
        mode_label = '单级' if ex.mode == 'single' else '逆流多级'
        stage_info = ex.stage_count if ex.stage_count is not None else 1
        lines.append(f"萃取模式: {mode_label}, 理论级数 = {stage_info}")
        if ex.stage_details:
            lines.append("级间组成：")
            for stage in ex.stage_details:
                lines.append(f"  第 {stage.stage_index} 级: x = {stage.raffinate_x:.4f}, y = {stage.extract_y:.4f}")

    lines.append("======== 理论级计算 ========")
    lines.append(f"芬斯克 N_T,min = {result.n_min:.2f}")
    lines.append(f"最小回流比 R_min = {result.r_min:.2f}")
    lines.append(f"吉利兰理论级 = {result.n_theoretical:.2f}")
    lines.append(f"McCabe-Thiele 逐板级数 = {result.stage_profile.stage_count}")
    if result.stage_profile.feed_stage is not None:
        lines.append(f"推荐进料板位 = 第 {result.stage_profile.feed_stage} 块理论板")
    if result.gilliland_warning:
        lines.append(f"Gilliland 警告: {result.gilliland_warning}（已采用逐板结果代替）")
    vapor_flow = (mb.reflux_ratio + 1.0) * mb.distillate_flow
    lines.append(f"塔内蒸汽流量 V ≈ {vapor_flow:.2f} kmol/h")

    if result.efficiency_result is not None:
        eff = result.efficiency_result
        lines.append("======== 效率校正 ========")
        lines.append(f"全塔板效率 = {eff.overall_efficiency * 100:.2f}%")
        lines.append(f"实际塔板数 ≈ {eff.actual_trays:.1f}")
    if result.murphree_value is not None:
        phase = '气相' if (result.murphree_type or 'gas') == 'gas' else '液相'
        lines.append("======== 默弗里效率 ========")
        lines.append(f"{phase} 默弗里效率 = {result.murphree_value * 100:.2f}%")
        if result.murphree_actual_stages is not None:
            lines.append(f"估算实际塔板数 ≈ {result.murphree_actual_stages:.1f}")
        if result.murphree_warning:
            lines.append(f"警告: {result.murphree_warning}")

    if result.diameter_estimate is not None:
        dia = result.diameter_estimate
        lines.append("======== 塔径估算 ========")
        lines.append(f"塔径 ≈ {dia.diameter:.3f} m (基于 {dia.basis})")

    comp_lines = _component_table_lines(result)
    if comp_lines:
        lines.append("======== 组分物流表 (kmol/h) ========")
        lines.extend(comp_lines)

    insight_lines = _energy_insights(result, vapor_flow)
    if insight_lines:
        lines.append("======== 能耗与改进提示 ========")
        lines.extend(insight_lines)

    return "\n".join(lines)


VLE_PRESET_CHOICES = list(vle.list_presets().keys())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="化工原理精馏解题助手")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", choices=list(cases.list_cases().keys()), help="调用预设案例")
    group.add_argument("--custom", action="store_true", help="自定义输入")

    parser.add_argument("--overall-eff", type=float, help="全塔板效率 (0-1)")
    parser.add_argument("--murphree-type", choices=["gas", "liquid"], help="默弗里效率类型")
    parser.add_argument("--murphree-eff", type=float, help="默弗里效率 (0-1)")
    parser.add_argument("--vapor-vel", type=float, help="空塔气速 m/s")
    parser.add_argument("--vapor-temp", type=float, help="汽相温度 K")
    parser.add_argument("--vapor-press", type=float, help="塔内压力 kPa")
    parser.add_argument("--out-dir", type=Path, help="图形输出目录")

    parser.add_argument("--feed-flow", type=float)
    parser.add_argument("--distillate-flow", type=float)
    parser.add_argument("--bottoms-flow", type=float)
    parser.add_argument("--feed-x", type=float)
    parser.add_argument("--distillate-x", type=float)
    parser.add_argument("--bottoms-x", type=float)
    parser.add_argument("--reflux", type=float)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--recovery", type=float)
    parser.add_argument("--vle-mode", choices=["constant_alpha", "table"], default="constant_alpha")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--vle-file", type=Path)
    if VLE_PRESET_CHOICES:
        parser.add_argument(
            "--vle-preset",
            choices=VLE_PRESET_CHOICES,
            help="使用内置 VLE 数据集，无需额外文件/α",
        )
    parser.add_argument("--extraction-mode", choices=["none", "single", "countercurrent"], default="none")
    parser.add_argument("--extract-k", type=float, help="萃取分配系数 K")
    parser.add_argument("--extract-recovery", type=float, help="萃取段目标回收率 (0-1)")
    parser.add_argument("--extract-solvent-flow", type=float, help="萃取剂流量 (kmol/h)")
    parser.add_argument("--extract-solvent-ratio", type=float, help="萃取剂/进料流量比，用于估算溶剂流量")
    parser.add_argument("--extract-stages", type=int, help="逆流萃取理论级数")
    parser.add_argument("--extract-solvent-x", type=float, default=0.0, help="萃取剂中杂质/溶质摩尔分率")
    parser.add_argument("--extract-max-stages", type=int, default=10, help="搜索级数上限，用于目标回收率求解")
    parser.add_argument(
        "--extract-solute-role",
        choices=["light", "heavy"],
        default="light",
        help="溶质在后续精馏中视作轻键(light)或重键(heavy)",
    )
    parser.add_argument("--label-solute", type=str, default="轻键", help="组分标签：溶质/轻键")
    parser.add_argument("--label-diluent", type=str, default="惰性/稀释剂", help="组分标签：原料惰性")
    parser.add_argument("--label-solvent", type=str, default="萃取剂/重键", help="组分标签：萃取剂/重键")

    return parser


def parse_custom(args: argparse.Namespace) -> CustomInput:
    required = ["feed_x", "distillate_x", "bottoms_x", "reflux"]
    for attr in required:
        if getattr(args, attr) is None:
            raise ValueError(f"自定义模式需提供参数 --{attr.replace('_', '-')}")

    feed_flow = args.feed_flow
    distillate_flow = args.distillate_flow
    bottoms_flow = args.bottoms_flow

    if feed_flow is None and distillate_flow is not None and bottoms_flow is not None:
        feed_flow = distillate_flow + bottoms_flow

    if feed_flow is None:
        raise ValueError("请至少提供 --feed-flow，或同时提供 --distillate-flow 与 --bottoms-flow")

    if distillate_flow is None and bottoms_flow is not None:
        distillate_flow = feed_flow - bottoms_flow
    if bottoms_flow is None and distillate_flow is not None:
        bottoms_flow = feed_flow - distillate_flow

    if feed_flow <= 0:
        raise ValueError("feed-flow 必须大于 0")
    if distillate_flow is not None and distillate_flow <= 0:
        raise ValueError("distillate-flow 必须大于 0")
    if bottoms_flow is not None and bottoms_flow <= 0:
        raise ValueError("bottoms-flow 必须大于 0")

    if args.vle_preset is None:
        if args.vle_mode == "constant_alpha" and args.alpha is None:
            raise ValueError("常数 α 模式需提供 --alpha")
        if args.vle_mode == "table" and args.vle_file is None:
            raise ValueError("表格模式需提供 --vle-file")

    extraction_config: Optional[CustomExtractionConfig] = None
    solute_is_light = args.extract_solute_role != "heavy"
    if args.extraction_mode != "none":
        if args.extract_k is None:
            raise ValueError("启用萃取模块需提供 --extract-k")
        if args.extract_solvent_x is not None and not 0 <= args.extract_solvent_x <= 1:
            raise ValueError("萃取剂中溶质摩尔分率需位于 [0, 1]")
        if args.extraction_mode == "single":
            if args.extract_recovery is None:
                raise ValueError("单级萃取需提供 --extract-recovery")
            extraction_config = CustomExtractionConfig(
                mode="single",
                distribution_coefficient=args.extract_k,
                target_recovery=args.extract_recovery,
                solvent_flow=None,
                solvent_to_feed=None,
                stage_count=1,
                solvent_solute_frac=args.extract_solvent_x,
                max_stages=args.extract_max_stages,
                solute_is_light_key=solute_is_light,
            )
        else:
            if args.extract_recovery is None and args.extract_stages is None:
                raise ValueError("逆流萃取需提供 --extract-recovery 或 --extract-stages")
            extraction_config = CustomExtractionConfig(
                mode="countercurrent",
                distribution_coefficient=args.extract_k,
                target_recovery=args.extract_recovery,
                solvent_flow=args.extract_solvent_flow,
                solvent_to_feed=args.extract_solvent_ratio,
                stage_count=args.extract_stages,
                solvent_solute_frac=args.extract_solvent_x,
                max_stages=args.extract_max_stages,
                solute_is_light_key=solute_is_light,
            )

    component_labels = cases.ComponentLabels(
        solute=args.label_solute,
        diluent=args.label_diluent,
        solvent=args.label_solvent,
    )

    component_labels = cases.ComponentLabels(
        solute=args.label_solute,
        diluent=args.label_diluent,
        solvent=args.label_solvent,
    )

    return CustomInput(
        feed_flow=feed_flow,
        feed_x=args.feed_x,
        distillate_x=args.distillate_x,
        bottoms_x=args.bottoms_x,
        reflux_ratio=args.reflux,
        q=args.q,
        recovery=args.recovery,
        vle_mode=args.vle_mode,
        alpha=args.alpha,
        vle_file=args.vle_file,
        pressure=None,
        extraction=extraction_config,
        component_labels=component_labels,
        distillate_flow=distillate_flow,
        bottoms_flow=bottoms_flow,
        vle_preset=args.vle_preset,
    )


def run_app(argv: Optional[list[str]] = None) -> WorkflowResult:
    parser = build_parser()
    args = parser.parse_args(argv)

    overall_eff = args.overall_eff
    murphree_type = args.murphree_type
    murphree_eff = args.murphree_eff
    if murphree_type and murphree_eff is None:
        raise ValueError("指定默弗里效率类型时需同时给出效率值")
    if murphree_eff is not None and not 0 < murphree_eff <= 1:
        raise ValueError("默弗里效率需在 (0, 1]")
    vapor_velocity = args.vapor_vel
    vapor_temp = args.vapor_temp
    vapor_press = args.vapor_press

    if args.case:
        result = run_case(
            args.case,
            overall_efficiency=overall_eff,
            vapor_velocity=vapor_velocity,
            vapor_temperature=vapor_temp,
            vapor_pressure=vapor_press,
            murphree_type=murphree_type,
            murphree_efficiency=murphree_eff,
        )
    else:
        custom_params = parse_custom(args)
        result = run_custom(
            custom_params,
            overall_efficiency=overall_eff,
            vapor_velocity=vapor_velocity,
            vapor_temperature=vapor_temp,
            vapor_pressure=vapor_press,
            murphree_type=murphree_type,
            murphree_efficiency=murphree_eff,
        )

    print(workflow_to_text(result))

    if args.out_dir:
        figures = export_figures(result, args.out_dir)
        print("\n图形已输出:")
        for name, path in figures.items():
            print(f" - {name}: {path}")

    return result


if __name__ == "__main__":  # pragma: no cover
    run_app()
