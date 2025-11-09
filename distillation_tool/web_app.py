"""Streamlit 可视化界面."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from . import absorption, app, cases, flash, material_balance, visualization, vle
except ImportError:  # 当以脚本执行时追加绝对导入
    import sys

    PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from distillation_tool import absorption, app, cases, flash, material_balance, visualization, vle  # type: ignore


@dataclass
class UIOptions:
    overall_efficiency: Optional[float]
    vapor_velocity: Optional[float]
    vapor_temperature: Optional[float]
    vapor_pressure: Optional[float]
    murphree_type: Optional[str]
    murphree_efficiency: Optional[float]


def _parse_optional_float(label: str, value: str) -> Optional[float]:
    text = (value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"{label} 需输入数值") from exc
    if number < 0:
        raise ValueError(f"{label} 必须大于等于 0")
    return number


def _format_line(slope: float, intercept: float) -> str:
    sign = "+" if intercept >= 0 else "-"
    return f"y = {slope:.4f} x {sign} {abs(intercept):.4f}"


def _line_equilibrium_intersection(model: vle.VLEModel, slope: float, intercept: float) -> Optional[tuple[float, float]]:
    grid = np.linspace(0.0, 1.0, 2001)

    def resid(x_val: float) -> float:
        return slope * x_val + intercept - model.y(x_val)

    values = np.array([resid(float(x)) for x in grid])
    signs = values[:-1] * values[1:]
    idx = np.where(signs <= 0)[0]
    if len(idx) == 0:
        return None
    lower = float(grid[idx[0]])
    upper = float(grid[idx[0] + 1])
    for _ in range(60):
        mid = 0.5 * (lower + upper)
        val = resid(mid)
        if resid(lower) * val <= 0:
            upper = mid
        else:
            lower = mid
    x_star = 0.5 * (lower + upper)
    return x_star, slope * x_star + intercept


def _q_line_details(result: app.WorkflowResult) -> dict[str, float | str]:
    q_val = result.q_value
    x_feed = result.material_balance.feed_x
    if abs(q_val - 1.0) < 1e-6:
        return {
            "mode": "vertical",
            "q": q_val,
            "x": x_feed,
            "y_eq": result.vle_model.y(x_feed),
        }
    slope = q_val / (q_val - 1.0)
    intercept = -x_feed / (q_val - 1.0)
    details: dict[str, float | str] = {
        "mode": "oblique",
        "q": q_val,
        "slope": slope,
        "intercept": intercept,
    }
    intersection = _line_equilibrium_intersection(result.vle_model, slope, intercept)
    if intersection is not None:
        details["x_q"], details["y_q"] = intersection
    return details


def _component_dataframe(result: app.WorkflowResult) -> Optional[pd.DataFrame]:
    entries = app._component_table_entries(result)
    if not entries:
        return None
    df = pd.DataFrame(entries)
    labels = result.component_labels
    df = df.rename(
        columns={
            "stream": "物流",
            "total": "总流量 (kmol/h)",
            "solute": labels.solute,
            "solvent": labels.solvent,
            "diluent": labels.diluent,
        }
    )
    return df


def _render_fig(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def _show_results(result: app.WorkflowResult) -> None:
    mb = result.material_balance
    rect_a, rect_b = result.operating_lines.rect_params
    strip_a, strip_b = result.operating_lines.strip_params
    q_details = _q_line_details(result)
    vapor_flow = (mb.reflux_ratio + 1.0) * mb.distillate_flow
    comp_df = _component_dataframe(result)

    tabs = st.tabs(["结果概览", "关键计算", "图形展示"])

    with tabs[0]:
        if result.vle_model.description:
            st.caption(f"VLE 数据：{result.vle_model.description}")
        st.markdown("#### 物流与产出")
        col_f, col_d, col_w = st.columns(3)
        col_f.metric("F (kmol/h)", f"{mb.feed_flow:.2f}", f"x_F={mb.feed_x:.4f}")
        col_d.metric("D (kmol/h)", f"{mb.distillate_flow:.2f}", f"x_D={mb.distillate_x:.4f}")
        col_w.metric("W (kmol/h)", f"{mb.bottoms_flow:.2f}", f"x_W={mb.bottoms_x:.4f}")
        if mb.recovery is not None:
            st.caption(f"轻键回收率：{mb.recovery * 100:.2f}%")

        st.markdown("#### 理论级与回流")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("R_min", f"{result.r_min:.3f}")
        col_r2.metric("R 实际", f"{mb.reflux_ratio:.3f}")
        ratio = mb.reflux_ratio / result.r_min if result.r_min > 0 else float("nan")
        col_r3.metric("R/R_min", f"{ratio:.2f}")
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("芬斯克 N_min", f"{result.n_min:.2f}")
        col_s2.metric("Gilliland N", f"{result.n_theoretical:.2f}")
        feed_stage = result.stage_profile.feed_stage if result.stage_profile.feed_stage else "未定位"
        col_s3.metric("McCabe 级数", result.stage_profile.stage_count, f"进料板：{feed_stage}")
        if result.gilliland_warning:
            st.warning(f"Gilliland 计算提示：{result.gilliland_warning}")
        st.caption(f"塔内蒸汽流量 V ≈ {vapor_flow:.2f} kmol/h")

        if result.extraction_balance is not None:
            st.markdown("#### 萃取段物流")
            ex = result.extraction_balance
            col_e1, col_e2, col_e3 = st.columns(3)
            col_e1.metric("S (kmol/h)", f"{ex.solvent_flow:.2f}")
            col_e2.metric("E (kmol/h)", f"{ex.extract_flow:.2f}", f"y_E={ex.extract_solute_frac:.4f}")
            col_e3.metric("R (kmol/h)", f"{ex.raffinate_flow:.2f}", f"x_R={ex.raffinate_solute_frac:.4f}")

        if result.efficiency_result is not None or result.murphree_value is not None:
            st.markdown("#### 效率校正")
            cols = st.columns(2)
            if result.efficiency_result is not None:
                eff = result.efficiency_result
                cols[0].metric("全塔板效率", f"{eff.overall_efficiency * 100:.2f}%")
                cols[0].metric("实际塔板数", f"{eff.actual_trays:.1f}")
            if result.murphree_value is not None:
                label = "气相" if (result.murphree_type or "gas") == "gas" else "液相"
                cols[1].metric(f"{label} 默弗里效率", f"{result.murphree_value * 100:.2f}%")
                if result.murphree_actual_stages is not None:
                    cols[1].metric("折算塔板数", f"{result.murphree_actual_stages:.1f}")
                if result.murphree_warning:
                    st.warning(result.murphree_warning)

        if result.diameter_estimate is not None:
            dia = result.diameter_estimate
            st.markdown("#### 塔径估算")
            st.metric("塔径 (m)", f"{dia.diameter:.3f}", dia.basis)

        if comp_df is not None:
            st.markdown("#### 组分物流表 (kmol/h)")
            st.dataframe(comp_df.round(4), use_container_width=True)

        insights = app._energy_insights(result, vapor_flow)
        if insights:
            st.markdown("#### 能耗与优化提示")
            for tip in insights:
                st.write(f"- {tip}")

    with tabs[1]:
        st.markdown("#### q 线与最小回流比")
        lines = [f"- 进料热状况 q = {q_details['q']:.3f}"]
        if q_details["mode"] == "vertical":
            lines.append(f"- q 线：x = {q_details['x']:.4f}，与平衡线交于 y = {q_details['y_eq']:.4f}")
        else:
            slope = q_details["slope"]  # type: ignore[index]
            intercept = q_details["intercept"]  # type: ignore[index]
            lines.append(f"- q 线：{_format_line(slope, intercept)}")
            x_q = q_details.get("x_q")
            y_q = q_details.get("y_q")
            if isinstance(x_q, float) and isinstance(y_q, float):
                lines.append(f"  交点：x_q={x_q:.4f}, y_q={y_q:.4f}")
        lines.append(f"- R_min = {result.r_min:.3f}")
        lines.append(f"- 实际回流比 R = {mb.reflux_ratio:.3f}")
        st.markdown("\n".join(lines))

        st.markdown("#### 操作线与进料交点")
        feed_x, feed_y = result.operating_lines.feed_intersection
        st.markdown(
            "\n".join(
                [
                    f"- 精馏段：{_format_line(rect_a, rect_b)}",
                    f"- 精馏段液相流 L = R·D = {mb.reflux_ratio * mb.distillate_flow:.2f} kmol/h",
                    f"- 提馏段：{_format_line(strip_a, strip_b)}",
                    f"- 进料交点：(x={feed_x:.4f}, y={feed_y:.4f})",
                ]
            )
        )

        st.markdown("#### 阶梯计算概览")
        stage_rows = []
        for idx, step in enumerate(result.stage_profile.steps, start=1):
            (x0, y0), (x1, y1) = step
            stage_rows.append(
                {
                    "级数": idx,
                    "x 起": x0,
                    "y 起": y0,
                    "x 终": x1,
                    "y 终": y1,
                }
            )
        if stage_rows:
            st.dataframe(stage_rows, use_container_width=True)

    with tabs[2]:
        st.markdown("**McCabe-Thiele**")
        fig_mt = visualization.plot_mccabe_thiele(
            result.vle_model,
            result.stage_profile,
            result.operating_lines,
            actual_profile=result.murphree_profile,
        )
        _render_fig(fig_mt)

        st.markdown("**物流图表**")
        fig_table = visualization.plot_flow_table(result.material_balance, result.extraction_balance)
        _render_fig(fig_table)


def _case_mode(options: UIOptions) -> None:
    case_map = cases.list_cases()
    label_to_id = {f"{record.title} ({record.identifier})": key for key, record in case_map.items()}
    choice = st.selectbox("选择案例", list(label_to_id.keys()))
    record = case_map[label_to_id[choice]]

    with st.expander("案例数据预览"):
        st.write(record)

    if st.button("运行求解", type="primary"):
        result = app.run_case(
            record.identifier,
            overall_efficiency=options.overall_efficiency,
            vapor_velocity=options.vapor_velocity,
            vapor_temperature=options.vapor_temperature,
            vapor_pressure=options.vapor_pressure,
            murphree_type=options.murphree_type,
            murphree_efficiency=options.murphree_efficiency,
        )
        _show_results(result)


def _build_vle_from_upload(uploaded_file) -> vle.VLEModel:
    if uploaded_file is None:
        raise ValueError("请上传包含 x,y 列的 VLE 数据文件")

    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = Path(tmp.name)

    try:
        if suffix in {".csv", ".txt"}:
            x_data, y_data = vle.load_vle_from_csv(temp_path)
        elif suffix in {".xls", ".xlsx"}:
            x_data, y_data = vle.load_vle_from_excel(temp_path)
        else:
            raise ValueError("仅支持 CSV 或 Excel 文件")
    finally:
        temp_path.unlink(missing_ok=True)

    sort_idx = np.argsort(x_data)
    return vle.VLEModel(
        mode="table",
        x_data=x_data[sort_idx],
        y_data=y_data[sort_idx],
        description=f"上传文件：{uploaded_file.name}",
    )


def _custom_mode(options: UIOptions) -> None:
    preset_map = vle.list_presets()
    preset_label_map = {
        (f"{desc} [{name}]" if desc else name): name for name, desc in preset_map.items()
    }

    with st.form("custom_form"):
        flow_col, comp_col, oper_col = st.columns(3)
        with flow_col:
            feed_flow_text = st.text_input("进料流量 F (kmol/h，可留空)", value="100")
            distillate_flow_text = st.text_input("塔顶流量 D (kmol/h，可留空)", value="")
            bottoms_flow_text = st.text_input("塔釜流量 W (kmol/h，可留空)", value="")
            st.caption("至少提供 F，或同时提供 D 与 W，程序可自动补算")
        with comp_col:
            feed_x = st.number_input("进料轻键摩尔分率 x_F", value=0.48, min_value=0.0, max_value=1.0)
            distillate_x = st.number_input("塔顶组成 x_D", value=0.90, min_value=0.0, max_value=1.0)
            bottoms_x = st.number_input("塔釜组成 x_W", value=0.01, min_value=0.0, max_value=1.0)
            q_val = st.number_input("进料热状况 q", value=1.0, min_value=0.0, max_value=5.0)
        with oper_col:
            reflux_ratio = st.number_input("回流比 R", value=0.6, min_value=0.0)
            use_recovery = st.checkbox("启用回收率约束", value=True)
            recovery = None
            if use_recovery:
                recovery = st.number_input("回收率 (0-1)", value=0.99, min_value=0.01, max_value=1.0)
            vle_choice = st.radio("VLE 模式", ["常数 α", "内置预设", "表格数据"])
            alpha = None
            uploaded = None
            preset_key = None
            if vle_choice == "常数 α":
                alpha = st.number_input("相对挥发度 α", value=3.8, min_value=1.0)
            elif vle_choice == "内置预设":
                if not preset_label_map:
                    st.info("暂无内置数据，请切换到其他模式")
                else:
                    label = st.selectbox("选择内置数据", list(preset_label_map.keys()))
                    preset_key = preset_label_map[label]
            else:
                uploaded = st.file_uploader("上传 VLE 文件", type=["csv", "txt", "xls", "xlsx"])

        st.markdown("---")
        st.subheader("萃取 / 萃取-精馏设置")
        extraction_choice = st.selectbox("萃取模块", ["不启用", "单级萃取", "逆流多级萃取"])
        extract_k = extract_recovery = extract_solvent_flow = extract_ratio = None
        extract_stage = None
        extract_solvent_x = 0.0
        extract_max_stage = 10
        control_mode = None
        extract_solute_is_light = True
        if extraction_choice != "不启用":
            extract_k = st.number_input("分配系数 K", value=1.0, min_value=0.05)
            extract_solvent_x = st.number_input("萃取剂中溶质摩尔分率", value=0.0, min_value=0.0, max_value=1.0)
            if extraction_choice == "单级萃取":
                extract_recovery = st.number_input("目标回收率", value=0.90, min_value=0.01, max_value=0.999)
            else:
                control_mode = st.radio("控制方式", ["目标回收率", "指定级数"], horizontal=True)
                extract_max_stage = st.number_input("最大搜索级数", value=10, min_value=1, step=1, format="%d")
                if control_mode == "目标回收率":
                    extract_recovery = st.number_input("萃取段目标回收率", value=0.95, min_value=0.01, max_value=0.999)
                else:
                    extract_stage = st.number_input("理论级数", value=4, min_value=1, step=1, format="%d")
                extract_solvent_flow = st.number_input("萃取剂流量 (kmol/h)", value=80.0, min_value=0.0)
                extract_ratio = st.number_input("萃取剂/进料比 (可选)", value=1.0, min_value=0.0)
            role_label = st.radio("萃取溶质在精馏中视为", ["轻键 (塔顶富集)", "重键 (塔釜富集)"], horizontal=True)
            extract_solute_is_light = role_label.startswith("轻键")

        submitted = st.form_submit_button("运行求解")

    if not submitted:
        return

    try:
        feed_flow = _parse_optional_float("进料流量 F", feed_flow_text)
        distillate_flow = _parse_optional_float("塔顶流量 D", distillate_flow_text)
        bottoms_flow = _parse_optional_float("塔釜流量 W", bottoms_flow_text)
    except ValueError as exc:
        st.error(str(exc))
        return

    if feed_flow is None and distillate_flow is not None and bottoms_flow is not None:
        feed_flow = distillate_flow + bottoms_flow
    if feed_flow is None:
        st.error("请至少提供 F，或同时提供 D 与 W")
        return
    if distillate_flow is None and bottoms_flow is not None:
        distillate_flow = feed_flow - bottoms_flow
    if bottoms_flow is None and distillate_flow is not None:
        bottoms_flow = feed_flow - distillate_flow
    if feed_flow <= 0:
        st.error("进料流量必须大于 0")
        return
    if distillate_flow is not None and distillate_flow <= 0:
        st.error("塔顶流量必须大于 0")
        return
    if bottoms_flow is not None and bottoms_flow <= 0:
        st.error("塔釜流量必须大于 0")
        return

    if vle_choice == "常数 α":
        model = vle.build_model("constant_alpha", alpha=alpha, description="自定义 α")
    elif vle_choice == "内置预设":
        if preset_key is None:
            st.error("请选择 VLE 预设或切换模式")
            return
        model = vle.build_preset_model(preset_key)
    else:
        try:
            model = _build_vle_from_upload(uploaded)
        except ValueError as exc:
            st.error(str(exc))
            return

    extraction_result = None
    eff_feed_flow = feed_flow
    eff_feed_x = feed_x
    if extraction_choice != "不启用":
        if extract_k is None:
            st.error("请提供分配系数 K")
            return
        try:
            if extraction_choice == "单级萃取":
                if extract_recovery is None:
                    st.error("请提供萃取回收率")
                    return
                extraction_result = material_balance.solve_single_stage_extraction(
                    feed_flow,
                    feed_x,
                    distribution_coefficient=extract_k,
                    recovery=extract_recovery,
                    solvent_solute_frac=extract_solvent_x,
                )
            else:
                solvent_flow = extract_solvent_flow if extract_solvent_flow and extract_solvent_flow > 0 else None
                ratio = extract_ratio if extract_ratio and extract_ratio > 0 else None
                if solvent_flow is None and ratio is not None:
                    solvent_flow = ratio * feed_flow
                if solvent_flow is None:
                    st.error("请提供萃取剂流量或流量比")
                    return
                stage_target = None
                target_recovery = extract_recovery
                if control_mode == "指定级数" and extract_stage is not None:
                    stage_target = int(extract_stage)
                    target_recovery = None
                extraction_result = material_balance.solve_countercurrent_extraction(
                    feed_flow,
                    feed_x,
                    solvent_flow=solvent_flow,
                    distribution_coefficient=extract_k,
                    stage_count=stage_target,
                    target_recovery=target_recovery,
                    solvent_solute_frac=extract_solvent_x,
                    max_stages=int(extract_max_stage),
                )
        except (ValueError, RuntimeError) as exc:
            st.error(f"萃取计算失败：{exc}")
            return
        eff_feed_flow = extraction_result.extract_flow
        light_key_frac = (
            extraction_result.extract_solute_frac if extract_solute_is_light else 1.0 - extraction_result.extract_solute_frac
        )
        eff_feed_x = max(0.0, min(1.0, light_key_frac))

    result = app.solve_workflow(
        feed_flow=eff_feed_flow,
        feed_x=eff_feed_x,
        distillate_x=distillate_x,
        bottoms_x=bottoms_x,
        reflux_ratio=reflux_ratio,
        q=q_val,
        recovery=recovery,
        vle_model=model,
        overall_efficiency=options.overall_efficiency,
        vapor_velocity=options.vapor_velocity,
        vapor_temperature=options.vapor_temperature,
        vapor_pressure=options.vapor_pressure,
        extraction_result=extraction_result,
        murphree_type=options.murphree_type,
        murphree_efficiency=options.murphree_efficiency,
    )
    _show_results(result)


def _absorption_mode() -> None:
    st.subheader("逆流吸收塔计算")
    with st.form("absorption_form"):
        col1, col2 = st.columns(2)
        with col1:
            gas_flow = st.number_input("气相流量 V (kmol/h)", value=100.0, min_value=0.0)
            y_feed = st.number_input("进塔摩尔分率 y_F", value=0.0100, min_value=0.0, max_value=0.9999, format="%.4f")
            y_target = st.number_input("塔顶目标 y_D", value=0.0010, min_value=0.0, max_value=0.9999, format="%.4f")
        with col2:
            equilibrium_m = st.number_input("平衡线斜率 m (y = m x)", value=0.40, min_value=0.0001, format="%.4f")
            x_solvent = st.number_input("溶剂入口 x_L,in", value=0.0000, min_value=0.0, max_value=0.9999, format="%.4f")
            specify_mode = st.radio("溶剂指定方式", ["按安全系数", "直接给定流量"], horizontal=True)
            if specify_mode == "按安全系数":
                safety_factor = st.number_input("安全系数", value=1.20, min_value=1.0, step=0.05)
                solvent_flow = None
            else:
                solvent_flow = st.number_input("溶剂流量 L (kmol/h)", value=45.0, min_value=0.0)
                safety_factor = None
        max_stages = st.number_input("最大理论级数", value=20, min_value=1, step=1)
        submitted = st.form_submit_button("运行吸收计算", type="primary")

    if not submitted:
        return

    try:
        l_min = absorption.minimum_solvent_flow(
            gas_flow,
            y_feed=y_feed,
            y_target=y_target,
            x_solvent_in=x_solvent,
            equilibrium_slope=equilibrium_m,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    liquid_flow = solvent_flow if solvent_flow is not None else safety_factor * l_min

    try:
        result = absorption.simulate_absorption(
            gas_flow=gas_flow,
            liquid_flow=liquid_flow,
            y_feed=y_feed,
            y_target=y_target,
            x_solvent_in=x_solvent,
            equilibrium_slope=equilibrium_m,
            max_stages=int(max_stages),
        )
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        return

    tab_summary, tab_steps, tab_plots = st.tabs(["结果概览", "级次明细", "图形输出"])

    with tab_summary:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("V (kmol/h)", f"{result.gas_flow:.2f}")
        col_b.metric("L_min (kmol/h)", f"{l_min:.2f}")
        col_c.metric("L (kmol/h)", f"{result.liquid_flow:.2f}")
        col_d, col_e, col_f = st.columns(3)
        col_d.metric("吸收因子 A", f"{result.absorption_factor:.3f}")
        col_e.metric("整数级数", f"{result.stage_count}")
        col_f.metric("插值级数", f"{result.effective_stages:.2f}")
        st.markdown(
            "\n".join(
                [
                    f"- 塔顶气相：y_D = {result.y_top:.4f}",
                    f"- 溶剂出口：x_B = {result.x_bottom:.4f}",
                    f"- 计算得到的进料气相 y_F = {result.y_feed_required:.4f}",
                ]
            )
        )

    with tab_steps:
        stage_rows = [
            {
                "级数": stage.stage_index,
                "x_in": stage.x_in,
                "x_out": stage.x_out,
                "y_out": stage.y_out,
                "y_in": stage.y_in,
            }
            for stage in result.stages
        ]
        st.dataframe(stage_rows, use_container_width=True)
        slope = result.liquid_flow / result.gas_flow
        intercept = result.y_top - slope * result.x_top
        st.markdown(
            "\n".join(
                [
                    f"- 平衡线：y = {equilibrium_m:.4f} x",
                    f"- 操作线：{_format_line(slope, intercept)}",
                ]
            )
        )

    with tab_plots:
        fig = visualization.plot_absorption_yx(result, equilibrium_m)
        _render_fig(fig)


def _sidebar() -> UIOptions:
    st.sidebar.header("可选参数")
    use_eff = st.sidebar.checkbox("启用效率校正", value=True)
    eff = st.sidebar.slider("全塔板效率", 0.5, 1.0, 0.9, step=0.01) if use_eff else None

    st.sidebar.markdown("---")
    use_murphree = st.sidebar.checkbox("考虑默弗里效率", value=False)
    murphree_type = None
    murphree_eff = None
    if use_murphree:
        type_label = st.sidebar.selectbox("效率类型", ["气相", "液相"])
        murphree_type = "gas" if type_label == "气相" else "liquid"
        murphree_eff = st.sidebar.number_input("默弗里效率 (0-1)", value=0.9, min_value=0.1, max_value=1.0, step=0.01)

    st.sidebar.markdown("---")
    use_diameter = st.sidebar.checkbox("计算塔径", value=False)
    vel = temp = press = None
    if use_diameter:
        vel = st.sidebar.number_input("空塔气速 (m/s)", value=0.8, min_value=0.1)
        temp = st.sidebar.number_input("汽相温度 (K)", value=351.0, min_value=200.0)
        press = st.sidebar.number_input("塔内压力 (kPa)", value=101.3, min_value=10.0)

    return UIOptions(
        overall_efficiency=eff,
        vapor_velocity=vel,
        vapor_temperature=temp,
        vapor_pressure=press,
        murphree_type=murphree_type,
        murphree_efficiency=murphree_eff,
    )


def _flash_mode() -> None:
    st.subheader("单级平衡 / 闪蒸计算")
    with st.form("flash_form"):
        st.caption("可用于单级加热、冷凝或逆流接触估算")
        col1, col2, col3 = st.columns(3)
        with col1:
            flow1 = st.number_input("物流1流量 (kmol/h)", value=100.0, min_value=0.0)
            comp1 = st.number_input("物流1轻键摩尔分率", value=0.48, min_value=0.0, max_value=1.0)
            phase1 = st.selectbox("物流1相态", ["液相", "气相"], key="phase1")
        with col2:
            use_second = st.checkbox("添加物流2", value=False)
            flow2 = comp2 = 0.0
            phase2 = "液相"
            if use_second:
                flow2 = st.number_input("物流2流量 (kmol/h)", value=35.0, min_value=0.0)
                comp2 = st.number_input("物流2轻键摩尔分率", value=0.90, min_value=0.0, max_value=1.0)
                phase2 = st.selectbox("物流2相态", ["液相", "气相"], key="phase2")
        with col3:
            temperature = st.number_input("操作温度 (°C)", value=80.0)
            x_eq = st.number_input("平衡液相组成 x_eq", value=0.40, min_value=0.0, max_value=1.0)
            y_eq = st.number_input("平衡汽相组成 y_eq", value=0.70, min_value=0.0, max_value=1.0)

        submitted = st.form_submit_button("计算")

    if not submitted:
        return

    try:
        streams = [flash.Stream(phase="L" if phase1 == "液相" else "V", flow=flow1, fraction=comp1)]
        if use_second and flow2 > 0:
            streams.append(flash.Stream(phase="L" if phase2 == "液相" else "V", flow=flow2, fraction=comp2))
        result = flash.flash_with_mixed_feeds(streams, x_eq=x_eq, y_eq=y_eq)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.success("计算完成")
    st.write(
        {
            "操作温度 (°C)": f"{temperature:.2f}",
            "总进料 (kmol/h)": f"{result.feed_flow:.2f}",
            "进料轻键分率": f"{result.feed_fraction:.4f}",
            "汽相流量 V": f"{result.vapor_flow:.2f}",
            "汽相组成 y": f"{result.vapor_fraction:.4f}",
            "液相流量 L": f"{result.liquid_flow:.2f}",
            "液相组成 x": f"{result.liquid_fraction:.4f}",
            "汽化分率 β": f"{result.vapor_ratio:.4f}",
        }
    )
    st.caption("提示：x_eq、y_eq 可来源于 T-x-y 表或相对挥发度估算")


def main() -> None:
    st.set_page_config(page_title="精馏解题助手", layout="wide")
    st.title("化工原理精馏解题助手 (Web)")
    st.caption("支持期中试题、作业快速求解")

    options = _sidebar()

    mode = st.radio("选择求解模式", ["预设案例", "自定义精馏", "逆流吸收", "单级平衡/闪蒸"], horizontal=True)
    if mode == "预设案例":
        _case_mode(options)
    elif mode == "自定义精馏":
        _custom_mode(options)
    elif mode == "逆流吸收":
        _absorption_mode()
    else:
        _flash_mode()

    st.markdown("---")
    st.markdown("提示：若上传表格数据，请确保包含 x、y 列；闪蒸模式下输入的平衡组成可来自教材 T-x-y 数据或图表读数。")


if __name__ == "__main__":  # pragma: no cover
    main()
