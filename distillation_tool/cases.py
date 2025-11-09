"""典型考试案例参数库。

本模块收集了课程作业和期中试题中出现的代表性精馏 / 萃取-精馏组合问题，
用于快速调用并自动填充求解流程所需的关键数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ComponentLabels:
    solute: str = "轻键"
    diluent: str = "惰性"
    solvent: str = "萃取剂/重键"


@dataclass
class VLEPreset:
    """VLE 数据设定。"""

    mode: str  # "constant_alpha" | "table"
    alpha: Optional[float] = None
    file: Optional[Path] = None
    preset: Optional[str] = None
    description: str = ""
    has_azeotrope: bool = False
    azeotrope_hint: Optional[str] = None


@dataclass
class FeedSpec:
    name: str
    total_flow: float
    light_key_molfrac: float
    q: float = 1.0
    temperature: Optional[float] = None
    pressure: Optional[float] = None


@dataclass
class SeparationSpec:
    light_key_distillate: float
    light_key_bottoms: float
    reflux_ratio: float
    recovery: Optional[float] = None
    pressure: Optional[float] = None


@dataclass
class ExtractionSpec:
    distribution_coefficient: float
    recovery: float
    solvent_to_feed_guess: Optional[float] = None
    notes: str = ""
    mode: str = "single"  # "single" | "countercurrent"
    stage_target: Optional[int] = None
    solvent_flow: Optional[float] = None
    solvent_solute_frac: float = 0.0
    max_stages: int = 8
    solute_is_light_key: bool = True


@dataclass
class CaseRecord:
    identifier: str
    title: str
    system: str
    feed: FeedSpec
    separation: SeparationSpec
    vle: VLEPreset
    operation: str = "distillation"
    extraction: Optional[ExtractionSpec] = None
    extra_notes: str = ""
    references: Dict[str, str] = field(default_factory=dict)
    components: ComponentLabels = field(default_factory=ComponentLabels)


def _build_library() -> Dict[str, CaseRecord]:
    """构建预设案例库。"""

    base_dir = Path(__file__).resolve().parent.parent

    library: Dict[str, CaseRecord] = {}

    # 甲醇-水 单塔精馏（作业 3-1）
    library["methanol_water_basic"] = CaseRecord(
        identifier="methanol_water_basic",
        title="甲醇-水 常压精馏（作业 3-1）",
        system="甲醇(轻键组分) - 水",
        feed=FeedSpec(name="进料", total_flow=100.0, light_key_molfrac=0.48, q=1.0),
        separation=SeparationSpec(
            light_key_distillate=0.90,
            light_key_bottoms=0.0108,
            reflux_ratio=0.6,
            recovery=0.99,
        ),
        vle=VLEPreset(
            mode="preset",
            preset="methanol_water_101kpa",
            description="甲醇-水 101.3 kPa VLE 数据",
        ),
        operation="distillation",
        extra_notes="泡点进料，作业 3-1 与 3-3 基础数据。",
        references={
            "homework": "2025 作业 3-1",
        },
        components=ComponentLabels(solute="甲醇", diluent="水", solvent="—"),
    )

    # 正丁醇-水-S 萃取 + 精馏（期中试题示例）
    library["nba_water_solvent"] = CaseRecord(
        identifier="nba_water_solvent",
        title="正丁醇-水-溶剂S 萃取-精馏",
        system="正丁醇(轻键) - 水 + 萃取剂 S",
        feed=FeedSpec(name="稀正丁醇水溶液", total_flow=100.0, light_key_molfrac=0.12, q=1.0),
        separation=SeparationSpec(
            light_key_distillate=0.90,
            light_key_bottoms=0.02,
            reflux_ratio=1.2,
            recovery=0.95,
        ),
        vle=VLEPreset(
            mode="table",
            file=base_dir / "data" / "nba_water_vle.csv",
            description="正丁醇-水 常压 VLE 表 (需自备或导入)",
            has_azeotrope=True,
            azeotrope_hint="共沸点 x_BuOH≈0.284, T≈92.4℃",
        ),
        operation="extractive_distillation",
        extraction=ExtractionSpec(
            distribution_coefficient=10.0,
            recovery=0.95,
            solvent_to_feed_guess=0.8,
            notes="期中试题常设 Y=10X 分配线，可快速估算萃取剂流量。",
            mode="single",
            solute_is_light_key=True,
        ),
        extra_notes="先萃取后精馏，萃余溶剂回收。",
        references={
            "midterm": "2024 期中 第 1-4 题变体",
        },
        components=ComponentLabels(solute="正丁醇", diluent="水", solvent="溶剂 S"),
    )

    # 甲酸-水-甲酸乙酯 萃取-精馏（2024 期中试题）
    library["formic_water_ester"] = CaseRecord(
        identifier="formic_water_ester",
        title="甲酸-水-甲酸乙酯 萃取-精馏",
        system="甲酸(重键) - 水 - 甲酸乙酯 (溶剂)",
        feed=FeedSpec(name="稀甲酸溶液", total_flow=80.0, light_key_molfrac=0.18, q=1.0),
        separation=SeparationSpec(
            light_key_distillate=0.98,
            light_key_bottoms=0.30,
            reflux_ratio=2.0,
            recovery=None,
        ),
        vle=VLEPreset(
            mode="constant_alpha",
            alpha=4.0,
            description="甲酸-水 α=4 假设 (2024 期中试题 3 给定)",
        ),
        operation="extractive_distillation",
        extraction=ExtractionSpec(
            distribution_coefficient=1.0,
            recovery=0.90,
            solvent_to_feed_guess=1.5,
            notes="题设常给出 Y=X 分配线，可直接迭代出萃取剂量。",
            mode="countercurrent",
            stage_target=None,
            solvent_solute_frac=0.0,
            max_stages=10,
            solute_is_light_key=False,
        ),
        extra_notes="萃取段以甲酸乙酯带出甲酸，精馏段回收溶剂。",
        references={
            "midterm": "2024 期中 第 1-5 题",
        },
        components=ComponentLabels(solute="甲酸", diluent="水", solvent="甲酸乙酯"),
    )

    return library


_LIB: Dict[str, CaseRecord] = _build_library()


def list_cases() -> Dict[str, CaseRecord]:
    """返回案例库拷贝，避免外部修改内部状态。"""

    return dict(_LIB)


def get_case(identifier: str) -> CaseRecord:
    """按标识符获取案例数据。"""

    try:
        return _LIB[identifier]
    except KeyError as exc:  # pragma: no cover - 防御性
        raise ValueError(f"未找到标识符为 {identifier!r} 的案例") from exc


def register_case(record: CaseRecord) -> None:
    """允许在运行时扩展案例库。"""

    if record.identifier in _LIB:
        raise ValueError(f"案例 {record.identifier} 已存在，请更换 identifier")
    _LIB[record.identifier] = record
