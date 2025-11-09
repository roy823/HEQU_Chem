"""VLE 数据读取与计算工具。"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"

_PRESET_TABLES: Dict[str, Dict[str, Path | str]] = {
    "methanol_water_101kpa": {
        "file": DATA_DIR / "vle_methanol_water_101kpa.csv",
        "description": "甲醇-水 VLE (101.3 kPa) —— 附表 1 数据",
    },
}

@dataclass
class VLEModel:
    """二元 VLE 模型，支持常数 α 与表格插值。"""

    mode: str  # "constant_alpha" | "table"
    description: str = ""
    alpha: Optional[float] = None
    x_data: Optional[np.ndarray] = None
    y_data: Optional[np.ndarray] = None

    def y(self, x):
        """Return vapor composition y for a given liquid composition x.

        Accepts scalar or array-like inputs for convenient vectorized calculations."""

        arr = np.asarray(x, dtype=float)
        scalar_input = arr.ndim == 0
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError("液相 x 必须位于 [0, 1]")

        if self.mode == "constant_alpha":
            if self.alpha is None:
                raise ValueError("缺少相对挥发度 α")
            alpha = self.alpha
            result = alpha * arr / (1 + (alpha - 1.0) * arr)
        elif self.mode == "table":
            if self.x_data is None or self.y_data is None:
                raise ValueError("未加载 VLE 表格数据")
            flat = arr.reshape(-1)
            interp = np.interp(flat, self.x_data, self.y_data)
            result = interp.reshape(arr.shape)
        else:
            raise ValueError(f"未知 VLE 模式: {self.mode}")

        if scalar_input:
            return float(np.asarray(result))
        return np.asarray(result)

    def x(self, y):
        """Return liquid composition x for a given vapor composition y.

        Accepts scalar or array-like inputs for convenient vectorized calculations."""

        arr = np.asarray(y, dtype=float)
        scalar_input = arr.ndim == 0
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError("汽相 y 必须位于 [0, 1]")

        if self.mode == "constant_alpha":
            if self.alpha is None:
                raise ValueError("缺少相对挥发度 α")
            alpha = self.alpha
            result = arr / (alpha - (alpha - 1.0) * arr)
        elif self.mode == "table":
            if self.x_data is None or self.y_data is None:
                raise ValueError("未加载 VLE 表格数据")
            flat = arr.reshape(-1)
            interp = np.interp(flat, self.y_data, self.x_data)
            result = interp.reshape(arr.shape)
        else:
            raise ValueError(f"未知 VLE 模式: {self.mode}")

        if scalar_input:
            return float(np.asarray(result))
        return np.asarray(result)

    def azeotrope(self, tol: float = 1e-3) -> Optional[Tuple[float, float]]:
        """检测共沸点，返回 (x, y)。若无则为 None。"""

        if self.mode == "constant_alpha":
            if math.isclose(self.alpha or 0.0, 1.0, rel_tol=1e-6):
                return 0.5, 0.5
            return None

        if self.x_data is None or self.y_data is None:
            return None

        diff = np.abs(self.x_data - self.y_data)
        idx = int(np.argmin(diff))
        if diff[idx] < tol:
            return float(self.x_data[idx]), float(self.y_data[idx])
        return None


def load_vle_from_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with path.open("r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not {"x", "y"}.issubset(reader.fieldnames or {}):
            raise ValueError("CSV 必须至少包含列 x 与 y")
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    return np.asarray(xs), np.asarray(ys)


def load_vle_from_excel(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - 可选依赖
        raise RuntimeError("读取 Excel 需安装 pandas，请改用 CSV 或安装 pandas") from exc

    df = pd.read_excel(path)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("Excel 表中需包含列 x 与 y")
    return df["x"].to_numpy(float), df["y"].to_numpy(float)


def build_model(mode: str, *, alpha: Optional[float] = None, file: Optional[Path] = None, description: str = "") -> VLEModel:
    """根据给定参数创建 VLE 模型。"""

    mode = mode.lower()
    if mode == "constant_alpha":
        if alpha is None:
            raise ValueError("常数 α 模式必须提供 alpha")
        return VLEModel(mode=mode, alpha=alpha, description=description)

    if mode == "table":
        if file is None:
            raise ValueError("表格模式必须提供文件路径")
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"未找到 VLE 数据文件: {path}")
        if path.suffix.lower() in {".csv", ".txt"}:
            x_data, y_data = load_vle_from_csv(path)
        elif path.suffix.lower() in {".xls", ".xlsx"}:
            x_data, y_data = load_vle_from_excel(path)
        else:
            raise ValueError("仅支持 csv/txt/xls/xlsx 格式的 VLE 文件")
        sort_idx = np.argsort(x_data)
        x_sorted = x_data[sort_idx]
        y_sorted = y_data[sort_idx]
        return VLEModel(mode=mode, x_data=x_sorted, y_data=y_sorted, description=description)

    raise ValueError(f"未知 VLE 模式: {mode}")


def list_presets() -> Dict[str, str]:
    """返回内置 VLE 数据集的描述."""

    return {name: str(info["description"]) for name, info in _PRESET_TABLES.items()}


def build_preset_model(name: str) -> VLEModel:
    """基于内置数据库创建 VLE 模型."""

    try:
        info = _PRESET_TABLES[name]
    except KeyError as exc:
        raise ValueError(f"未找到名为 {name!r} 的 VLE 预设") from exc

    file_path = Path(info["file"])
    if not file_path.exists():
        raise FileNotFoundError(f"VLE 数据文件缺失: {file_path}")

    x_data, y_data = load_vle_from_csv(file_path)
    sort_idx = np.argsort(x_data)
    x_sorted = x_data[sort_idx]
    y_sorted = y_data[sort_idx]
    return VLEModel(
        mode="table",
        x_data=x_sorted,
        y_data=y_sorted,
        description=str(info["description"]),
    )


def enrich_curve(model: VLEModel, points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """生成平衡曲线点集，便于绘图。"""

    xs = np.linspace(0.0, 1.0, points)
    ys = np.array([model.y(x) for x in xs])
    return xs, ys


def ensure_iterable(data: Iterable[float]) -> np.ndarray:
    return np.asarray(list(data), dtype=float)

