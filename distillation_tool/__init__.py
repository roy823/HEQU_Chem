"""化工原理精馏解题工具包。

该包聚合了典型考试题所需的关键计算模块，包括：

- 参数与案例管理
- VLE 数据处理
- 物料衡算
- 理论级与操作线计算
- 塔效率校正
- 塔径估算
- 可视化输出

主入口请使用 ``distillation_tool.app`` 中的 ``run_app`` 函数。
"""

from . import (
    cases,
    material_balance,
    stages,
    efficiency,
    diameter,
    visualization,
    vle,
    flash,
    absorption,
    app,
    web_app,
    absorption_app,
)

__all__ = [
    "cases",
    "material_balance",
    "stages",
    "efficiency",
    "diameter",
    "visualization",
    "vle",
    "flash",
    "absorption",
    "app",
    "web_app",
    "absorption_app",
]
