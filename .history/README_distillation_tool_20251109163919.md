# 精馏解题自动化程序使用指南

BY——荷城合取

本工具针对化工原理课程的典型精馏与萃取-精馏组合题（甲醇-水、正丁醇-萃取剂 S、甲酸-甲酸乙酯等）提供从数据输入、物料衡算、理论级计算到图形输出的全流程求解。

## 功能概览

- **案例库**：内置作业与期中试题的参数，`--case <id>` 一键求解。
- **自定义输入**：支持任意二元体系，进料/塔顶/塔釜流量可任意缺省（系统自动补算），VLE 可选常数 α、内置预设或 CSV/Excel 表格。
- **VLE 数据库**：内置 `methanol_water_101kpa`（甲醇-水 101.3 kPa 附表 1 数据），一键复现课堂/附表图线。
- **物料衡算**：自动输出 F/D/W 及萃取剂需求，生成课程同款物流表。
- **理论级计算**：芬斯克最少板、吉利兰关联、McCabe-Thiele 逐板图全覆盖。
- **萃取-精馏联动**：提供单级萃取与逆流多级（Kremser）萃取模块，并将萃取相直接作为后续精馏进料。
- **效率与尺寸**：全塔板效率、气/液相默弗里效率、空塔气速估算塔径。
- **可视化**：平衡/操作线阶梯图、流程框图、物流表 PNG，一键嵌入报告。
- **单级平衡/闪蒸**：内置杠杆法与 Rachford-Rice 辅助函数，方便期中附加题。

## 快速开始

```powershell
python -c "import sys; sys.path.append(r'D:/<路径>'); from distillation_tool import app; app.run_app(['--case','methanol_water_basic','--out-dir','output'])"
```

```powershell
# 仅给定 D/W，自动补算 F，并使用内置甲醇-水数据
python -c "import sys; sys.path.append(r'D:/<路径>'); from distillation_tool import app; app.run_app([
    '--custom',
    '--distillate-flow','32',
    '--bottoms-flow','48',
    '--feed-x','0.45',
    '--distillate-x','0.92',
    '--bottoms-x','0.04',
    '--reflux','0.9',
    '--q','1.0',
    '--vle-preset','methanol_water_101kpa'
])"
```

> Windows 建议使用目录的 8.3 短路径（`cmd /c "dir /x"`）以避免中文/空格导致的导入问题。

运行后会得到：

- 物料衡算与回收率检查
- 芬斯克最少板、Rmin、Gilliland 理论板数
- McCabe-Thiele 阶梯图可视化
- （可选）效率校正、塔径估算
- `output/` 目录下的 `mccabe_thiele.png` / `stream_table.png` / `process_flow.png`

## 自定义示例（逆流萃取 + 精馏）

```powershell
python -c "import sys; sys.path.append(r'D:/<路径>'); from distillation_tool import app; app.run_app([
    '--custom',
    '--feed-flow','80',
    '--feed-x','0.18',
    '--distillate-x','0.05',
    '--bottoms-x','0.70',
    '--reflux','1.5',
    '--q','1',
    '--vle-mode','constant_alpha',
    '--alpha','4.0',
    '--extraction-mode','countercurrent',
    '--extract-k','1.0',
    '--extract-recovery','0.92',
    '--extract-solute-role','heavy',
    '--extract-solvent-ratio','1.5',
    '--extract-max-stages','12',
    '--overall-eff','0.85',
    '--vapor-vel','0.9',
    '--vapor-temp','360',
    '--vapor-press','101.3'
])"
```

常用萃取参数：

- `--extraction-mode single|countercurrent`
- `--extract-k <K>`：线性分配系数。
- `--extract-recovery <0-1>`：目标回收率（逆流模式下可用于级数搜索）。
- `--extract-stages <N>`：直接指定逆流理论级数。
- `--extract-solvent-flow / --extract-solvent-ratio`：溶剂流量或溶剂/进料比。
- `--extract-max-stages <N>`：逆流搜索上限。
- `--extract-solute-role light|heavy`：指定溶质在后续精馏中是轻键（塔顶）还是重键（塔釜）。

> 小贴士：`--feed-flow` 可省略，只要给出 `--distillate-flow` 与 `--bottoms-flow`（或任意两股流量）即可满足物料守恒；无需上传文件时，可通过 `--vle-preset <id>` 直接引用内置 VLE 数据。

## Web 界面

1. 安装依赖：`pip install streamlit matplotlib numpy pandas openpyxl`
2. 启动：`streamlit run distillation_tool/web_app.py`
3. 在浏览器中选择“预设案例 / 自定义精馏 / 单级平衡”，即可交互调整参数、上传 VLE 表、切换萃取模式并导出图形。

## 单级平衡/闪蒸模块

- 输入平衡点 `(x_eq, y_eq)` 即可计算汽液流量、汽化分率。
- 支持两股流混合后闪蒸，与期中“冷凝-再接触”题型一致。
- `flash.rachford_rice` 可扩展到多组分 K 值求解。

## 扩展与定制

- 新案例：`distillation_tool.cases.register_case` 运行期内动态扩展。
- 萃取模型：`material_balance.solve_single_stage_extraction` 与 `solve_countercurrent_extraction` 已抽象，便于替换更复杂的关联。
- 可视化：`visualization.*` 函数返回 `matplotlib` Figure，可嵌入 Jupyter/报告。

## 常见问题

- **ModuleNotFoundError**：确认项目根目录已加入 `sys.path` 或设置 `PYTHONPATH`。
- **中文路径报错**：使用目录短名或将仓库移动到无中文路径的位置。
- **VLE 表导入失败**：CSV 需包含 `x,y` 表头；Excel 需安装 `pandas + openpyxl`。
- **Streamlit 无响应**：检查端口占用，可用 `--server.port 8502` 更换端口。
- **逆流萃取未收敛**：提高 `--extract-max-stages` 或增大溶剂流量/比值。

如需多级萃取耦合能耗分析、塔径更精细设计等，可在现有模块基础上二次开发。
