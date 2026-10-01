# 数据获取说明

本仓库**不包含任何数据集**。请按下面的说明自行获取，放到指定路径即可。

预期目录结构：

```
data/
├── water_dataset.mat                      # 真实时空水数据集
├── README.docx                            # 该数据集的原始说明（可选）
└── ssl/
    ├── complete.csv                       # 公开表格数据（完整标签）
    └── missing_20pct.csv                  # 在 complete.csv 基础上加两列
```

`data/` 已在 `.gitignore` 中，不会被提交。

---

## 1. Water Potability（表格二分类）

**用途**：`exp3_ssl_tabular.py`、`exp4_pure_ssl.py`
**规模**：3276 行 × 9 个已 z-score 化的理化指标，二分类，正例率 39.0%

### 获取

Kaggle 上的 *Water Potability* 数据集（关键词 `water potability`）。
下载后得到原始的 `water_potability.csv`（含缺失值）。

### 本仓库期望的格式

`complete.csv` 应具备以下列：

| 列名 | 说明 |
|---|---|
| `row_id` | 行号 |
| `split` | `train` / `validation` / `test` |
| `z_ph`, `z_Hardness`, `z_Solids`, `z_Chloramines`, `z_Sulfate`, `z_Conductivity`, `z_Organic_carbon`, `z_Trihalomethanes`, `z_Turbidity` | 9 个标准化后的特征 |
| `label_true` | 0/1 标签 |

划分建议：`train` 2293 / `validation` 491 / `test` 492。

### 生成 `missing_20pct.csv`

在 `complete.csv` 基础上**只对 train 部分随机移除 20% 标签**，追加两列：

| 列名 | 说明 |
|---|---|
| `label_partial` | 被观察到的标签值，未观察到为空 |
| `label_observed` | 0/1，是否被观察到 |

参考实现（与仓库中的实验一致）：

```python
import numpy as np, pandas as pd
df = pd.read_csv("data/ssl/complete.csv")
rng = np.random.default_rng(0)
obs = np.ones(len(df), dtype=int)
tr = (df["split"] == "train").values
obs[tr] = (rng.random(tr.sum()) >= 0.20).astype(int)
df["label_observed"] = obs
df["label_partial"] = np.where(obs == 1, df["label_true"], np.nan)
df.to_csv("data/ssl/missing_20pct.csv", index=False)
```

### 缺失机制核验

```bash
python analysis/verify_missing.py
```

会输出：train/val/test 各自的观察率、观察组与未观察组的正例率（应当接近，
说明缺失与标签独立）、以及一个「用特征预测是否被观察」的 MCAR 探针 AUC。

### 关于这份数据的一个提醒

`analysis/diag_dataset_reality.py` 发现这份数据在统计上**不具真实场景特性**：

- 9 个物理上强耦合的指标几乎互不相关（平均 \|r\| = 0.029，最大 0.150）
- 9 列全部近似独立高斯（偏度 −0.09~0.62，超额峰度 −0.28~1.78）
- 标签近乎随机：模型可把训练标签拟合到 **98.1%**，但测试 AUC 上限只有 **0.612**
  （置换标签对照 0.504）
- 没有站点列、没有时间列

因此它更适合作为**负面对照**（「当标签近乎随机时，所有 SSL 方法都无效」），
而不是「真实场景」的主证据。详见 `docs/纯半监督训练结果与数据真实性评估.md`。

---

## 2. USGS 水数据集（真实时空回归）

**用途**：`exp2_water.py`、`exp5_water_ssl.py`、`analysis/explore_data.py`
**规模**：37 个监测站 × 705 天 × 11 个水质指标，目标是**次日 pH**（连续回归）

### 来源

> Liang Zhao, Olga Gkountouna, Dieter Pfoser.
> *Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial
> Topological Constraints.*
> **ACM Transactions on Spatial Algorithms and Systems (TSAS)**, 5(3), Article 19, 2019.
> DOI: [10.1145/3339823](https://doi.org/10.1145/3339823)

数据源自 United States Geological Survey (USGS) 的地表水监测记录，
经上述论文整理为 `water_dataset.mat`。

### 预期变量（MATLAB v5 `.mat`）

| 变量 | 类型 / 尺寸 | 说明 |
|---|---|---|
| `features` | cell 1×11 | 11 个水质指标的名单 |
| `location_ids` | 37×1 | 监测站 ID |
| `location_group` | cell 1×3 | 站点空间分组（含一个单点组与两个连通水系统） |
| `X_tr` | cell 1×423 | 训练期每天一个 37×11 矩阵（站点 × 指标） |
| `X_te` | cell 1×282 | 测试期，同上 |
| `Y_tr` | 37×423 | 训练期 pH 目标 |
| `Y_te` | 37×282 | 测试期 pH 目标 |

放到 `data/water_dataset.mat` 即可。**本仓库自带纯 Python 的 `.mat` 读取器**
（`matio.py`，不依赖 scipy）：

```bash
python matio.py data/water_dataset.mat     # 打印变量名、形状与取值范围
python analysis/explore_data.py            # 完整数据诊断
python analysis/explore_align.py           # 时间对齐与可学习性诊断
```

### 使用该数据时的三条注意事项

来自 `analysis/explore_align.py` 与 `docs/研究报告.md`：

1. **不能按 K 个锚点做中心化**——协变量预测的正是站点的基线水平，
   按支持集均值中心化会把这个唯一可预测的信号删掉
   （池化岭回归：中心化 0.628 vs 不中心化 0.337）。
2. **站点间水平差异极大**（标准化后均值范围 −1.44 ~ +2.30），
   因此必须报告**逐站点的配对差**，而不是池化 MSE。
3. **标签传播在这份数据上会过平滑**（K=30 个锚点只占 704 天的 4.3%），
   无标签数据的正确用法是**自训练**，不是图传播。

---

## 3. 引用要求

使用上述任一数据集时，请引用其原始出处（见上文）。本仓库的代码许可是 MIT，
但**数据集许可独立于本仓库**，请遵守各数据集自己的条款。
