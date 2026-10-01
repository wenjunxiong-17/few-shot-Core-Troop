# 半监督信号该放在哪里？
### MAML 内循环 vs. 适应之后 —— 机理分析与受控实验

<p align="left">
  <img alt="deps" src="https://img.shields.io/badge/dependencies-numpy%20only-green">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

**研究问题**：把半监督损失放进 MAML 的 inner loop，能不能提升小样本性能？
如果不能，应该放在哪里？

**一句话结论**：框架上放得进去，但会掉点；放到**适应之后**才有效，
并在真实时空数据上把 MSE 降低了 **11.8%–24.1%**。

---

## 主要结论

### 1. 内层 SSL 一致有害

同一 MAML 主干（合成 5-way 1-shot，准确率 %）：

| 无标签用法 | 准确率 | 变化 |
|---|---|---|
| 不用（纯监督适应） | **75.86** | — |
| 内层 + 熵最小化 | 63.05 | **−12.81** |
| 内层 + 原型蒸馏 | 71.64 | **−4.22** |
| 内层 + LP 蒸馏 | 65.96 | **−9.90** |
| **适应之后 + 标签传播** | **82.64** | **+6.78** |

### 2. 不是调参问题：`(α, m)` 扫描

内循环熵最小化的 Δ准确率（百分点，正值 = 内层 SSL 更好）：

| | m=1 | m=3 | m=8 |
|---|---|---|---|
| α=0.05 | +0.83 | +0.40 | −0.68 |
| α=0.1 | +0.19 | −0.09 | −4.35 |
| α=0.2 | −0.09 | **−12.36** | **−16.69** |
| α=0.4 | −0.17 | **−12.81** | **−23.33** |
| α=0.8 | −0.28 | −7.28 | −17.68 |

**只有「死区」和「断崖」，没有「先升后降」。** 说明问题在**方向**，不在**力度**。
`α·m ≲ 0.2` 时 |Δ| ≤ 0.9 个百分点；越过就雪崩。

**对照实验**（同一张表里的 `frozen` 列）：把一致性项的教师**冻结在 $\theta_0$**，
则 $m=1$ 时它与纯支持集 CE **逐位完全相同**（69.74/69.74、75.78/75.78、77.31/77.31、
77.39/77.39、76.41/76.41）——因为在 $\theta_0$ 处损失和梯度都恰为 0。
这直接证明「内层 SSL 拿到的信号主要来自共享初始化 $\theta_0$」。

原始数据：`results/exp1b_alpha_sweep.json`，每个 `rows` 条目直接给出
`sup` / `ent` / `frozen` 三个准确率，上表就是 `100*(ent - sup)`。

### 3. 机理

内层只走 1–5 步，所以 $\theta' = \theta_0 + O(\alpha m)$。
伪标签由 $\theta'$ 产生，而 $\theta_0$ 是**跨任务共享**的
→ 伪标签里**任务特异的部分只有 $O(\alpha m)$**
→ 这一项实际退化成「对初始化的正则」，而不是「每个任务各自的半监督适应」。

**证据**：同一个熵最小化项，放内层 **−12.81**，放外层（元正则）**+3.58**
（`results/exp1_synth.json`：`ssl_ent` 主干下的纯监督适应 79.44 vs
纯 MAML 主干 75.86）。

### 4. 一个可证明的理论结果

对 softmax 模型，$D_{KL}(p(y|x)\,\|\,p(y|x{+}r))$ 在 $r=0$ 处的 Hessian 满足

```
rank(H) = min(I, C − 1)
```

- **二分类**（C=2）→ 秩 1，VAT 精确退化为**无监督 FGSM**
- **标量回归** → 秩 1

数值验证：`verify/verify_vat_rank1.py`、`verify/verify_vat_regression.py`
（**不需要任何数据**，随机初始化即可，因为该恒等式对 $x,\theta$ 逐点成立）

```
 C  rank(H) FD  min(I,C-1)      # 分类
 2         [1]           1
 3         [2]           2
 5         [4]           4
 9         [8]           8
```

- 有限差分 Hessian 与闭式解 $H=\sum_c p_c (J_c-\bar J)(J_c-\bar J)^\top$
  的相对误差 **≤ 4.5e−8**；
- 二分类：rank 恰为 1，特征向量与梯度 $|\cos| = \mathbf{1.000000}$，
  $\lambda_1$ 相对误差 ≤ 2.7e−8；
- 标量回归：同样 rank 1，$|\cos| = \mathbf{1.000000}$，相对误差 ~1e−12；
- **一步 VAT 方向 vs 最大特征向量**：C=2 时 $|\cos| = 0.999976$（幂迭代确实是多余的），
  C=3/5/9 时掉到 0.91/0.91/0.84（方向真的不确定了）。

→ **VAT 只在输出维度 ≥ 2 时才真正有用**：多分类 $C\ge3$（rank $=\min(I,C-1)$）
或多维回归 $D\ge2$（rank $=\min(I,D)$）。二分类和标量回归下它没有可找的方向。

### 5. 真实数据（USGS，37 站 × 705 天 × 11 指标，次日 pH）

| 设定 | 纯监督 MSE | Pseudo-Label MSE | 配对 ΔMSE | 相对 |
|---|---|---|---|---|
| 随机划分 K=10 | 0.378 | **0.299** | −0.0792 ± 0.0165 | **−20.9%** |
| 随机划分 K=30 | 0.387 | **0.294** | −0.0935 ± 0.0459 | **−24.1%** |
| 空间划分 K=30 | 0.283 | **0.250** | −0.0333 ± 0.0066 | **−11.8%** |

共比较四种半监督项（都写成回归形式）：

| 方法 | 源域网格搜索选出的 λ |
|---|---|
| **Pseudo-Label**（滞后 EMA 教师） | **1.0**（唯一有效） |
| **FixMatch 的回归对应形式**（`pi_reg`：弱/强视图一致性；置信阈值在回归里没有对应物，去掉后即 Π-model） | 0（λ∈{0.005,0.02,0.1} 全部变差，最多 +50%） |
| **VAT**（虚拟对抗扰动下的一致性） | 0（同样全部变差，最多 +74%） |
| 图 Laplacian 平滑（流形正则） | 0.1（微小、不稳定） |

**三个设定里被网格搜索选中的 λ 完全一致**，所以这不是调参运气。

> **工程细节**：Pseudo-Label 直接用在**回归**上梯度**恒等于 0**（目标就是预测值本身）。
> 必须改成**滞后 EMA 教师**（Mean Teacher）才有效。这一点在分类里被 argmax 掩盖了。

---

## 包内容（10 个顶层条目）

```
├── README.md                     本文件
├── LICENSE                       MIT
├── requirements.txt              只有 numpy
├── .gitignore                    排除数据 / 论文 PDF / 缓存
├── DATA.md                       两个数据集怎么拿（本仓库不含数据）
│
├── ml/                          核心库（纯 NumPy，无框架依赖）
│   ├── core.py                  MLP 前反向、inner_adapt、MAML 二阶元梯度（有限差分 HVP）
│   ├── ssl.py                   6 种半监督项
│   └── tasks.py                 任务采样器（含噪声维度与任务特异性）
│
├── experiments/                 关键实验
│   ├── exp1_synth.py            ★ 三种注入位置对比（合成，无需数据）
│   ├── exp1b_alpha.py           ★ (α, m) 扫描，机理检验（合成，无需数据）
│   ├── exp2_water.py            真实水数据集的元学习 + 三段式
│   ├── exp5_water_ssl.py        ★ 四种 SSL 方法在真实数据上的回归形式对比
│   │                              （Pseudo-Label / Π 一致性 / VAT / 流形正则）
│   └── matio.py                 纯 Python 的 MATLAB .mat 读取器
│
├── verify/                      正确性与理论验证
│   ├── test_maml_grad.py        ★ 元梯度有限差分校验（相对误差 1.8e−5）
│   ├── verify_vat_rank1.py      ★ VAT 秩定理（分类 C=2/3/5/9 → rank 1/2/4/8）
│   └── verify_vat_regression.py VAT 秩定理（标量回归）
│
├── docs/                        报告（6 份核心 + 3 张图）
│   ├── 组会汇报材料_完整详解版.md          ★ 最完整（11 章）
│   ├── 结论推导链_为什么半监督不该进inner_loop.md  ★ 论证过程
│   ├── 三种SSL方法对比与结论.md            ★ 含 VAT 秩定理证明
│   ├── 真实水数据集的SSL方法选择.md
│   ├── 研究报告.md                      机理全文 M1–M9
│   ├── 纯半监督训练结果与数据真实性评估.md
│   └── figures/*.svg
│
└── results/                     实验结果 JSON 与表格
```

> 报告里偶尔出现的 `reports/*.txt`（训练日志、数据诊断）**没有随包分发**，
> 它们是运行过程中的中间产物；其余引用（`results/*.json`、`docs/*.md`、
> `docs/figures/*.svg`）都已改写到本仓库的实际路径。

---

## 快速开始

```bash
pip install -r requirements.txt      # 只有 numpy

# 1) 先验证元梯度实现是对的（有限差分校验，相对误差应 ~1e-5）
python verify/test_maml_grad.py

# 2) 跑实验一：三种注入位置对比（合成数据，无需外部数据）
python experiments/exp1_synth.py --out results/exp1_synth.json
#    约 25 分钟。加 --quick 只用来确认代码能跑通（几分钟）；
#    元训练迭代被大幅削减，"适应后涨点" 的幅度出不来，
#    正式数字请直接看随包提供的 results/exp1_synth.json

# 3) 跑 (α, m) 扫描
python experiments/exp1b_alpha.py
```

**依赖**：只需要 `numpy`。元学习、二阶梯度、半监督损失、优化器、`.mat` 读取
全部手写实现，不依赖 PyTorch / TensorFlow / scipy / scikit-learn。

---

## 数据

**本仓库不包含数据集**（版权原因）。两个数据集的获取方式、目标路径、
预处理脚本、以及使用注意事项都写在 **`DATA.md`** 里：

| 数据 | 来源 |
|---|---|
| Water Potability | Kaggle 公开数据集 |
| USGS 水数据集 | Zhao, Gkountouna, Pfoser, ACM TSAS 5(3), 2019 |

拿到后放到：

```
data/
├── water_dataset.mat        # 37 站 × 705 天 × 11 指标
└── ssl/{complete,missing_20pct}.csv
```

---

## 统计协议

- **配对差**：所有增益都是同一任务、同一随机种子下 `method − baseline`，
  报告均值 ± 标准误。
- **源域选参**：所有半监督权重都在**源域任务**上网格搜索，**完全不碰目标域标签**。

---

## 引用

```bibtex
@misc{fewshot-ssl-maml,
  title  = {Where should the semi-supervised signal go? Inner-loop vs. post-adaptation in MAML},
  author = {wenjunxiong-17},
  year   = {2025},
  url    = {https://github.com/wenjunxiong-17/few-shot-Core-Troop}
}
```

**引用的论文**（本仓库只引用、**不分发** PDF 或全文）：

1. Finn, Abbeel, Levine. *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML 2017.
2. Sohn et al. *FixMatch.* NeurIPS 2020.
3. Lee. *Pseudo-Label.* ICML 2013 Workshop.
4. Miyato et al. *Virtual Adversarial Training.* IEEE TPAMI 2019.
5. Zhu. *Semi-Supervised Learning Literature Survey.* UW-Madison TR 1530, 2008.
6. Ren et al. *Meta-Learning for Semi-Supervised Few-Shot Classification.* ICLR 2018.
7. Zhao, Gkountouna, Pfoser. *ACM TSAS* 5(3), 2019.

**许可**：代码 MIT（见 `LICENSE`）。数据集与论文版权归各自作者，请遵守其条款。
