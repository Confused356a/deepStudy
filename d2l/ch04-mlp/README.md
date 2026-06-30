# Ch04 — 多层感知机 (Multilayer Perceptron, MLP)

> 对应 d2l 第 4 章  |  从感知机到深层全连接网络  |  2026-06-30

---

## 核心思想

单层感知机 = 线性分类器, 无法解决 XOR。**加隐藏层 + 非线性激活函数 → MLP**, 突破线性限制。

```
感知机:   y = sign(w·x + b)                    ← 线性, 只能分线性可分的类
MLP:     h = phi(W1·x + b1)                   ← 隐藏层 + 激活 (非线性)
         y = softmax(W2·h + b2)               ← 输出层
```

隐藏层将数据映射到新的空间, 使得原本线性不可分的数据变得线性可分 — 这是所有深度学习的基础。

---

## 章节文件

| # | 文件 | d2l 小节 | 内容 | 关键产出 |
|---|------|----------|------|----------|
| 01 | `01_perceptron.py` | 4.1 感知机 | 感知机模型、XOR 问题、MLP 解决方案 | 决策边界可视化、XOR 空间变换 |
| 02 | `02_activation.py` | 4.1 激活函数 | ReLU/Sigmoid/Tanh 对比、梯度消失演示 | 函数+梯度图、10 层梯度衰减 |
| 03 | `03_mlp_scratch.py` | 4.2 从零实现 | MLP 从零训练 Fashion-MNIST | ~85% 准确率 vs 线性 83% |
| 04 | `04_mlp_concise.py` | 4.3 简洁实现 | nn.Sequential 一行模型 | 从零 vs 框架对照表 |
| 05 | `05_underfit_overfit.py` | 4.4 模型选择 | 多项式回归、欠拟合/过拟合 | 1~20 阶误差曲线 + 泛化差距 |
| 06 | `06_weight_decay.py` | 4.5 权重衰减 | L2 正则化、weight decay 对比 | p>>n 实验、MLP 提升 |
| 07 | `07_dropout.py` | 4.6 Dropout | 从零实现 Dropout、nn.Dropout | 小数据训练对比 p=0/0.3/0.5 |
| 08 | `08_numerical_stability.py` | 4.7-4.8 数值稳定性 | 梯度消失/爆炸、Xavier/He 初始化 | 方差传播曲线、训练对比 |

---

## 核心公式

### 前向传播
```
h = phi(W^(1) @ x + b^(1))    隐藏层 (含激活函数)
o = W^(2) @ h + b^(2)         输出层 (logits)
y_hat = softmax(o)            概率输出
```

### 激活函数
```
ReLU(z)    = max(0, z)           ReLU'(z) = 1 if z>0 else 0
Sigmoid(z) = 1/(1+e^{-z})        Sigmoid'(z) = sig(z)(1-sig(z))  ∈ (0, 0.25]
Tanh(z)    = (e^z-e^{-z})/(e^z+e^{-z})   Tanh'(z) = 1-tanh^2(z)  ∈ (0, 1]
```

### 正则化
```
L2 正则:    L_total(w) = L(w) + (λ/2)·||w||²
Dropout:    h' = h ⊙ mask / (1-p),   mask_i ~ Bernoulli(1-p)
```

### 初始化
```
Xavier:  Var(w) = 2 / (fan_in + fan_out)    — 适合 sigmoid/tanh
He:      Var(w) = 2 / fan_in                — 适合 ReLU (补偿负半轴截断)
```

---

## 实验结果汇总

| 实验 | 数据集 | 关键指标 | 结论 |
|------|--------|----------|------|
| XOR 感知机 | 4 点 XOR | Loss 永不收敛 | 线性模型有根本局限 |
| XOR MLP | 4 点 XOR | 500 epoch 收敛 | 加一隐藏层→非线性决策边界 |
| 梯度消失 | 模拟 10 层 | Sigmoid 梯度 ~1e-8, ReLU ~1e-2 | ReLU 缓解梯度消失 |
| MLP 从零 | Fashion-MNIST | ~85.5% (vs linear 83%) | 非线性 + 更多参数有效 |
| MLP 框架 | Fashion-MNIST | ~85.5% | 从零和框架一致 |
| 欠拟合/过拟合 | 合成 sin | 1 阶 vs 20 阶, U 形误差 | 模型容量 vs 泛化的经典权衡 |
| 权重衰减 | 合成线性 | λ=0: 过拟合, λ=20: 正常 | p>>n 时 weight_decay 必不可少 |
| Dropout | Fashion-MNIST (5K) | p=0.3: 最好, p=0.5: 略差 | 数据少时 Dropout 提升明显 |
| 参数初始化 | Fashion-MNIST | He~85%, Xavier~85%, Random~82% | 好初始化让训练更稳定更快 |

---

## 工程踩坑

| 坑 | 表现 | 解决 |
|----|------|------|
| 深层 Sigmoid 梯度消失 | loss 不下降, 浅层权重不更新 | 换 ReLU + He 初始化 |
| 随机初始化方差太大/太小 | 激活值 → 0 或 → inf | 用 Xavier/He 初始化 |
| weight_decay 太大 | 欠拟合, 所有权重→0 | 从小的 λ (如 1e-4) 开始调 |
| Dropout 忘记 eval() | 测试时还在随机丢弃, 结果不稳定 | `model.eval()` 自动关闭 Dropout |
| 多项式阶数=数据量 | 完美拟合训练集但测试误差极大 | 训练/验证 split 监控泛化 |
| CrossEntropyLoss 前加 softmax | 双重 softmax, loss 不降 | `model` 最后一层不加 softmax |

---

## 从 Ch03 到 Ch04 的进阶

```
Ch03 (线性神经网络):           Ch04 (多层感知机):
  x → W·x+b → softmax           x → W1·x+b1 → ReLU → W2·h+b2 → softmax
  只有输出层                      多了隐藏层 + 非线性

  线性决策边界                    非线性决策边界
  83% on Fashion-MNIST           ~85.5% on Fashion-MNIST
  7.8K 参数                      ~203K 参数
  无正则化                       有 Dropout/Weight Decay
```

关键进步: 从线性到非线性, 从无正则到有正则, 从浅层到深层。但训练循环 (forward→loss→backward→step) 完全不变 — 这是深度学习框架抽象的价值。

---

## 导航

- 上一章: [Ch03 线性神经网络](../ch03-linear/README.md)
- 下一章: Ch05 深度学习计算 (待开始)
- 返回: [d2l 总览](../README.md)
