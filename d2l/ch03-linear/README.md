# Ch03 — 线性神经网络 (Linear Neural Networks)

> d2l.ai 第 3 章 · PyTorch 从零实现 · 2026-06-28 开始

---

## 3.1 线性回归 (Linear Regression)

### 什么是线性回归

线性回归是深度学习的 "Hello World" —— 用一个线性模型预测连续值。虽然简单，但包含了**模型、损失、优化**这三个深度学习的核心组件。

### 模型

$$
\hat{y} = X w + b
$$

- $X \in \mathbb{R}^{n \times d}$ — n 个样本，每个 d 个特征
- $w \in \mathbb{R}^d$ — 权重向量（每个特征的重要性）
- $b \in \mathbb{R}$ — 偏置（截距，无输入时的基础值）

**为什么叫 "线性"？** 因为 $\hat{y}$ 是 $w$ 和 $b$ 的线性函数——没有非线性激活，没有层叠。

### 损失函数：均方误差 (MSE)

$$
L(w, b) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})^2
$$

- 为什么平方？① 误差正负抵消问题（$+3-3 \neq 0$）；② 大误差惩罚更重（平方放大）；③ 处处可导
- 为什么除以 n？归一化到单样本平均损失，不同 batch size 可比较

### 解析解 (Analytical Solution)

线性回归有闭式解——这是它和深度学习最本质的区别：

$$
w^* = (X^\top X)^{-1} X^\top y
$$

- 来源：L 对 w 求导 = 0，解出 w
- **能求解析解的条件：** $X^\top X$ 可逆（特征不完美线性相关）
- **深度学习为什么不用解析解？** 神经网络是非线性的，没有闭式解；必须用梯度下降

### 梯度下降 (Gradient Descent)

$$
w \leftarrow w - \eta \cdot \nabla_w L(w, b)
$$

- $\eta$ = 学习率：控制每一步多大
- $\nabla_w L$ = w 的梯度（MSE 对 w 的偏导）

**直觉：** 把 loss 想象成山谷，你在山顶，每次沿最陡方向下坡一步。步长 = lr，方向 = 梯度。

### 三种梯度下降变体

| 类型 | batch 大小 | 优点 | 缺点 |
|------|-----------|------|------|
| 批量梯度下降 | n（全部） | 梯度准确，稳定收敛 | 大数据集太慢 |
| 随机梯度下降 (SGD) | 1 | 快，可跳出局部极小 | 梯度震荡大，收敛不稳 |
| **小批量 SGD** | 32~256 | ⭐ 平衡速度和稳定性 | 需调 batch size |

**小批量 SGD 是深度学习的默认选择。**

### 学习率的关键性

- 太大 → loss 振荡/发散
- 太小 → 收敛慢
- 合适 → loss 稳定下降

---

## 3.2 线性回归的从零开始实现

> 只用 tensor 和 autograd，不调用 nn.Linear、nn.MSELoss 等高层 API。目的是理解底层在发生什么。

### 3.2.1 生成数据集 (Generate Dataset)

```python
y = X @ w + b + noise
```

- `X ~ N(0, 1)`, shape `(n, d)` — 从标准正态采样特征
- `w = [2, -3.4]`, `b = 4.2` — 真实参数（后面要学回来）
- `noise ~ N(0, 0.01)` — 观测噪声（真实世界数据永远不完美）

**关键认知：** 生成数据时知道真实参数——这在真实场景不可能。这里的目的是**验证**：如果算法正确，学出来的 w 和 b 应该接近真实的 w 和 b。

### 3.2.2 读取数据集 (Data Iterator)

```python
def data_iter(batch_size, features, labels):
    # 随机打乱索引，每次 yield 一个 batch
    indices = torch.randperm(n)
    for i in range(0, n, batch_size):
        yield features[batch_indices], labels[batch_indices]
```

- `yield` 生成器 — 惰性返回，内存在 O(batch)，不是 O(n)
- `randperm` — 随机排列，每 epoch 不同顺序
- **为什么不用 Python list 切片？** `yield` 内存效率：100 万样本，每次只在内存里放 32 个

### 3.2.3 初始化模型参数

```python
w = torch.normal(0, 0.01, size=(d, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

- `w` 初始化为小随机值 — 打破对称性（不然所有权重梯度相同）
- `b` 初始化为 0 — 偏置不需要对称性打破
- **`requires_grad=True`** — 告诉 autograd 追踪这两个 tensor，训练时才能求梯度

### 3.2.4 定义模型

```python
def linreg(X, w, b):
    return X @ w + b       # (n, d) @ (d, 1) + (1,) → (n, 1)
```

- 这是整个"神经网络"：一行矩阵乘法 + 加法
- **`@` 是矩阵乘法**，`*` 是逐元素乘 — 又见面了

### 3.2.5 定义损失函数

```python
def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape)) ** 2) / 2
```

- 除以 2 不是笔误！是技巧：求导后 $\frac{d}{dx}\frac{1}{2}x^2 = x$，常数抵消
- 实际返回的是向量（每个样本的 loss），训练时取 mean
- `y.reshape(y_hat.shape)` — 防御性形状对齐

### 3.2.6 定义优化算法 (SGD)

```python
def sgd(params, lr, batch_size):
    with torch.no_grad():              # 参数更新不参与计算图
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()          # 清零！不然下次 backward 累加
```

- **`with torch.no_grad()`** — 权重更新不是前向计算，不需要被追踪
- **`param.grad / batch_size`** — loss 是 sum（没取 mean），所以梯度除以 batch_size
- **`param.grad.zero_()`** — Ch02 学的 grad 累加机制在这里发挥作用

### 3.2.7 训练 (Training Loop)

```python
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = squared_loss(linreg(X, w, b), y)   # 前向 + 损失
        l.sum().backward()                       # 反向传播（sum 把向量变标量）
        sgd([w, b], lr, batch_size)             # 更新参数
    train_l = squared_loss(linreg(features, w, b), labels).mean()
    print(f'epoch {epoch+1}, loss {train_l:f}')
```

**训练循环 = Ch02 的所有知识汇聚：**

```
zero_grad (在 sgd 内) → forward → loss → backward → step (sgd 内)
```

### 运行结果

```
epoch 1, loss 0.032633
epoch 2, loss 0.000115
epoch 3, loss 0.000050

w 真实值: [2.0, -3.4]    学到的 w: [[ 1.9998], [-3.3996]]
b 真实值: 4.2              学到的 b: [4.1998]
```

- 3 epoch 就从随机初始值学到了几乎完美的参数
- 参数误差在 $10^{-4}$ 量级
- 为什么这么准？① 数据是线性生成的；② 模型也是线性的；③ 没有过拟合

---

## 贯穿 Ch03 的收获

1. **深度学习三组件：模型 → 损失 → 优化** — 所有深度学习问题都是这个框架
2. **解析解 vs 梯度下降** — 线性回归是唯一有解析解的"神经网络"；一旦加一层非线性，只能用梯度下降
3. **`/2` 不是魔法** — loss 平方除以 2 是为了求导后常数消失，这种"为了梯度而设计"的思路贯穿整个深度学习
4. **`no_grad()` 块** — 权重更新不是计算图的一部分，PyTorch 不会帮你区分，必须显式声明
5. **小批量 SGD** — 不是"数据太多算不动"的妥协，而是 batch size 本身就是超参数，影响泛化和收敛速度
6. **从零实现的价值** — `nn.Linear` 一行搞定的事拆成 7 步，每一步都看清楚了：数据怎么来的、参数怎么初始化的、梯度怎么传的、参数怎么更新的
7. **`yield` 的设计** — Python 生成器的惰性求值，是大数据深度学习的必备工程技能

---

## 代码文件

| 文件 | 内容 |
|------|------|
| `01_linear_regression_theory.py` | 3.1 解析解 + 梯度下降一维可视化 + loss 分解 |
| `02_scratch_linear_regression.py` | 3.2 完整从零实现 + 真实参数 vs 学习参数对比图 |
