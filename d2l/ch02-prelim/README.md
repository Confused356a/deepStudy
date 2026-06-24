# Ch02 — 预备知识 (Preliminaries)

> d2l.ai 第 2 章 · PyTorch 实现 · 2026-06-24 开始

---

## 2.1 数据操作 (ndarray)

**Tensor = 深度学习的数据容器**，等价于 NumPy ndarray + GPU 支持。

### 创建 Tensor

| API | 说明 |
|-----|------|
| `torch.arange(n)` | 生成 [0, n) 整数序列 |
| `x.reshape(r, c)` | 改变形状（可能复制） |
| `x.reshape(-1, 4)` | -1 自动推断该维度 |
| `torch.zeros(shape)` | 全 0 |
| `torch.ones(shape)` | 全 1 |
| `torch.randn(shape)` | 标准正态 N(0,1) |
| `torch.tensor(list)` | 从 Python list 构造 |

### 运算符

- 逐元素运算：`+ - * / **` 全部是 element-wise
- `torch.exp(x)` — 指数
- **`*` 不是矩阵乘法！** 矩阵乘法用 `@` 或 `torch.mm()`

### 广播机制 (Broadcasting)

规则：从最后一维向前对齐，维度为 1 的轴自动复制扩展。
`(3,1) + (1,2) → (3,2)` — 自动复制到共同形状。

### 节省内存

```python
X += Y        # 原地加法，id 不变
Z[:] = X + Y  # 切片赋值，id 不变
```

### PyTorch ↔ NumPy

- `tensor.numpy()` — 共享底层内存！改一个另一个也变
- `torch.tensor(ndarray)` — 复制数据
- `tensor.item()` — 标量 tensor → Python 原生数字

---

## 2.2 数据预处理 (Data Preprocessing)

深度学习的第一步：把真实世界的脏数据变成 tensor。

### pandas 三件套

| 操作 | API | 作用 |
|------|-----|------|
| 读 CSV | `pd.read_csv(path)` | 加载表格数据 |
| 缺失值填充 | `df.fillna(df.mean())` | 数值列用均值替代 NaN |
| One-hot 编码 | `pd.get_dummies(df, dummy_na=True)` | 类别列 + NaN 转 one-hot 向量 |

### pandas → Tensor

```python
X = torch.tensor(df.values, dtype=torch.float32)
```

---

## 2.3 线性代数 (Linear Algebra)

### 张量阶数 & 深度学习

| 阶 | 形状 | 含义 |
|:--:|------|------|
| 0 | `()` | 标量（loss 值） |
| 1 | `(d,)` | 向量（bias） |
| 2 | `(m, n)` | 矩阵（权重 W） |
| 3 | `(B, T, D)` | batch × 序列 × hidden（Transformer） |
| 4 | `(B, C, H, W)` | batch × 通道 × 高 × 宽（CNN） |

### 聚合操作 — dim 参数是核心

- `A.sum(dim=0)` — 沿行方向压缩 → 每列求和
- `A.sum(dim=1)` — 沿列方向压缩 → 每行求和
- `keepdim=True` — 保持维度，方便广播除法

### 矩阵乘法

- **Hadamard 积** `A * B` — 逐元素乘（对应位置）
- **矩阵乘法** `A @ B` — 真正的线性代数乘法，形状：`(m,n) @ (n,k) → (m,k)`
- **点积** `torch.dot(x, y)` — 两向量 → 标量

### 范数

- L2: `torch.norm(x)` — 欧氏距离
- L1: `torch.abs(x).sum()` — 绝对值之和
- Frobenius: `torch.norm(M)` — 矩阵的 L2

---

## 2.4 微积分 (Calculus)

### 核心公式

**梯度** = 所有偏导组成的向量：∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]

**链式法则** = 反向传播的理论基础：dz/dx = dz/dy · dy/dx

**梯度下降**：w ← w − lr · ∇L(w)

### 在 PyTorch 中

```python
x = torch.tensor(2.0, requires_grad=True)
z = (x + 3) ** 2
z.backward()
x.grad  # = 2*(2+3) = 10 ✓  自动算出，无需手推公式
```

---

## 2.5 自动求导 (Autograd) ⭐

**整个深度学习框架的灵魂。**

### 三步走

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)  # ① 开启追踪
y = some_computation(x)
y.backward()                                          # ② 反向传播
print(x.grad)                                         # ③ 读取梯度
```

### 关键机制

| 概念 | 说明 |
|------|------|
| `requires_grad=True` | 追踪该 tensor 的计算图 |
| `.backward()` | 反向传播（只能对标量调用） |
| `.grad` | 存储梯度值（**会累加！**） |
| `.grad.zero_()` | 清零梯度（每次 backward 前必须做） |
| `.detach()` | 从计算图分离，共享数据 |
| `torch.no_grad():` | 上下文管理器，临时关闭梯度追踪 |

### ⚠️ 最大坑：grad 累加机制

```python
y.backward()   # x.grad = [4, 8, 12]
y.backward()   # x.grad = [8, 16, 24]  ← 累加了！不是期望的值
```

→ **这就是训练循环中 `optimizer.zero_grad()` 存在的理由！**

### 动态计算图

PyTorch 使用 define-by-run：每次前向都重建计算图，Python 控制流（if/for/while）也能求导。这是比 TensorFlow 1.x 静态图更灵活的设计。

---

## 2.6 概率统计 (Probability)

### 关键 API

| API | 用途 |
|-----|------|
| `torch.multinomial(weights, n)` | 多项分布抽样 |
| `torch.randn(size)` | 标准正态 N(0,1) |
| `torch.normal(mean, std, size)` | 一般正态分布 |
| `torch.rand(size)` | 均匀分布 U[0,1) |
| `torch.softmax(logits, dim)` | logits → 概率分布 |

### NLL → 交叉熵

负对数似然是分类损失的本质：
```
NLL = -log P(y_true | x)
```

- 预测概率高 → NLL 低 → 模型信心足
- 预测概率低 → NLL 高 → 模型被惩罚
- `CrossEntropyLoss = LogSoftmax + NLLLoss`

---

## 贯穿 Ch02 的收获

1. **Tensor 是深度学习的统一语言** — 从数据到模型到 loss 都是 tensor
2. **dim 参数是聚合操作的精髓** — 理解 dim=0（行方向）vs dim=1（列方向）
3. **广播是有规则的内存优化** — 不是魔法，是从最后一维向前对齐
4. **autograd 是框架的灵魂** — 手动推导反向传播的时代结束了
5. **grad 累加不是 bug 是 feature** — 支持梯度累积（大 batch 模拟），但训练时必须清零
6. **概率→NLL→交叉熵** — 这是理解所有分类损失函数的钥匙
