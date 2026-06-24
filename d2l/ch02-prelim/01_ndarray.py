"""
d2l 2.1 — 数据操作 (Data Manipulation)

PyTorch Tensor 是深度学习的数据容器，等价于 NumPy ndarray 但支持 GPU。
本节覆盖：创建、运算、广播、索引、节省内存、与 NumPy 互转。
"""
import torch

print("=" * 50)
print("2.1.1 — 创建 Tensor")
print("=" * 50)

# arange: 生成 [0, 12) 的整数序列
x = torch.arange(12)
print(f"x = torch.arange(12)          → {x}")
print(f"x.shape                        → {x.shape}")    # torch.Size([12])
print(f"x.numel()                      → {x.numel()}")   # 元素总数 = 12

# reshape: 改变形状，不改变数据（view 是引用，reshape 可能复制）
X = x.reshape(3, 4)
print(f"x.reshape(3, 4)                → shape={X.shape}\n{X}")

# -1 自动推断维度
X2 = x.reshape(-1, 4)    # 等价于 reshape(3, 4)
print(f"x.reshape(-1, 4)               → shape={X2.shape}")

# 常用初始化
print(f"\ntorch.zeros((2, 3, 4))         → shape (2,3,4) 全 0")
print(f"torch.ones((2, 3, 4))          → shape (2,3,4) 全 1")
print(f"torch.randn(3, 4)              → shape (3,4) 标准正态分布 N(0,1)")

# 从 Python 列表直接构造
t = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(f"\ntorch.tensor([[2,1,4,3],...])   →\n{t}")

# 从已有 tensor 克隆形状（randn_like 要求 float 类型）
Y = torch.randn_like(X.float())  # X 的形状 + 标准正态
print(f"\ntorch.randn_like(X)             → shape={Y.shape} 随机值")


print("\n" + "=" * 50)
print("2.1.2 — 运算符")
print("=" * 50)

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

print(f"x = {x}")
print(f"y = {y}")
print(f"x + y   = {x + y}")        # 逐元素加法
print(f"x - y   = {x - y}")        # 逐元素减法
print(f"x * y   = {x * y}")        # 逐元素乘法（不是矩阵乘法！）
print(f"x / y   = {x / y}")        # 逐元素除法
print(f"x ** y  = {x ** y}")       # 逐元素幂
print(f"torch.exp(x) = {torch.exp(x)}")  # e^x


print("\n" + "=" * 50)
print("2.1.3 — 拼接 (Concatenation)")
print("=" * 50)

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2., 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(f"X (3×4):\n{X}")
print(f"Y (3×4):\n{Y}")
print(f"\ntorch.cat((X, Y), dim=0) → shape (6, 4):\n{torch.cat((X, Y), dim=0)}")
print(f"\ntorch.cat((X, Y), dim=1) → shape (3, 8):\n{torch.cat((X, Y), dim=1)}")


print("\n" + "=" * 50)
print("2.1.4 — 广播机制 (Broadcasting)")
print("=" * 50)

# 广播规则: 从最后一维向前对齐，维度为 1 的轴沿该方向复制
a = torch.arange(3).reshape((3, 1))   # shape (3, 1)
b = torch.arange(2).reshape((1, 2))   # shape (1, 2)

print(f"a shape (3,1):\n{a}")
print(f"b shape (1,2):\n{b}")
print(f"a + b → (3,2):\n{a + b}")
# 等价于 a (3,2) + b (3,2)，其中 a 沿 col 复制，b 沿 row 复制


print("\n" + "=" * 50)
print("2.1.5 — 索引与切片 (Indexing & Slicing)")
print("=" * 50)

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(f"X:\n{X}")

# 索引规则和 Python list 一样: [start:stop:step]
print(f"X[-1]       → 最后一行: {X[-1]}")
print(f"X[1:3]      → 第 1~2 行:\n{X[1:3]}")

# 写入
X[1, 2] = 99
print(f"X[1, 2] = 99 →\n{X}")

X[0:2, :] = 12
print(f"X[0:2, :] = 12 →\n{X}")


print("\n" + "=" * 50)
print("2.1.6 — 节省内存 (In-place Operations)")
print("=" * 50)

# ❌ 会分配新内存: Y = X + Y
# [OK] 原地操作，不分配新内存:
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2., 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(f"X id before: {id(X)}")
X += Y                     # 原地加法，id 不变
print(f"X id after : {id(X)}  (相同 → 原地)")

# 或用切片赋值
Z = torch.zeros_like(Y)
before_id = id(Z)
Z[:] = X + Y               # 通过 [:] 赋值，id 不变
print(f"Z[:] = X+Y → id 不变: {id(Z) == before_id}")


print("\n" + "=" * 50)
print("2.1.7 — PyTorch &lt;-&gt; NumPy 互转")
print("=" * 50)

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
A = X.numpy()              # Tensor → NumPy（共享内存！）
B = torch.tensor(A)         # NumPy → Tensor（复制数据）

print(f"type(A) = {type(A)}")
print(f"type(B) = {type(B)}")

# 修改 tensor 会同步影响 numpy（共享内存）
X[0, 0] = 999
print(f"X.numpy()[0,0] after X mod = {A[0, 0]}  ← 证明共享内存")

# 标量 tensor → Python 原生类型
a = torch.tensor([3.5])
print(f"\na.item()  = {a.item()}")     # 3.5
print(f"float(a) = {float(a)}")        # 3.5
print(f"int(a)   = {int(a)}")          # 3


print("\n" + "=" * 50)
print("[OK] 2.1 数据操作 — 完成")
print("=" * 50)
