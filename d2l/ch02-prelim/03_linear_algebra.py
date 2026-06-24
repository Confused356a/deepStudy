"""
d2l 2.3 — 线性代数 (Linear Algebra)

深度学习的"语言"就是线性代数：数据是矩阵，运算靠矩阵乘法，
维度变换贯穿整个训练过程。本节同时覆盖 torch 的聚合操作。
"""
import torch

print("=" * 50)
print("2.3.1 — 标量、向量、矩阵")
print("=" * 50)

# 标量: 0 阶张量
scalar = torch.tensor(3.0)
print(f"标量: {scalar}, shape={scalar.shape}")

# 向量: 1 阶张量
vector = torch.arange(4)
print(f"向量: {vector}, shape={vector.shape}")

# 矩阵: 2 阶张量
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(f"矩阵 A (5×4):\n{A}")


print("\n" + "=" * 50)
print("2.3.2 — 矩阵操作")
print("=" * 50)

# 转置: A^T, 行列互换
print(f"A.T → shape {A.T.shape}:\n{A.T}")

# 对称矩阵判定: B == B.T


print("\n" + "=" * 50)
print("2.3.3 — 张量 (N 阶数组)")
print("=" * 50)

# 深度学习常用的张量是 3~5 阶
# [batch, channels, height, width] = 4 阶 (CNN)
# [batch, seq_len, hidden_dim]    = 3 阶 (RNN/Transformer)
X = torch.arange(24).reshape(2, 3, 4)
print(f"3 阶张量 shape (2,3,4):\n{X}")


print("\n" + "=" * 50)
print("2.3.4 — 张量运算的基本性质")
print("=" * 50)

# 逐元素运算：对应位置独立计算
# 矩阵乘法  ：涉及维度缩减

# Hadamard 积 (逐元素乘): A ⊙ B
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = torch.ones(2, 3) * 2
print(f"A:\n{A}")
print(f"B:\n{B}")
print(f"A * B (Hadamard积):\n{A * B}")


print("\n" + "=" * 50)
print("2.3.5 — 降维与聚合 (Reduction)")
print("=" * 50)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(f"A (5×4):\n{A}")

# 求和
print(f"\nA.sum()              → {A.sum()}")                      # 所有元素
print(f"A.sum(dim=0)         → shape {A.sum(dim=0).shape}: {A.sum(dim=0)}")   # 沿行 → 每列求和
print(f"A.sum(dim=1)         → shape {A.sum(dim=1).shape}: {A.sum(dim=1)}")   # 沿列 → 每行求和

# 均值 (需要 float 类型！)
print(f"\nA.mean()             → {A.mean()}")
print(f"A.mean(dim=0)        → {A.mean(dim=0)}")

# keepdim=True 保持维度，方便广播
print(f"\nA.sum(dim=1, keepdim=True) → shape {A.sum(dim=1, keepdim=True).shape}:\n{A.sum(dim=1, keepdim=True)}")
# shape 从 (5,) 变成 (5,1)，可以直接做广播除法
print(f"A / A.sum(dim=1, keepdim=True) → 每行归一化:\n{A / A.sum(dim=1, keepdim=True)}")

# 累积和
print(f"\nA.cumsum(dim=0):\n{A.cumsum(dim=0)}")  # 沿 dim=0 逐行累加


print("\n" + "=" * 50)
print("2.3.6 — 点积 (Dot Product) & 矩阵-向量乘 (mv)")
print("=" * 50)

x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4)
print(f"x = {x}")
print(f"y = {y}")

# 点积: torch.dot(x, y) 或 (x * y).sum() 或 x @ y (1D 向量)
print(f"torch.dot(x, y)     = {torch.dot(x, y)}")     # 0+1+2+3 = 6
print(f"(x * y).sum()      = {(x * y).sum()}")        # 等价写法

# 矩阵-向量乘 (matrix-vector): A(5×4) @ x(4) = (5)
print(f"\nA @ x → shape {A.shape} @ {x.shape} = {(A @ x).shape}:\n{A @ x}")


print("\n" + "=" * 50)
print("2.3.7 — 矩阵乘法 (Matrix-Matrix Multiply)")
print("=" * 50)

# A(m×n) @ B(n×k) → C(m×k)
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = torch.ones(3, 4)

print(f"A (2×3):\n{A}")
print(f"B (3×4):\n{B}")

C = A @ B                    # 等价于 torch.mm(A, B) 或 torch.matmul(A, B)
print(f"A @ B → shape {C.shape}:\n{C}")
# C[i,j] = sum_k A[i,k] * B[k,j]


print("\n" + "=" * 50)
print("2.3.8 — 范数 (Norm)")
print("=" * 50)

u = torch.tensor([3.0, -4.0])

# L2 范数 (欧氏距离): sqrt(sum(x^2))
print(f"torch.norm(u)         = {torch.norm(u)}")     # sqrt(9+16) = 5

# L1 范数: sum(|x|)
print(f"torch.abs(u).sum()    = {torch.abs(u).sum()}")  # |3| + |-4| = 7

# Frobenius 范数 (矩阵的 L2)
M = torch.arange(4, dtype=torch.float32).reshape(2, 2)
print(f"\nM:\n{M}")
print(f"torch.norm(M) (Frobenius) = {torch.norm(M)}")
# sqrt(0²+1²+2²+3²) = sqrt(14)
print(f"等价于 sqrt((M**2).sum()) = {torch.sqrt((M**2).sum())}")


print("\n" + "=" * 50)
print("[OK] 2.3 线性代数 — 完成")
print("=" * 50)
print("\n核心 API 速查:")
print("  A.T                  → 转置")
print("  A.sum(dim=k)         → 沿 k 轴求和")
print("  A.sum(keepdim=True)  → 保持维度，方便广播")
print("  A.cumsum(dim=k)      → 累积和")
print("  torch.dot(x, y)      → 向量点积")
print("  A @ B (或 torch.mm)  → 矩阵乘法")
print("  torch.norm(x)        → L2/Frobenius 范数")
