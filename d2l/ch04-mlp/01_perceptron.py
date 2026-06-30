"""
d2l 4.1 — 感知机 (Perceptron)

感知机是神经网络的起源。Frank Rosenblatt 1957 年提出，是第一个
用算法来定义的神经网络。

核心公式:
    y = sign(w · x + b)    — 二分类线性分类器

关键局限: 无法解决 XOR 问题 → 直接导致了 1970s 的"AI 寒冬"
突破: 加一层隐藏层就能解决 XOR → 多层感知机 (MLP)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 感知机模型: y = sign(w · x + b)
# ============================================================
def perceptron(X, w, b):
    """感知机前向计算: sign(w·x + b) → {+1, -1}"""
    return torch.sign(X @ w + b)


def perceptron_loss(X, y, w, b):
    """感知机损失: 只对分错的样本计算 -y(w·x+b) 的和"""
    margins = y * (X @ w + b)          # 正确分类 → margin > 0
    misclassified = margins <= 0       # 分错的样本
    if misclassified.any():
        return -margins[misclassified].sum()
    return torch.tensor(0.0)


print("=" * 60)
print("1. 感知机在二维数据上的决策边界")
print("=" * 60)

torch.manual_seed(42)
n_pts = 100

# 生成线性可分数据
X_pos = torch.randn(n_pts // 2, 2) + torch.tensor([2.0, 2.0])   # 正类 (右上)
X_neg = torch.randn(n_pts // 2, 2) + torch.tensor([-2.0, -2.0])  # 负类 (左下)
X = torch.cat([X_pos, X_neg], dim=0)
y = torch.cat([torch.ones(n_pts // 2), -torch.ones(n_pts // 2)])

# 初始化参数
w = torch.randn(2, 1, dtype=torch.float32) * 0.01
b = torch.zeros(1, dtype=torch.float32)
lr = 0.1

# 感知机学习算法: 对每个错分样本，w ← w + lr * y_i * x_i
loss_history = []
for epoch in range(20):
    total_loss = 0.0
    for i in range(len(X)):
        x_i = X[i:i+1]
        y_i = y[i:i+1]
        margin = y_i * (x_i @ w + b)
        if margin.item() <= 0:  # 分错了
            w = w + lr * y_i.item() * x_i.T
            b = b + lr * y_i.item()
            total_loss += -margin.item()
    loss_history.append(total_loss)
    if total_loss == 0:
        print(f"  epoch {epoch+1}: loss=0, 完美分类! 提前停止")
        break
    if (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch+1}: loss={total_loss:.3f}")

# 可视化决策边界
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左: 数据 + 决策边界
ax = axes[0]
ax.scatter(X_pos[:, 0], X_pos[:, 1], c='blue', s=40, alpha=0.7, label='正类 (+1)')
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='red', s=40, alpha=0.7, label='负类 (-1)')

x1_vals = torch.linspace(-5, 5, 100)
if abs(w[1].item()) > 1e-6:
    x2_vals = -(w[0].item() * x1_vals + b.item()) / w[1].item()
    ax.plot(x1_vals, x2_vals, 'k--', linewidth=2, label='决策边界 w·x+b=0')
ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('感知机: 线性决策边界')
ax.legend(); ax.grid(True, alpha=0.3)

# 右: loss 下降
axes[1].plot(loss_history, 'b-o', markersize=5)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].set_title('感知机损失下降')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'perceptron_boundary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/perceptron_boundary.png — 感知机决策边界 + 损失下降")


# ============================================================
# 2. XOR 问题 — 感知机的致命缺陷
# ============================================================
print("\n" + "=" * 60)
print("2. XOR 问题: 单层感知机无法解决")
print("=" * 60)

# XOR 数据
X_xor = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_xor = torch.tensor([[-1.], [1.], [1.], [-1.]])  # (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0

# 尝试用感知机学 XOR — 永远不会收敛
torch.manual_seed(0)
w_xor = torch.randn(2, 1, dtype=torch.float64) * 0.01
b_xor = torch.zeros(1, dtype=torch.float64)
loss_xor = []
for epoch in range(100):
    total = 0.0
    for i in range(4):
        margin = y_xor[i:i+1] * (X_xor[i:i+1] @ w_xor + b_xor)
        if margin.item() <= 0:
            w_xor = w_xor + 0.1 * y_xor[i].item() * X_xor[i:i+1].T
            b_xor = b_xor + 0.1 * y_xor[i].item()
            total += -margin.item()
    loss_xor.append(total)

print(f"  训练 100 epoch 后 loss = {loss_xor[-1]:.3f} (永不收敛到 0)")
print(f"  原因: XOR 不是线性可分的 — 没有一条直线能分开这两类")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# XOR 数据分布
ax = axes[0]
colors = ['blue' if yi == 1 else 'red' for yi in y_xor.flatten()]
markers = ['o' if yi == 1 else 'x' for yi in y_xor.flatten()]
for i in range(4):
    ax.scatter(X_xor[i, 0], X_xor[i, 1], c='blue' if y_xor[i] == 1 else 'red',
               s=200, marker='o' if y_xor[i] == 1 else 'x',
               edgecolors='black', linewidths=1.5, zorder=5)
    ax.annotate(f"({int(X_xor[i,0])},{int(X_xor[i,1])})", (X_xor[i, 0]+0.05, X_xor[i, 1]+0.05))
ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('XOR 问题: 没有直线能分开')
ax.grid(True, alpha=0.3)

# loss 不收敛
axes[1].plot(loss_xor, 'r-', linewidth=1.5)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].set_title('XOR 感知机训练: Loss 永不收敛')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'xor_unsolvable.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/xor_unsolvable.png — XOR 问题: 感知机无法解决")


# ============================================================
# 3. 用两层网络解决 XOR
# ============================================================
print("\n" + "=" * 60)
print("3. 加一层隐藏层 → 解决 XOR")
print("=" * 60)
print("  架构: 2输入 → 2隐藏(ReLU) → 1输出")

# 手算: 隐藏层学到 h1 = x1 AND x2, h2 = x1 OR x2, 输出层组合
# 这里用 PyTorch 自动学
torch.manual_seed(3)
X_xor_t = X_xor.to(torch.float32)
y_xor_t = y_xor.to(torch.float32)

# 两层网络
W1 = torch.randn(2, 2, dtype=torch.float32) * 0.5
b1 = torch.zeros(2, dtype=torch.float32)
W2 = torch.randn(2, 1, dtype=torch.float32) * 0.5
b2 = torch.zeros(1, dtype=torch.float32)

params = [W1, b1, W2, b2]
for p in params:
    p.requires_grad = True

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params, lr=0.5)

loss_mlp = []
for epoch in range(500):
    # forward
    h = torch.relu(X_xor_t @ W1 + b1)   # 隐藏层 + ReLU
    y_hat = h @ W2 + b2                   # 输出层
    loss = loss_fn(y_hat, y_xor_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_mlp.append(loss.item())

    if loss.item() < 0.001 and epoch > 50:
        print(f"  epoch {epoch+1}: loss={loss.item():.6f} → 收敛!")
        break

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# XOR 决策边界
ax = axes[0]
x1 = torch.linspace(-0.5, 1.5, 200)
x2 = torch.linspace(-0.5, 1.5, 200)
X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
grid = torch.stack([X1.flatten(), X2.flatten()], dim=1)
with torch.no_grad():
    h_grid = torch.relu(grid @ W1 + b1)
    Z = (h_grid @ W2 + b2).reshape(200, 200)

ax.contourf(X1, X2, Z, levels=50, cmap='RdBu', alpha=0.5)
ax.contour(X1, X2, Z, levels=[0], colors='black', linewidths=2)
for i in range(4):
    ax.scatter(X_xor[i, 0], X_xor[i, 1],
               c='blue' if y_xor[i] == 1 else 'red',
               s=150, edgecolors='black', linewidths=2, zorder=5)
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('MLP 解决 XOR: 非线性决策边界')

# 隐藏层学到的表示
axes[1].scatter([0, 0, 1, 1], [0, 1, 0, 1],
                c=['red', 'blue', 'blue', 'red'], s=200, edgecolors='black')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
axes[1].set_title('原始空间: 线性不可分')
axes[1].set_xlim(-0.5, 1.5); axes[1].set_ylim(-0.5, 1.5)
axes[1].grid(True, alpha=0.3)

with torch.no_grad():
    H = torch.relu(X_xor_t @ W1 + b1)
axes[2].scatter(H[:, 0], H[:, 1],
                c=['red', 'blue', 'blue', 'red'], s=200, edgecolors='black')
axes[2].set_xlabel('h1'); axes[2].set_ylabel('h2')
axes[2].set_title('隐藏空间: 变得线性可分!')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'xor_mlp_solution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/xor_mlp_solution.png — MLP 解决 XOR: 非线性边界 + 空间变换")


# ============================================================
# 关键总结
# ============================================================
print("\n" + "=" * 60)
print("关键总结")
print("=" * 60)
print("  1. 单层感知机 = 线性分类器, 只能分线性可分的类")
print("  2. XOR 问题直接导致了第一次 AI 寒冬 (Minsky & Papert, 1969)")
print("  3. 加隐藏层 + 非线性激活 = 突破线性限制")
print("  4. 隐藏层把数据映射到新空间, 使线性可分")
print("  5. 核心思想: 堆叠多层 → 学习层次化特征")
