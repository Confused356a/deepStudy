"""
d2l 3.1 — 线性回归理论 (Linear Regression Theory)

深度学习的 "Hello World"——用一个线性模型预测连续值。
本节覆盖：线性模型、MSE 损失、解析解、梯度下降、学习率影响。

只用 numpy + matplotlib，不讲框架，只讲数学直觉。
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 全局中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("3.1.1 — 线性模型: y = X @ w + b")
print("=" * 60)

# 人造数据：2 个特征，1000 个样本
torch.manual_seed(42)
n, d = 1000, 2
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2

X = torch.normal(0, 1, (n, d))           # 特征 ~ N(0, 1)
noise = torch.normal(0, 0.01, (n, 1))    # 观测噪声
y = X @ true_w.reshape(-1, 1) + true_b + noise   # 标签 = 线性变换 + 噪声

print(f"X  shape: {X.shape}    (n=1000, d=2)")
print(f"y  shape: {y.shape}    (n=1000, 1)")
print(f"true_w: {true_w.numpy()}, true_b: {true_b:.1f}")
print(f"前 3 个样本:")
for i in range(3):
    print(f"  x=[{X[i,0]:6.3f}, {X[i,1]:6.3f}]  y={y[i,0]:8.4f}")


print("\n" + "=" * 60)
print("3.1.2 — 解析解: w* = (X^T X)^(-1) X^T y")
print("=" * 60)

# 正规方程 (Normal Equation)
# L = (1/n) * ||y - Xw - b||^2，对 w 求导 = 0 得到闭式解
# 把 b 合并进 w：X 加一列全 1，w_ext = [w; b]
X_with_bias = torch.cat([X, torch.ones(n, 1)], dim=1)   # (n, d+1)

# w* = (X^T X)^(-1) X^T y
XTX = X_with_bias.T @ X_with_bias                        # (d+1, d+1)
XTy = X_with_bias.T @ y                                  # (d+1, 1)
w_analytical = torch.linalg.inv(XTX) @ XTy               # (d+1, 1)

print(f"X^T X shape: {XTX.shape}")
print(f"解析解 w*: {w_analytical[:-1].flatten().numpy()}  (真实: {true_w.numpy()})")
print(f"解析解 b*: {w_analytical[-1].item():.4f}           (真实: {true_b})")
print()
print("意义: 线性回归是唯一有解析解的'神经网络'")
print("      一旦模型有非线性 -> 没有闭式解 -> 必须用梯度下降")


print("\n" + "=" * 60)
print("3.1.3 — 梯度下降: w := w - lr * grad")
print("=" * 60)

# 一维例子：f(w) = (w - 3)^2，最小值在 w=3
def f(w):
    return (w - 3) ** 2

def grad_f(w):
    return 2 * (w - 3)

# 梯度下降迭代
w = torch.tensor(0.0)         # 初始值（离最小值很远）
lr = 0.1
history = [w.item()]

for step in range(20):
    w = w - lr * grad_f(w)
    history.append(w.item())

print(f"初始 w: 0.0,  最小值在 w=3")
print(f"lr=0.1, 20 步后 w = {w.item():.6f}  (应接近 3)")
print()

# 验证: lr 太大导致振荡
w_big = torch.tensor(0.0)
history_big = [w_big.item()]
for step in range(20):
    w_big = w_big - 1.8 * grad_f(w_big)    # lr=1.8 (太大！)
    history_big.append(w_big.item())

w_small = torch.tensor(0.0)
history_small = [w_small.item()]
for step in range(20):
    w_small = w_small - 0.01 * grad_f(w_small)  # lr=0.01 (太小)
    history_small.append(w_small.item())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 左: 梯度下降直觉
w_curve = np.linspace(-1, 7, 200)
for ax, h, title in zip(axes,
                         [history, history_big, history_small],
                         ['lr=0.1 (合适): 稳定收敛',
                          'lr=1.8 (太大): 振荡发散',
                          'lr=0.01 (太小): 收敛缓慢']):
    ax.plot(w_curve, f(w_curve), 'b-', linewidth=2, label='f(w)=(w-3)^2')
    for i, wi in enumerate(h):
        color = plt.cm.viridis(i / len(h))
        ax.plot(wi, f(wi), 'o', color=color, markersize=6)
    ax.plot(3, 0, 'r*', markersize=15, label='最小值 w=3')
    ax.set_xlabel('w')
    ax.set_ylabel('f(w)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'gradient_descent_lr.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/gradient_descent_lr.png — 学习率影响可视化")


print("\n" + "=" * 60)
print("3.1.4 — 损失函数分解: MSE = Bias^2 + Variance + Noise")
print("=" * 60)

# 生成新的测试数据（不同 noise 水平）
torch.manual_seed(123)
n_test = 200

# 用解析解预测
y_pred = X_with_bias @ w_analytical
residuals = (y - y_pred).flatten().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 左: 预测 vs 真实
axes[0].scatter(y.numpy(), y_pred.numpy(), alpha=0.5, s=20, edgecolors='none')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='完美预测')
axes[0].set_xlabel('真实 y')
axes[0].set_ylabel('预测 y_hat')
axes[0].set_title(f'预测 vs 真实 (MSE={((y_pred - y)**2).mean().item():.6f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右: 残差分布
axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('残差 (y - y_hat)')
axes[1].set_ylabel('频数')
axes[1].set_title(f'残差分布 (std={residuals.std():.4f}, 真实 noise std=0.01)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'linear_regression_fit.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/linear_regression_fit.png — 预测 vs 真实 + 残差分布")


print("\n" + "=" * 60)
print("3.1.5 — 批量 vs 小批量 vs 随机梯度下降")
print("=" * 60)

# 用真实数据展示：不同 batch_size 的梯度路径
# 目标: 二维 w = [w1, w2]，真实值 (2.0, -3.4)
# loss = mean((Xw + b_true - y)^2)

def compute_gradient(X_batch, y_batch, w_current):
    """MSE 对 w 的梯度: (2/n) * X^T (Xw - y)"""
    n_b = X_batch.shape[0]
    pred = X_batch @ w_current.reshape(-1, 1)
    grad = (2.0 / n_b) * (X_batch.T @ (pred - y_batch.reshape(-1, 1)))
    return grad.flatten()

# 准备数据（用真实的 true_w, true_b 生成）
X2 = torch.normal(0, 1, (500, 2))
y2 = X2 @ true_w.reshape(-1, 1) + true_b + torch.normal(0, 0.01, (500, 1))

w_init = torch.tensor([0.0, 0.0])  # 从原点出发
lr_compare = 0.02
epochs_compare = 50

batch_sizes = {'批量 (n=500)': 500, '小批量 (n=32)': 32, '随机 (n=1)': 1}
paths = {}

for name, bs in batch_sizes.items():
    w = w_init.clone()
    path = [w.numpy().copy()]
    indices = torch.randperm(500)
    for epoch in range(epochs_compare):
        for i in range(0, 500, bs):
            batch_idx = indices[i:i + bs]
            g = compute_gradient(X2[batch_idx], y2[batch_idx], w)
            w = w - lr_compare * g
        path.append(w.numpy().copy())
    paths[name] = np.array(path)

fig, ax = plt.subplots(figsize=(8, 6))

# Loss 等高线
w1_range = np.linspace(-1, 4, 200)
w2_range = np.linspace(-5, 1, 200)
W1, W2 = np.meshgrid(w1_range, w2_range)
L = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w_grid = torch.tensor([W1[i, j], W2[i, j]], dtype=torch.float32)
        pred = X2 @ w_grid.reshape(-1, 1)
        L[i, j] = ((pred - y2) ** 2).mean().item()

levels = np.logspace(np.log10(L.min()), np.log10(L.max()), 20)
ax.contour(W1, W2, L, levels=levels, alpha=0.4, cmap='Blues')
ax.contourf(W1, W2, L, levels=levels, alpha=0.15, cmap='Blues')

colors = {'批量 (n=500)': 'green', '小批量 (n=32)': 'blue', '随机 (n=1)': 'orange'}
for name, path in paths.items():
    ax.plot(path[:, 0], path[:, 1], '-o', color=colors[name], label=name,
            linewidth=1.5, markersize=3, alpha=0.8)
    ax.plot(path[0, 0], path[0, 1], 'ko', markersize=8, label='_nolegend_')

ax.plot(2.0, -3.4, 'r*', markersize=20, label='最优解 (2.0, -3.4)')
ax.plot(paths['批量 (n=500)'][0, 0], paths['批量 (n=500)'][0, 1], 'ko', markersize=8, label='起点 (0, 0)')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_title('三种梯度下降的优化路径 (loss 等高线)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'gradient_descent_paths.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/gradient_descent_paths.png — 三种 GD 变体的优化路径")


print("\n" + "=" * 60)
print("[OK] 3.1 线性回归理论 — 完成")
print("     核心: 模型(y=Xw+b) → 损失(MSE) → 优化(GD/SGD)")
print("     理解: 解析解推导 + 学习率影响 + batch size 选择")
print("=" * 60)
