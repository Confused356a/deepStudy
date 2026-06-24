"""
d2l 2.4 — 微积分 (Calculus)

深度学习优化的核心思想：损失函数对参数求导 → 沿负梯度方向更新。
梯度是损失函数下降最快的方向。链式法则是反向传播的理论基础。
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

print("=" * 50)
print("2.4.1 — 导数与微分")
print("=" * 50)

# 导数 = 函数在一点的变化率 = 切线斜率
# f'(x) = lim_{h→0} [f(x+h) - f(x)] / h

# 示例: f(x) = x^2, f'(x) = 2x
# 在 x=1 处导数 = 2


print("=" * 50)
print("2.4.2 — 梯度 (Gradient) 与偏导数")
print("=" * 50)

# 梯度: 将导数推广到多元函数
# ∇f(x₁, x₂, ..., xₙ) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]^T
#
# 例: f(x, y) = 3x^2 + 2y
# ∇f = [∂f/∂x, ∂f/∂y] = [6x, 2]

print("\n可以用 2.5 节介绍的 autograd 自动算梯度，无需手推公式。")


print("=" * 50)
print("2.4.3 — 梯度下降直觉 (可视化)")
print("=" * 50)

# 可视化: f(x) = x^2 —— 梯度下降沿负梯度方向移动
x = np.linspace(-3, 3, 100)
y = x ** 2

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, 'b-', label='f(x) = x^2')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('梯度下降直觉: 沿负梯度方向移动')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)
ax.legend()

# 画几个"步"
positions = [2.5, 1.5, 0.8, 0.3]
for i, pos in enumerate(positions):
    ax.plot(pos, pos**2, 'ro', markersize=6)
    if i > 0:
        ax.annotate('', xy=(positions[i], positions[i]**2),
                     xytext=(positions[i-1], positions[i-1]**2),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

plt.tight_layout()
import os
save_dir = os.path.join(os.path.dirname(__file__), "notes")
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "gradient_descent_intuition.png"), dpi=100)
plt.close()
print("→ 图像已保存至 notes/gradient_descent_intuition.png")
print("  红点从 x=2.5 逐渐向 x=0 (最小值) 移动")


print("\n" + "=" * 50)
print("2.4.4 — 链式法则 (Chain Rule)")
print("=" * 50)

# 复合函数求导: dz/dx = dz/dy · dy/dx
# 这是反向传播的理论基础！
#
# 例: z = (x + y)^2
# 设 u = x + y, 则 z = u^2
# dz/dx = dz/du · du/dx = 2u · 1 = 2(x + y)
# dz/dy = dz/du · du/dy = 2u · 1 = 2(x + y)

# PyTorch 自动做这件事——只要定义前向计算，backward() 自动求梯度
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = (x + y) ** 2
z.backward()
print(f"\nx=2, y=3, z=(x+y)^2={z.item()}")
print(f"dz/dx = {x.grad}")   # 2*(2+3) = 10 ✓
print(f"dz/dy = {y.grad}")   # 2*(2+3) = 10 ✓


print("\n" + "=" * 50)
print("[OK] 2.4 微积分 — 完成")
print("=" * 50)
print("\n关键公式:")
print("  梯度:   grad f = [df/dx1, ..., df/dxn]")
print("  链式法则: dz/dx = dz/dy * dy/dx  <- 反向传播核心")
print("  梯度下降: w <- w - lr * grad L(w)")
