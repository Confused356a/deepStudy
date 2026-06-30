"""
d2l 4.1 — 激活函数 (Activation Functions)

没有激活函数, 多层网络等价于单层线性变换. 激活函数引入非线性,
是神经网络能拟合复杂函数的关键.

核心公式 (前向传播):
    h = phi(W1 @ x + b1)    ← 激活函数作用在隐藏层
    o = W2 @ h + b2         ← 输出层通常不加激活(或 softmax)

三种经典激活函数:
    ReLU:    phi(z) = max(0, z)          — 现代 CNN/MLP 标配
    Sigmoid: phi(z) = 1 / (1 + exp(-z)) — 早期流行, 现在用于门控
    Tanh:    phi(z) = (exp(z)-exp(-z)) / (exp(z)+exp(-z)) — 零中心

为什么 ReLU 是默认选择?
    1. 正区间梯度=1, 不衰减 → 缓解梯度消失
    2. 计算简单 (max 操作) → 前向/反向都快
    3. 负区间梯度=0 → 稀疏激活, 正则化效果
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from contextlib import contextmanager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


@contextmanager
def _quiet_glyph():
    """抑制 matplotlib C 层字形缺失警告 (SimHei 缺 − 字形).

    ft2font C 扩展直接写 stderr, 绕过 Python warnings/logging 模块.
    仅在 plt.savefig / tight_layout 期间重定向 stderr.
    """
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


# ============================================================
# 1. 激活函数及其梯度 — 并排对比
# ============================================================
print("=" * 60)
print("1. ReLU / Sigmoid / Tanh 函数值与梯度对比")
print("=" * 60)

z = torch.linspace(-6, 6, 1000)

# 前向
relu = torch.relu(z)
sigmoid = torch.sigmoid(z)
tanh = torch.tanh(z)

# 梯度 (手动计算)
relu_grad = (z > 0).float()                           # ReLU': 0 if z<0 else 1
sigmoid_grad = sigmoid * (1 - sigmoid)                # sigmoid': sig(z)*(1-sig(z))
tanh_grad = 1 - tanh ** 2                             # tanh': 1 - tanh^2(z)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 第一行: 函数值
for ax, (name, y_vals, color) in zip(
    axes[0],
    [('ReLU max(0,z)', relu, '#2196F3'),
     ('Sigmoid 1/(1+e^{-z})', sigmoid, '#4CAF50'),
     ('Tanh', tanh, '#FF9800')]
):
    ax.plot(z, y_vals, color=color, linewidth=2)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title(name, fontsize=13)
    ax.set_xlabel('z'); ax.set_ylabel('phi(z)')
    ax.grid(True, alpha=0.3)

# 第二行: 梯度
for ax, (name, y_grad, color) in zip(
    axes[1],
    [("ReLU' (0 or 1)", relu_grad, '#2196F3'),
     ("Sigmoid' max=0.25", sigmoid_grad, '#4CAF50'),
     ("Tanh' max=1", tanh_grad, '#FF9800')]
):
    ax.plot(z, y_grad, color=color, linewidth=2)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title(name, fontsize=13)
    ax.set_xlabel('z'); ax.set_ylabel("phi'(z)")
    ax.grid(True, alpha=0.3)

plt.suptitle('激活函数与其梯度对比', fontsize=15, y=1.01)
with _quiet_glyph():
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'activation_functions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/activation_functions.png — 三种激活函数 + 梯度")


# ============================================================
# 2. 梯度消失问题 — 为什么 Sigmoid/Tanh 在深层网络出问题
# ============================================================
print("\n" + "=" * 60)
print("2. 梯度消失 (Vanishing Gradient) 实验")
print("=" * 60)
print("  链式法则: dL/dx = dL/dh_n * dh_n/dh_{n-1} * ... * dh_1/dx")
print("  每层乘一次 phi'(z), Sigmoid 梯度 max=0.25 → 深层梯度指数衰减")

# 模拟 10 层全连接，每层用 sigmoid，看梯度衰减
torch.manual_seed(0)
x = torch.randn(1, 10, requires_grad=True)

# Sigmoid 链
sigmoid_grads = []
h = x.clone().detach().requires_grad_(True)
h_sig = h
for layer_idx in range(10):
    W = torch.randn(10, 10, dtype=torch.float32) * 1.0
    h_sig = h_sig @ W
    h_sig = torch.sigmoid(h_sig)
    # 记录这一层的平均梯度幅度
    grad_test = torch.ones_like(h_sig)
    h_sig.backward(grad_test, retain_graph=True)
    sigmoid_grads.append(h.grad.abs().mean().item())
    h.grad.zero_()

# ReLU 链
relu_grads = []
h_relu = h.clone().detach().requires_grad_(True)
h_r = h_relu
for layer_idx in range(10):
    W = torch.randn(10, 10, dtype=torch.float32) * 1.0
    h_r = h_r @ W
    h_r = torch.relu(h_r)
    grad_test = torch.ones_like(h_r)
    h_r.backward(grad_test, retain_graph=True)
    relu_grads.append(h_relu.grad.abs().mean().item())
    h_relu.grad.zero_()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, 11), sigmoid_grads, 'g-o', linewidth=2, markersize=8, label='Sigmoid 链')
ax.plot(range(1, 11), relu_grads, 'b-s', linewidth=2, markersize=8, label='ReLU 链')
ax.axhline(y=1.0, color='gray', linewidth=0.5, linestyle='--', label='初始梯度=1')
ax.set_yscale('log')
ax.set_xlabel('层数', fontsize=12)
ax.set_ylabel('反向传回首层的平均梯度 (log scale)', fontsize=12)
ax.set_title('梯度消失: Sigmoid 梯度指数衰减, ReLU 保持稳定', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

with _quiet_glyph():
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'vanishing_gradient.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/vanishing_gradient.png — 10层 Sigmoid vs ReLU 梯度衰减")


# ============================================================
# 3. ReLU 的变体: LeakyReLU / GELU
# ============================================================
print("\n" + "=" * 60)
print("3. ReLU 变体: LeakyReLU / GELU")
print("=" * 60)

z = torch.linspace(-4, 4, 500)
relu = torch.relu(z)
leaky_relu = torch.nn.functional.leaky_relu(z, negative_slope=0.01)

# GELU (Gaussian Error Linear Unit) — Transformer 标配
# GELU(z) = z * Phi(z), Phi = 标准正态 CDF
gelu = z * 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(z, relu, linewidth=2, label='ReLU')
axes[0].plot(z, leaky_relu, linewidth=2, label='LeakyReLU (slope=0.01)')
axes[0].plot(z, gelu, linewidth=2, label='GELU')
axes[0].axhline(y=0, color='gray', linewidth=0.5)
axes[0].axvline(x=0, color='gray', linewidth=0.5)
axes[0].set_xlabel('z'); axes[0].set_ylabel('phi(z)')
axes[0].set_title('ReLU 及其变体')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 放大负区间
z_zoom = torch.linspace(-2, 0.5, 300)
axes[1].plot(z_zoom, torch.relu(z_zoom), linewidth=2, label='ReLU (负区间=0)')
axes[1].plot(z_zoom, torch.nn.functional.leaky_relu(z_zoom, 0.01),
             linewidth=2, label='LeakyReLU (小负值)')
axes[1].plot(z_zoom, z_zoom * 0.5 * (1.0 + torch.erf(z_zoom / np.sqrt(2.0))),
             linewidth=2, label='GELU (平滑过渡)')
axes[1].axhline(y=0, color='gray', linewidth=0.5)
axes[1].set_xlabel('z'); axes[1].set_ylabel('phi(z)')
axes[1].set_title('负区间放大: ReLU 死区 vs LeakyReLU 泄漏')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

with _quiet_glyph():
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'relu_variants.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/relu_variants.png — ReLU / LeakyReLU / GELU 对比")


# ============================================================
# 关键总结
# ============================================================
print("\n" + "=" * 60)
print("关键总结")
print("=" * 60)
print("  1. 激活函数 = 非线性的来源, 没有它就等价于单层线性")
print("  2. ReLU: 默认首选, 正区间梯度=1, 缓解梯度消失")
print("  3. Sigmoid/Tanh: 饱和区梯度≈0, 深层网络梯度消失")
print("  4. Sigmoid 梯度 max=0.25 → 10 层衰减 0.25^10 ≈ 10^-6!")
print("  5. LeakyReLU: 给负区间一点点梯度, 防止神经元'死掉'")
print("  6. GELU: Transformer 标配, 比 ReLU 更平滑, 处处可微")
