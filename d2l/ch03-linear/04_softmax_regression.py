"""
d2l 3.4 — Softmax 回归 (Softmax Regression)

从线性回归到分类的桥梁:
  线性回归: y ∈ R           (连续值, MSE loss)
  Softmax:  y ∈ {0,...,k-1} (离散类别, 交叉熵 loss)

核心思想: 线性输出 logits → softmax 转概率 → 交叉熵计算损失
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


print("=" * 60)
print("3.4.1 — 为什么线性回归不适用于分类?")
print("=" * 60)

# 问题: 10 个类别, 给每个类别一个"分数", 最高分的类别 = 预测
# 线性回归输出一个实数, 分类需要概率向量

print("线性回归:   输出 1 个实数 (比如房价 $350k)")
print("Softmax 回归: 输出 k 个概率 (比如 [0.7, 0.1, 0.05, ...])")
print()
print("对 k 类分类, 模型输出 k 个值 (logits), 然后 softmax 转为概率")


print("\n" + "=" * 60)
print("3.4.2 — Softmax 函数: R^k → [0,1]^k (概率单纯形)")
print("=" * 60)

# softmax(o)_i = exp(o_i) / sum_j(exp(o_j))
# 三个关键性质:
#   1. 非负: exp(x) > 0 对任何实数 x
#   2. 和为 1: 分母是全部 exp 的和 → 归一化
#   3. 保持顺序: exp 单调 → 最大 logit 对应最大概率

# 演示
torch.manual_seed(42)
logits = torch.randn(6) * 3   # 6 个类别的 "原始分数"
probs = torch.softmax(logits, dim=0)

print(f"logits (原始输出):  {logits.numpy()}")
print(f"softmax 后 (概率): {probs.detach().numpy()}")
print(f"概率之和:           {probs.sum().item():.6f}  (应为 1)")
print(f"最大 logit 的类别:   {logits.argmax().item()}")
print(f"最大概率的类别:      {probs.argmax().item()}")
print(f"最大概率值:         {probs.max().item():.4f}")

# 可视化: 不同"温度"下的 softmax
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# 左: 单个 logits 向量的 softmax 转换
x = np.arange(6)
width = 0.35
ax1.bar(x - width/2, logits.numpy(), width, label='Logits (原始分数)', alpha=0.7)
ax1.bar(x + width/2, probs.detach().numpy(), width, label='Softmax (概率)', alpha=0.7)
ax1.set_xticks(x)
ax1.set_xlabel('类别')
ax1.set_title('Logits → Softmax 转换')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右: softmax 温度效应
temps = [0.1, 0.5, 1.0, 2.0, 10.0]
demo_logits = torch.tensor([1.0, 2.0, 0.5])
for t in temps:
    p = torch.softmax(demo_logits / t, dim=0).numpy()
    ax2.plot([1, 2, 3], p, '-o', linewidth=2, markersize=8, label=f'T={t}')

ax2.set_xticks([1, 2, 3])
ax2.set_xlabel('类别')
ax2.set_ylabel('概率')
ax2.set_title('温度参数 T 对 Softmax 的影响\nT→0: 趋近 one-hot (硬决策)  T→∞: 趋近均匀分布')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d2l/ch03-linear/notes/softmax_demo.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] notes/softmax_demo.png — softmax 转换 + 温度效应")


print("\n" + "=" * 60)
print("3.4.3 — 交叉熵损失 (Cross-Entropy Loss)")
print("=" * 60)

# CE(y_hat, y) = -log(y_hat[y])
# 其中 y_hat[y] 是预测概率向量中, 真实类别 y 对应的概率
#
# 直觉:
#   y_hat[y] = 1.0  → -log(1.0) = 0      (完美预测, loss=0)
#   y_hat[y] = 0.5  → -log(0.5) = 0.693  (不太确定)
#   y_hat[y] = 0.01 → -log(0.01) = 4.605  (几乎错了, loss 很大)

p_values = np.logspace(-2, 0, 100)   # 从 0.01 到 1.0
ce_values = -np.log(p_values)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p_values, ce_values, 'b-', linewidth=2.5)
ax.set_xlabel('预测概率 y_hat[y] (真实类别的概率)')
ax.set_ylabel('交叉熵损失 -log(y_hat[y])')
ax.set_title('交叉熵损失曲线')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

# 标注关键点
for p in [0.01, 0.1, 0.5, 0.9, 1.0]:
    ax.annotate(f'loss={-np.log(p):.2f}', xy=(p, -np.log(p)),
                textcoords="offset points", xytext=(10, 10), fontsize=9)

plt.tight_layout()
plt.savefig('d2l/ch03-linear/notes/cross_entropy_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/cross_entropy_curve.png — 交叉熵损失曲线")


print("\n" + "=" * 60)
print("3.4.4 — 数值稳定性: softmax 的溢出问题")
print("=" * 60)

# 问题: exp(1000) → inf, exp(-1000) → 0 → log(0) = -inf
# 解决: softmax(x) = softmax(x - max(x))  (减去最大值, 数学等价)

big_logits = torch.tensor([1000.0, 0.0, -1000.0])

# 不稳定版本
try:
    naive = torch.exp(big_logits) / torch.exp(big_logits).sum()
    print(f"naive softmax:  {naive}")
except:
    print("naive softmax:  overflow! exp(1000) = inf")

# 稳定版本
shifted = big_logits - big_logits.max()
stable = torch.exp(shifted) / torch.exp(shifted).sum()
print(f"shifted logits: {shifted.numpy()}")
print(f"stable softmax: {stable.numpy()}")
print()
print("关键技巧: softmax(o) = softmax(o - max(o))")
print("  减掉最大值: exp 不会溢出, 且数学上等价 (分子分母同时除以 exp(max))")


print("\n" + "=" * 60)
print("3.4.5 — Softmax 回归 = 线性模型 + Softmax + 交叉熵")
print("=" * 60)

print("完整模型:")
print("  1. 线 性层:   o = X @ W + b      (batch, d) @ (d, k) + (k,) → (batch, k)")
print("  2. Softmax:   y_hat = softmax(o)  每行是概率分布, 和为 1")
print("  3. 交叉熵:    L = -log(y_hat[y])  真实类别的负对数概率")
print()
print("参数规模:")
print(f"  Fashion-MNIST: d=784 (28*28), k=10 (类别)")
print(f"  W: 784 × 10 = 7,840 个参数")
print(f"  b: 10 个参数")
print(f"  总计: 7,850  ← 极少! (相比 CNN 的百万级)")
print()
print("与线性回归的本质区别:")
print("  线性回归: 输出 ∈ R    | 损失 = MSE     | 目标 = 连续值")
print("  Softmax:  输出 ∈ R^k  | 损失 = 交叉熵   | 目标 = 类别")


print("\n" + "=" * 60)
print("3.4.6 — Softmax 回归 vs 逻辑回归 (Logistic Regression)")
print("=" * 60)

print("逻辑回归 = softmax 回归在 k=2 时的特例:")
print()
print("  逻辑回归:     P(y=1) = sigmoid(o) = 1 / (1 + exp(-o))")
print("  Softmax(k=2): P(y=1) = exp(o1) / (exp(o1) + exp(o0))")
print("                       = 1 / (1 + exp(o0 - o1))")
print("                       = sigmoid(o1 - o0)")
print()
print("→ 二分类时, softmax 和 sigmoid 等价 (差一个线性变换)")


print("\n" + "=" * 60)
print("[OK] 3.4 Softmax 回归理论 — 完成")
print("     核心: softmax 转概率 + 交叉熵作为分类损失")
print("     关键技巧: 减去 max 保证数值稳定")
print("=" * 60)
