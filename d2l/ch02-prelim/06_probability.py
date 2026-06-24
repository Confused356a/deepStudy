"""
d2l 2.6 — 概率统计 (Probability & Statistics)

深度学习中概率无处不在：
- 损失函数 → 最大似然 → 交叉熵
- 分类器输出 → softmax → 概率分布
- 评估指标 → 期望/方差
- Dropout / VAE / GAN → 抽样

本节快速过核心 API，不展开数学推导。
"""
import torch
import matplotlib.pyplot as plt
import os

print("=" * 50)
print("2.6.1 — 离散概率分布抽样")
print("=" * 50)

# torch.multinomial: 从多项分布中抽样
# weights: 各元素的相对权重（不需要归一化，自动归一化）
fair_probs = torch.ones([6]) / 6   # 均匀分布 → 骰子
print(f"fair_probs = {fair_probs}")

# 抽样 1 次
sample = torch.multinomial(fair_probs, num_samples=1)
print(f"multinomial 1次: {sample.item()}")

# 抽样 10 次（有放回）
samples = torch.multinomial(fair_probs, num_samples=10, replacement=True)
print(f"multinomial 10次: {samples.tolist()}")


print("\n" + "=" * 50)
print("2.6.2 — 大数定律可视化")
print("=" * 50)

# 随着抽样次数增加，频率趋近真实概率
counts = torch.multinomial(fair_probs, num_samples=1000, replacement=True)
# 统计每个面的出现次数
freq = torch.bincount(counts, minlength=6) / 1000
print(f"1000次抽样后各面频率: {freq}")
print(f"理论概率: {fair_probs}")
print(f"绝对误差: {(freq - fair_probs).abs().max().item():.4f}")

# 累积估计
estimates_list = []
n_samples = 10000
counts = torch.multinomial(fair_probs, num_samples=n_samples, replacement=True)
for i in range(1, n_samples + 1):
    if i % 100 == 0:  # 每 100 步记录一次
        freq = torch.bincount(counts[:i], minlength=6).float() / i
        estimates_list.append(freq)

estimates = torch.stack(estimates_list)  # (n_steps, 6)

fig, ax = plt.subplots(figsize=(8, 5))
for j in range(6):
    ax.plot(range(100, n_samples + 1, 100), estimates[:, j].numpy(),
            label=f'P(面={j+1})')
ax.axhline(y=1/6, color='black', linestyle='--', linewidth=0.8, label='理论 1/6 ≈ 0.167')
ax.set_xlabel('抽样次数')
ax.set_ylabel('估计概率')
ax.set_title('大数定律: 频率趋近概率')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

save_dir = os.path.join(os.path.dirname(__file__), "notes")
os.makedirs(save_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "law_of_large_numbers.png"), dpi=100)
plt.close()
print("→ 图像已保存至 notes/law_of_large_numbers.png")


print("\n" + "=" * 50)
print("2.6.3 — 正态分布抽样")
print("=" * 50)

# torch.randn: 标准正态分布 N(0, 1)
# torch.normal(mean, std, size): 一般正态分布
samples = torch.normal(mean=0.0, std=1.0, size=(10000,))
print(f"N(0,1) 10000样本: mean={samples.mean():.4f}, std={samples.std():.4f}")

# torch.rand: 均匀分布 U[0, 1)
uniform_samples = torch.rand(10000)
print(f"U[0,1) 10000样本: mean={uniform_samples.mean():.4f} (理论 0.5)")


print("\n" + "=" * 50)
print("2.6.4 — 最大似然与损失函数的关系")
print("=" * 50)

# 负对数似然 (Negative Log Likelihood, NLL) —— 损失的源头
# 对于分类器: P(y|x) = softmax(logits)[y]
# NLL = -log P(y|x) — 概率越高，损失越小
# 交叉熵 = 负对数似然（当真分布是 one-hot 时）

logits = torch.tensor([2.0, 1.0, 0.1])           # 三个类别的 raw logits
probs = torch.softmax(logits, dim=0)              # 转为概率
print(f"logits = {logits}")
print(f"softmax(logits) = {probs}")
print(f"预测类别 = {probs.argmax().item()} (概率 {probs.max():.4f})")

# 假设真实标签是类别 0
true_class = 0
nll = -torch.log(probs[true_class])               # 负对数似然
print(f"\n真实类别 = {true_class}")
print(f"P(y=0|x) = {probs[true_class]:.4f}")
print(f"NLL = -log({probs[true_class]:.4f}) = {nll:.4f}")
print(f"\n→ 如果概率接近 1，NLL 接近 0；如果概率接近 0，NLL → ∞")
print(f"→ 这就是为什么 CrossEntropyLoss 长这样！")


print("\n" + "=" * 50)
print("[OK] 2.6 概率统计 — 完成")
print("=" * 50)
print("\n核心 API 速查:")
print("  torch.multinomial(weights, n)   → 多项分布抽样")
print("  torch.randn(size)               → 标准正态 N(0,1)")
print("  torch.normal(mean, std, size)   → 通用正态")
print("  torch.rand(size)                → 均匀分布 U[0,1)")
print("  torch.softmax(logits, dim)      → logits → 概率")
print("  -log P(y_true|x)                → NLL → 交叉熵本质")
