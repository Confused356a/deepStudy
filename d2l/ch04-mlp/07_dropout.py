"""
d2l 4.6 — Dropout

Dropout 是一种简单但极其有效的正则化技术。
训练时: 以概率 p 随机"丢弃"(置零)神经元 → 每轮训练不同子网络
测试时: 所有权重乘以 (1-p) → 相当于所有子网络的集成

直觉:
    - 强迫每个神经元独立学有用特征, 不能依赖特定搭档
    - 相当于训练大量子网络的 ensemble
    - 和 L2 正则不同: L2 惩罚大权重, Dropout 打破神经元"共适应"

公式:
    训练: h' = h * mask / (1-p)    mask ~ Bernoulli(1-p)
    测试: h (不丢弃, scaling 已在训练阶段通过除以(1-p)完成)

PyTorch: nn.Dropout(p) — 自动处理 train/eval 模式切换!
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Dropout — 随机丢弃正则化")
print("=" * 60)


# ============================================================
# 1. Dropout 原理演示
# ============================================================
torch.manual_seed(0)
h = torch.arange(16, dtype=torch.float32).reshape(4, 4)
print("\n[1] Dropout 原理 — 训练时随机置零")
print(f"  原始激活:\n{h}")

# Dropout: p=0.5 → 训练时每个神经元有 50% 概率被丢弃
dropout = nn.Dropout(p=0.5)

dropout.train()  # 训练模式
h_train = dropout(h)

dropout.eval()   # 测试模式
h_eval = dropout(h)

print(f"\n  p=0.5, 训练模式 (随机丢弃):\n{h_train}")
print(f"  保留神经元数: {(h_train > 0).sum().item()}/{h.numel()}")
print(f"\n  p=0.5, 测试模式 (原样通过):\n{h_eval}")
print("  测试模式不丢弃, 因为训练时已除以 (1-p) 做了缩放补偿")


# ============================================================
# 2. 从零实现 Dropout
# ============================================================
print("\n[2] 从零实现 Dropout...")

def dropout_layer(X, p_drop):
    """从零实现 Dropout: 训练时随机丢弃, 除以 (1-p) 做期望补偿"""
    assert 0 <= p_drop < 1
    if p_drop == 0:
        return X
    if not torch.is_grad_enabled():  # 测试模式 (no_grad 上下文)
        return X
    mask = (torch.rand(X.shape, device=X.device) > p_drop).float()
    return mask * X / (1 - p_drop)  # 除以 (1-p) 保持期望不变!


# 验证: Dropout 保持输出的期望值不变
X_demo = torch.ones(10000, 100)  # 全 1, 期望=1
X_drop = dropout_layer(X_demo, p_drop=0.5)
print(f"  输入期望: {X_demo.mean().item():.4f}")
print(f"  Dropout(p=0.5) 后期望: {X_drop.mean().item():.4f} (应 ≈ 1.0, 因为除以了 0.5)")


# ============================================================
# 3. Dropout 可视化 — 逐 neuron 丢弃效果
# ============================================================
torch.manual_seed(5)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 原始激活矩阵 [20 neurons × 30 samples]
H_orig = torch.randn(20, 30)

for idx, (p, ax) in enumerate(zip([0.0, 0.5, 0.8], axes)):
    if p == 0:
        H_drop = H_orig
    else:
        mask = (torch.rand(20, 30) > p).float()
        H_drop = mask * H_orig / (1 - p)

    im = ax.imshow(H_drop, cmap='RdBu', aspect='auto', vmin=-3, vmax=3)
    ax.set_xlabel(f'样本'); ax.set_ylabel('神经元')
    active = (H_drop != 0).float().mean().item()
    ax.set_title(f'Dropout p={p}\n活跃神经元: {active*100:.0f}%')

plt.colorbar(im, ax=axes, shrink=0.6, label='激活值')
plt.suptitle('Dropout 效果: 逐神经元可视化', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'dropout_visual.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] notes/dropout_visual.png — Dropout p=0/0.5/0.8 逐神经元可视化")


# ============================================================
# 4. Fashion-MNIST: Dropout vs No Dropout
# ============================================================
print("\n[4] Fashion-MNIST 实验: Dropout vs 无 Dropout...")

import torchvision
import torchvision.transforms as transforms
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), '..', 'datasets')

transform = transforms.ToTensor()
data_root = os.path.join(DATA_DIR, 'fashion_mnist')
try:
    train_set = torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=transform, download=False)
    test_set = torchvision.datasets.FashionMNIST(root=data_root, train=False, transform=transform, download=False)
    if len(train_set) == 0:
        raise RuntimeError
except Exception:
    train_set = torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.FashionMNIST(root=data_root, train=False, transform=transform, download=True)

batch_size = 256
train_ld = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_ld = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


def make_mlp(dropout_p=0):
    """构建 MLP, 可选 Dropout"""
    layers = [nn.Flatten(), nn.Linear(784, 256), nn.ReLU()]
    if dropout_p > 0:
        layers.append(nn.Dropout(dropout_p))
    layers.extend([nn.Linear(256, 128), nn.ReLU()])
    if dropout_p > 0:
        layers.append(nn.Dropout(dropout_p))
    layers.append(nn.Linear(128, 10))
    return nn.Sequential(*layers)


def train_evaluate(model, epochs=15, lr=0.1):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    test_accs = []

    for epoch in range(epochs):
        model.train()
        for X, y in train_ld:
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_ld:
                logits = model(X)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.numel()
        test_accs.append(correct / total)

    return test_accs


torch.manual_seed(0)
# 用一个比 Fashion-MNIST 更小的集合模拟过拟合场景
# 只取 5000 训练样本
train_subset = torch.utils.data.Subset(train_set, range(5000))
train_small = torch.utils.data.DataLoader(train_subset, batch_size=256, shuffle=True)

# 替换全局 train_ld
train_ld_backup = train_ld
train_ld = train_small

print("  使用 5000 训练样本 (模拟数据不足, 容易过拟合)")
print("  MLP 架构: 784→256→ReLU→Dropout→128→ReLU→Dropout→10")

torch.manual_seed(0)
model_no_drop = make_mlp(dropout_p=0)
acc_no = train_evaluate(model_no_drop)

torch.manual_seed(0)
model_drop = make_mlp(dropout_p=0.3)
acc_drop = train_evaluate(model_drop)

torch.manual_seed(0)
model_drop5 = make_mlp(dropout_p=0.5)
acc_drop5 = train_evaluate(model_drop5)

print(f"  无 Dropout:    最终 test_acc={acc_no[-1]:.3f}")
print(f"  Dropout p=0.3: 最终 test_acc={acc_drop[-1]:.3f}")
print(f"  Dropout p=0.5: 最终 test_acc={acc_drop5[-1]:.3f}")

# 恢复
train_ld = train_ld_backup


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 准确率曲线
epochs = range(1, 16)
axes[0].plot(epochs, acc_no, 'gray', linewidth=2, marker='o', markersize=5, label='无 Dropout')
axes[0].plot(epochs, acc_drop, 'blue', linewidth=2, marker='s', markersize=5, label='Dropout p=0.3')
axes[0].plot(epochs, acc_drop5, 'red', linewidth=2, marker='^', markersize=5, label='Dropout p=0.5')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test Accuracy')
axes[0].set_title('Dropout 对测试准确率的影响 (5000 训练样本)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 最终准确率柱状图
axes[1].bar(['无 Dropout', 'p=0.3', 'p=0.5'], [acc_no[-1], acc_drop[-1], acc_drop5[-1]],
            color=['gray', 'steelblue', 'coral'], edgecolor='black')
for i, (v, label) in enumerate(zip([acc_no[-1], acc_drop[-1], acc_drop5[-1]],
                                    [acc_no[-1], acc_drop[-1], acc_drop5[-1]])):
    axes[1].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Test Accuracy'); axes[1].set_title('最终准确率对比')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'dropout_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] notes/dropout_results.png — Dropout p=0/0.3/0.5 准确率对比")


# ============================================================
# 关键总结
# ============================================================
print("\n" + "=" * 60)
print("关键总结")
print("=" * 60)
print("  1. Dropout: 训练时随机丢弃神经元, 打破'共适应'")
print("  2. 除以 (1-p): 保持激活值的期望不变, 测试时不需要调整")
print("  3. nn.Dropout: 自动在 train/eval 模式间切换, 无需手动")
print("  4. p 的选择: 输入层 p=0~0.2, 隐藏层 p=0.3~0.5")
print("  5. 和 weight_decay 互补: L2 惩罚权重大小, Dropout 打破依赖")
print("  6. 数据少时效果更明显, 数据多时可能不需要 Dropout")
