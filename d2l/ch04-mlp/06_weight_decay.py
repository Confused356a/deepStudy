"""
d2l 4.5 — 权重衰减 (Weight Decay / L2 Regularization)

过拟合的一个特征: 权重值很大, 对输入微小变化剧烈反应.
权重衰减 = 在损失函数中惩罚大权重, 迫使模型学得更"平滑".

数学:
    原始:    minimize  L(w)                    — 只关心拟合数据
    L2 正则: minimize  L(w) + (lambda/2) * ||w||^2  — 同时惩罚权重大小

梯度更新:
    w ← w - lr * (dL/dw + lambda * w)      ← 每次更新权重额外减去 lambda*w
    w ← (1 - lr*lambda) * w - lr * dL/dw   ← 等价于每次先"衰减"权重

PyTorch: optimizer = SGD(params, lr, weight_decay=lambda)
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
print("权重衰减 (L2 Regularization)")
print("=" * 60)


# ============================================================
# 1. 高维线性回归: 少量数据 + 大量特征 → 容易过拟合
# ============================================================
torch.manual_seed(0)
n_train, n_test, num_inputs = 20, 100, 200
# 只有 20 个样本, 但有 200 个特征 → 经典过拟合场景

# 真实权重: 只有前 5 维非零 (稀疏真相)
true_w = torch.zeros(num_inputs, 1)
true_w[:5] = torch.tensor([5, 3, 2, -4, -1]).reshape(-1, 1)
true_b = 0.5

X_train = torch.randn(n_train, num_inputs) * 0.5
y_train = X_train @ true_w + true_b + 0.01 * torch.randn(n_train, 1)

X_test = torch.randn(n_test, num_inputs) * 0.5
y_test = X_test @ true_w + true_b + 0.01 * torch.randn(n_test, 1)

print(f"  数据: {n_train} 训练样本, {num_inputs} 特征")
print(f"  真实权重: 只有前 5 维非零 (稀疏)")
print(f"  场景: p >> n → 不正则化必然过拟合")


# ============================================================
# 2. 训练函数
# ============================================================
def train_linear(X_train, y_train, X_test, y_test, weight_decay=0, lr=0.01, epochs=100):
    """训练线性回归模型, 返回训练/测试误差历史"""
    w = torch.zeros(num_inputs, 1, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    train_losses, test_losses = [], []
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        # forward
        y_hat = X_train @ w + b
        l = loss_fn(y_hat, y_train)

        # backward (L2 正则由 optimizer 的 weight_decay 自动添加)
        w.grad = None; b.grad = None
        l.backward()

        # 手动 SGD with weight decay
        with torch.no_grad():
            w -= lr * (w.grad + weight_decay * w)
            b -= lr * b.grad

        train_losses.append(l.item())
        with torch.no_grad():
            y_pred = X_test @ w + b
            test_losses.append(loss_fn(y_pred, y_test).item())

    return train_losses, test_losses, w.detach().clone(), b.detach().clone()


# ============================================================
# 3. 对比: 无正则化 vs weight_decay=1 vs weight_decay=10
# ============================================================
print("\n训练对比: 不同 weight_decay...")

configs = [
    (0, '无正则化 (λ=0)'),
    (3, '轻量正则 (λ=3)'),
    (20, '强正则 (λ=20)'),
]

all_results = []
for wd, label in configs:
    train_l, test_l, w, b = train_linear(X_train, y_train, X_test, y_test, weight_decay=wd)
    all_results.append((wd, label, train_l, test_l, w, b))
    print(f"  λ={wd:2d}: train_loss={train_l[-1]:.4f}, test_loss={test_l[-1]:.4f}  |w|={w.norm().item():.1f}")


# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

# 第一行: 训练/测试 loss 曲线
for idx, (wd, label, train_l, test_l, w, b) in enumerate(all_results):
    ax = axes[0][idx]
    ax.semilogy(train_l, 'b-', alpha=0.7, label='train loss')
    ax.semilogy(test_l, 'r-', alpha=0.7, label='test loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
    ax.set_title(f'{label}')
    ax.legend(); ax.grid(True, alpha=0.3)
    # 标注泛化差距
    gap = test_l[-1] - train_l[-1]
    ax.text(0.95, 0.85, f'泛化差距={gap:.4f}', transform=ax.transAxes,
            fontsize=9, ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 第二行: 权重分布
for idx, (wd, label, train_l, test_l, w, b) in enumerate(all_results):
    ax = axes[1][idx]
    w_np = w.flatten().numpy()
    colors = ['red' if i < 5 else 'steelblue' for i in range(num_inputs)]
    ax.bar(range(num_inputs), w_np, color=colors, width=1.0)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_xlabel('特征维度'); ax.set_ylabel('权重值')
    ax.set_title(f'{label}\n前5维(红)=真实信号, 其余=噪声')

    # 标注非零权重的恢复情况
    true_signal = true_w[:5].flatten().numpy()
    learned_signal = w_np[:5]
    signal_mse = np.mean((true_signal - learned_signal) ** 2)
    ax.text(0.95, 0.85, f'信号 mse={signal_mse:.4f}\n噪声 std={w_np[5:].std():.4f}',
            transform=ax.transAxes, fontsize=8, ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'weight_decay.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] notes/weight_decay.png — 无正则/轻正则/强正则 对比")


# ============================================================
# 5. weight_decay 扫描
# ============================================================
print("\n扫描 weight_decay 值...")
lambdas = [0, 10**-1, 10**0, 10**1, 10**2, 10**3]
final_test = []
final_wnorm = []

for wd in lambdas:
    train_l, test_l, w, b = train_linear(X_train, y_train, X_test, y_test, weight_decay=wd)
    final_test.append(test_l[-1])
    final_wnorm.append(w.norm().item())

best_idx = np.argmin(final_test)
print(f"  最优 λ={lambdas[best_idx]}, test_loss={final_test[best_idx]:.4f}")

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.semilogx(lambdas, final_test, 'r-s', linewidth=2, markersize=8, label='Test Loss')
ax1.set_xlabel('Weight Decay (λ)', fontsize=12)
ax1.set_ylabel('Test Loss', color='red', fontsize=12)
ax1.tick_params(axis='y', labelcolor='red')
ax1.grid(True, alpha=0.3)

ax2.semilogx(lambdas, final_wnorm, 'b-o', linewidth=2, markersize=8, label='|w| (L2 norm)')
ax2.set_ylabel('|w| (L2 norm)', color='blue', fontsize=12)
ax2.tick_params(axis='y', labelcolor='blue')

ax1.axvline(x=lambdas[best_idx], color='gray', linestyle='--', linewidth=1.5, label=f'最优 λ={lambdas[best_idx]}')
ax1.set_title('Weight Decay 对 Test Loss 和权重范数的影响', fontsize=13)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'weight_decay_scan.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/weight_decay_scan.png — weight_decay 扫描")


# ============================================================
# 6. MLP + weight_decay 在 Fashion-MNIST 上的效果
# ============================================================
print("\n在 MLP 上测试 weight_decay...")
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

train_ld = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
test_ld = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)


def train_mlp_wd(weight_decay, lr=0.1, epochs=15):
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    return correct / total


acc_no_wd = train_mlp_wd(weight_decay=0)
acc_wd = train_mlp_wd(weight_decay=0.001)
print(f"  MLP (Fashion-MNIST) 无 weight_decay: test_acc={acc_no_wd:.3f}")
print(f"  MLP (Fashion-MNIST) weight_decay=0.001: test_acc={acc_wd:.3f}")
print(f"  提升: {(acc_wd-acc_no_wd)*100:+.1f}pp")


# ============================================================
# 关键总结
# ============================================================
print("\n" + "=" * 60)
print("关键总结")
print("=" * 60)
print("  1. 权重衰减 = L2 正则化: loss += λ/2 * ||w||^2")
print("  2. 效果: 权重值被压小, 模型更平滑 → 泛化更好")
print("  3. p >> n (特征远多于样本) 时不正则化必过拟合")
print("  4. λ 太小 → 没用; λ 太大 → 欠拟合 (所有权重趋近 0)")
print("  5. PyTorch: optimizer = SGD(params, lr, weight_decay=λ)")
print("  6. weight_decay 对真实信号(大权重)惩罚轻, 对噪声维度(小权重)惩罚重")
