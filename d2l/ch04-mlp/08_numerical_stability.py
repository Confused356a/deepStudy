"""
d2l 4.7–4.8 — 数值稳定性与参数初始化 (Numerical Stability & Init)

深层网络训练的两个核心问题:
    梯度消失 (Vanishing): 梯度逐层缩小 → 浅层参数几乎不更新
    梯度爆炸 (Exploding):  梯度逐层放大 → 参数剧烈震荡, loss NaN

根本原因: 链式法则导致梯度乘法叠加.
    dL/dx_1 = dL/dx_n * prod_{i=2..n} (Wi^T * phi'(z_i))

如果每层 Wi 的特征值 < 1 → 梯度消失
如果每层 Wi 的特征值 > 1 → 梯度爆炸

解决方案:
    Xavier:  Var(w) = 2 / (fan_in + fan_out)   — 适合 sigmoid/tanh
    He:     Var(w) = 2 / fan_in                 — 适合 ReLU 系列
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
print("数值稳定性 & 参数初始化")
print("=" * 60)


# ============================================================
# 1. 梯度爆炸/消失演示
# ============================================================
print("\n[1] 50 层全连接网络: 观察梯度传播...")

torch.manual_seed(0)
x = torch.randn(1, 256)

# ---- 场景1: 权重太大 → 梯度爆炸 ----
weights_big = []
h = x.clone()
for i in range(50):
    W = torch.randn(256, 256) * 0.5   # std=0.5, 太大
    h = h @ W
    weights_big.append(W)

# 用 MSE loss
loss = h.mean()
loss.backward()

# 记录每层的梯度范数 (从最后一层反向查)
# 实际上 backward 后每层 param 的 grad 记录了 dL/dW
# 我们要看 dL/dh_i, 但这里简化: 直接看参数梯度的范数
big_grads = []
h = x.clone()
for W in weights_big:
    W_clone = W.clone().requires_grad_(False)
    # 重新前向并计算这个 W 的梯度
    pass

# 换个更清晰的方法: 用 requires_grad 的中间值
torch.manual_seed(0)
x_g = torch.randn(1, 256, requires_grad=True)
h_g = x_g

# 爆炸场景: 大权重
for i in range(50):
    W = torch.randn(256, 256) * 0.3
    h_g = h_g @ W
h_g.mean().backward()
grad_explode = x_g.grad.abs().mean().item()

# 消失场景: 小权重
torch.manual_seed(0)
x_g2 = torch.randn(1, 256, requires_grad=True)
h_g2 = x_g2
for i in range(50):
    W = torch.randn(256, 256) * 0.01
    h_g2 = h_g2 @ W
h_g2.mean().backward()
grad_vanish = x_g2.grad.abs().mean().item()

print(f"  50 层, 权重 std=0.3  → 首层梯度均值: {grad_explode:.2e} (爆炸!)")
print(f"  50 层, 权重 std=0.01 → 首层梯度均值: {grad_vanish:.2e} (消失!)")
print(f"  理想: 梯度在 1e-4 ~ 1e-2 之间")


# ============================================================
# 2. Xavier 和 He 初始化公式
# ============================================================
print("\n[2] Xavier 和 He 初始化的数学...")

fan_in, fan_out = 256, 256

# 三种初始化的标准差
std_random = 0.01
std_xavier = np.sqrt(2.0 / (fan_in + fan_out))   # Xavier: 2/(fan_in+fan_out)
std_he = np.sqrt(2.0 / fan_in)                    # He: 2/fan_in

print(f"  fan_in={fan_in}, fan_out={fan_out}")
print(f"  随机初始化 std:        {std_random:.4f}")
print(f"  Xavier 初始化 std:     {std_xavier:.4f}   ← 适合 sigmoid/tanh")
print(f"  He 初始化 std:         {std_he:.4f}        ← 适合 ReLU (因为负半轴被截断, 方差减半)")

# 可视化三种初始化的权重分布
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

torch.manual_seed(0)
init_configs = [
    ('随机 (std=0.01)', torch.randn(10000) * 0.01),
    (f'Xavier (std={std_xavier:.3f})', torch.randn(10000) * std_xavier),
    (f'He (std={std_he:.3f})', torch.randn(10000) * std_he),
]

for ax, (name, weights) in zip(axes, init_configs):
    ax.hist(weights.numpy(), bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_title(name, fontsize=12)
    ax.set_xlabel('权重值'); ax.set_ylabel('概率密度')
    ax.axvline(x=0, color='gray', linewidth=0.5)

plt.suptitle('三种权重初始化分布对比', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'initialization_dist.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] notes/initialization_dist.png — 随机/Xavier/He 权重分布")


# ============================================================
# 3. 初始化对训练的影响 — MLP on Fashion-MNIST
# ============================================================
print("\n[3] 不同初始化对 MLP 训练的影响...")

import torchvision
import torchvision.transforms as transforms
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), '..', 'datasets')

transform = transforms.ToTensor()
try:
    train_set = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, transform=transform, download=False)
    test_set = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False, transform=transform, download=False)
    if len(train_set) == 0:
        raise RuntimeError
except Exception:
    train_set = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=False, transform=transform, download=True)

train_ld = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
test_ld = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)


def init_model(init_type='random', scale=0.01):
    """创建 MLP 并按指定方式初始化"""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )

    for layer in model:
        if isinstance(layer, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(layer.weight)
            elif init_type == 'he':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif init_type == 'random':
                nn.init.normal_(layer.weight, std=scale)
            nn.init.zeros_(layer.bias)
    return model


def train_and_eval(model, epochs=10, lr=0.1):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    test_accs, train_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X, y in train_ld:
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y)
        train_losses.append(epoch_loss / len(train_set))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_ld:
                logits = model(X)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.numel()
        test_accs.append(correct / total)

    return train_losses, test_accs


print("  训练 3 种初始化...")
torch.manual_seed(0)
m_random, (tl_random, ta_random) = init_model('random', 0.01), train_and_eval(init_model('random', 0.01))
torch.manual_seed(0)
m_xavier, (tl_xavier, ta_xavier) = init_model('xavier'), train_and_eval(init_model('xavier'))
torch.manual_seed(0)
m_he, (tl_he, ta_he) = init_model('he'), train_and_eval(init_model('he'))

print(f"  随机 N(0, 0.01): 最终 test_acc={ta_random[-1]:.3f}")
print(f"  Xavier uniform:   最终 test_acc={ta_xavier[-1]:.3f}")
print(f"  He (kaiming):     最终 test_acc={ta_he[-1]:.3f}")


# ============================================================
# 4. 可视化训练曲线
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

epochs = range(1, 11)

# Test Accuracy
axes[0].plot(epochs, ta_random, 'gray', linewidth=2, marker='o', markersize=5, label='Random N(0,0.01)')
axes[0].plot(epochs, ta_xavier, 'blue', linewidth=2, marker='s', markersize=5, label='Xavier')
axes[0].plot(epochs, ta_he, 'red', linewidth=2, marker='^', markersize=5, label='He (Kaiming)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test Accuracy')
axes[0].set_title('不同初始化的测试准确率')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Train Loss
axes[1].semilogy(epochs, tl_random, 'gray', linewidth=2, marker='o', markersize=5, label='Random N(0,0.01)')
axes[1].semilogy(epochs, tl_xavier, 'blue', linewidth=2, marker='s', markersize=5, label='Xavier')
axes[1].semilogy(epochs, tl_he, 'red', linewidth=2, marker='^', markersize=5, label='He (Kaiming)')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Train Loss')
axes[1].set_title('不同初始化的训练 Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 最终准确率柱状图
axes[2].bar(['Random 0.01', 'Xavier', 'He (Kaiming)'],
            [ta_random[-1], ta_xavier[-1], ta_he[-1]],
            color=['gray', 'steelblue', 'coral'], edgecolor='black')
for i, v in enumerate([ta_random[-1], ta_xavier[-1], ta_he[-1]]):
    axes[2].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Test Accuracy'); axes[2].set_title('最终准确率对比')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'initialization_training.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] notes/initialization_training.png — 不同初始化训练效果对比")


# ============================================================
# 5. 激活值传播: 观察每层输出的方差
# ============================================================
print("\n[5] 激活值方差传播...")

torch.manual_seed(0)
x = torch.randn(1000, 256)  # 1000 个样本, 各 256 维

def check_variance_propagation(init_fn, init_name, n_layers=10):
    """模拟 n 层网络前向传播, 观察每层激活值的方差"""
    variances = [x.var().item()]
    h = x.clone()

    for i in range(n_layers):
        W = torch.empty(256, 256)
        init_fn(W)
        h = h @ W
        # 通过 ReLU
        h = torch.relu(h)
        variances.append(h.var().item())

    return variances

# 三种初始化
random_var = check_variance_propagation(
    lambda W: nn.init.normal_(W, std=0.01), 'Random 0.01')
xavier_var = check_variance_propagation(
    lambda W: nn.init.xavier_uniform_(W), 'Xavier')
he_var = check_variance_propagation(
    lambda W: nn.init.kaiming_uniform_(W, nonlinearity='relu'), 'He')

fig, ax = plt.subplots(figsize=(10, 5))
layers = range(11)
ax.plot(layers, random_var, 'gray', linewidth=2, marker='o', markersize=6, label='Random N(0, 0.01)')
ax.plot(layers, xavier_var, 'blue', linewidth=2, marker='s', markersize=6, label='Xavier')
ax.plot(layers, he_var, 'red', linewidth=2, marker='^', markersize=6, label='He (Kaiming)')
ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, label='理想方差=1')
ax.set_yscale('log')
ax.set_xlabel('层数'); ax.set_ylabel('激活值方差 (log scale)')
ax.set_title('10 层网络: 每层激活值方差变化 (ReLU)')
ax.legend(); ax.grid(True, alpha=0.3)
ax.text(0.5, 0.15, 'Random: 方差指数衰减 → 梯度消失\nHe: 方差保持稳定 → 训练稳定',
        transform=ax.transAxes, fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'variance_propagation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/variance_propagation.png — 10 层网络激活值方差传播")


# ============================================================
# 关键总结
# ============================================================
print("\n" + "=" * 60)
print("关键总结")
print("=" * 60)
print("  1. 梯度消失: 深层梯度指数衰减, 浅层几乎不更新 → 用 ReLU + He 初始化")
print("  2. 梯度爆炸: 梯度指数增长, loss→NaN → 梯度裁剪 / 更好的初始化")
print("  3. Xavier: Var(w) = 2/(fan_in+fan_out), 适合 sigmoid/tanh")
print("  4. He: Var(w) = 2/fan_in, 适合 ReLU (补偿负半轴截断的方差损失)")
print("  5. 好初始化 = 各层激活值方差 ≈ 1 → 梯度流畅通")
print("  6. PyTorch 默认 kaiming_uniform (He), 所以通常不需要手动初始化")
print("  7. ReLU + He + BatchNorm 是现在深层网络的标准配方")
