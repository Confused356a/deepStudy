"""
d2l 4.2 — 多层感知机从零实现 (MLP from Scratch)

架构: Flatten(784) → Linear(784→256) → ReLU → Linear(256→10) → Softmax
比 Ch03 Softmax 回归多了一个隐藏层 + ReLU 激活。

关键区别:
    Softmax 回归: y = softmax(X @ W + b)     — 线性模型
    MLP:          h = ReLU(X @ W1 + b1)        — 隐藏层
                  y = softmax(h @ W2 + b2)     — 输出层

训练循环同上: forward → loss → backward → SGD step
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), '..', 'datasets')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("MLP 从零实现 — Fashion-MNIST 分类")
print("=" * 60)


# ============================================================
# 1. 数据加载
# ============================================================
print("\n[1] 加载 Fashion-MNIST...")

transform = transforms.ToTensor()
data_root = os.path.join(DATA_DIR, 'fashion_mnist')
train_dataset = torchvision.datasets.FashionMNIST(
    root=data_root, train=True, transform=transform, download=False)
test_dataset = torchvision.datasets.FashionMNIST(
    root=data_root, train=False, transform=transform, download=False)

# 先用 download=True 尝试，如果已存在则 download=False 即可
if len(train_dataset) == 0:
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, transform=transform, download=True)

num_inputs = 28 * 28   # 784
num_hidden = 256
num_outputs = 10
batch_size = 256

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
print(f"  测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")
print(f"  架构: {num_inputs} → {num_hidden}(ReLU) → {num_outputs}")


# ============================================================
# 2. 参数初始化
# ============================================================
print("\n[2] 参数初始化 (Xavier)...")

torch.manual_seed(42)
# Xavier 初始化: 权重方差 = 2 / (fan_in + fan_out)
W1 = torch.randn(num_inputs, num_hidden, dtype=torch.float32) * 0.01
b1 = torch.zeros(num_hidden, dtype=torch.float32)
W2 = torch.randn(num_hidden, num_outputs, dtype=torch.float32) * 0.01
b2 = torch.zeros(num_outputs, dtype=torch.float32)

params = [W1, b1, W2, b2]
for p in params:
    p.requires_grad_(True)

# Xavier 初始化改进版本
torch.nn.init.xavier_uniform_(W1)
torch.nn.init.xavier_uniform_(W2)

print(f"  W1: {list(W1.shape)} 均值={W1.mean().item():.6f}  std={W1.std().item():.4f}")
print(f"  W2: {list(W2.shape)} 均值={W2.mean().item():.6f}  std={W2.std().item():.4f}")
print(f"  参数量: {W1.numel() + b1.numel() + W2.numel() + b2.numel():,}")
print(f"    (Ch03 Softmax 回归: {num_inputs * num_outputs + num_outputs:,})")


# ============================================================
# 3. 模型 + 损失 + 优化器 (从零定义)
# ============================================================
def relu(X):
    """ReLU 激活函数: max(0, X)"""
    a = torch.zeros_like(X)
    return torch.max(X, a)


def softmax_stable(X):
    """数值稳定的 softmax: softmax(o) = softmax(o - max(o))"""
    X_max = X.max(dim=1, keepdim=True).values
    X_exp = torch.exp(X - X_max)
    return X_exp / X_exp.sum(dim=1, keepdim=True)


def net(X, W1, b1, W2, b2):
    """MLP 前向: X → Linear → ReLU → Linear → Softmax"""
    X = X.reshape(-1, num_inputs)       # [batch, 1, 28, 28] → [batch, 784]
    H = relu(X @ W1 + b1)               # [batch, 784] → [batch, 256]
    O = H @ W2 + b2                     # [batch, 256] → [batch, 10]
    return softmax_stable(O)


def cross_entropy(y_hat, y):
    """交叉熵: -log(y_hat[range(n), y])"""
    return -torch.log(y_hat[range(len(y_hat)), y] + 1e-9)


def sgd(params, lr, batch_size):
    """小批量 SGD: param -= lr * grad / batch_size"""
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad.zero_()


class Accumulator:
    """累加器 — 用于统计 epoch 级别指标"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """计算预测准确率"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net_fn, data_iter, W1, b1, W2, b2):
    """在测试集上评估准确率"""
    acc = Accumulator(2)  # [correct, total]
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net_fn(X, W1, b1, W2, b2)
            acc.add(accuracy(y_hat, y), y.numel())
    return acc[0] / acc[1]


# ============================================================
# 4. 训练
# ============================================================
print("\n[3] 开始训练...")
num_epochs = 20
lr = 0.1

train_losses = []
test_accs = []

for epoch in range(num_epochs):
    # 训练
    metric = Accumulator(3)  # [train_loss, train_acc, total]
    for X, y in train_loader:
        y_hat = net(X, W1, b1, W2, b2)
        l = cross_entropy(y_hat, y)
        l.sum().backward()
        sgd(params, lr, len(y))
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    train_loss = metric[0] / len(train_loader)
    train_acc = metric[1] / metric[2]
    test_acc = evaluate_accuracy(net, test_loader, W1, b1, W2, b2)

    train_losses.append(train_loss)
    test_accs.append(test_acc)

    if (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch+1:2d}/{num_epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

final_test_acc = test_accs[-1]
print(f"\n  最终测试准确率: {final_test_acc:.3f} ({final_test_acc*100:.1f}%)")
print(f"  Ch03 Softmax 回归 (线性): ~83%")
print(f"  MLP (一个隐藏层): → {final_test_acc*100:.1f}% (提升了 {final_test_acc*100-83:.1f}pp)")


# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Loss 曲线
axes[0].plot(range(1, num_epochs+1), train_losses, 'b-o', markersize=6)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Train Loss')
axes[0].set_title('MLP 训练 Loss (从零实现)')
axes[0].grid(True, alpha=0.3)

# 准确率曲线
axes[1].plot(range(1, num_epochs+1), test_accs, 'r-s', markersize=6, label='Test Acc')
axes[1].axhline(y=0.83, color='gray', linestyle='--', linewidth=1.5,
                label='Softmax 线性 baseline (83%)')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('MLP 测试准确率 (vs 线性模型)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 权重可视化 — 只可视化 W2 (256→10 的列向量 reshape 为 16×16)
axes[2].axis('off')
# 取 W2 的每一列 reshape 为 16×16
W2_vis = W2.detach().T.reshape(10, 16, 16)
grid = np.zeros((4*16, 3*16))
for i in range(10):
    r, c = i // 3, i % 3
    if r < 4:
        grid[r*16:(r+1)*16, c*16:(c+1)*16] = W2_vis[i]
axes[2].imshow(grid, cmap='RdBu', aspect='auto')
axes[2].set_title('W2 权重的 16×16 视图 (10 类, 256→10)')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'mlp_scratch_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] notes/mlp_scratch_results.png — Loss + 准确率 + 权重可视化")


# ============================================================
# 6. 不同隐藏层大小的对比
# ============================================================
print("\n[4] 隐藏层大小对比实验...")
hidden_sizes = [64, 128, 256, 512, 1024]
compare_accs = []

for h_size in hidden_sizes:
    torch.manual_seed(42)
    W1_h = torch.empty(num_inputs, h_size, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(W1_h)
    b1_h = torch.zeros(h_size, dtype=torch.float32)
    W2_h = torch.empty(h_size, num_outputs, dtype=torch.float32)
    torch.nn.init.xavier_uniform_(W2_h)
    b2_h = torch.zeros(num_outputs, dtype=torch.float32)
    params_h = [W1_h, b1_h, W2_h, b2_h]
    for p in params_h:
        p.requires_grad_(True)

    for epoch in range(10):  # 只跑 10 epoch 快速对比
        for X, y in train_loader:
            H = relu(X.reshape(-1, num_inputs) @ W1_h + b1_h)
            y_hat = softmax_stable(H @ W2_h + b2_h)
            l = cross_entropy(y_hat, y)
            l.sum().backward()
            sgd(params_h, lr, len(y))

    acc = evaluate_accuracy(lambda X, w1, b1, w2, b2: net(X, w1, b1, w2, b2),
                             test_loader, W1_h, b1_h, W2_h, b2_h)
    compare_accs.append(acc)
    print(f"  hidden={h_size:4d}: test_acc={acc:.3f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar([str(s) for s in hidden_sizes], compare_accs, color='steelblue', edgecolor='black')
ax.set_xlabel('Hidden Size', fontsize=12)
ax.set_ylabel('Test Accuracy (10 epoch)', fontsize=12)
ax.set_title('隐藏层大小对准确率的影响', fontsize=13)
for i, (s, a) in enumerate(zip(hidden_sizes, compare_accs)):
    ax.text(i, a + 0.003, f'{a:.3f}', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0.80, 0.87)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'mlp_hidden_size.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/mlp_hidden_size.png — 不同隐藏层大小对比")


# ============================================================
# 关键总结
# ============================================================
print("\n" + "=" * 60)
print("关键总结")
print("=" * 60)
print(f"  1. MLP = 线性模型 + 隐藏层 + 激活函数 ({num_inputs}→{num_hidden}→{num_outputs})")
print(f"  2. 一个隐藏层就从 ~83% 提升到 ~{final_test_acc*100:.1f}%, 证明非线性 + 更多参数有效")
print("  3. Xavier 初始化: 保持每层激活值方差稳定, 训练更稳定")
print("  4. 训练循环不变: 与 Ch03 Softmax 回归一模一样的四步曲")
print("  5. 隐藏层越大 → 容量越大, 但边际收益递减 (256→512 提升很小)")
print("  6. 参数量: 784×256+256 + 256×10+10 = ~203K (Softmax 回归只有 7.8K)")
