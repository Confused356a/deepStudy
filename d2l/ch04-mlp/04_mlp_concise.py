"""
d2l 4.3 — 多层感知机简洁实现 (MLP Concise Implementation)

一行 nn.Sequential 替代从零实现的 ~30 行代码.

架构:
    nn.Flatten()          — [batch, 1, 28, 28] → [batch, 784]
    nn.Linear(784, 256)   — 隐藏层
    nn.ReLU()             — 激活函数
    nn.Linear(256, 10)    — 输出层 (raw logits!)

关键: CrossEntropyLoss 期望 raw logits, 不要 softmax!
"""
import torch
import torch.nn as nn
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
print("MLP 简洁实现 — Fashion-MNIST 分类")
print("=" * 60)

# ============================================================
# 1. 数据加载
# ============================================================
print("\n[1] 加载数据...")

transform = transforms.ToTensor()
data_root = os.path.join(DATA_DIR, 'fashion_mnist')
train_dataset = torchvision.datasets.FashionMNIST(
    root=data_root, train=True, transform=transform, download=False)
test_dataset = torchvision.datasets.FashionMNIST(
    root=data_root, train=False, transform=transform, download=False)
if len(train_dataset) == 0:
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, transform=transform, download=True)

batch_size = 256
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)


# ============================================================
# 2. 模型定义 — 一行 nn.Sequential
# ============================================================
num_inputs = 28 * 28   # 784
num_hidden = 256
num_outputs = 10

# 从零 (03): 需要手动定义 relu(), softmax(), cross_entropy(), sgd(), net()
# 框架: 全部用 nn 模块一行搞定
model = nn.Sequential(
    nn.Flatten(),                # [batch, 1, 28, 28] → [batch, 784]
    nn.Linear(num_inputs, num_hidden),   # [batch, 784] → [batch, 256]
    nn.ReLU(),                   # 非线性激活
    nn.Linear(num_hidden, num_outputs),  # [batch, 256] → [batch, 10]
)

# Xavier 初始化
for layer in model:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)

loss_fn = nn.CrossEntropyLoss()          # = LogSoftmax + NLLLoss
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print(f"  模型:\n{model}")
print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"  损失函数: CrossEntropyLoss (期望 raw logits!)")


# ============================================================
# 3. 训练 + 评估
# ============================================================
def evaluate(model, data_loader, device='cpu'):
    """在给定数据加载器上评估准确率"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / total


print("\n[2] 开始训练...")
num_epochs = 20

train_losses = []
test_accs = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for X, y in train_loader:
        logits = model(X)                    # raw logits!
        loss = loss_fn(logits, y)            # CrossEntropyLoss 内部做 softmax

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(y)

    train_loss = epoch_loss / len(train_dataset)
    test_acc = evaluate(model, test_loader)

    train_losses.append(train_loss)
    test_accs.append(test_acc)

    if (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch+1:2d}: train_loss={train_loss:.4f}, test_acc={test_acc:.3f}")

final_acc = test_accs[-1]
print(f"\n  最终测试准确率: {final_acc:.3f} ({final_acc*100:.1f}%)")


# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Loss
axes[0].plot(range(1, num_epochs+1), train_losses, 'b-o', markersize=6)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Train Loss')
axes[0].set_title('MLP 训练 Loss (简洁实现)')
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(range(1, num_epochs+1), test_accs, 'r-s', markersize=6, label='MLP (框架)')
axes[1].axhline(y=0.83, color='gray', linestyle='--', linewidth=1.5, label='线性 baseline 83%')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Test Accuracy')
axes[1].set_title('MLP vs Softmax 线性模型')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 权重可视化 — 第一层 Linear(784→256) 的部分权重 reshape 成 28×28
W1 = model[1].weight.detach()  # shape [256, 784]
# 取前 16 个神经元的权重, 各 reshape 为 28×28
grid = np.zeros((4*28, 4*28))
for i in range(16):
    r, c = i // 4, i % 4
    w_img = W1[i].reshape(28, 28)
    # 归一化到 [-1, 1] 便于可视化
    grid[r*28:(r+1)*28, c*28:(c+1)*28] = w_img / (abs(w_img).max() + 1e-8)
axes[2].imshow(grid, cmap='RdBu', aspect='auto')
axes[2].set_title('W1 前 16 个神经元 (28×28 权重模板)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'mlp_concise_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] notes/mlp_concise_results.png — Loss + 准确率 + W1 可视化")


# ============================================================
# 5. 从零 vs 框架对照
# ============================================================
print("\n" + "=" * 60)
print("从零 vs 框架 对照")
print("=" * 60)

print(f"""
  | 组件 | 从零 (~80 行) | 框架 (~8 行) |
  |------|--------------|-------------|
  | 模型 | def net(X): H=relu(X@W1+b1); return softmax(H@W2+b2) | nn.Sequential(Flatten, Linear, ReLU, Linear) |
  | 损失 | def cross_entropy(y_hat,y): -log(y_hat[range(n),y]) | nn.CrossEntropyLoss() |
  | 优化 | def sgd(params,lr,bs): p-=lr*p.grad/bs; p.grad.zero_() | optimizer.zero_grad(); optimizer.step() |
  | 数据 | 手写 yield 生成器 | DataLoader(TensorDataset) |
""")

print("关键提醒:")
print("  1. CrossEntropyLoss 输入 raw logits, 不要在模型里加 Softmax!")
print("  2. loss.mean() → optimizer 自动处理 batch_size, 不需要手动除")
print("  3. model.train() / model.eval() — Dropout/BatchNorm 行为不同")
print(f"  4. 从零准确率 ~{test_accs[-1]:.3f}, 框架准确率 ~{final_acc:.3f} — 几乎一致")
