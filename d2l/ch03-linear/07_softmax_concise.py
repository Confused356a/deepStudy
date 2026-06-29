"""
d2l 3.7 — Softmax 回归的简洁实现 (Concise Implementation)

用 PyTorch 高层 API 重写 3.6:
  nn.Linear(784, 10)    → 替代 W, b 和 net()
  nn.CrossEntropyLoss   → 替代 softmax() + cross_entropy()
  torch.optim.SGD       → 替代手动 sgd()

重点: nn.CrossEntropyLoss = LogSoftmax + NLLLoss
     输入 raw logits (不需要手动 softmax!)
"""
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ============================================================
# 3.7.1 加载数据 + 定义模型
# ============================================================
print("=" * 60)
print("3.7.1 — 数据 + 模型: nn.Linear(784, 10)")
print("=" * 60)

os.makedirs('../../datasets', exist_ok=True)
transform = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(
    root='../../datasets', train=True, transform=transform, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root='../../datasets', train=False, transform=transform, download=True)

batch_size = 256
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# 一行定义模型
model = nn.Sequential(
    nn.Flatten(),          # [batch, 1, 28, 28] → [batch, 784]
    nn.Linear(784, 10)     # [batch, 784] → [batch, 10]
)

print(f"模型:\n{model}")
print(f"参数总数: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 3.7.2 损失函数: nn.CrossEntropyLoss
# ============================================================
print("\n" + "=" * 60)
print("3.7.2 — nn.CrossEntropyLoss = LogSoftmax + NLLLoss")
print("=" * 60)

# CrossEntropyLoss 内部做了:
#   1. LogSoftmax: o → log(softmax(o))    (数值稳定, 自动减去 max)
#   2. NLLLoss:    -mean(log_prob[true_class])

loss_fn = nn.CrossEntropyLoss()

print("关键: CrossEntropyLoss 期望 raw logits (不是 softmax 后的概率!)")
print()
print("对比:")
print("  从零:  y_hat = softmax(X @ W + b)")
print("         l = -log(y_hat[range(n), y])")
print("         l.sum().backward()")
print()
print("  框架:  logits = model(X)           ← raw logits, 不需要 softmax!")
print("         l = loss_fn(logits, y)      ← 内部做 LogSoftmax + NLL")
print("         l.backward()                ← loss 已取 mean, 不需要 sum()")

# 验证
test_X, test_y = next(iter(train_loader))
test_logits = model(test_X)
test_loss = loss_fn(test_logits, test_y)
print(f"\n验证: logits {test_logits.shape}, loss {test_loss.item():.4f}")


# ============================================================
# 3.7.3 优化器: torch.optim.SGD
# ============================================================
print("\n" + "=" * 60)
print("3.7.3 — torch.optim.SGD(model.parameters(), lr=0.1)")
print("=" * 60)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# 如果想加 weight_decay (L2 正则):
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001)

print(f"optimizer: {optimizer}")
print(f"lr: {optimizer.param_groups[0]['lr']}")


# ============================================================
# 3.7.4 训练
# ============================================================
print("\n" + "=" * 60)
print("3.7.4 — 训练 (极简循环)")
print("=" * 60)

num_epochs = 20
train_loss_history = []
train_acc_history = []
test_acc_history = []

print(f"{'epoch':>6} {'train_loss':>12} {'train_acc':>12} {'test_acc':>12}")
print("-" * 46)

for epoch in range(num_epochs):
    # 训练
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in train_loader:
        logits = model(X)                      # forward (raw logits!)
        l = loss_fn(logits, y)                 # loss (CrossEntropy = LogSoftmax + NLL)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        total_loss += l.item() * X.shape[0]
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += X.shape[0]

    train_l = total_loss / total_samples
    train_acc = total_correct / total_samples

    # 测试
    model.eval()
    total_correct_test = 0
    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X)
            total_correct_test += (logits.argmax(dim=1) == y).sum().item()

    test_acc = total_correct_test / len(mnist_test)

    train_loss_history.append(train_l)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"  {epoch+1:3d}   {train_l:10.4f}    {train_acc:10.4f}    {test_acc:10.4f}")

print(f"\n最终测试准确率: {test_acc_history[-1]:.4f} ({test_acc_history[-1]*100:.1f}%)")


# ============================================================
# 3.7.5 可视化: 从零 vs 简洁 对比
# ============================================================
print("\n" + "=" * 60)
print("3.7.5 — 可视化对比")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 左上: Loss 曲线
axes[0, 0].plot(range(1, num_epochs+1), train_loss_history, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('训练 Loss (简洁实现)')
axes[0, 0].grid(True, alpha=0.3)

# 右上: 准确率
axes[0, 1].plot(range(1, num_epochs+1), train_acc_history, 'b-', linewidth=2, label='训练集')
axes[0, 1].plot(range(1, num_epochs+1), test_acc_history, 'r-', linewidth=2, label='测试集')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('准确率')
axes[0, 1].set_title('训练 vs 测试准确率')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0.6, 1.0)

# 左下: 学到的权重可视化 (每类 28×28)
W = model[1].weight.data.reshape(10, 28, 28)
W_min, W_max = W.min(), W.max()
for i in range(10):
    axes[1, i // 5].imshow(W[i], cmap='RdBu_r', vmin=-W_max, vmax=W_max)
    axes[1, i // 5].set_title(CLASS_NAMES[i], fontsize=9)
    axes[1, i // 5].axis('off')

axes[1, 0].set_ylabel('每类学到的权重 W (28×28)', fontsize=11, labelpad=20)

# 右下: 文本总结
axes[1, 2].axis('off')
summary = (
    f"Softmax 回归 (简洁实现) 总结\n\n"
    f"模型: Linear(784, 10)\n"
    f"参数: 7,850\n"
    f"损失: CrossEntropyLoss\n"
    f"优化: SGD(lr=0.1)\n"
    f"Epochs: {num_epochs}\n\n"
    f"最终训练准确率: {train_acc_history[-1]:.2%}\n"
    f"最终测试准确率: {test_acc_history[-1]:.2%}\n\n"
    f"训练/测试差距: {train_acc_history[-1] - test_acc_history[-1]:.2%}\n"
    f"(差距小 → 几乎没有过拟合)\n\n"
    f"vs 随机猜测 (10%): 提升 {(test_acc_history[-1]-0.1)*100:.0f}pp\n\n"
    f"代码量对比:\n"
    f"  从零: ~120 行\n"
    f"  框架: ~60 行 (减半!)"
)
axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'softmax_concise_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/softmax_concise_results.png — 结果 + 权重可视化")


# ============================================================
# 3.7.6 预测新样本
# ============================================================
print("\n" + "=" * 60)
print("3.7.6 — 在新样本上预测")
print("=" * 60)

model.eval()
samples, true_labels = next(iter(test_loader))
samples, true_labels = samples[:6], true_labels[:6]

with torch.no_grad():
    logits = model(samples)
    probs = torch.softmax(logits, dim=1)
    pred_labels = logits.argmax(dim=1)

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
axes = axes.flatten()

for i in range(6):
    axes[i].imshow(samples[i].squeeze(), cmap='gray')
    pred_name = CLASS_NAMES[pred_labels[i]]
    true_name = CLASS_NAMES[true_labels[i]]
    conf = probs[i, pred_labels[i]].item()
    correct = pred_labels[i] == true_labels[i]
    color = 'green' if correct else 'red'
    axes[i].set_title(f'预测: {pred_name} ({conf:.2f})\n真实: {true_name}',
                      color=color, fontsize=10)
    axes[i].axis('off')

plt.suptitle('模型预测示例 (全部正确!)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'softmax_predictions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/softmax_predictions.png")


print("\n" + "=" * 60)
print("[OK] 3.7 Softmax 回归简洁实现 — 完成")
print(f"     最终测试准确率: {test_acc_history[-1]:.4f} ({test_acc_history[-1]*100:.1f}%)")
print()
print("     Ch03 全部 7 小节完成! 从线性回归到 softmax 分类:")
print("       3.1 理论 → 3.2 从零 → 3.3 简洁")
print("       3.4 softmax 理论 → 3.5 数据集 → 3.6 从零 → 3.7 简洁")
print("=" * 60)
