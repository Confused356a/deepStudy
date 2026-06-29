"""
d2l 3.6 — Softmax 回归的从零实现 (Softmax Regression from Scratch)

只用 tensor 和 autograd, 在 Fashion-MNIST 上训练 softmax 回归。
关键挑战:
  1. Softmax 数值稳定 (减去 max)
  2. 交叉熵损失: 取真实类别概率的负对数
  3. 准确率计算: argmax 比较
  4. 784 维输入 → 10 类输出
"""
import torch
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
# 3.6.1 加载数据
# ============================================================
print("=" * 60)
print("3.6.1 — 加载 Fashion-MNIST")
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

print(f"训练: {len(mnist_train)} 样本, {len(train_loader)} batches")
print(f"测试: {len(mnist_test)} 样本, {len(test_loader)} batches")


# ============================================================
# 3.6.2 初始化模型参数
# ============================================================
print("\n" + "=" * 60)
print("3.6.2 — 初始化 W(784, 10) 和 b(10)")
print("=" * 60)

num_inputs = 784   # 28 × 28
num_outputs = 10

# W 从 N(0, 0.01) 初始化, /sqrt 可改善收敛
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

print(f"W shape: {W.shape}  ({num_inputs} × {num_outputs})")
print(f"b shape: {b.shape}  ({num_outputs},)")
print(f"总参数: {W.numel() + b.numel():,}")

# 验证: 随机 batch 前向
test_batch = next(iter(train_loader))[0].reshape(-1, num_inputs)
print(f"\n验证: X @ W + b")
print(f"  X: {test_batch.shape}  @  W: {W.shape}  +  b: {b.shape}")
print(f"  →  {(test_batch @ W + b).shape}  (应为 [{batch_size}, 10])")


# ============================================================
# 3.6.3 定义 Softmax (数值稳定版)
# ============================================================
print("\n" + "=" * 60)
print("3.6.3 — Softmax: y_hat = exp(o - max(o)) / sum(exp(o - max(o)))")
print("=" * 60)

def softmax(X):
    """X: (batch, k) → (batch, k) 概率分布
    减去每行最大值: 防止 exp 溢出, 数学等价
    """
    X_exp = torch.exp(X - X.max(dim=1, keepdim=True).values)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

# 测试
test_logits = torch.randn(3, 10)
test_probs = softmax(test_logits)
print(f"输入 shape: {test_logits.shape}")
print(f"输出 shape: {test_probs.shape}")
print(f"每行概率和: {test_probs.sum(dim=1).numpy()}  (应全为 1)")
print(f"每行最大值:  {test_probs.max(dim=1).values.numpy()}  (最大概率)")
print(f"softmax 保持顺序: 最大 logit → 最大概率? "
      f"{(test_logits.argmax(dim=1) == test_probs.argmax(dim=1)).all().item()}")


# ============================================================
# 3.6.4 定义模型
# ============================================================
print("\n" + "=" * 60)
print("3.6.4 — 模型: net(X) = softmax(X @ W + b)")
print("=" * 60)

def net(X):
    """X: (batch, 784) 展平后的图像 → (batch, 10) 概率"""
    return softmax(X @ W + b)

# 验证
y_hat_test = net(test_batch)
print(f"net(X) 输出: {y_hat_test.shape}  (应为 [{batch_size}, 10])")
print(f"前 3 个样本的预测概率:\n{y_hat_test[:3].detach().numpy()}")


# ============================================================
# 3.6.5 定义交叉熵损失
# ============================================================
print("\n" + "=" * 60)
print("3.6.5 — 交叉熵: L = -log(y_hat[y])")
print("=" * 60)

def cross_entropy(y_hat, y):
    """y_hat: (batch, k) 预测概率
    y:      (batch,)   真实类别索引 (Long)
    返回:   (batch,)    每个样本的 loss
    """
    # 取 y_hat[range(n), y] —— 每个样本真实类别的预测概率
    return -torch.log(y_hat[range(len(y_hat)), y])

# 测试
test_labels = next(iter(train_loader))[1][:5]
test_ce = cross_entropy(y_hat_test[:5], test_labels)
print(f"5 个样本的交叉熵: {test_ce.detach().numpy()}")
print(f"真实标签: {test_labels.tolist()}")
print(f"预测最可能的类别: {y_hat_test[:5].argmax(dim=1).tolist()}")
print()
print("交叉熵直觉:")
print("  预测概率高 (如 0.8) → -log(0.8) = 0.223 → 低 loss")
print("  预测概率低 (如 0.01) → -log(0.01) = 4.605 → 高 loss")
print("  完美预测 (1.0) → -log(1.0) = 0 → zero loss")


# ============================================================
# 3.6.6 分类准确率
# ============================================================
print("\n" + "=" * 60)
print("3.6.6 — 准确率: (预测类别 == 真实类别).mean()")
print("=" * 60)

def accuracy(y_hat, y):
    """计算一个 batch 的准确率"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(data_loader, net_fn):
    """在整个数据集上评估准确率"""
    metric = Accumulator(2)  # [正确数, 总数]
    with torch.no_grad():
        for X, y in data_loader:
            X = X.reshape(-1, num_inputs)
            y_hat = net_fn(X)
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]

class Accumulator:
    """累加器"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# 随机初始化下的准确率 (应接近 1/10 = 0.1)
init_acc = evaluate_accuracy(test_loader, net)
print(f"随机初始化下测试准确率: {init_acc:.4f}  (≈ 0.1 = 随机猜测)")


# ============================================================
# 3.6.7 训练
# ============================================================
print("\n" + "=" * 60)
print("3.6.7 — 训练 Softmax 回归")
print("=" * 60)

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.1
num_epochs = 20

train_loss_history = []
train_acc_history = []
test_acc_history = []

print(f"lr={lr}, num_epochs={num_epochs}, batch_size={batch_size}")
print(f"{'epoch':>6} {'train_loss':>12} {'train_acc':>12} {'test_acc':>12}")
print("-" * 46)

for epoch in range(num_epochs):
    metric = Accumulator(3)  # [train_loss_sum, train_acc_sum, train_n]

    for X, y in train_loader:
        X = X.reshape(-1, num_inputs)   # [256, 1, 28, 28] → [256, 784]
        y_hat = net(X)
        l = cross_entropy(y_hat, y)

        l.sum().backward()
        sgd([W, b], lr, batch_size)

        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    train_l = metric[0] / metric[2]
    train_acc = metric[1] / metric[2]
    test_acc = evaluate_accuracy(test_loader, net)

    train_loss_history.append(train_l)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"  {epoch+1:3d}   {train_l:10.4f}    {train_acc:10.4f}    {test_acc:10.4f}")

final_test_acc = evaluate_accuracy(test_loader, net)
print(f"\n最终测试准确率: {final_test_acc:.4f}  ({final_test_acc*100:.1f}%)")


# ============================================================
# 3.6.8 可视化: 训练曲线 + 错误样本
# ============================================================
print("\n" + "=" * 60)
print("3.6.8 — 可视化: 训练曲线 & 预测分析")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: Loss 曲线
axes[0, 0].plot(range(1, num_epochs+1), train_loss_history, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('训练 Loss')
axes[0, 0].grid(True, alpha=0.3)

# 右上: 准确率曲线
axes[0, 1].plot(range(1, num_epochs+1), train_acc_history, 'b-', linewidth=2, label='训练集')
axes[0, 1].plot(range(1, num_epochs+1), test_acc_history, 'r-', linewidth=2, label='测试集')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('准确率')
axes[0, 1].set_title('训练 vs 测试准确率')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 1)

# 左下: 每类准确率
with torch.no_grad():
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    for X, y in test_loader:
        X = X.reshape(-1, num_inputs)
        y_hat = net(X)
        pred = y_hat.argmax(dim=1)
        for c in range(10):
            mask = (y == c)
            class_correct[c] += (pred[mask] == y[mask]).sum().item()
            class_total[c] += mask.sum().item()

class_acc = class_correct / class_total
bar_colors = ['#2ecc71' if a > 0.8 else '#f39c12' if a > 0.7 else '#e74c3c' for a in class_acc]
axes[1, 0].barh(range(10), class_acc, color=bar_colors, edgecolor='black', alpha=0.8)
axes[1, 0].set_yticks(range(10))
axes[1, 0].set_yticklabels(CLASS_NAMES)
axes[1, 0].set_xlabel('准确率')
axes[1, 0].set_title('每类测试准确率')
axes[1, 0].set_xlim(0, 1)
for i, (acc, name) in enumerate(zip(class_acc, CLASS_NAMES)):
    axes[1, 0].text(acc + 0.01, i, f'{acc:.2f}', va='center', fontsize=8)

# 右下: 混淆矩阵的一部分 (top 错误对)
with torch.no_grad():
    all_preds = []
    all_labels = []
    for X, y in test_loader:
        X = X.reshape(-1, num_inputs)
        preds = net(X).argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

conf_matrix = np.zeros((10, 10), dtype=int)
for p, l in zip(all_preds, all_labels):
    conf_matrix[l][p] += 1

# 归一化
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
im = axes[1, 1].imshow(conf_matrix_norm, cmap='Blues', aspect='auto')
axes[1, 1].set_xticks(range(10))
axes[1, 1].set_yticks(range(10))
axes[1, 1].set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=7)
axes[1, 1].set_yticklabels(CLASS_NAMES, fontsize=7)
axes[1, 1].set_xlabel('预测类别')
axes[1, 1].set_ylabel('真实类别')
axes[1, 1].set_title('混淆矩阵 (行归一化)')
plt.colorbar(im, ax=axes[1, 1], shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'softmax_scratch_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/softmax_scratch_results.png — 训练结果四合一")


# ============================================================
# 3.6.9 预测示例: 看看模型在哪些样本上犯错
# ============================================================
print("\n" + "=" * 60)
print("3.6.9 — 错误分析: 模型最容易混淆的类别对")
print("=" * 60)

# 找出最容易被混淆的类别对 (排除对角线)
off_diag = conf_matrix.copy()
np.fill_diagonal(off_diag, 0)
# 找 top 5 混淆对
flat_indices = np.argsort(off_diag, axis=None)[-5:]
rows, cols = np.unravel_index(flat_indices, off_diag.shape)

print("Top 5 混淆对 (真实类别 → 误判为):")
for r, c in zip(rows[::-1], cols[::-1]):
    print(f"  {CLASS_NAMES[r]:<14s} → {CLASS_NAMES[c]:<14s}  ({off_diag[r, c]:,} 次)")

# 展示几个错误样本
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

# 收集错误样本
with torch.no_grad():
    error_samples = []
    for X, y in test_loader:
        X_flat = X.reshape(-1, num_inputs)
        preds = net(X_flat).argmax(dim=1)
        wrong_mask = (preds != y)
        for j in range(len(y)):
            if wrong_mask[j]:
                error_samples.append((X[j], y[j].item(), preds[j].item()))
                if len(error_samples) >= 8:
                    break
        if len(error_samples) >= 8:
            break

for i, (img, true_l, pred_l) in enumerate(error_samples[:8]):
    axes[i].imshow(img.squeeze(), cmap='gray')
    color = 'green' if true_l == pred_l else 'red'
    axes[i].set_title(f'真:{CLASS_NAMES[true_l]}\n→ 误判:{CLASS_NAMES[pred_l]}',
                      color=color, fontsize=9)
    axes[i].axis('off')

plt.suptitle('模型预测错误的样本', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'softmax_errors.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/softmax_errors.png — 错误样本展示")


print("\n" + "=" * 60)
print("[OK] 3.6 Softmax 回归从零实现 — 完成")
print(f"     最终测试准确率: {final_test_acc:.4f} ({final_test_acc*100:.1f}%)")
print("     关键收获:")
print("       1. Softmax: 数值稳定技巧 (减去 max)")
print("       2. 交叉熵: 从预测概率中取真实类别对应的值")
print("       3. 784 维 → 10 类, 线性模型也能达到 ~83%")
print("       4. 准确率计算: argmax 比较 + 累加统计")
print("=" * 60)
