"""
d2l 3.2 — 线性回归从零实现 (Linear Regression from Scratch)

只用 tensor 和 autograd，不调 nn.Linear / nn.MSELoss。
目的: 看清楚底层每一步在发生什么。

7 个子步骤:
  3.2.1 生成数据集
  3.2.2 读取数据集 (yield 生成器)
  3.2.3 初始化模型参数
  3.2.4 定义模型 (X @ w + b)
  3.2.5 定义损失函数 (MSE / 2)
  3.2.6 定义优化算法 (小批量 SGD)
  3.2.7 训练循环
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 3.2.1 生成数据集
# ============================================================
print("=" * 60)
print("3.2.1 — 生成合成数据集: y = X @ w + b + noise")
print("=" * 60)

torch.manual_seed(42)

# 超参数
num_examples = 1000
num_inputs = 2        # 特征数 d
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2

# 从标准正态生成特征
features = torch.normal(0, 1, (num_examples, num_inputs))
# 按线性模型生成标签，加观测噪声
labels = features @ true_w.reshape(-1, 1) + true_b
labels += torch.normal(0, 0.01, labels.shape)

print(f"features  shape: {features.shape}  ({num_examples} samples, {num_inputs} features)")
print(f"labels    shape: {labels.shape}")
print(f"真实参数: w = {true_w.numpy()}, b = {true_b}")
print(f"标签范围: [{labels.min().item():.2f}, {labels.max().item():.2f}]")

# 可视化: 两个特征分别对 label 的散点图
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i in range(2):
    axes[i].scatter(features[:, i].numpy(), labels[:, 0].numpy(),
                    alpha=0.4, s=15, edgecolors='none')
    axes[i].set_xlabel(f'特征 x{i+1} (w{i+1}={true_w[i]:.1f})')
    axes[i].set_ylabel('标签 y')
    axes[i].set_title(f'特征 {i+1} vs 标签')
    axes[i].grid(True, alpha=0.3)

# 右: 噪声分布
noise_samples = (labels - features @ true_w.reshape(-1, 1) - true_b).flatten().numpy()
axes[2].hist(noise_samples, bins=30, edgecolor='black', alpha=0.7)
axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[2].set_xlabel('noise')
axes[2].set_ylabel('频数')
axes[2].set_title(f'观测噪声分布 (std={noise_samples.std():.4f}, 真实 0.01)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d2l/ch03-linear/notes/synthetic_data.png', dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/synthetic_data.png — 合成数据可视化")

print("\n关键认知: 生成数据时知道真实参数 → 可以验证算法是否正确学到它们")


# ============================================================
# 3.2.2 读取数据集
# ============================================================
print("\n" + "=" * 60)
print("3.2.2 — 数据迭代器: yield batch")
print("=" * 60)

def data_iter(batch_size, features, labels):
    """每次 yield 一个随机 mini-batch"""
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机打乱 —— 每 epoch 样本顺序不同
    shuffled = torch.randperm(num_examples).tolist()

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(shuffled[i: i + batch_size])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
print(f"batch_size = {batch_size}")
first_X, first_y = next(data_iter(batch_size, features, labels))
print(f"第一个 batch: X shape {first_X.shape}, y shape {first_y.shape}")
print(f"  X[0] = {first_X[0].numpy()}, y[0] = {first_y[0, 0]:.4f}")
print()
print("为什么用 yield?")
print("  惰性求值: 100 万样本, 每次只在内存放 batch_size 个")
print("  randperm: 每 epoch 打乱顺序, 避免模型记住样本顺序")


# ============================================================
# 3.2.3 初始化模型参数
# ============================================================
print("\n" + "=" * 60)
print("3.2.3 — 初始化模型参数: w ~ N(0, 0.01), b = 0")
print("=" * 60)

w = torch.normal(0, 0.01, size=(num_inputs, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print(f"w 初始值 (requires_grad={w.requires_grad}): {w.flatten().detach().numpy()}")
print(f"b 初始值 (requires_grad={b.requires_grad}): {b.item():.4f}")
print()
print("w 用小随机值: 打破对称性 (如果全 0, 所有权重梯度相同, 无法差异化学习)")
print("b 初始化为 0:  偏置不需要对称性打破")
print("requires_grad=True: 告诉 autograd 追踪它们 → 训练时才能求梯度")


# ============================================================
# 3.2.4 定义模型
# ============================================================
print("\n" + "=" * 60)
print("3.2.4 — 定义模型: linreg(X, w, b) = X @ w + b")
print("=" * 60)

def linreg(X, w, b):
    """线性回归: 整个'神经网络'就是一行矩阵乘法 + 加法"""
    return X @ w + b

# 验证形状
test_out = linreg(first_X, w, b)
print(f"linreg(X[{first_X.shape}], w[{w.shape}], b[{b.shape}])")
print(f"  → output shape: {test_out.shape}  (应为 [{batch_size}, 1])")


# ============================================================
# 3.2.5 定义损失函数
# ============================================================
print("\n" + "=" * 60)
print("3.2.5 — 损失函数: squared_loss = (y_hat - y)^2 / 2")
print("=" * 60)

def squared_loss(y_hat, y):
    """均方误差 / 2
    /2 是技巧: 求导后 d/dx (x^2/2) = x, 常数抵消, 梯度更简洁
    返回向量 (每个样本的 loss), 训练时 sum 再 backward
    """
    return ((y_hat - y.reshape(y_hat.shape)) ** 2) / 2

# 验证
test_loss = squared_loss(test_out, first_y)
print(f"squared_loss shape: {test_loss.shape}  (每个样本一个 loss)")
print(f"前 3 个样本 loss: {test_loss[:3].flatten().detach().numpy()}")
print()
print("为什么 /2 ?")
print("  d/dw ((Xw - y)^2 / 2) = X^T (Xw - y)  ← 干净的梯度, 没有系数 2")
print("y.reshape(y_hat.shape): 防御性形状对齐, 避免广播意外")


# ============================================================
# 3.2.6 定义优化算法 (小批量 SGD)
# ============================================================
print("\n" + "=" * 60)
print("3.2.6 — 小批量 SGD: param -= lr * param.grad / batch_size")
print("=" * 60)

def sgd(params, lr, batch_size):
    """小批量随机梯度下降 (Mini-batch Stochastic Gradient Descent)

    关键细节:
    1. torch.no_grad(): 参数更新不属于计算图, 不能被 autograd 追踪
    2. grad / batch_size: loss 是 sum (没取 mean), 所以梯度要除以 batch_size
    3. grad.zero_(): 清零! Ch02 学的 grad 累加机制 — 不清零下次 backward 会累加
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

print("sgd 函数定义完成")
print("  核心: no_grad() 块 + 梯度除以 batch_size + zero_() 清零")
print()
print("三个关键细节:")
print("  1. no_grad():    权重更新不是前向计算, autograd 不需要追踪")
print("  2. /batch_size:  loss=sum → grad=sum → 除以 batch_size 取平均")
print("  3. grad.zero_(): Ch02 重点: grad 会累加! 每次更新前必须清零")


# ============================================================
# 3.2.7 训练
# ============================================================
print("\n" + "=" * 60)
print("3.2.7 — 训练循环 (Ch02 知识全部用上)")
print("=" * 60)

lr = 0.03
num_epochs = 5
net = linreg
loss = squared_loss

# 记录训练过程
loss_history = []
w_history = []  # 记录 w 的变化
b_history = []  # 记录 b 的变化

for epoch in range(num_epochs):
    # 每个 epoch 遍历全部数据
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)      # ① forward + loss
        l.sum().backward()             # ② backward (sum 把向量变标量)
        sgd([w, b], lr, batch_size)    # ③ 更新参数 (内含 zero_grad)

    # 每个 epoch 后计算整体 loss
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        epoch_loss = train_l.mean().item()
        loss_history.append(epoch_loss)
        w_history.append(w.detach().flatten().numpy().copy())
        b_history.append(b.item())

    print(f"epoch {epoch+1}, loss {epoch_loss:f},  "
          f"w=[{w[0,0].item():7.4f}, {w[1,0].item():7.4f}],  b={b.item():7.4f}")

# 最终比较
print(f"\n{'='*40}")
print(f"真实 w:   {true_w.numpy()}     学到的 w: {w.flatten().detach().numpy()}")
print(f"真实 b:   {true_b}                  学到的 b: {b.item():.4f}")
print(f"w 误差:   {(w.flatten() - true_w).abs().max().item():.6f}")
print(f"b 误差:   {abs(b.item() - true_b):.6f}")


# ============================================================
# 可视化: 训练过程
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: loss 曲线
axes[0, 0].plot(range(1, num_epochs+1), loss_history, 'b-o', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('训练 Loss 下降曲线')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')
for i, l in enumerate(loss_history):
    axes[0, 0].annotate(f'{l:.5f}', (i+1, l), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8)

# 右上: w 学习轨迹
w_history = np.array(w_history)
axes[0, 1].plot(range(1, num_epochs+1), w_history[:, 0], 'b-o',
                linewidth=2, markersize=8, label=f'w1 → 目标 {true_w[0]:.1f}')
axes[0, 1].plot(range(1, num_epochs+1), w_history[:, 1], 'r-s',
                linewidth=2, markersize=8, label=f'w2 → 目标 {true_w[1]:.1f}')
axes[0, 1].axhline(y=true_w[0].item(), color='b', linestyle='--', alpha=0.5)
axes[0, 1].axhline(y=true_w[1].item(), color='r', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('权重值')
axes[0, 1].set_title('权重 w 学习轨迹')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 左下: b 学习轨迹
axes[1, 0].plot(range(1, num_epochs+1), b_history, 'g-o', linewidth=2, markersize=8)
axes[1, 0].axhline(y=true_b, color='g', linestyle='--', alpha=0.5, label=f'目标 b={true_b:.1f}')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('偏置值')
axes[1, 0].set_title('偏置 b 学习轨迹')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 右下: 真实 vs 预测散点
with torch.no_grad():
    y_pred_final = net(features, w, b)
axes[1, 1].scatter(labels.numpy(), y_pred_final.numpy(), alpha=0.4, s=15, edgecolors='none')
axes[1, 1].plot([labels.min(), labels.max()], [labels.min(), labels.max()],
                'r--', linewidth=2, label='完美预测')
axes[1, 1].set_xlabel('真实 y')
axes[1, 1].set_ylabel('预测 y_hat')
axes[1, 1].set_title('训练后: 真实 vs 预测')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d2l/ch03-linear/notes/training_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] notes/training_results.png — 训练结果四合一")


# ============================================================
# 补充实验: 不同学习率对比
# ============================================================
print("\n" + "=" * 60)
print("补充实验 — 学习率的影响: lr=0.001 vs 0.03 vs 0.5 vs 2.0")
print("=" * 60)

def train_with_lr(lr, num_epochs=8):
    w_test = torch.normal(0, 0.01, size=(num_inputs, 1), requires_grad=True)
    b_test = torch.zeros(1, requires_grad=True)
    loss_curve = []

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = squared_loss(linreg(X, w_test, b_test), y)
            l.sum().backward()
            sgd([w_test, b_test], lr, batch_size)
        with torch.no_grad():
            train_l = squared_loss(linreg(features, w_test, b_test), labels).mean()
            loss_curve.append(train_l.item())
    return loss_curve

lrs = [0.001, 0.01, 0.03, 0.1, 0.5, 2.0]
fig, ax = plt.subplots(figsize=(10, 5))

for lr_val in lrs:
    curve = train_with_lr(lr_val, num_epochs=8)
    label = f'lr={lr_val}' + (' (最优)' if lr_val == 0.03 else '')
    ax.plot(range(1, len(curve)+1), curve, '-o', linewidth=2, markersize=6, label=label)

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('不同学习率下 Loss 下降对比')
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d2l/ch03-linear/notes/learning_rate_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/learning_rate_comparison.png — 学习率对比")
print("结论: lr=0.03 最快收敛; lr=0.001 太慢; lr=2.0 发散 (loss 爆炸)")


print("\n" + "=" * 60)
print("[OK] 3.2 线性回归从零实现 — 完成")
print("")
print("回顾: 完整的训练循环 =")
print("  for epoch:")
print("      for batch in data_iter():")
print("          l = loss(model(X, w, b), y)   # forward")
print("          l.sum().backward()             # backward")
print("          sgd([w, b], lr, batch_size)   # update (内含 grad.zero_())")
print("")
print("这 4 行代码是整个深度学习的核心循环 —— 无论是 CNN、RNN 还是 Transformer。")
print("=" * 60)
