"""
d2l 3.3 — 线性回归的简洁实现 (Concise Implementation)

用 PyTorch 高层 API 重写 3.2 的从零实现:
  nn.Linear        → 替代手动 w, b 和 linreg()
  nn.MSELoss       → 替代 squared_loss()
  torch.optim.SGD  → 替代手动 sgd()
  DataLoader       → 替代 data_iter()

目的: 体会框架帮你做了什么, 对比从零实现理解每一层的抽象。
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 生成同样的合成数据 (和 3.2 完全一致)
# ============================================================
torch.manual_seed(42)
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
n, d = 1000, 2

features = torch.normal(0, 1, (n, d))
labels = features @ true_w.reshape(-1, 1) + true_b
labels += torch.normal(0, 0.01, labels.shape)

print("=" * 60)
print("3.3.1 — 数据集: TensorDataset + DataLoader")
print("=" * 60)

# 方式一(从零): 手写 data_iter + yield
# 方式二(框架): TensorDataset + DataLoader
dataset = TensorDataset(features, labels)
batch_size = 10
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"dataset  类型: {type(dataset).__name__}")
print(f"dataloader 类型: {type(data_loader).__name__}")
print(f"batch_size: {batch_size}")
print(f"n_batches:  {len(data_loader)}  ({n} / {batch_size})")
print()
print("对比:")
print("  从零: def data_iter(batch_size, features, labels):")
print("          shuffled = randperm(n)")
print("          for i in range(0, n, batch_size):")
print("              yield features[indices[i:i+bs]], labels[...]")
print("  框架: DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True)")
print("  省了 (~12 行 → 1 行)")


# ============================================================
# 3.3.2 模型: nn.Linear
# ============================================================
print("\n" + "=" * 60)
print("3.3.2 — 模型: nn.Linear(2, 1)")
print("=" * 60)

# 从零: w = torch.normal(0, 0.01, (2,1), requires_grad=True)
#        b = torch.zeros(1, requires_grad=True)
#        def linreg(X, w, b): return X @ w + b
#
# 框架一行:
model = nn.Linear(2, 1)

print(f"model: {model}")
print(f"model.weight shape: {model.weight.shape}  (等价于 w)")
print(f"model.bias   shape: {model.bias.shape}    (等价于 b)")
print(f"model.weight 初始值: {model.weight.data.flatten().numpy()}")
print(f"model.bias   初始值: {model.bias.item():.4f}")
print()
print("nn.Linear 在内部做了:")
print("  1. 创建 weight ~ U(-sqrt(k), sqrt(k))  (kaiming 初始化)")
print("  2. 创建 bias = 0")
print("  3. 注册为可学习参数 (requires_grad=True)")
print("  4. forward(): y = X @ W^T + b")

# 验证: 修改权重为随机初始化 (模拟我们之前的 N(0, 0.01))
nn.init.normal_(model.weight, mean=0, std=0.01)
nn.init.zeros_(model.bias)
print(f"\ninit 后 weight: {model.weight.data.flatten().numpy()}")


# ============================================================
# 3.3.3 损失函数: nn.MSELoss
# ============================================================
print("\n" + "=" * 60)
print("3.3.3 — 损失函数: nn.MSELoss(reduction='none')")
print("=" * 60)

# 从零: def squared_loss(y_hat, y):
#           return ((y_hat - y.reshape(y_hat.shape)) ** 2) / 2
#
# 框架:
loss_fn = nn.MSELoss(reduction='none')   # 'none' = 保留每个样本的 loss

test_X = features[:3]
test_y = labels[:3]
with torch.no_grad():
    test_pred = model(test_X)
    test_loss = loss_fn(test_pred, test_y)

print(f"MSE (3 样本): {test_loss.flatten().numpy()}")
print()
print("对比:")
print(f"  从零: ((y_hat - y)^2) / 2         ← 除以 2 让梯度简洁")
print(f"  框架: (y_hat - y)^2               ← 标准 MSE, 不减半")
print(f"  梯度差别只是常数因子 2, 可通过 lr 吸收 → 最终收敛结果等价")


# ============================================================
# 3.3.4 优化器: torch.optim.SGD
# ============================================================
print("\n" + "=" * 60)
print("3.3.4 — 优化器: torch.optim.SGD(model.parameters(), lr=0.03)")
print("=" * 60)

# 从零: def sgd(params, lr, batch_size):
#           with torch.no_grad():
#               for p in params:
#                   p -= lr * p.grad / batch_size
#                   p.grad.zero_()
#
# 框架:
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

print(f"optimizer: {optimizer}")
print(f"param_groups: lr={optimizer.param_groups[0]['lr']}")
print()
print("对比:")
print("  从零: 手动遍历 [w, b], 手动 -= lr * grad / bs, 手动 zero_()")
print("  框架: optimizer.zero_grad() + optimizer.step()  两行搞定")


# ============================================================
# 3.3.5 训练 (和 3.2 完全相同的循环, 但用 API)
# ============================================================
print("\n" + "=" * 60)
print("3.3.5 — 训练循环对比")
print("=" * 60)

print("\n[从零实现]                          [简洁实现]")
print("for epoch:                           for epoch:")
print("  for X,y in data_iter():              for X,y in data_loader:")
print("    l = linreg(X,w,b)   # forward        l = model(X)        # forward")
print("    l = squared_loss(l,y) # loss         l = loss_fn(l, y)   # loss")
print("    l.sum().backward()   # bwd           l.mean().backward() # bwd")
print("    sgd([w,b], lr, bs)  # update        optimizer.step()    # update")
print("                            (含 zero_grad)  optimizer.zero_grad()")

# 实际训练
num_epochs = 5
loss_history = []
w_history = []

for epoch in range(num_epochs):
    for X, y in data_loader:
        l = loss_fn(model(X), y)      # 前向 + 损失
        l.mean().backward()           # 反向 (mean 转标量)
        optimizer.step()              # 更新参数
        optimizer.zero_grad()         # 清零梯度

    # 每个 epoch 后评估
    with torch.no_grad():
        train_l = loss_fn(model(features), labels).mean()
        loss_history.append(train_l.item())
        w_history.append(model.weight.data.flatten().numpy().copy())

    w_now = model.weight.data.flatten().numpy()
    b_now = model.bias.item()
    print(f"epoch {epoch+1}, loss {train_l.item():f},  "
          f"w=[{w_now[0]:7.4f}, {w_now[1]:7.4f}],  b={b_now:.4f}")

print(f"\n真实 w: {true_w.numpy()}  学到的 w: {model.weight.data.flatten().numpy()}")
print(f"真实 b: {true_b}           学到的 b: {model.bias.item():.4f}")


# ============================================================
# 可视化: 从零 vs 简洁 训练轨迹对比
# ============================================================
# 用从零实现再训一次, 方便对比
torch.manual_seed(42)
w_scratch = torch.normal(0, 0.01, size=(d, 1), requires_grad=True)
b_scratch = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    return X @ w + b

def sgd_scratch(params, lr, bs):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / bs
            p.grad.zero_()

loss_scratch = []
for epoch in range(num_epochs):
    for i in range(0, n, batch_size):
        idx = torch.randint(0, n, (batch_size,))
        Xb, yb = features[idx], labels[idx]
        l = ((linreg(Xb, w_scratch, b_scratch) - yb) ** 2) / 2
        l.sum().backward()
        sgd_scratch([w_scratch, b_scratch], 0.03, batch_size)
    with torch.no_grad():
        tl = ((linreg(features, w_scratch, b_scratch) - labels) ** 2).mean()
        loss_scratch.append(tl.item())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(range(1, num_epochs+1), loss_scratch, 'b-o', linewidth=2, markersize=8, label='从零实现 (squared_loss/2)')
ax1.plot(range(1, num_epochs+1), loss_history, 'r-s', linewidth=2, markersize=8, label='简洁实现 (nn.MSELoss)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('从零 vs 简洁 — Loss 下降对比')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.bar(['w1', 'w2', 'b'],
        [2.0, -3.4, 4.2],
        width=0.25, label='真实值', alpha=0.7)
ax2.bar(np.arange(3) + 0.25,
        [w_scratch[0,0].item(), w_scratch[1,0].item(), b_scratch.item()],
        width=0.25, label='从零', alpha=0.7)
ax2.bar(np.arange(3) + 0.5,
        [model.weight.data[0,0].item(), model.weight.data[0,1].item(), model.bias.item()],
        width=0.25, label='简洁', alpha=0.7)
ax2.set_xticks(np.arange(3) + 0.25)
ax2.set_xticklabels(['w1', 'w2', 'b'])
ax2.set_title('学到的参数对比')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d2l/ch03-linear/notes/scratch_vs_concise.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] notes/scratch_vs_concise.png — 从零 vs 简洁对比")


print("\n" + "=" * 60)
print("[OK] 3.3 线性回归简洁实现 — 完成")
print("     核心收获: 框架把 ~40 行手工代码压缩到 ~10 行")
print("     但理解了从零实现, 才知道每行 API 在背后做了什么")
print("=" * 60)
