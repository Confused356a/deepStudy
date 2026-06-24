"""
d2l 2.5 — 自动求导 (Automatic Differentiation)

PyTorch 的计算图会自动追踪前向运算，调用 backward() 反向传播求梯度。
这是整个深度学习框架的灵魂——无需手动推导反向传播公式。

核心概念: 计算图 → 动态图（define-by-run）、grad 累加、detach/no_grad
"""
import torch

print("=" * 50)
print("2.5.1 — 基础例子: y = 2x^T x 对 x 求导")
print("=" * 50)

# requires_grad=True 告诉 PyTorch 追踪这个 tensor 上的所有运算
x = torch.arange(4.0, requires_grad=True)
print(f"x = {x}")
print(f"x.requires_grad = {x.requires_grad}")
print(f"x.grad = {x.grad}  ← 初始为 None")

# 前向计算: y = 2 * x^T @ x = 2 * (x0² + x1² + x2² + x3²)
y = 2 * torch.dot(x, x)
print(f"\ny = 2 * dot(x, x) = {y.item()}")
# y = 2 * (0² + 1² + 2² + 3²) = 2 * 14 = 28

# 反向传播: ∂y/∂x = ∂(2∑x_i²)/∂x_i = 4x_i
y.backward()
print(f"x.grad = {x.grad}")
# 理论: 4x = [0, 4, 8, 12]
# 验证
assert torch.allclose(x.grad, torch.tensor([0., 4., 8., 12.])), "梯度计算错误！"
print("[OK] 梯度正确: [0, 4, 8, 12] = 4x")


print("\n" + "=" * 50)
print("2.5.2 — 非标量输出的 backward")
print("=" * 50)

# backward() 只能对标量调用（或传入与输出同形的 gradient 参数）
x = torch.arange(4.0, requires_grad=True)
y = x * x                      # y 是向量 (4,)，非标量

# 方式1: 先求和变成标量再 backward
y.sum().backward()
print(f"x.grad (via sum): {x.grad}")  # d(sum)/dx = 2x = [0, 2, 4, 6]

# 方式2: 传入 gradient 参数（用于 Jacobian 向量积）
# 注意: 之前的计算图已被 y.sum().backward() 释放，需重新建图
x2 = torch.arange(4.0, requires_grad=True)
y2 = x2 * x2
y2.backward(torch.ones_like(y2))  # 等价于 sum().backward()
print(f"x2.grad (via ones): {x2.grad}")


print("\n" + "=" * 50)
print("2.5.3 — grad 是累加的！必须手动清零")
print("=" * 50)

x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)

# 第一次 backward（retain_graph=True 保留计算图，否则第二次 backward 报错）
y.backward(retain_graph=True)
print(f"第一次 backward: x.grad = {x.grad}")

# 忘记清零直接再 backward —— grad 会累加！
y.backward(retain_graph=True)
print(f"第二次 backward (未清零): x.grad = {x.grad}  <- 累积了！")

# [OK] 正确做法: 每次 backward 前清零
x.grad.zero_()
y.backward()
print(f"清零后 backward: x.grad = {x.grad}  ← 恢复了")

# 这就是训练循环中 optimizer.zero_grad() 的原因！
print("\n→ 训练中 optimizer.zero_grad() 就是这个作用！")


print("\n" + "=" * 50)
print("2.5.4 — detach: 从计算图中分离")
print("=" * 50)

# detach() 返回一个共享数据但不需要梯度的 tensor
# 常用于: 固定部分参数、记录中间值而不断开梯度流
x = torch.arange(4.0, requires_grad=True)
y = x * x
u = y.detach()     # u 与 y 数值相同，但不需要梯度
z = u * x          # 梯度只流到 x (不经过 u → y)

z.sum().backward()
print(f"x.grad = {x.grad}")     # u 被视为常数，所以 dz/dx = u = x^2
print(f"期望: x^2 = {x**2}")


print("\n" + "=" * 50)
print("2.5.5 — torch.no_grad(): 临时关闭梯度")
print("=" * 50)

# 用在测试/推理阶段，省显存 + 加速
x = torch.arange(4.0, requires_grad=True)

with torch.no_grad():
    y = x * x                     # 不需要梯度，不建计算图
    print(f"y.requires_grad = {y.requires_grad}")  # False

print(f"x.requires_grad = {x.requires_grad}")       # True (不受影响)


print("\n" + "=" * 50)
print("2.5.6 — Python 控制流也能求导")
print("=" * 50)

# PyTorch 的计算图是动态的 (define-by-run)，if/for 都能求导
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

# 验证: f(a) = k * a, 其中 k 取决于 while 条件（运行时确定）
print(f"a = {a.item():.4f}")
print(f"d = {d.item():.4f}")
print(f"d/a = {d.item()/a.item():.4f}")
print(f"a.grad = {a.grad.item():.4f}")
print("[OK] Python 控制流也能求导——这是动态图的威力！")


print("\n" + "=" * 50)
print("[OK] 2.5 自动求导 — 完成")
print("=" * 50)
print("\n核心知识清单:")
print("  requires_grad=True   → 开始追踪")
print("  .backward()          → 反向传播（只能对 tensor 调用且必须清零）")
print("  .grad                → 存储梯度（会累加，需 zero_grad）")
print("  .detach()            → 从计算图分离（共享数据）")
print("  torch.no_grad():     → 临时关闭梯度（测试/推理用）")
print("  grad 累加机制        → 正是 optimizer.zero_grad() 存在的理由")
