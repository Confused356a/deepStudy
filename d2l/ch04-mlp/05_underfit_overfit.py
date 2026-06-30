"""
d2l 4.4 — 模型选择、欠拟合与过拟合 (Underfitting & Overfitting)

深度学习最核心的问题之一: 模型复杂度 vs 泛化能力.

三种现象:
    欠拟合 (Underfit):  模型太简单, 训练误差和测试误差都高
    正常 (Good fit):     模型复杂度适中, 测试误差接近训练误差
    过拟合 (Overfit):    模型太复杂, 训练误差很低但测试误差很高
                         (记住了训练数据的噪声, 而不是学习规律)

实验设计: 多项式回归 → 阶数 = 模型容量
    - 低阶 (1): 欠拟合 — 直线拟合曲线
    - 中阶 (4): 正常 — 接近真实函数
    - 高阶 (20): 过拟合 — 完美穿过训练点但在别处剧烈摆动
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("模型选择: 欠拟合 vs 过拟合")
print("=" * 60)

# ============================================================
# 1. 生成数据: sin(x) + noise
# ============================================================
torch.manual_seed(0)
n_train, n_test = 80, 80
true_func = lambda x: torch.sin(2 * np.pi * x)  # 真实函数: sin(2πx)

# 训练数据: 加噪声
X_train = torch.rand(n_train, 1) * 2 - 1  # [-1, 1]
y_train = true_func(X_train) + 0.2 * torch.randn(n_train, 1)

# 测试数据: 真实函数 (无噪声, 用于评估泛化)
X_test = torch.linspace(-1, 1, n_test).reshape(-1, 1)
y_test = true_func(X_test)

print(f"  训练集: {n_train} 点 (带噪声)")
print(f"  测试集: {n_test} 点 (无噪声, 真实函数)")
print(f"  真实函数: y = sin(2πx)")


# ============================================================
# 2. 多项式特征构造
# ============================================================
def poly_features(X, degree):
    """构造多项式特征: [x] → [1, x, x^2, ..., x^degree]"""
    features = [torch.ones_like(X)]
    for d in range(1, degree + 1):
        features.append(X ** d)
    return torch.cat(features, dim=1)


# ============================================================
# 3. 训练不同阶数的多项式
# ============================================================
print("\n训练 1/4/20 阶多项式模型...")

degrees = [1, 4, 20]
results = {}

for degree in degrees:
    # 特征转换
    X_poly = poly_features(X_train, degree)

    # 正规方程求解: w* = (X^T X)^(-1) X^T y
    # 加一个小正则项保证数值稳定
    I = torch.eye(degree + 1) * 1e-4
    w = torch.linalg.solve(X_poly.T @ X_poly + I, X_poly.T @ y_train)

    # 训练和测试预测
    X_test_poly = poly_features(X_test, degree)
    y_train_pred = X_poly @ w
    y_test_pred = X_test_poly @ w

    train_mse = torch.nn.functional.mse_loss(y_train_pred, y_train).item()
    test_mse = torch.nn.functional.mse_loss(y_test_pred, y_test).item()

    results[degree] = {
        'w': w, 'train_mse': train_mse, 'test_mse': test_mse,
        'y_pred': y_test_pred
    }
    print(f"  degree={degree:2d}: train_mse={train_mse:.6f}, test_mse={test_mse:.6f}")


# ============================================================
# 4. 可视化: 三个模型对比
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, degree in zip(axes, degrees):
    r = results[degree]

    # 训练数据点
    ax.scatter(X_train, y_train, s=15, alpha=0.5, color='gray', label='训练数据 (带噪)')

    # 真实函数
    ax.plot(X_test.flatten(), y_test.flatten(), 'k--', linewidth=2, alpha=0.7, label='真实 sin(2πx)')

    # 模型预测
    ax.plot(X_test.flatten(), r['y_pred'].flatten(), 'r-', linewidth=2,
            label=f'多项式({degree}阶)')

    ax.set_xlabel('x'); ax.set_ylabel('y')
    status = '欠拟合' if degree == 1 else ('正常' if degree == 4 else '过拟合')
    ax.set_title(f'{status}\n{degree}阶, train={r["train_mse"]:.4f}, test={r["test_mse"]:.4f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'underfit_overfit.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[saved] notes/underfit_overfit.png — 欠拟合(1阶) / 正常(4阶) / 过拟合(20阶)")


# ============================================================
# 5. 全面扫描: 1~20 阶的 train/test 误差曲线
# ============================================================
print("\n扫描 1~20 阶, 画训练/测试误差曲线...")

scan_degrees = range(1, 21)
train_errors, test_errors = [], []

for degree in scan_degrees:
    X_poly = poly_features(X_train, degree)
    I = torch.eye(degree + 1) * 1e-4
    w = torch.linalg.solve(X_poly.T @ X_poly + I, X_poly.T @ y_train)

    train_mse = torch.nn.functional.mse_loss(X_poly @ w, y_train).item()

    X_test_poly = poly_features(X_test, degree)
    y_test_pred = X_test_poly @ w
    test_mse = torch.nn.functional.mse_loss(y_test_pred, y_test).item()

    train_errors.append(train_mse)
    test_errors.append(test_mse)

# 找出最优阶数
best_degree = scan_degrees[np.argmin(test_errors)]
print(f"  最优阶数: {best_degree} (test_mse={min(test_errors):.6f})")
print(f"  此后 test error 开始上升 → 过拟合")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 误差曲线
ax = axes[0]
ax.semilogy(list(scan_degrees), train_errors, 'b-o', markersize=6, label='训练误差')
ax.semilogy(list(scan_degrees), test_errors, 'r-s', markersize=6, label='测试误差')
ax.axvline(x=best_degree, color='gray', linestyle='--', linewidth=1.5,
           label=f'最优阶数={best_degree}')
ax.set_xlabel('多项式阶数 (模型容量)'); ax.set_ylabel('MSE (log)')
ax.set_title('训练误差 vs 测试误差 — 阶数增加')
ax.legend(); ax.grid(True, alpha=0.3)

# 泛化差距 (generalization gap = test - train)
gap = np.array(test_errors) - np.array(train_errors)
axes[1].bar(list(scan_degrees), gap, color='orange', edgecolor='black')
axes[1].axhline(y=0, color='gray', linewidth=1)
axes[1].set_xlabel('多项式阶数'); axes[1].set_ylabel('泛化差距 (test - train)')
axes[1].set_title('泛化差距随容量增大而增大')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'overfit_error_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/overfit_error_curves.png — 1~20 阶误差曲线 + 泛化差距")


# ============================================================
# 6. 数据量对过拟合的影响
# ============================================================
print("\n数据量对过拟合的影响...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
n_list = [10, 40, 160]
degree = 18  # 高阶模型

for ax, n_data in zip(axes, n_list):
    X_n = X_train[:n_data]
    y_n = y_train[:n_data]

    X_poly = poly_features(X_n, degree)
    I = torch.eye(degree + 1) * 1e-4
    w = torch.linalg.solve(X_poly.T @ X_poly + I, X_poly.T @ y_n)

    X_test_poly = poly_features(X_test, degree)
    y_pred = X_test_poly @ w

    test_mse = torch.nn.functional.mse_loss(y_pred, y_test).item()

    ax.scatter(X_n, y_n, s=20, alpha=0.6, color='gray', label=f'{n_data} 训练点')
    ax.plot(X_test.flatten(), y_test.flatten(), 'k--', linewidth=1.5, alpha=0.5, label='真实')
    ax.plot(X_test.flatten(), y_pred.flatten(), 'r-', linewidth=2, label=f'18阶拟合 test={test_mse:.4f}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'{n_data} 个训练点 → test_mse={test_mse:.4f}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 2)

plt.suptitle('同一高阶模型(18阶), 数据越多 → 过拟合越轻', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'data_size_vs_overfit.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/data_size_vs_overfit.png — 10/40/160 点, 18 阶模型")


# ============================================================
# 关键总结
# ============================================================
print("\n" + "=" * 60)
print("关键总结")
print("=" * 60)
print("  1. 欠拟合: 模型太简单 → train 和 test 都高 (增加容量)")
print("  2. 过拟合: 模型太复杂 → train 低但 test 高 (需要正则化/更多数据)")
print("  3. 模型容量 ↑ → 训练误差单调 ↓, 测试误差先 ↓ 再 ↑ (U 形)")
print("  4. 泛化差距 = test_error - train_error, 随容量增大而扩大")
print("  5. 更多训练数据 → 缓解过拟合 (不修改模型的前提下)")
print("  6. 实践中: 看 train/val 误差曲线判断欠拟合还是过拟合")
