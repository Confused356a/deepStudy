"""
d2l 3.5 — 图像分类数据集 (Image Classification Dataset)

Fashion-MNIST: 28×28 灰度图, 10 个服装类别, 70,000 样本。
相当于"深度学习的 MNIST 升级版"——比手写数字更难, 更有代表性。
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Fashion-MNIST 类别名
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print("=" * 60)
print("3.5.1 — 加载 Fashion-MNIST")
print("=" * 60)

# 数据预处理: PIL Image → Tensor (自动缩放到 [0, 1])
transform = transforms.ToTensor()

# 检查本地是否已有数据集 (缓存优先原则)
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_root = '../../datasets'
os.makedirs(data_root, exist_ok=True)

# 训练集
mnist_train = torchvision.datasets.FashionMNIST(
    root=data_root, train=True, transform=transform, download=True)
# 测试集
mnist_test = torchvision.datasets.FashionMNIST(
    root=data_root, train=False, transform=transform, download=True)

print(f"训练集大小: {len(mnist_train)}")
print(f"测试集大小: {len(mnist_test)}")
print(f"图像形状:   {mnist_train[0][0].shape}  (通道=1, 高=28, 宽=28)")
print(f"像素范围:   [{mnist_train[0][0].min():.2f}, {mnist_train[0][0].max():.2f}]")
print(f"标签范围:   0 ~ 9")


print("\n" + "=" * 60)
print("3.5.2 — 类别分布 & 样本展示")
print("=" * 60)

# 统计每类样本数
train_labels = mnist_train.targets.numpy()
class_counts = np.bincount(train_labels)

for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
    print(f"  {i}: {name:<14s} {count:>5d}  {'|' + '='*(count//100)}")

# 展示每个类别的一张样本
fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
axes = axes.flatten()

# 找每个类别的第一个样本
for class_idx in range(10):
    idx = np.where(train_labels == class_idx)[0][0]
    img, label = mnist_train[idx]
    axes[class_idx].imshow(img.squeeze(), cmap='gray')
    axes[class_idx].set_title(f'{label}: {CLASS_NAMES[label]}', fontsize=10)
    axes[class_idx].axis('off')

plt.suptitle('Fashion-MNIST: 每类一个样本', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'fashion_mnist_samples.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/fashion_mnist_samples.png")


print("\n" + "=" * 60)
print("3.5.3 — DataLoader: 批量加载")
print("=" * 60)

batch_size = 256
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

print(f"batch_size: {batch_size}")
print(f"train batches: {len(train_loader)}  ({len(mnist_train)}/{batch_size})")
print(f"test batches:  {len(test_loader)}  ({len(mnist_test)}/{batch_size})")

# 取一个 batch 验证
first_images, first_labels = next(iter(train_loader))
print(f"\n一个 batch: images {first_images.shape}, labels {first_labels.shape}")
print(f"images dtype: {first_images.dtype} (float32)")
print(f"labels dtype: {first_labels.dtype} (int64)")
print(f"前 10 个标签: {first_labels[:10].tolist()}")
print(f"  对应类别: {[CLASS_NAMES[l] for l in first_labels[:10].tolist()]}")


print("\n" + "=" * 60)
print("3.5.4 — 数据形状理解 (核心!)")
print("=" * 60)

print("原始形状: [batch, 1, 28, 28]")
print("  batch: 256   ← 每批 256 张图")
print("  1:     1      ← 单通道灰度")
print("  28:    28     ← 高度")
print("  28:    28     ← 宽度")
print()
print("Softmax 回归需要展平: [batch, 784]")
print("  784 = 28 × 28  ← 每个像素是一个特征")
print()
print("这是最简单的'图像模型': 把每个像素当成独立特征, 无视空间结构")
print("后面 CNN 会利用空间结构 (卷积核), 效果会好很多")


print("\n" + "=" * 60)
print("3.5.5 — 随机样本可视化 (带标签)")
print("=" * 60)

# 展示 4×4 = 16 张随机训练样本
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
axes = axes.flatten()
random_indices = np.random.choice(len(mnist_train), 16, replace=False)

for i, idx in enumerate(random_indices):
    img, label = mnist_train[idx]
    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].set_title(CLASS_NAMES[label], fontsize=9)
    axes[i].axis('off')

plt.suptitle('Fashion-MNIST: 随机 16 张训练样本', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'notes', 'fashion_mnist_grid.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[saved] notes/fashion_mnist_grid.png")


print("\n" + "=" * 60)
print("3.5.6 — 数据预处理流水线 (思考)")
print("=" * 60)

print("当前: transforms.ToTensor()  ← 仅转成 tensor [0,1]")
print()
print("常见增强 (后面章节会用到):")
print("  transforms.RandomHorizontalFlip()   随机水平翻转")
print("  transforms.RandomCrop(28, padding=4) 随机裁剪")
print("  transforms.Normalize(mean, std)      标准化 (像 ImageNet)")
print()
print("数据预处理 = 深度学习工程的半壁江山")


print("\n" + "=" * 60)
print("[OK] 3.5 图像分类数据集 — 完成")
print("     Fashion-MNIST: 70k 样本, 10 类, 28×28 灰度")
print("     DataLoader 批量加载, 为 3.6 训练做准备")
print("=" * 60)
