# 04-CNN 学习日志

> 从零开始，逐组件理解卷积神经网络。基于 PyTorch + CIFAR-10。

---

## 学习路线图

```
Conv2d → MaxPool2d → ReLU/Sigmoid → Linear → Sequential → Loss → 训练循环 → SGD优化器 → 预训练模型
 01        02           03           04         05         06       07          08           09
```

---

## 01 — Conv2d 卷积层

**文件：** `01_conv2d_cifar10.py`

**知识点：**
- `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`
- 输出尺寸公式：**H_out = (H_in + 2P − K) / S + 1**
  - 例：32×32 输入，K=3, P=0, S=1 → 30×30
- **in_channels** = 输入图像的通道数（RGB=3）
- **out_channels** = 卷积核个数，也是输出特征图的通道数（可任意设，这里设 6）
- CIFAR-10 图像 shape：`[batch, 3, 32, 32]`
- TensorBoard `add_images` 可视化输入/输出（输出取前 3 通道映射到 RGB）

**实验：** 纯 Conv2d 输出 `[64, 6, 30, 30]`，确认尺寸计算正确。

---

## 02 — MaxPool2d 池化层

**文件：** `02_pooling_cifar10.py`

**知识点：**
- `nn.MaxPool2d(kernel_size, ceil_mode)`
- 输出尺寸公式同卷积：**H_out = (H_in − K) / S + 1**（S 默认 = K）
- **ceil_mode=True**：向上取整，保留边缘信息（默认 False 向下取整会丢弃边缘）
- 先用小 tensor `[1,1,5,5]` 手动验证，再上 CIFAR-10
- 池化本质：**保留局部最强信号，减少数据量，提取关键特征**
- `dtype=torch.float32` 必须写，整数 tensor 进 MaxPool2d 会报错

**坑：** 数据集路径 `"datasets"` vs `"../../datasets"` 不一致会导致重复下载。

---

## 03 — 激活函数 ReLU / Sigmoid

**文件：** `03_relu.py`

**知识点：**
- `nn.ReLU()` — 负值置 0，正值不变。简单高效，缓解梯度消失
- `nn.Sigmoid()` — 压缩到 (0,1)，两端梯度趋近 0，深层网络容易梯度消失
- 激活函数引入**非线性**，没有它多层网络等价于单层线性变换
- 同样用 TensorBoard 可视化激活前后对比

**注意：** 此时只在 CIFAR-10 图像上跑激活函数看效果，还没嵌入网络。

---

## 04 — Linear 全连接层

**文件：** `04_linear.py`

**知识点：**
- `nn.Linear(in_features, out_features)` — 矩阵乘法 y = xW^T + b
- CIFAR-10 图像展平：`torch.flatten(imgs, start_dim=1)` → `[batch, 3072]`
- **3072 = 3 × 32 × 32**
- Linear(3072, 10)：将 3072 维像素直接映射到 10 个类别 logits
- 纯 Linear 没有卷积，没有非线性，表达能力有限

**输出验证：** `[64, 3, 32, 32]` → flatten → `[64, 3072]` → Linear → `[64, 10]` ✅

---

## 05 — Sequential 容器

**文件：** `05_seq.py`

**知识点：**
- `nn.Sequential()` 把多层打包成一个模块，forward 自动按顺序执行
- 对照写法：定义独立层 vs 放进 Sequential
- 3 层 CNN 架构：Conv→Pool → Conv→Pool → Conv→Pool → Flatten → Linear → Linear
- 尺寸推算：32 → 32(pad) → 16(pool) → 16(pad) → 8(pool) → 8(pad) → 4(pool) → 1024 → 64 → 10
- **1024 = 64 × 4 × 4**（pool3 后：64 通道 × 4×4）
- `writer.add_graph(model, input)` 把计算图写入 TensorBoard，可视化网络结构

**经验：** `add_graph` 报错时翻墙试试——国外 IP 能连 TensorBoard 的 GitHub 资源。

---

## 06 — 损失函数

**文件：** `06_loss.py`

**知识点：**
- `nn.L1Loss(reduction='sum')` — MAE，|pred − target| 之和，对异常值不敏感
- `nn.MSELoss()` — MSE，(pred − target)² 均值，放大异常值惩罚
- **dtype 陷阱**：`torch.tensor([1,2,3])` 默认 int64，损失函数要求 float32，报 `NotImplementedError for Long`
  - 解决：加 `dtype=torch.float32`
- 先在小 tensor `[1,1,1,3]` 上验证，再上真实数据

**对比：** L1Loss(sum)=2.0, MSELoss=1.333

---

## 07 — 完整训练循环

**文件：** `07_loss_network.py`

**知识点：**
- **CrossEntropyLoss**：内置 LogSoftmax + NLLLoss，`outputs` 传 raw logits（不加 softmax），`targets` 传 Long 类型类别索引
- **SGD 优化器**：`torch.optim.SGD(model.parameters(), lr=0.01)`
- **训练四步曲**：`zero_grad → forward → loss → backward → step`
- `running_loss += loss.item()` — 用 `.item()` 取标量，断开计算图
- TensorBoard `add_scalar("train/loss", loss.item(), step)` 记录 loss 曲线
- 2 epoch，avg loss 从 2.31 降到 2.25

**关键发现：** 没有 ReLU 的纯卷积+线性层，loss 下降极慢——缺少非线性，网络表达能力受限。

---

## 08 — SGD 优化器深入

**文件：** `08_optim.py`

**知识点：**
- `torch.optim.SGD(params, lr=0.01)` — 随机梯度下降
- `optim.zero_grad()` — 清零梯度（否则跨 batch 累加）
- `optim.step()` — w = w − lr × grad
- 20 epoch 循环观察 loss 趋势
- **坑：** `running_loss += result_loss`（保留 tensor）会保留计算图，越累越大可能爆显存
  - 正确：`running_loss += loss.item()`

---

## 09 — 预训练模型 VGG16

**文件：** `09_model_pretrained.py`

**知识点：**
- `torchvision.models.vgg16(pretrained=True)` — 下载 ImageNet 预训练权重（~528MB）
- `pretrained=False` — 仅网络结构，参数随机初始化
- ImageNet 1000 类 → CIFAR-10 只需 10 类，需要修改输出层
- `model.add_module(name, layer)` — 动态添加层到 `_modules`
- **关键问题：** VGG16 的 `forward()` 固定走 `features → classifier`，add_module 加的层不会被调用
  - 正确做法：`model.classifier[6] = nn.Linear(4096, 10)` 替换分类头

---

## 贯穿全程的工程经验

| 经验 | 来源 |
|------|------|
| 数据集先检查 `datasets/` 是否已有，避免重复下载 | 01~09 通用 |
| `torch.tensor()` 默认 int64，损失函数要 float32 | 06 |
| `pip` 的 `nn` 包会覆盖 `torch.nn`，必须写 `import torch.nn as nn` | 环境 |
| TensorBoard 可视化计算图、特征图、loss 曲线 | 01,02,03,05,07 |
| Git Bash 中 conda 不在 PATH，用完整路径调 python | 环境 |
| 先在小 tensor 上验证逻辑，再上真实数据 | 02,06 |

---

## 下一步方向

- [ ] ReLU 嵌入网络（目前 07/08 的模型没有激活函数）
- [ ] GPU 训练（`.cuda()`，RTX 3070）
- [ ] 准确率计算（`argmax` + 比较）
- [ ] 验证集切分 + model.eval()
- [ ] 预训练模型迁移学习（替换分类头 + 微调）

---

*最后更新：2026-06-22*
